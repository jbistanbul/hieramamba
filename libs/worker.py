from collections import OrderedDict
from copy import deepcopy
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler, MultiScaleMaskedContrastive, MultiScaleMaskedGTPointContrastive
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str, generate_multiscale_gt_masks, generate_multiscale_gt_masks_contrastive

from .modeling.model import make_models_net
from torch.cuda.amp import autocast
import json

AUX_LOSS_REGISTRY = {
    'ds_contrastive': MultiScaleMaskedContrastive,
    'gt_point_contrastive': MultiScaleMaskedGTPointContrastive,
}



class TrainerOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 20))
        # rng = None
        # print('no seed set')
        # build model and EMA
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()],) #find_unused_parameters=True)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True) # (bs, c_v, t)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list]) # (bs * num_queries, 2)
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs * num_queries, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs * num_queries, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs * num_queries, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_adaptive(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved2(points, targets)
        
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets
    
    def _annotate_points_per_video_fine_scale_fix(self, points, target):
        """
        Conservative fine-scale fix that reduces regression loss.
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # (1) Standard center sampling
        if self.center_sampling == 'radius':
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        strides = points[:, 3]
        
        # More conservative approach: only minimal expansion for very short segments
        fine_scale_mask = strides <= 4
        very_short_segment = segment_length < 15  # Only help very short segments
        
        # Only expand upper bound, and only modestly
        adapted_reg_max = torch.where(
            fine_scale_mask & very_short_segment,
            points[:, 2] * 1.3,  # Only 30% expansion vs your 200%+ expansion
            points[:, 2]  # Keep original for others
        )
        
        # Keep original minimum to avoid too many low-quality positives
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1],  # Original minimum
            max_reg_dist < adapted_reg_max  # Slightly expanded maximum
        )
        
        # Additional quality filter for very short segments
        if segment_length < 15:
            # Only keep points with reasonable symmetry to reduce regression difficulty
            asymmetry_ratio = torch.abs(pt2start - pt2end) / torch.maximum(pt2start, pt2end)
            symmetry_filter = asymmetry_ratio < 0.8  # Allow up to 80% asymmetry
            inside_range = torch.logical_and(inside_range, symmetry_filter)
        
        labels = torch.logical_and(inside_window, inside_range)
        return labels, offsets
    
    def _annotate_points_improved2(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video_fine_scale_fix(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets
    
    def _annotate_points_per_video_short_segments(self, points, target):
        """
        Improved annotation method for very short segments (e.g., length ~10 in videos of length ~900).
        Uses more conservative and targeted improvements.
        
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # Calculate segment length and center
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])

        # (1) Multi-scale aware center sampling
        if self.center_sampling == 'radius':
            base_radius = points[:, 3] * self.center_sampling_radius
            
            # Strategy 1: Scale-adaptive radius with conservative expansion
            stride_ratio = segment_length / points[:, 3]
            
            # Only boost for very fine scales where segment is smaller than 2x stride
            scale_boost = torch.where(
                stride_ratio < 1.0,  # Very short relative to stride
                torch.clamp(1.0 / torch.sqrt(stride_ratio), 1.0, 1.5),  # Conservative sqrt-based boost
                torch.ones_like(stride_ratio)
            )
            
            # Strategy 2: Ensure minimum effective radius but cap it
            min_effective_radius = torch.minimum(
                segment_length * 0.2,  # 20% of segment length
                points[:, 3] * 0.5     # Or half the stride, whichever is smaller
            )
            
            adaptive_radius = torch.maximum(base_radius * scale_boost, min_effective_radius)
            
            # Apply adaptive radius
            t_min = (ctr - adaptive_radius).clamp_(min=target[0])
            t_max = (ctr + adaptive_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        
        # Strategy 3: Gradual regression range relaxation based on segment/stride ratio
        if segment_length < points[:, 3].min() * 2:  # Only for very short segments
            # Conservative expansion: only 25% relaxation
            relaxation_factor = 1.25
            expanded_reg_min = points[:, 1] / relaxation_factor
            expanded_reg_max = points[:, 2] * relaxation_factor
            
            # Use weighted combination of original and relaxed ranges
            weight = torch.minimum(segment_length / (points[:, 3] * 2), torch.tensor(1.0))
            final_reg_min = weight * points[:, 1] + (1 - weight) * expanded_reg_min
            final_reg_max = weight * points[:, 2] + (1 - weight) * expanded_reg_max
            
            inside_range = torch.logical_and(
                max_reg_dist >= final_reg_min, max_reg_dist < final_reg_max
            )
        else:
            # Normal regression range constraint
            inside_range = torch.logical_and(
                max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
            )

        # Strategy 4: Additional quality filtering for very short segments
        if segment_length < 20:
            # Prefer points closer to segment center for very short segments
            dist_to_center = torch.abs(points[:, 0] - ctr)
            center_weight = torch.exp(-dist_to_center / (segment_length * 0.5))
            
            # Only keep points with reasonable center alignment (top 80% by center weight)
            center_thresh = torch.quantile(center_weight[inside_window], 0.2) if inside_window.sum() > 5 else 0.0
            center_filter = center_weight >= center_thresh
            
            labels = torch.logical_and(
                torch.logical_and(inside_window, inside_range),
                center_filter
            )
        else:
            labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _annotate_points_ultra_short(self, points, target):
        """
        Specialized handling for ultra-short segments (< 5 time units).
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # Very tight center sampling
        if self.center_sampling == 'radius':
            # Use fixed small radius for ultra-short segments
            tight_radius = torch.minimum(
                segment_length * 0.3,  # 30% of segment
                points[:, 3] * 0.3     # 30% of stride
            )
            
            t_min = (ctr - tight_radius).clamp_(min=target[0])
            t_max = (ctr + tight_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # Strict regression range - only minimal relaxation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1] * 0.8,  # Minimal relaxation
            max_reg_dist < points[:, 2] * 1.2
        )
        
        # Only keep the most relevant scale levels for ultra-short segments
        scale_filter = points[:, 3] <= segment_length * 2  # Only use fine scales
        
        labels = torch.logical_and(
            torch.logical_and(inside_window, inside_range),
            scale_filter
        )
        
        return labels, offsets

    def _annotate_points_multi_strategy(self, points, target):
        """
        Multi-strategy annotation that combines different approaches based on segment characteristics.
        """
        segment_length = target[1] - target[0]
        
        # Strategy selection based on segment length
        if segment_length < 5:
            # Ultra-short segments: very conservative, focus on highest quality points
            return self._annotate_points_ultra_short(points, target)
        elif segment_length < 20:
            # Short segments: use improved conservative method
            return self._annotate_points_per_video_short_segments(points, target)
        else:
            # Normal segments: use original method
            return self._annotate_points_per_video(points, target)

    def _annotate_points_improved(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_multi_strategy(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state_ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()





class TrainerAuxiliary(TrainerOriginal):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        vid_embd_dim = opt['model']['vid_net']['embd_dim']
        self.ds_contrastive = opt['train']['loss_aux']['ds_contrast']['enable']
        self.gt_contrastive = opt['train']['loss_aux']['gt_contrast']['enable']
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        if self.early_fusion:
            self.logger.write("Early fusion enabled")

        if self.ds_contrastive:
            ds_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['ds_contrast'].get('type', 'ds_contrastive')]
            self.ds_contrastive_loss = ds_loss_type(opt['train']['loss_aux']['ds_contrast'], vid_embd_dim).cuda()
            self.ds_contrastive_weight = opt['train']['loss_aux']['ds_contrast']['weight']
        else: 
            self.ds_contrastive_weight = 0.0
        
        if self.gt_contrastive:
            gt_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['gt_contrast'].get('type', 'gt_point_contrastive')]
            self.gt_contrastive_loss = gt_loss_type(opt['train']['loss_aux']['gt_contrast'], vid_embd_dim).cuda()
            self.gt_contrastive_weight = opt['train']['loss_aux']['gt_contrast']['weight']
            self.loss_aux_gt_type = opt['train']['loss_aux']['gt_contrast'].get('gt_type', 'point')
            self.loss_aux_span_radius = opt['train']['loss_aux']['gt_contrast'].get('span_radius', self.center_sampling_radius)
            self.span_contr_gt = opt['train']['loss_aux']['gt_contrast'].get('span_contr_gt', False)       
        else:
            self.gt_contrastive_weight = 0.0

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = ds_contrast = gt_contrast = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            ds_contrast += loss_dict['ds_contrast']
            gt_contrast += loss_dict['gt_contrast']
        
        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'ds_contrast': ds_contrast, 'gt_contrast': gt_contrast, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks= \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks= \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        if self.ds_contrastive:
            ds_contrastive_loss = self.ds_contrastive_loss(
                fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks
            ) / self.loss_norm * get_world_size()
            # normalize by number of queries if not early fusion
            if not self.early_fusion:
                ds_contrastive_loss = text_size.float().mean() * ds_contrastive_loss
        else:
            ds_contrastive_loss = torch.tensor(0.0).cuda()

        if self.gt_contrastive:
            gt_labels_split = gt_labels.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            fpn_masks_split = fpn_masks.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            # fpn_logits_split = fpn_logits.split(fpn_n_points, dim=1)

            # gt labels SPAN
            gt_labels_span = generate_multiscale_gt_masks(targets, fpn_n_points)
            gt_labels_span = gt_labels_span.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # gt labels SPAN CONTRASTIVE
            gt_labels_span_contrastive = generate_multiscale_gt_masks_contrastive(points, targets, self.loss_aux_span_radius)
            gt_labels_span_contrastive = gt_labels_span_contrastive.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # replace gt labels with contrastive gt labels that was sampled with configured radius
            if self.span_contr_gt:
                gt_labels_split = gt_labels_span_contrastive
            
            # Expand fpn from (bs, d, T_l) to (bs*num_queries, d, T_l) for non-early fusion
            if not self.early_fusion:
                fpn_expanded = tuple(torch.repeat_interleave(fpn_layer, text_size, dim=0) for fpn_layer in fpn)
                # fpn_expanded = fpn
            else:
                fpn_expanded = fpn
            
            gt_contrastive_loss = self.gt_contrastive_loss(
                fpn_expanded, fpn_masks_split, gt_labels_split, gt_labels_span
            ) / self.loss_norm * get_world_size()

            # not masking entire gt span
            # gt_contrastive_loss = self.gt_contrastive_loss(
            #     fpn, fpn_masks_split, gt_labels_split, gt_labels_split
            # ) / self.loss_norm * get_world_size()
        else:
            gt_contrastive_loss = torch.tensor(0.0).cuda()

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        if self.use_mst:
            fpn_logits2 = torch.cat(fpn_logits2, dim=1) # (bs, p)
            cls_loss2 = self._calc_focal_loss(
                logits=fpn_logits2[fpn_masks], labels=gt_labels[fpn_masks]
            ) / self.loss_norm * get_world_size()
            cls_loss = (cls_loss + cls_loss2) / 2
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss + \
            self.ds_contrastive_weight * ds_contrastive_loss + \
            self.gt_contrastive_weight * gt_contrastive_loss
        total_loss.backward()

        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'ds_contrast': ds_contrastive_loss.detach(),
            'gt_contrast': gt_contrastive_loss.detach()
        }

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            if k == 'ds_contrast' or k == 'gt_contrast':
                log_str += f"{k} {float(v.item()):.6f} | "
            else:
                log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()




class EvaluatorOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))
        
        # initialize prediction storage
        self.predictions = {}

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        self.max_text_len = opt['eval'].get('max_text_len', 24)
        self.batchify_text_queries = opt['eval'].get('batchify_text_queries', True)
        self.text_batch_size = opt['eval'].get('text_batch_size', 0)
        if self.batchify_text_queries:
            print("Batchify text queries for evaluation")
        else:
            print("Single text query processing for evaluation")

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print0("Evaluation started.")
        start_time = time.time()
        for data_list in self.dataloader:
            data = data_list[0]
            results = self.predict(data)
            targets = data['segment']
            vid_id = data['vid_id']
            assert len(results) == len(targets)

            # Store predictions for this video
            if vid_id not in self.predictions:
                self.predictions[vid_id] = {
                    'queries': [],
                    'recall_at_iou': {}
                }

            video_iou_counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
            
            for query_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                video_iou_counts += (iou_n[:, None] >= self.iou_threshs[None])
                
                # Store query predictions (top-5 predictions)
                top5_segs = segs[:5] if len(segs) >= 5 else segs
                top5_scores = scores[:5] if len(scores) >= 5 else scores
                
                query_data = {
                    'query_id': query_idx,
                    'ground_truth': target[0].cpu().numpy().tolist(),
                    'predictions': []
                }
                
                for seg, score in zip(top5_segs, top5_scores):
                    query_data['predictions'].append({
                        'segment': seg.cpu().numpy().tolist(),
                        'score': score.item()
                    })
                
                self.predictions[vid_id]['queries'].append(query_data)
            
            # Calculate recall at IoU for this video
            video_metrics = video_iou_counts / len(targets)
            for i, rank in enumerate(self.ranks):
                for j, thresh in enumerate(self.iou_threshs):
                    key = f"Rank@{rank}_IoU@{thresh:.1f}"
                    self.predictions[vid_id]['recall_at_iou'][key] = video_metrics[i, j].item()
            
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        end_time = time.time()
        self.log(is_last=True)
        completion_msg = f"Evaluation completed in {time_str(time.time() - start_time)}."
        print0(completion_msg)
        self.logger.write(completion_msg)
        
        # Save predictions to JSON file
        self.save_predictions()

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text'] # all text queries for the single video
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_masks, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_masks = text_masks.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_masks = self.model.encode_text2(text, text_masks, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )
        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(window, window_mask)
                fpn_logits_list, fpn_offsets_list = tuple(), tuple()
                if self.batchify_text_queries:
                    fpn_logits, fpn_offsets, _ = self.model.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)
                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    for text, text_mask in zip(text_list, text_mask_list):
                        fpn_logits, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )

            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks
    
    def _batchify_text2(self, text_list):
        bs = len(text_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return text, text_masks, text_size
    
    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
        self.logger.write(log_str)
    
    def save_predictions(self):
        """Save predictions to JSON file"""
        predictions_file = os.path.join(self.opt['_root'], f"predictions_{self.opt['_ckpt']}.json")
        
        # Add overall metrics to the predictions
        overall_metrics = self.counts / self.text_cnt
        summary = {
            'overall_recall_at_iou': {},
            'total_queries': int(self.text_cnt),
            'total_videos': len(self.predictions)
        }
        
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                key = f"Rank@{rank}_IoU@{thresh:.1f}"
                summary['overall_recall_at_iou'][key] = overall_metrics[i, j].item()
        
        output_data = {
            'summary': summary,
            'videos': self.predictions
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print0(f"Predictions saved to {predictions_file}")
        self.logger.write(f"Predictions saved to {predictions_file}")






class EvaluatorAuxiliary(EvaluatorOriginal):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        if self.early_fusion:
            self.logger.write("Early fusion enabled")
        
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_mask, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_mask = text_mask.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_mask = self.model.encode_text2(text, text_mask, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )


        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)



            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            with torch.no_grad():
                if self.batchify_text_queries:
                    if self.early_fusion:
                        window, window_mask = self.model.vid_proj(window, window_mask)
                        window, window_mask = self.model.fusion(window, window_mask, text, text_mask, text_size)
                    fpn, fpn_masks, _, _ = self.model.encode_video(window, window_mask)
                    
                    if self.use_mst:
                        fpn_logits, fpn_logits2, fpn_offsets, _ = \
                            self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask, text_size)
                    else:
                        fpn_logits, _, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask, text_size)

                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        # query_logits = tuple((fl1[query_idx:query_idx+1] + fl2[query_idx:query_idx+1]) / 2 for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        # query_logits = tuple(torch.maximum(fl1[query_idx:query_idx+1], fl2[query_idx:query_idx+1]) for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    window_orig = window.clone()
                    window_mask_orig = window_mask.clone()
                    for text, text_mask in zip(text_list, text_mask_list):
                        # window: (1, dim, T)
                        # window_mask: (1, 1, T)
                        if self.early_fusion:
                            window, window_mask = self.model.vid_proj(window_orig, window_mask_orig)
                            window, window_mask = self.model.fusion(window, window_mask, text, text_mask)
                        fpn, fpn_masks, _, _ = self.model.encode_video(window, window_mask)
                        if self.use_mst:
                            fpn_logits, fpn_logits2, fpn_offsets, _ = \
                                self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask)
                        else:
                            fpn_logits, _, fpn_offsets, _ = \
                                self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )
            # fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results