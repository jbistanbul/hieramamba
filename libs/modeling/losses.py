import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Callable

from libs.dist_utils import print0
from .contrastive_losses import contrastive_subsample_negative_mp


def build_single_level_loss(loss_type: str) -> Callable:
    if loss_type == 'contr_mp':
        return lambda *args, **kw: contrastive_subsample_negative_mp(*args, **kw)
    raise ValueError(f"Unsupported released contrastive type: {loss_type}")


class MultiScaleMaskedContrastive(nn.Module):
    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        contrastive_type = opt.get('contr_type', 'contr_mp')
        self.loss_fn = build_single_level_loss(contrastive_type)
        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 0.20)
        self.gap_ratio = opt.get('gap_ratio', 0.30)
        self.radius = opt.get('radius', 0)
        self.hard_neg = opt.get('hard_neg', False)
        self.cross_video_neg = opt.get('cross_video_neg', False)
        proj_outdim = opt.get('proj_outdim', 256)
        proj_expand = opt.get('proj_expand', 1.0)
        proj_num_layers = opt.get('proj_num_layers', 2)
        self.projector = LNProjector(
            in_dim=vid_embd_dim,
            out_dim=proj_outdim,
            expand=proj_expand,
            num_layers=proj_num_layers,
        )
        print0(
            f'Using Contrastive Loss of type: {contrastive_type}, '
            f'temp: {self.temp}, neg_ratio: {self.neg_ratio}, gap_ratio: {self.gap_ratio}, '
            f'radius: {self.radius}, hard_neg: {self.hard_neg}, '
            f'cross_video_neg: {self.cross_video_neg}, weight: {opt.get("weight", 1.0)}'
        )

    def forward(
        self,
        sequence_fpn: Tuple[Tensor],
        sequence_fpn_mask: Tuple[Tensor],
        anchor_fpn: Tuple[Tensor],
        anchor_fpn_mask: Tuple[Tensor],
    ) -> Tensor:
        total = torch.tensor(0.0, device=sequence_fpn[0].device)
        for seq, seq_m, anc, anc_m in zip(
            sequence_fpn, sequence_fpn_mask,
            anchor_fpn, anchor_fpn_mask,
        ):
            total = total + self.loss_fn(
                anchors=anc,
                seq_tokens=seq,
                anchor_mask=anc_m,
                seq_mask=seq_m,
                projector=self.projector,
                temperature=self.temp,
                neg_ratio=self.neg_ratio,
                gap_ratio=self.gap_ratio,
                radius=self.radius,
                hard_neg=self.hard_neg,
                cross_video_neg=self.cross_video_neg,
            )
        return total


class MultiScaleMaskedGTPointContrastive(nn.Module):
    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        contrastive_type = opt.get('contr_type', 'point_gt_contr_pooled')
        if contrastive_type != 'point_gt_contr_pooled':
            raise ValueError(
                'Unsupported released GT contrastive type: '
                f'{contrastive_type}. The code release only supports point_gt_contr_pooled.'
            )

        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 1.0)
        self.use_projector = opt.get('use_projector', True)
        if self.use_projector:
            proj_outdim = opt.get('proj_outdim', 256)
            proj_expand = opt.get('proj_expand', 1.0)
            proj_num_layers = opt.get('proj_num_layers', 2)
            self.projector = LNProjector(
                in_dim=vid_embd_dim,
                out_dim=proj_outdim,
                expand=proj_expand,
                num_layers=proj_num_layers,
            )
        else:
            self.projector = None
            print0('No projector used')

        print0(
            f'Using GT Contrastive Loss of type: {contrastive_type}, '
            f'temp: {self.temp}, neg_ratio: {self.neg_ratio}, '
            f'span_contr_gt: {opt.get("span_contr_gt", False)}, weight: {opt.get("weight", 1.0)}'
        )

    def forward(
        self,
        fpn_fused: Tuple[Tensor],
        fpn_fused_masks: Tuple[Tensor],
        gt_labels: Tuple[Tensor],
        gt_labels_span_list: Tuple[Tensor],
    ) -> Tensor:
        total = torch.tensor(0.0, device=fpn_fused[0].device)
        for fpn, fpn_mask, gt_label, gt_labels_span in zip(
            fpn_fused, fpn_fused_masks, gt_labels, gt_labels_span_list
        ):
            total = total + self._compute_infonce_loss_pooled(
                fpn, fpn_mask, gt_label, gt_labels_span
            )
        return total

    def _compute_infonce_loss_pooled(
        self,
        fpn: Tensor,
        fpn_mask: Tensor,
        gt_label: Tensor,
        gt_labels_span: Tensor,
    ) -> Tensor:
        device = fpn.device
        total_loss = torch.zeros(1, device=device)
        total_anchors = 0

        for batch_idx in range(fpn.shape[0]):
            valid_idx = fpn_mask[batch_idx].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            if self.use_projector and self.projector is not None:
                proj = self.projector(fpn[batch_idx, :, valid_idx].T).T
            else:
                proj = F.normalize(fpn[batch_idx, :, valid_idx], p=2, dim=0)
            proj = F.normalize(proj, p=2, dim=0)

            labels = gt_label[batch_idx, valid_idx]
            pos_idx = labels.nonzero(as_tuple=True)[0]
            neg_idx = torch.logical_and(
                ~labels,
                ~gt_labels_span[batch_idx, valid_idx],
            ).nonzero(as_tuple=True)[0]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            k_neg = int(pos_idx.numel() * self.neg_ratio)
            if neg_idx.numel() > k_neg:
                sel = torch.randperm(neg_idx.numel(), device=device)[:k_neg]
                neg_idx = neg_idx[sel]
            if neg_idx.numel() == 0:
                continue

            anchor = proj[:, pos_idx].mean(dim=1, keepdim=True)
            anchor = F.normalize(anchor, p=2, dim=0)

            sim_pos = (anchor.T @ proj[:, pos_idx]) / self.temp
            sim_neg = (anchor.T @ proj[:, neg_idx]) / self.temp
            log_pos = torch.logsumexp(sim_pos, dim=1)
            log_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=1), dim=1)
            total_loss += -(log_pos - log_all)
            total_anchors += 1

        if total_anchors == 0:
            return torch.tensor(0.0, device=device)
        return (total_loss / total_anchors).squeeze()


class LNProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, expand: float = 2.0, num_layers: int = 2):
        super().__init__()
        if num_layers not in [1, 2, 3]:
            raise ValueError(f'num_layers must be 1, 2, or 3, got {num_layers}')

        self.num_layers = num_layers
        hid_dim = int(expand * in_dim)

        if num_layers == 1:
            self.fc = nn.Linear(in_dim, out_dim)
        elif num_layers == 2:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(hid_dim, out_dim)
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
        else:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(hid_dim, hid_dim, bias=False)
            self.ln2 = nn.LayerNorm(hid_dim)
            self.act2 = nn.GELU()
            self.fc3 = nn.Linear(hid_dim, out_dim)
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
            nn.init.ones_(self.ln2.weight)
            nn.init.zeros_(self.ln2.bias)

    def forward(self, x):
        if self.num_layers == 1:
            z = self.fc(x)
        elif self.num_layers == 2:
            z = self.fc2(self.act1(self.ln1(self.fc1(x))))
        else:
            h1 = self.act1(self.ln1(self.fc1(x)))
            h2 = self.act2(self.ln2(self.fc2(h1)))
            z = self.fc3(h2)
        return F.normalize(z, p=2, dim=1)
