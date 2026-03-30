import torch
import torch.nn.functional as F


def contrastive_subsample_negative_mp(
    anchors: torch.Tensor,
    seq_tokens: torch.Tensor,
    anchor_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    projector: torch.nn.Module,
    radius: int = 0,
    temperature: float = 0.07,
    neg_ratio: float = 0.20,
    gap_ratio: float = 0.30,
    hard_neg: bool = False,
    cross_video_neg: bool = False,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE with either hard-negative top-M or random-negative M.
    This is the only released ds_contrast helper used by the current code release.
    """
    assert radius >= 0
    batch_size, dim, anchor_len = anchors.shape
    seq_len = seq_tokens.shape[2]
    device = anchors.device

    num_anchors = batch_size * anchor_len
    flat_anchors = anchors.permute(0, 2, 1).reshape(num_anchors, dim)
    flat_seq = seq_tokens.permute(0, 2, 1).reshape(batch_size * seq_len, dim)
    flat_anchor_mask = anchor_mask.squeeze(1).reshape(num_anchors)
    flat_seq_mask = seq_mask.squeeze(1).reshape(batch_size * seq_len)

    vid_id = torch.arange(num_anchors, device=device) // anchor_len
    time_id = torch.arange(num_anchors, device=device) % anchor_len

    z_seq = F.normalize(projector(flat_seq), dim=-1)

    positive_features = []
    positive_masks = []
    for offset in range(-radius, radius + 1):
        shifted_t = time_id + offset
        in_range = (0 <= shifted_t) & (shifted_t < anchor_len)

        even_idx = (vid_id * seq_len + 2 * shifted_t).clamp(0, batch_size * seq_len - 1)
        odd_idx = (even_idx + 1).clamp(0, batch_size * seq_len - 1)

        even_mask = in_range & flat_seq_mask[even_idx]
        odd_mask = in_range & flat_seq_mask[odd_idx]

        positive_features.extend([z_seq[even_idx], z_seq[odd_idx]])
        positive_masks.extend([even_mask, odd_mask])

    z_pos = torch.stack(positive_features, dim=1)
    pos_mask = torch.stack(positive_masks, dim=1)

    valid = flat_anchor_mask & (pos_mask.sum(1) > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_idxs = valid.nonzero(as_tuple=False).view(-1)
    z_anchor = F.normalize(projector(flat_anchors[valid_idxs]), dim=-1)
    z_pos = z_pos[valid_idxs]
    pos_mask = pos_mask[valid_idxs]

    valid_vid = vid_id[valid_idxs]
    valid_t = time_id[valid_idxs]
    num_valid = z_anchor.size(0)

    sim_pos = (z_anchor.unsqueeze(1) * z_pos).sum(-1) / temperature
    exp_pos = torch.exp(sim_pos) * pos_mask.float()
    exp_pos = exp_pos.sum(1)

    sim_all = torch.matmul(z_anchor, z_anchor.T) / temperature
    eye = torch.eye(num_valid, device=device, dtype=torch.bool)
    same_vid = valid_vid.unsqueeze(0).eq(valid_vid.unsqueeze(1))

    gap = max(1, int(gap_ratio * anchor_len))
    near = same_vid & (torch.abs(valid_t.unsqueeze(0) - valid_t.unsqueeze(1)) <= gap)

    if cross_video_neg:
        candidates = ~(eye | near)
    else:
        candidates = same_vid & ~(eye | near)

    exp_sim = torch.exp(sim_all) * candidates.float()

    valid_neg_counts = candidates.sum(1)
    pos_counts = pos_mask.sum(1)
    samples_per_row = (neg_ratio * pos_counts.float()).ceil().long().clamp(min=1)
    samples_per_row = torch.minimum(samples_per_row, valid_neg_counts)
    max_samples = int(samples_per_row.max())

    if hard_neg:
        masked_sim = exp_sim.masked_fill(~candidates, -1)
        _, top_idx = torch.topk(masked_sim, k=max_samples, dim=1, largest=True)
    else:
        random_scores = torch.rand_like(exp_sim) * candidates.float() - (~candidates).float()
        _, top_idx = torch.topk(random_scores, k=max_samples, dim=1, largest=True)

    row_ids = torch.arange(num_valid, device=device).unsqueeze(1).expand_as(top_idx)
    keep_mask = torch.arange(max_samples, device=device).unsqueeze(0) < samples_per_row.unsqueeze(1)
    selected = torch.zeros_like(exp_sim, dtype=torch.bool)
    selected[row_ids, top_idx] = keep_mask

    neg_sum = (exp_sim * selected.float()).sum(1)
    loss = -torch.log(exp_pos / (exp_pos + neg_sum + 1e-9))
    return loss.mean()
