import torch
import torch.nn.functional as F

CAT_DISC_RESOLUTIONS = (32, 16, 8, 4)


def resize_latent(x, size):
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(
        x,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


def build_cat_fake_pyramid(stage_outputs):
    h0, h1, h2, h3 = stage_outputs
    return [
        resize_latent(h3, 32),
        resize_latent(h2, 16),
        resize_latent(h1, 8),
        resize_latent(h0, 4),
    ]


def build_cat_real_pyramid(real_latent):
    return [
        resize_latent(real_latent, 32),
        resize_latent(real_latent, 16),
        resize_latent(real_latent, 8),
        resize_latent(real_latent, 4),
    ]


def cat_consistency_loss(stage_outputs, weights=(1 / 3, 1 / 2, 1.0)):
    h0, h1, h2, h3 = stage_outputs
    losses = [
        weights[0] * F.mse_loss(h0, h3),
        weights[1] * F.mse_loss(h1, h3),
        weights[2] * F.mse_loss(h2, h3),
    ]
    return sum(losses) / 3


@torch.no_grad()
def cat_alignment_metrics(stage_outputs):
    h0, h1, h2, h3 = [stage.detach() for stage in stage_outputs]
    metrics = {
        "align/discrep_h0_final": F.mse_loss(h0, h3),
        "align/discrep_h1_final": F.mse_loss(h1, h3),
        "align/discrep_h2_final": F.mse_loss(h2, h3),
        "align/rewrite_h0_h1": F.mse_loss(h0, h1),
        "align/rewrite_h1_h2": F.mse_loss(h1, h2),
        "align/rewrite_h2_h3": F.mse_loss(h2, h3),
    }

    final = h3.flatten(1)
    for idx, (current, next_stage) in enumerate(((h0, h1), (h1, h2), (h2, h3))):
        update = (next_stage - current).flatten(1)
        target = final - current.flatten(1)
        metrics[f"align/cos_update_h{idx}"] = F.cosine_similarity(
            update, target, dim=1, eps=1e-8
        ).mean()

    return metrics


def build_block_diag_attention_mask(group_lengths, device, dtype):
    total = sum(group_lengths)
    mask = torch.full((total, total), float("-inf"), device=device, dtype=dtype)
    start = 0
    for length in group_lengths:
        mask[start:start + length, start:start + length] = 0
        start += length
    return mask
