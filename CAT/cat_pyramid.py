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


def build_block_diag_attention_mask(group_lengths, device, dtype):
    total = sum(group_lengths)
    mask = torch.full((total, total), float("-inf"), device=device, dtype=dtype)
    start = 0
    for length in group_lengths:
        mask[start:start + length, start:start + length] = 0
        start += length
    return mask
