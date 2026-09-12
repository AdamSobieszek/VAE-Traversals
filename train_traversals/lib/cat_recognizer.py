"""CAT's shared scale-wise representation, adapted to ordered image pairs."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cat_components import (
    TransformerBlock, VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed,
)


# Capacity follows the generated image resolution before Recognizer.pool_size.
# Narrow the quadratic-cost projections first; retain three or four mixing steps.
# Head dimensions stay RoPE-compatible (24, 32 or 64); heads alone do not save weights.
CAT_PRESETS = {
    32: dict(hidden_size=48, depth=3, num_heads=2, num_scales=2),
    64: dict(hidden_size=64, depth=3, num_heads=2, num_scales=2),
    128: dict(hidden_size=96, depth=4, num_heads=3, num_scales=2),
    256: dict(hidden_size=128, depth=4, num_heads=4, num_scales=3),
    512: dict(hidden_size=160, depth=4, num_heads=5, num_scales=3),
    1024: dict(hidden_size=192, depth=4, num_heads=3, num_scales=3),
}


class _ScalePosition(nn.Module):
    """Bounded, nonpersistent cache: fixed positions for one grid per scale.

    There are no resolution-dependent trainable parameters. Buffers follow device
    and dtype moves and do not make checkpoints depend on the last input size.
    """
    def __init__(self, hidden_size, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.register_buffer("pos_embed", torch.empty(0), persistent=False)
        self.rope = None

    def forward(self, tokens, grid):
        if (self.pos_embed.shape != (1, grid * grid, self.hidden_size)
                or self.pos_embed.device != tokens.device
                or self.pos_embed.dtype != tokens.dtype):
            # A first call under inference_mode must not poison cached constants
            # for later training: RoPE multiplications save these for backward.
            with torch.inference_mode(False):
                pos = get_2d_sincos_pos_embed(self.hidden_size, grid)
                self.pos_embed = torch.from_numpy(pos).unsqueeze(0).to(
                    device=tokens.device, dtype=tokens.dtype)
                rope = VisionRotaryEmbeddingFast(dim=self.head_dim // 2, pt_seq_len=grid)
                # CAT's fixed RoPE tables are regenerated, never checkpoint weights.
                for name, buffer in list(rope.named_buffers()):
                    rope.register_buffer(name, buffer, persistent=False)
                self.rope = rope.to(device=tokens.device, dtype=tokens.dtype)
        return self.pos_embed, self.rope


class CATPairEncoder(nn.Module):
    """One ordered 2C-channel image per scale -> shared backbone -> K logits.

    Inputs are square images, with each scale divisible by patch_size. pool_size
    is handled by the outer Recognizer *before* this relative half-size pyramid.
    ``forward_pyramid`` also accepts precomputed paired scales for diagnostics or
    future generator pyramid integration; the public Recognizer stays two-image.
    """
    def __init__(self, dim_index, channels=3, *, hidden_size=None, depth=None,
                 num_heads=None, patch_size=4, num_scales=None, mlp_ratio=4.0,
                 layerscale=0.1, fused_attn=True, gradient_checkpointing=False,
                 preset=None):
        super().__init__()
        if preset is not None and preset not in CAT_PRESETS:
            raise ValueError(f"Unknown CAT preset {preset!r}; choose {tuple(CAT_PRESETS)}.")
        defaults = (CAT_PRESETS[preset] if preset is not None else
                    dict(hidden_size=192, depth=4, num_heads=3, num_scales=2))
        hidden_size = defaults['hidden_size'] if hidden_size is None else hidden_size
        depth = defaults['depth'] if depth is None else depth
        num_heads = defaults['num_heads'] if num_heads is None else num_heads
        num_scales = defaults['num_scales'] if num_scales is None else num_scales
        self.preset = preset
        for name, value in dict(dim_index=dim_index, channels=channels,
                                hidden_size=hidden_size, depth=depth,
                                num_heads=num_heads, patch_size=patch_size,
                                num_scales=num_scales).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if hidden_size % num_heads or (hidden_size // num_heads) % 4:
            raise ValueError("hidden_size must divide into heads whose dimension is a multiple of 4 for 2D RoPE.")
        if not mlp_ratio > 0 or int(2 / 3 * int(hidden_size * mlp_ratio)) < 1:
            raise ValueError("mlp_ratio must produce a positive SwiGLU hidden dimension.")
        self.in_channels = 2 * channels
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_proj = nn.Conv2d(self.in_channels, hidden_size, patch_size, patch_size)
        self.scale_embed = nn.Parameter(torch.empty(num_scales, hidden_size))
        self.cls_tokens = nn.Parameter(torch.empty(num_scales, 1, hidden_size))
        self.positions = nn.ModuleList([
            _ScalePosition(hidden_size, hidden_size // num_heads) for _ in range(num_scales)
        ])
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                             layerscale=layerscale, qk_norm=True, fused_attn=fused_attn)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.RMSNorm(hidden_size, eps=1e-6),
                                  nn.Linear(hidden_size, dim_index))
        self.apply(self._init_weights)
        nn.init.normal_(self.scale_embed, std=0.02)
        nn.init.normal_(self.cls_tokens, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def make_pyramid(self, paired):
        self._validate_scale(paired)
        divisor = self.patch_size * 2 ** (self.num_scales - 1)
        if paired.shape[-1] % divisor:
            raise ValueError(
                f"CAT working resolution {paired.shape[-1]} must be divisible by {divisor} "
                f"(patch_size * 2**(num_scales-1)); adjust pool_size, patch_size or num_scales.")
        pyramid = [paired]
        for _ in range(1, self.num_scales):
            # Exact factor-two box downsampling: low-pass, differentiable, MPS native.
            pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))
        return pyramid

    def _validate_scale(self, paired):
        if paired.ndim != 4 or paired.shape[1] != self.in_channels:
            raise ValueError(f"CAT expects [B, {self.in_channels}, H, W] paired images.")
        h, w = paired.shape[-2:]
        if h != w or h < self.patch_size or h % self.patch_size:
            raise ValueError("CAT expects square images with resolution divisible by patch_size.")

    def forward_pyramid(self, paired_scales, *, return_per_scale=False):
        if len(paired_scales) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} paired scales, got {len(paired_scales)}.")
        cls_features = []
        for idx, paired in enumerate(paired_scales):
            self._validate_scale(paired)
            tokens = self.patch_proj(paired).flatten(2).transpose(1, 2)
            position, rope = self.positions[idx](tokens, paired.shape[-1] // self.patch_size)
            scale = self.scale_embed[idx].to(tokens.dtype)
            cls = self.cls_tokens[idx].to(tokens.dtype).expand(paired.shape[0], -1, -1) + scale
            seq = torch.cat([cls, tokens + position + scale], dim=1)
            # Complete each scale independently, with exactly the same block weights.
            # No patch or CLS token ever attends to a different resolution.
            for block in self.blocks:
                if self.gradient_checkpointing and torch.is_grad_enabled():
                    seq = checkpoint(block, seq, None, rope, None, use_reentrant=False)
                else:
                    seq = block(seq, feat_rope=rope)
            cls_features.append(seq[:, 0])
        per_scale = self.head(torch.stack(cls_features, dim=1))
        logits = per_scale.mean(dim=1)
        return (logits, per_scale) if return_per_scale else logits

    def forward(self, paired):
        return self.forward_pyramid(self.make_pyramid(paired))
