# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Shared, unconditional CAT transformer representation components.

Extracted unchanged from discriminator.py for reuse by the Traversals recognizer.
These layers have no GAN objective or class-label dependency.
"""
import numpy as np
import torch
import torch.nn as nn

from timm.models.vision_transformer import Mlp
from .custom_layers import Attention
from .pos_embed import VisionRotaryEmbeddingFast
from .swiglu_ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layerscale=1e-1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            fused_attn=block_kwargs["fused_attn"],
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        use_swiglu = True
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.ls_attn = nn.Parameter(torch.ones(hidden_size) * layerscale)
        self.ls_mlp = nn.Parameter(torch.ones(hidden_size) * layerscale)

    def forward(self, x, c=None, feat_rope=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope, attn_mask=attn_mask) * self.ls_attn
        x = x + self.mlp(self.norm2(x)) * self.ls_mlp
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


