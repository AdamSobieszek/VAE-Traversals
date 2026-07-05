# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Mlp
from models.custom_layers import Attention, EqualLinear
from models.swiglu_ffn import SwiGLUFFN
from cat_pyramid import build_block_diag_attention_mask


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.latent_embedder = nn.Sequential(
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01),
            nn.SiLU(),
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01),
        )

    def forward(self, labels, train):
        embeddings = self.embedding_table(labels)
        return self.latent_embedder(embeddings)


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


class CATDiscriminator(nn.Module):
    SCALE_RESOLUTIONS = (32, 16, 8, 4)
    SCALE_TOKEN_GRIDS = (16, 8, 4, 2)
    GROUP_LENGTHS = (257, 65, 17, 5)
    CLS_INDICES = (0, 257, 322, 339)

    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        z_dims=(768,),
        projector_dim=2048,
        cmap_dim=2048,
        fused_attn=True,
        qk_norm=True,
        **block_kwargs,
    ):
        super().__init__()
        block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm, **block_kwargs}
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.hidden_size = hidden_size
        self.depth = depth
        self.cmap_dim = cmap_dim

        self.patch_proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )

        pos_embeds = []
        for grid in self.SCALE_TOKEN_GRIDS:
            pos = get_2d_sincos_pos_embed(hidden_size, grid)
            pos_embeds.append(
                nn.Parameter(torch.from_numpy(pos).float().unsqueeze(0), requires_grad=False)
            )
        self.pos_embeds = nn.ParameterList(pos_embeds)

        self.scale_embed = nn.Parameter(torch.zeros(4, hidden_size))
        self.cls_tokens = nn.Parameter(torch.randn(4, 1, hidden_size) * 0.02)

        layer_gain = 1e-1
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    layerscale=layer_gain,
                    **block_kwargs,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(hidden_size, cmap_dim, bias=True),
        )
        self.y_embedder = LabelEmbedder(num_classes, cmap_dim, class_dropout_prob)

        self.aux_feat_size = z_dims[0] if len(z_dims) > 0 else 0
        if self.aux_feat_size > 0:
            self.proj = build_mlp(hidden_size, projector_dim, z_dims[0])

        self.register_buffer(
            "attn_mask",
            build_block_diag_attention_mask(self.GROUP_LENGTHS, torch.device("cpu"), torch.float32),
            persistent=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.scale_embed, std=0.02)

    def patchify(self, x):
        tokens = self.patch_proj(x).flatten(2).transpose(1, 2)
        return tokens

    def encode_scale(self, x, scale_idx):
        tokens = self.patchify(x) + self.pos_embeds[scale_idx] + self.scale_embed[scale_idx]
        cls = self.cls_tokens[scale_idx].expand(x.shape[0], -1, -1) + self.scale_embed[scale_idx]
        return torch.cat([cls, tokens], dim=1)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def forward(self, xs, y, return_aux=False):
        assert len(xs) == 4
        expected = self.SCALE_RESOLUTIONS
        for x, size in zip(xs, expected):
            assert x.shape[1] == self.in_channels
            assert x.shape[-2:] == (size, size), f"Expected {size}x{size}, got {x.shape[-2:]}"

        y_emb = self.y_embedder(y, self.training).squeeze(dim=1)

        seq_parts = [self.encode_scale(x, idx) for idx, x in enumerate(xs)]
        seq = torch.cat(seq_parts, dim=1)

        attn_mask = self.attn_mask.to(device=seq.device, dtype=seq.dtype)
        for block in self.blocks:
            seq = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block),
                seq,
                None,
                None,
                attn_mask,
                use_reentrant=False,
            )

        self.recent_x_std = seq.std()

        cls_feats = torch.stack([seq[:, idx] for idx in self.CLS_INDICES], dim=1)
        logits = (self.final_layer(cls_feats) * y_emb.unsqueeze(1)).sum(dim=-1)

        if self.aux_feat_size > 0:
            x32_spatial = seq_parts[0][:, 1:]
            x32_cls = seq_parts[0][:, :1]
            x_feat = [self.proj(x32_cls), self.proj(x32_spatial)]
        else:
            x_feat = None

        if return_aux:
            return logits, {"x_feat": x_feat}
        return logits


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


def CAT_D_B_2(**kwargs):
    return CATDiscriminator(
        depth=12,
        hidden_size=768,
        patch_size=2,
        num_heads=12,
        **kwargs,
    )


CATD_models = {
    "CAT-D-B/2": CAT_D_B_2,
}
