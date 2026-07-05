# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp
from models.custom_layers import Attention, FourierFeature, EqualLinear
from models.pos_embed import VisionRotaryEmbeddingFast
from models.swiglu_ffn import SwiGLUFFN


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulate(x, scale, shift=None):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        return self.embedding_table(labels)


class CATBlock(nn.Module):
    def __init__(self, hidden_size, w_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            fused_attn=block_kwargs["fused_attn"],
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(w_dim, 4 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, w_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(w_dim, hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        return self.linear(x)


class CATGenerator(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        latent_size=64,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        z_dims=(768,),
        projector_dim=2048,
        output_layers=None,
        fused_attn=True,
        qk_norm=True,
        **block_kwargs,
    ):
        super().__init__()
        block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm, **block_kwargs}
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        w_dim = self.hidden_size

        if output_layers is None:
            rgb_every = self.depth // 4
            output_layers = tuple((i + 1) * rgb_every for i in range(4))
        self.output_layers = tuple(output_layers)
        self.output_layer_indices = {layer: idx for idx, layer in enumerate(self.output_layers)}

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = LabelEmbedder(num_classes, latent_size, class_dropout_prob)
        st_resolution = self.input_size // self.patch_size
        self.num_st_patches = st_resolution ** 2

        self.use_fourierfeat = False
        if self.use_fourierfeat:
            self.pos_embed = FourierFeature(hidden_size, resolution=st_resolution)
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, st_resolution ** 2, hidden_size), requires_grad=False
            )

        self.use_rope = True
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.latent_embedder = nn.Sequential(
            EqualLinear(latent_size * 2, w_dim, lr_mult=0.01),
            nn.SiLU(),
            EqualLinear(w_dim, w_dim, lr_mult=0.01),
        )

        self.blocks = nn.ModuleList(
            [
                CATBlock(hidden_size, w_dim, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layers = nn.ModuleList(
            [
                FinalLayer(hidden_size, w_dim, patch_size, self.out_channels)
                for _ in range(len(self.output_layers))
            ]
        )

        self.initialize_weights()
        self.w_avg_beta = 0.995
        self.register_buffer("w_avg", torch.zeros([num_classes, w_dim]))

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

        if self.use_fourierfeat:
            self.pos_embed.reset_parameters()
        else:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.num_st_patches ** 0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        layer_gain = 1e-1
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight, gain=layer_gain)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for final_layer in self.final_layers:
            nn.init.xavier_uniform_(final_layer.adaLN_modulation[-1].weight, gain=layer_gain)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)

        if isinstance(self.x_embedder, PatchEmbed):
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

    def unpatchify(self, x):
        c = x.shape[-1] // (self.patch_size ** 2)
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def forward(
        self,
        x,
        y,
        z,
        guidance_scale=1.0,
        update_ema=False,
        truncation_psi=0.0,
        return_stages=False,
    ):
        if self.use_fourierfeat:
            x = self.x_embedder(x) * 0.0 + self.pos_embed(z)
        else:
            x = self.pos_embed + self.x_embedder(x) * 0.0

        y_idx = y
        y = self.y_embedder(y, self.training)
        c = torch.cat([normalize_2nd_moment(y.squeeze(1)), normalize_2nd_moment(z)], dim=1)
        c = self.latent_embedder(c)

        if truncation_psi != 0.0:
            c = c.lerp(self.w_avg[y_idx], truncation_psi)

        if self.w_avg_beta is not None and update_ema:
            unique_labels = y_idx.unique()
            for label in unique_labels:
                mask = y_idx.squeeze() == label
                avg_c = c[mask].mean(dim=0)
                self.w_avg[label].copy_(
                    avg_c.detach().lerp(self.w_avg[label].to(c.dtype), self.w_avg_beta)
                )

        if torch.is_tensor(guidance_scale):
            scales = guidance_scale
        else:
            scales = torch.full((y.shape[0],), guidance_scale, device=y.device, dtype=z.dtype)

        if (scales != 1.0).any():
            indices = torch.arange(self.num_classes).to(y_idx.device)
            y_null = self.y_embedder(indices, self.training)
            y_null = y_null.unsqueeze(0).repeat(z.shape[0], 1, 1)
            z_null = z.unsqueeze(1).repeat(1, self.num_classes, 1)
            w_null = torch.cat(
                [normalize_2nd_moment(y_null), normalize_2nd_moment(z_null)], dim=-1
            )
            w_null = self.latent_embedder(w_null).detach().mean(dim=1)
            c = w_null + scales.unsqueeze(1) * (c - w_null)

        xs = []
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block), x, c, self.feat_rope, use_reentrant=False
            )
            xs.append(x)

        self.recent_x_std = x.std()

        stage_outputs = []
        rgb_accum = 0
        head_idx = 0
        for block_idx, block_tokens in enumerate(xs, start=1):
            if block_idx not in self.output_layer_indices:
                continue
            rgb = self.final_layers[head_idx](block_tokens, c)
            rgb_accum = rgb_accum + rgb
            stage_outputs.append(self.unpatchify(rgb_accum))
            head_idx += 1

        if return_stages:
            return stage_outputs
        return stage_outputs[-1]


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


def CAT_G_B_2(**kwargs):
    return CATGenerator(
        depth=12,
        hidden_size=768,
        patch_size=2,
        num_heads=12,
        output_layers=(3, 6, 9, 12),
        **kwargs,
    )


def CAT_G_M_2(**kwargs):
    return CATGenerator(
        depth=24,
        hidden_size=768,
        patch_size=2,
        num_heads=12,
        output_layers=(6, 12, 18, 24),
        **kwargs,
    )


def CAT_G_H_2(**kwargs):
    return CATGenerator(
        depth=32,
        hidden_size=1280,
        patch_size=2,
        num_heads=16,
        output_layers=(8, 16, 24, 32),
        **kwargs,
    )


CAT_models = {
    "CAT-G-B/2": CAT_G_B_2,
    "CAT-G-M/2": CAT_G_M_2,
    "CAT-G-H/2": CAT_G_H_2,
}
