"""
PCA-residual StyleGAN feature codec.

Compresses frozen StyleGAN2 ``b256`` activations ``[B, 128, 256, 256]`` into a
full-resolution stochastic latent ``[B, 4, 256, 256]`` by learning nonlinear
residuals around a frozen rank-4 channel PCA analysis/synthesis shortcut.

The exported latent keeps PCA-aligned channel ordering so each axis retains a
stable appearance correspondence, while ConvNeXt-style residual branches and an
internal context tower improve on the linear PCA floor.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_CHANNELS = 128
FEATURE_RESOLUTION = 256
LATENT_CHANNELS = 4


# ---------------------------------------------------------------------------
# Frozen rank-4 PCA analysis / synthesis
# ---------------------------------------------------------------------------

class FrozenRank4PCA(nn.Module):
    """
    Exact channel-wise rank-K PCA with whitening.

    Analysis:  ``z = Λ^{-1/2} P^T (x - mean)``
    Synthesis: ``x = mean + P Λ^{1/2} z``
    """

    def __init__(
        self,
        in_channels: int = FEATURE_CHANNELS,
        latent_channels: int = LATENT_CHANNELS,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("components", torch.eye(in_channels, latent_channels))
        self.register_buffer("eigenvalues", torch.ones(latent_channels))
        self.register_buffer("channel_std", torch.ones(in_channels))
        self.register_buffer("n_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @property
    def scale(self) -> torch.Tensor:
        return (self.eigenvalues.clamp_min(1e-8)).sqrt()

    @property
    def inv_scale(self) -> torch.Tensor:
        return 1.0 / self.scale

    @torch.no_grad()
    def fit_from_covariance(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        n_samples: int = 0,
        channel_std: Optional[torch.Tensor] = None,
    ) -> "FrozenRank4PCA":
        mean = mean.detach().float().cpu()
        cov = 0.5 * (cov.detach().float().cpu() + cov.detach().float().cpu().T)
        evals, evecs = torch.linalg.eigh(cov)
        comps = evecs[:, -self.latent_channels :].flip(1)  # largest first
        vals = evals[-self.latent_channels :].flip(0).clamp_min(1e-8)
        for k in range(self.latent_channels):
            v = comps[:, k]
            i = int(torch.argmax(v.abs()).item())
            if v[i] < 0:
                comps[:, k] = -v
        self.mean.copy_(mean.to(self.mean.device, dtype=self.mean.dtype))
        self.components.copy_(comps.to(self.components.device, dtype=self.components.dtype))
        self.eigenvalues.copy_(vals.to(self.eigenvalues.device, dtype=self.eigenvalues.dtype))
        if channel_std is None:
            channel_std = torch.diag(cov).clamp_min(1e-8).sqrt()
        self.channel_std.copy_(channel_std.to(self.channel_std.device, dtype=self.channel_std.dtype))
        self.n_samples.fill_(int(n_samples))
        self.fitted.fill_(True)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,C,H,W] -> [B,K,H,W]`` whitened PCA codes."""
        x = x.float()
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        comps = self.components.to(dtype=x.dtype, device=x.device)
        inv_scale = self.inv_scale.to(dtype=x.dtype, device=x.device)
        weight = (comps * inv_scale[None, :]).T[:, :, None, None]
        return F.conv2d(x - mean[None, :, None, None], weight)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``[B,K,H,W] -> [B,C,H,W]`` PCA reconstruction."""
        z = z.float()
        comps = self.components.to(dtype=z.dtype, device=z.device)
        scale = self.scale.to(dtype=z.dtype, device=z.device)
        mean = self.mean.to(dtype=z.dtype, device=z.device)
        weight = (comps * scale[None, :])[:, :, None, None]
        return F.conv2d(z, weight, bias=mean)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_mse(
        self,
        x: torch.Tensor,
        z_pca: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exact rank-K reconstruction MSE without materializing a C-channel reconstruction."""
        x = x.float()
        centered = x - self.mean.to(dtype=x.dtype, device=x.device)[None, :, None, None]
        if z_pca is None:
            z_pca = self.encode(x)
        projected = z_pca.float() * self.scale.to(
            dtype=x.dtype, device=x.device
        )[None, :, None, None]
        residual_energy = centered.square().sum() - projected.square().sum()
        return residual_energy.clamp_min(0.0) / x.numel()

    def state_dict_compact(self) -> Dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "eigenvalues": self.eigenvalues.detach().cpu(),
            "channel_std": self.channel_std.detach().cpu(),
            "n_samples": self.n_samples.detach().cpu(),
            "fitted": self.fitted.detach().cpu(),
            "in_channels": torch.tensor(self.in_channels),
            "latent_channels": torch.tensor(self.latent_channels),
        }

    def load_compact(self, state: Dict[str, torch.Tensor]) -> "FrozenRank4PCA":
        self.mean.copy_(state["mean"].to(self.mean.device, dtype=self.mean.dtype))
        self.components.copy_(state["components"].to(self.components.device, dtype=self.components.dtype))
        self.eigenvalues.copy_(state["eigenvalues"].to(self.eigenvalues.device, dtype=self.eigenvalues.dtype))
        self.channel_std.copy_(state["channel_std"].to(self.channel_std.device, dtype=self.channel_std.dtype))
        self.n_samples.copy_(state["n_samples"].to(self.n_samples.device))
        self.fitted.copy_(state["fitted"].to(self.fitted.device))
        return self


class StreamingPCACalibrator:
    """Accumulate mean/cov from feature maps with a vectorized Welford update."""

    def __init__(self, in_channels: int = FEATURE_CHANNELS):
        self.in_channels = in_channels
        self.mean = torch.zeros(in_channels, dtype=torch.float64)
        self.m2 = torch.zeros(in_channels, in_channels, dtype=torch.float64)
        self.n = 0

    @torch.no_grad()
    def update(self, maps: torch.Tensor, max_samples: int = 8192) -> None:
        c = self.in_channels
        flat = maps.detach().float().permute(0, 2, 3, 1).reshape(-1, c)
        if flat.shape[0] > max_samples:
            idx = torch.randint(0, flat.shape[0], (max_samples,), device=flat.device)
            flat = flat[idx]
        batch = flat.detach().to(device="cpu", dtype=torch.float32).double()
        b = batch.shape[0]
        if b == 0:
            return
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_m2 = centered.T @ centered
        if self.n == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.n = b
            return
        # Parallel / Chan update for combining sufficient statistics
        n_a, n_b = self.n, b
        delta = batch_mean - self.mean
        n = n_a + n_b
        self.m2 = self.m2 + batch_m2 + torch.outer(delta, delta) * (n_a * n_b / n)
        self.mean = self.mean + delta * (n_b / n)
        self.n = n

    def finalize(self, latent_channels: int = LATENT_CHANNELS) -> FrozenRank4PCA:
        if self.n < 2:
            raise RuntimeError("Need at least 2 samples to fit PCA")
        cov = (self.m2 / (self.n - 1)).float()
        channel_std = torch.diag(cov).clamp_min(1e-8).sqrt()
        pca = FrozenRank4PCA(self.in_channels, latent_channels)
        return pca.fit_from_covariance(
            self.mean.float(), cov, n_samples=self.n, channel_std=channel_std
        )


# ---------------------------------------------------------------------------
# ConvNeXt-style residual blocks (MPS-friendly)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Optional Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtResidualBlock(nn.Module):
    """Depthwise 7×7 + pointwise expansion residual block."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dilation: int = 1,
        drop_path: float = 0.0,
        use_grn: bool = False,
    ):
        super().__init__()
        padding = dilation * 3
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=padding,
            dilation=dilation, groups=channels, bias=True,
        )
        self.norm = LayerNorm2d(channels)
        hidden = channels * expansion
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden) if use_grn else nn.Identity()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)
        self.drop_path = float(drop_path)
        self.gamma = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pw2(h)
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
            h = h * mask / keep
        return x + self.gamma * h


class ContextTower(nn.Module):
    """Temporary lower-resolution context producing FiLM scale/shift at full res."""

    def __init__(self, channels: int, use_grn: bool = False):
        super().__init__()
        self.down = nn.AvgPool2d(4)
        self.blocks = nn.Sequential(
            ConvNeXtResidualBlock(channels, dilation=1, use_grn=use_grn),
            ConvNeXtResidualBlock(channels, dilation=2, use_grn=use_grn),
        )
        self.to_film = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.blocks(self.down(h))
        c = F.interpolate(c, size=h.shape[-2:], mode="bilinear", align_corners=False)
        scale, shift = self.to_film(c).chunk(2, dim=1)
        return scale, shift


# ---------------------------------------------------------------------------
# Diagonal Gaussian posterior
# ---------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.deterministic = deterministic

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """Per-sample KL summed over C,H,W."""
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )

    def kl_per_latent(self) -> torch.Tensor:
        """Mean KL per latent scalar (normalized by 4*H*W)."""
        n = float(self.mean.shape[1] * self.mean.shape[2] * self.mean.shape[3])
        return self.kl() / n


# ---------------------------------------------------------------------------
# PCA-residual variational codec
# ---------------------------------------------------------------------------

class ResidualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        num_blocks: int = 4,
        use_context: bool = True,
        use_grn: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        dilations = [1, 2, 4, 1] * ((num_blocks + 3) // 4)
        self.blocks = nn.ModuleList([
            ConvNeXtResidualBlock(
                hidden_channels,
                dilation=dilations[i],
                drop_path=drop_path * i / max(num_blocks - 1, 1),
                use_grn=use_grn,
            )
            for i in range(num_blocks)
        ])
        self.context = ContextTower(hidden_channels, use_grn=use_grn) if use_context else None
        self.to_mean = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.to_mean.weight)
        nn.init.zeros_(self.to_mean.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -9.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        if self.context is not None:
            scale, shift = self.context(h)
            h = h * (1.0 + scale) + shift
        return self.to_mean(h), self.to_logvar(h)


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_channels: int,
        num_blocks: int = 4,
        use_context: bool = True,
        use_grn: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # No normalization on raw latent input (preserves PCA scale/mean info).
        self.stem = nn.Conv2d(latent_channels, hidden_channels, kernel_size=1)
        dilations = [1, 2, 4, 1] * ((num_blocks + 3) // 4)
        self.blocks = nn.ModuleList([
            ConvNeXtResidualBlock(
                hidden_channels,
                dilation=dilations[i],
                drop_path=drop_path * i / max(num_blocks - 1, 1),
                use_grn=use_grn,
            )
            for i in range(num_blocks)
        ])
        self.context = ContextTower(hidden_channels, use_grn=use_grn) if use_context else None
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.stem(z)
        for blk in self.blocks:
            h = blk(h)
        if self.context is not None:
            scale, shift = self.context(h)
            h = h * (1.0 + scale) + shift
        return self.to_out(h)


class PCAResidualFeatureCodec(nn.Module):
    """
    Nonlinear residual variational codec around frozen rank-4 PCA.

    At zero residual gates, deterministic encode/decode equals PCA exactly.
    """

    def __init__(
        self,
        pca: FrozenRank4PCA,
        hidden_channels: int = 64,
        num_blocks: int = 5,
        use_context: bool = True,
        use_grn: bool = False,
        drop_path: float = 0.05,
        residual_gate_init: float = 1.0,
    ):
        super().__init__()
        if not bool(pca.fitted.item()):
            raise ValueError("FrozenRank4PCA must be fitted before building the codec")
        self.pca = pca
        for p in self.pca.parameters():
            p.requires_grad_(False)
        self.pca.eval()

        self.in_channels = pca.in_channels
        self.latent_channels = pca.latent_channels
        self.hidden_channels = hidden_channels
        self.encoder = ResidualEncoder(
            in_channels=self.in_channels,
            hidden_channels=hidden_channels,
            latent_channels=self.latent_channels,
            num_blocks=num_blocks,
            use_context=use_context,
            use_grn=use_grn,
            drop_path=drop_path,
        )
        self.decoder = ResidualDecoder(
            out_channels=self.in_channels,
            hidden_channels=hidden_channels,
            latent_channels=self.latent_channels,
            num_blocks=num_blocks,
            use_context=use_context,
            use_grn=use_grn,
            drop_path=drop_path,
        )
        # Unit gates by default; exact PCA-at-init comes from zero-initialized residual heads.
        self.enc_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self.dec_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def train(self, mode: bool = True):
        super().train(mode)
        self.pca.eval()
        return self

    def encode(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution, torch.Tensor, torch.Tensor]:
        z_pca = self.pca.encode(x)
        mean_res, logvar = self.encoder(x)
        mean = z_pca + self.enc_gate * mean_res
        posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic or (not sample_posterior)
        )
        z = posterior.sample() if sample_posterior and not deterministic else posterior.mode()
        return z, posterior, z_pca, mean_res

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pca = self.pca.decode(z)
        x_res = self.decoder(z)
        recon = x_pca + self.dec_gate * x_res
        return recon, x_pca

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        deterministic: bool = False,
    ) -> Dict[str, object]:
        z, posterior, z_pca, mean_res = self.encode(
            x, sample_posterior=sample_posterior, deterministic=deterministic
        )
        recon, x_pca = self.decode(z)
        return {
            "recon": recon,
            "posterior": posterior,
            "z": z,
            "z_pca": z_pca,
            "mean_res": mean_res,
            "x_pca": x_pca,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def feature_codec_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    posterior: DiagonalGaussianDistribution,
    pca: FrozenRank4PCA,
    z: Optional[torch.Tensor] = None,
    z_pca: Optional[torch.Tensor] = None,
    mean_res: Optional[torch.Tensor] = None,
    kl_weight: float = 0.0,
    free_bits: float = 0.0,
    lambda_charb: float = 0.5,
    lambda_grad: float = 0.1,
    lambda_ms: float = 0.1,
    lambda_anchor: float = 0.05,
    lambda_cov: float = 0.01,
    deterministic: bool = False,
    return_stats_tensors: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Compound Phase-1 feature loss with PCA-relative distortion."""
    with torch.no_grad():
        pca_mse = pca.reconstruction_mse(target, z_pca=z_pca).clamp_min(1e-8)

    std = pca.channel_std.view(1, -1, 1, 1).to(device=target.device, dtype=target.dtype).clamp_min(1e-4)
    err = (recon - target) / std
    mse = F.mse_loss(recon, target)
    nmse = mse / pca_mse
    charb = charbonnier(err).mean()

    gx = recon[:, :8, :, 1:] - recon[:, :8, :, :-1]
    gy = recon[:, :8, 1:, :] - recon[:, :8, :-1, :]
    tx = target[:, :8, :, 1:] - target[:, :8, :, :-1]
    ty = target[:, :8, 1:, :] - target[:, :8, :-1, :]
    grad = F.l1_loss(gx, tx) + F.l1_loss(gy, ty)

    ms = target.new_zeros(())
    for s in (2, 4):
        ms = ms + F.mse_loss(F.avg_pool2d(recon, s), F.avg_pool2d(target, s))
    ms = ms / 2.0

    anchor = target.new_zeros(())
    if mean_res is not None:
        anchor = mean_res.pow(2).mean()
    if z is not None and z_pca is not None:
        anchor = anchor + F.mse_loss(z.mean(dim=(0, 2, 3)), z_pca.mean(dim=(0, 2, 3)))

    cov_pen = target.new_zeros(())
    if z is not None:
        flat = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        cov = (flat @ flat.T) / max(flat.shape[1] - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        cov_pen = off.pow(2).mean()

    if deterministic:
        kl_raw = target.new_zeros(())
        kl_pl = target.new_zeros(())
        kl_term = target.new_zeros(())
    else:
        kl_raw = posterior.kl().mean()
        n_latents = float(
            posterior.mean.shape[1] * posterior.mean.shape[2] * posterior.mean.shape[3]
        )
        kl_pl = kl_raw / n_latents
        if kl_weight <= 0:
            kl_term = target.new_zeros(())
        else:
            kl_term = torch.clamp(kl_pl - free_bits, min=0.0) if free_bits > 0 else kl_pl

    loss = (
        nmse
        + lambda_charb * charb
        + lambda_grad * grad
        + lambda_ms * ms
        + lambda_anchor * anchor
        + lambda_cov * cov_pen
        + kl_weight * kl_term
    )
    tensor_stats = {
        "loss": loss.detach(),
        "mse": mse.detach(),
        "pca_mse": pca_mse.detach(),
        "nmse": nmse.detach(),
        "rel_improve": (1.0 - nmse.detach()).clamp(-10, 10),
        "charb": charb.detach(),
        "grad": grad.detach(),
        "ms": ms.detach(),
        "anchor": anchor.detach(),
        "cov_pen": cov_pen.detach(),
        "kl": kl_raw.detach(),
        "kl_per_latent": kl_pl.detach(),
    }
    if return_stats_tensors:
        return loss, tensor_stats

    # One device synchronization for callers that need Python values, rather
    # than synchronizing once for every metric.
    keys = tuple(tensor_stats)
    values = torch.stack([tensor_stats[k] for k in keys]).cpu().tolist()
    return loss, dict(zip(keys, values))


@torch.no_grad()
def eval_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    pca: FrozenRank4PCA,
    posterior: Optional[DiagonalGaussianDistribution] = None,
) -> Dict[str, float]:
    mse = F.mse_loss(recon, target)
    pca_mse = pca.reconstruction_mse(target).clamp_min(1e-8)
    nmse = mse / pca_mse
    cos = F.cosine_similarity(recon.flatten(1), target.flatten(1), dim=1).mean()
    out = {
        "mse": float(mse),
        "pca_mse": float(pca_mse),
        "nmse": float(nmse),
        "rel_improve": float(1.0 - nmse),
        "cosine": float(cos),
    }
    if posterior is not None:
        out["kl"] = float(posterior.kl().mean())
        out["kl_per_latent"] = float(posterior.kl_per_latent().mean())
        out["post_std"] = float(posterior.std.mean())
    return out
