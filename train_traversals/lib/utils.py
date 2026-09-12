"""Training helpers for devices, sampling, optimizers, schedules, and gradient clipping."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Iterable, Sequence, Union, Optional


def choose_device() -> torch.device:
        # Device selection
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

    # Set default tensor type for CUDA only (no MPS default tensor type exists)
    if cuda_available:
        torch.set_default_device(torch.device('cuda'))
    elif mps_available:
        torch.set_default_device(torch.device('mps'))
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_device(torch.device('cpu'))

    return device


@torch.no_grad()
def sample_z(batch_size, generator, params, device = torch.device('cuda')):
    """
    Instead of sampling batch_size independent random vectors,
    sample one random vector and generate the rest as an orthonormal basis
    (Gram-Schmidt) to it. If batch_size > generator.dim_z, will pad with zeros.
    """
    dim_z = generator.dim_z if hasattr(generator, 'dim_z') else generator.latent_size

    # Reduced QR is the vectorized equivalent of Gram-Schmidt on Gaussian
    # columns. Correcting the arbitrary QR signs also makes the first basis
    # vector equal to the normalized first sample, as in the legacy code.
    basis_size = min(batch_size, dim_z)
    raw = torch.randn(dim_z, basis_size, device=device)
    z0_norm = raw[:, 0].norm()
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = r.diagonal().sign().masked_fill_(r.diagonal() == 0, 1)
    z = q.mul(signs.unsqueeze(0)).mT.mul_(z0_norm)
    # If batch_size > dim_z, pad with zeros
    if batch_size > dim_z:
        pad = torch.zeros(batch_size - dim_z, dim_z, device=device)
        z = torch.cat([z, pad], dim=0)
    # If batch_size < dim_z, truncate
    if z.shape[0] > batch_size:
        z = z[:batch_size]

    # Move to correct device if needed
    if z.device.type == 'cuda':
        z = z.cuda(non_blocking=True)

    # Optionally shift in w-space and apply truncation
    if getattr(generator, "shift_in_w_space", False):
        z = generator.get_w(z)
        if getattr(params, "z_truncation", None) is not None:
            z_mean = z.mean(dim=0, keepdim=True)
            z = (z - z_mean) * params.z_truncation + z_mean
    else:
        if getattr(params, "z_truncation", None) is not None:
            z = z * params.z_truncation

    return z


def _norm_classes():
    return (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )


def split_weight_decay_groups(
    model: nn.Module,
    weight_decay: float,
    extra_no_decay_names: Sequence[str] = (),
    extra_no_decay_params: Sequence[Optional[torch.nn.Parameter]] = (),
    include_1d_as_no_decay: bool = True,
):
    """
    Build AdamW param groups:
      - Decay: 'true' weights.
      - No-decay: biases, norm params, explicitly provided params, and (optionally) 1D tensors.
    Uses PARAM IDENTITY to ensure correct bucketing.

    Args:
        model: the module to scan for parameters.
        weight_decay: WD to apply to the decay group.
        extra_no_decay_names: param names or suffixes to exclude from WD (exact or ".suffix" match).
        extra_no_decay_params: explicit Parameter objects to exclude (by identity).
        include_1d_as_no_decay: if True, any 1D tensor (e.g., LayerNorm/Bias) goes to no-decay.

    Returns:
        A list of param-group dicts suitable for torch.optim.AdamW.
    """
    norm_types = _norm_classes()

    # 1) Collect no-decay by identity (explicit params)
    no_decay_ids = set(id(p) for p in extra_no_decay_params if p is not None)

    # 2) Add norm layer params by identity
    for m in model.modules():
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                no_decay_ids.add(id(p))

    # 3) Name-based rules (bias and explicit suffix matches)
    name_rules = set(extra_no_decay_names or ())

    def name_is_extra(n: str) -> bool:
        # exact or endswith(".name")
        return (n in name_rules) or any(n.endswith(f".{x}") for x in name_rules)

    # 4) Final bucketing
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            id(p) in no_decay_ids
            or n.endswith("bias")
            or name_is_extra(n)
            or (include_1d_as_no_decay and p.ndim == 1)
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _as_param_groups(obj, weight_decay: float):
    """Best-effort conversion of various inputs into AdamW param groups.

    Accepts:
      - nn.Module -> single WD-specified group (caller typically wants split_weight_decay_groups instead)
      - Iterable[Parameter] -> one group
      - Sequence[dict] (already param groups) -> returned as-is
    """
    if isinstance(obj, nn.Module):
        return [{"params": [p for p in obj.parameters() if p.requires_grad], "weight_decay": float(weight_decay)}]

    # Already param groups
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict):
        return list(obj)

    # Iterable of parameters
    try:
        it = iter(obj)  # type: ignore
    except TypeError:
        raise TypeError("build_adamw: unsupported input type for 'model'/'params' argument.")
    params = [p for p in it if isinstance(p, torch.nn.Parameter)]
    if not params:
        raise ValueError("build_adamw: received an iterable with no Parameters.")
    return [{"params": params, "weight_decay": float(weight_decay)}]


def build_adamw(
    model: Union[nn.Module, Sequence[dict], Iterable[torch.nn.Parameter]],
    lr: float,
    weight_decay: float,
    extra_no_decay_names: Sequence[str] = (),
    extra_no_decay_params: Sequence[Optional[torch.nn.Parameter]] = (),
    betas=(0.9, 0.999),
    eps=1e-8,
    include_1d_as_no_decay: bool = True,
):
    """
    Flexible AdamW builder.

    If `model` is an nn.Module -> build groups using split_weight_decay_groups (norms/bias excluded).
    If `model` is an iterable of Parameters -> single group (uses provided weight_decay).
    If `model` is a sequence of param-group dicts -> passed through unchanged.

    Note: We always pass `weight_decay=0.0` to the optimizer itself and rely on group-level WD.
    """
    if isinstance(model, nn.Module):
        groups = split_weight_decay_groups(
            model,
            weight_decay,
            extra_no_decay_names=extra_no_decay_names,
            extra_no_decay_params=extra_no_decay_params,
            include_1d_as_no_decay=include_1d_as_no_decay,
        )
    else:
        groups = _as_param_groups(model, weight_decay)

    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)


class CosineScheduleWithWarmup(_LRScheduler):
    """
    Linear warmup followed by a configurable decay (cosine by default).
    Pickle-safe: no lambdas or local callables are stored.
    """
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        # decay config (primitive types only -> pickle-safe)
        self.decay_kind = 'cosine'
        self.decay_power = 1.0
        # Minimum LR factor after decay (e.g., 0.1 => decay to 1/10 of base LR)
        self.min_lr_factor = 0.1
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _decay_factor(self, p: float) -> float:
        """
        Return a multiplicative LR factor for the decay phase.

        For decaying schedules, the factor goes from 1.0 at p=0 to min_lr_factor at p=1
        (so it does NOT decay to 0).
        """
        # p is in [0,1]
        min_f = float(self.min_lr_factor)
        if min_f < 0.0 or min_f > 1.0:
            raise ValueError(f"min_lr_factor must be in [0,1], got {min_f}")

        if self.decay_kind == 'cosine':
            raw = 0.5 * (1.0 + math.cos(math.pi * p))     # 1 -> 0
            return min_f + (1.0 - min_f) * raw            # 1 -> min_f
        elif self.decay_kind == 'linear':
            raw = 1.0 - p                                 # 1 -> 0
            return min_f + (1.0 - min_f) * raw            # 1 -> min_f
        elif self.decay_kind == 'constant':
            return 1.0                                    # hold LR
        elif self.decay_kind == 'poly':
            raw = (1.0 - p) ** float(self.decay_power)    # 1 -> 0
            return min_f + (1.0 - min_f) * raw            # 1 -> min_f
        else:
            raise ValueError(f"Unknown decay schedule '{self.decay_kind}'")

    def get_lr(self):
        # Warmup: 0 -> 1
        if self.last_epoch < self.num_warmup_steps:
            progress = self._clamp01(self.last_epoch / max(1, self.num_warmup_steps))
            return [base_lr * progress for base_lr in self.base_lrs]

        # Decay phase
        denom = max(1, self.num_training_steps - self.num_warmup_steps)
        p = self._clamp01((self.last_epoch - self.num_warmup_steps) / denom)
        factor = self._decay_factor(p)
        return [base_lr * factor for base_lr in self.base_lrs]

    def set_decay(self, schedule: str = 'cosine', **kwargs):
        """
        Change post-warmup decay on the fly (pickle-safe).
        schedule ∈ {'cosine', 'linear', 'constant', 'poly'}
        For 'poly', you can pass power=2.0, etc.
        """
        schedule = (schedule or 'cosine').lower()
        if schedule not in {'cosine', 'linear', 'constant', 'poly'}:
            raise ValueError("schedule must be one of {'cosine','linear','constant','poly'}")

        self.decay_kind = schedule
        if 'min_lr_factor' in kwargs:
            self.min_lr_factor = float(kwargs['min_lr_factor'])
        if schedule == 'poly':
            self.decay_power = float(kwargs.get('power', 1.0))


@torch.no_grad()
def _schedule_scalar(step: int, total_steps: int, start: float, end: float,
                     schedule: str = "cosine", anneal_fraction: float = 1.0) -> float:
    schedule = (schedule or "cosine").lower()
    anneal_steps = max(1, int(round(total_steps * max(0.0, min(1.0, anneal_fraction)))))
    p = min(1.0, max(0.0, step / float(anneal_steps)))
    if schedule == "linear":
        return float(start + (end - start) * p)
    # cosine default
    cos_p = 0.5 * (1.0 + math.cos(math.pi * p))  # 1 -> 0
    return float(end + (start - end) * cos_p)


@torch.no_grad()
def _as_beta_k(trainer, K: int, device, dtype) -> torch.Tensor:
    """
    Try to obtain per-k beta from trainer / stat_tracker.
    Expected shapes:
      - [K], [1,K], [1,K,1], [K,1]
    Fallback: ones.
    """
    beta = None

    # preferred: a method
    for obj in (getattr(trainer, "stat_tracker", None), trainer):
        if obj is None:
            continue
        if hasattr(obj, "get_beta_k") and callable(getattr(obj, "get_beta_k")):
            beta = obj.get_beta_k()
            break

    # common attrs
    if beta is None:
        for obj in (getattr(trainer, "stat_tracker", None), trainer):
            if obj is None:
                continue
            for name in ("beta_k", "beta", "confusion_beta_k"):
                if hasattr(obj, name):
                    beta = getattr(obj, name)
                    break
            if beta is not None:
                break

    if beta is None:
        return torch.ones((1, K), device=device, dtype=dtype)

    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, device=device, dtype=dtype)

    beta = beta.to(device=device, dtype=dtype)

    # squeeze/reshape to [1,K]
    if beta.ndim == 1:
        beta = beta.view(1, -1)
    elif beta.ndim == 2:
        if beta.shape[0] != 1 and beta.shape[1] == 1:
            beta = beta.view(1, -1)
    elif beta.ndim == 3:
        beta = beta.view(beta.shape[0], beta.shape[1])  # drop last dim if [1,K,1]
        if beta.shape[0] != 1:
            beta = beta[:1]
    else:
        beta = beta.view(1, -1)

    if beta.shape[1] != K:
        # last-resort: truncate / pad
        if beta.shape[1] > K:
            beta = beta[:, :K]
        else:
            pad = torch.ones((1, K - beta.shape[1]), device=device, dtype=dtype)
            beta = torch.cat([beta, pad], dim=1)

    return beta


@torch.no_grad()
def _dt_legacy_uniform(B: int, device, half_range: int, low: int = 5000, high: int = 5001, dtype=torch.float32):
    dt = torch.randint(low, high, (1, 1), device=device) / 5000.0
    dt_scale = 2.0 / max(1, (half_range - 1))
    return dt.to(dtype).mul(dt_scale).repeat(B, 1)  # [B,1]


@torch.no_grad()
def _dt_chi_temp(B: int, device, *, step: int, total_opt_steps: int, half_range: int,
                 temp_start: float, temp_end: float, schedule: str, anneal_fraction: float,
                 clip_max: float, dtype=torch.float32):
    base_a = math.sqrt(math.pi / 8.0)
    temp = _schedule_scalar(step, total_opt_steps, temp_start, temp_end, schedule, anneal_fraction)
    a = float(base_a * temp)

    x = torch.randn((B, 3), device=device, dtype=dtype).mul_(a)
    dt_raw = x.norm(dim=1, keepdim=True)  # [B,1]
    if clip_max is not None and clip_max > 0:
        dt_raw = dt_raw.clamp(max=float(clip_max))

    dt_scale = 2.0 / max(1, (half_range - 1))
    return dt_raw.mul(dt_scale)  # [B,1]


@torch.no_grad()
def _dt_lognormal(B: int, device, half_range: int, *, mean: float = -1.5, std: float = 0.35,
                  clip_max: float = 5.0, dtype=torch.float32):
    # exp(N(mean,std^2)) then scaled to your step scale
    x = torch.randn((B, 1), device=device, dtype=dtype).mul_(float(std)).add_(float(mean)).exp_()
    if clip_max is not None and clip_max > 0:
        x = x.clamp(max=float(clip_max))
    dt_scale = 2.0 / max(1, (half_range - 1))
    return x.mul(dt_scale)  # [B,1]


@torch.no_grad()
def sample_dt(trainer, B: int, half_range: int, total_opt_steps: int, *, dtype=torch.float32) -> torch.Tensor:
    """
    Returns dt in shape:
      - [B,1] if beta_mode == "none"
      - [B,K] if beta_mode != "none"   (per-field energy scale)
    Reads trainer.params.* and caches small stats on trainer._dt_cache.

    Base modes: legacy_uniform | chi_temp | lognormal
    Beta modes: none | confusion
      (confusion expects trainer/stat_tracker to expose beta via get_beta_k() or beta_k/beta.)
    """
    p = trainer.params
    # if not hasattr(p, "dt_beta_mode"):
    #     # base dt sampling
    #     p.dt_mode = "chi_temp"
    #     p.dt_temp_start = 1.0
    #     p.dt_temp_end = 0.05
    #     p.dt_temp_schedule = "cosine"
    #     p.dt_temp_anneal_fraction = 1.0
    #     p.dt_clip_max = 5.0

    #     # beta coupling
    #     p.dt_beta_mode = "confusion"      # or "none"
    #     p.dt_beta_min = 0.5
    #     p.dt_beta_max = 2.0
    #     p.dt_beta_renorm = True
    #     p.dt_beta_power = 1.0             # <1 softens, >1 sharpens    
    device = trainer.device

    K = int(getattr(trainer, "K", getattr(p, "num_traversal_sets", 1)))
    step = int(getattr(getattr(trainer, "stat_tracker", None), "global_opt_step", 0))

    # ---- base dt sampler ----
    base_mode = str(getattr(p, "dt_mode", "legacy_uniform")).lower()

    if base_mode in {"legacy", "legacy_uniform", "uniform"}:
        dt = _dt_legacy_uniform(B, device, half_range, dtype=dtype)
    elif base_mode in {"chi", "chi_temp", "chi-temperature"}:
        dt = _dt_chi_temp(
            B, device,
            step=step,
            total_opt_steps=int(total_opt_steps),
            half_range=int(half_range),
            temp_start=float(getattr(p, "dt_temp_start", 1.0)),
            temp_end=float(getattr(p, "dt_temp_end", 0.05)),
            schedule=str(getattr(p, "dt_temp_schedule", "cosine")),
            anneal_fraction=float(getattr(p, "dt_temp_anneal_fraction", 1.0)),
            clip_max=float(getattr(p, "dt_clip_max", 5.0)),
            dtype=dtype,
        )
    elif base_mode in {"lognormal", "ln"}:
        dt = _dt_lognormal(
            B, device, half_range,
            mean=float(getattr(p, "dt_logn_mean", -1.5)),
            std=float(getattr(p, "dt_logn_std", 0.35)),
            clip_max=float(getattr(p, "dt_clip_max", 5.0)),
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown dt_mode={base_mode!r}")

    # ---- optional sign (kept separate; default positive) ----
    sign_mode = str(getattr(p, "dt_sign_mode", "positive")).lower()
    if sign_mode in {"rademacher", "pm"}:
        s = torch.randint(0, 2, (B, 1), device=device).mul_(2).sub_(1).to(dtype)
        dt = dt.mul(s)  # [B,1]

    # ---- confusion-based beta amplification (per-k energy scale) ----
    beta_mode = str(getattr(p, "dt_beta_mode", "none")).lower()
    if beta_mode in {"none", "off", ""}:
        beta = None
        out = dt  # [B,1]
    elif beta_mode in {"confusion", "confusion_beta", "beta"}:
        beta = _as_beta_k(trainer, K, device=device, dtype=dtype)  # [1,K]

        # soften / shape beta
        beta_pow = float(getattr(p, "dt_beta_power", 1.0))
        if beta_pow != 1.0:
            beta = beta.clamp_min(1e-12).pow(beta_pow)

        # clamp
        bmin = float(getattr(p, "dt_beta_min", 0.5))
        bmax = float(getattr(p, "dt_beta_max", 2.0))
        beta = beta.clamp(min=bmin, max=bmax)

        # optional renorm to mean 1 (stabilizes global step scale)
        if bool(getattr(p, "dt_beta_renorm", True)):
            beta = beta * (float(K) / beta.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # apply per-k scale: dt [B,1] -> [B,K]
        out = dt * beta  # broadcast to [B,K]
    else:
        raise ValueError(f"Unknown dt_beta_mode={beta_mode!r}")

    # ---- optional lightweight stats cache (for TB/debug) ----
    # Disabled by default because every .item() synchronizes accelerator work.
    if bool(getattr(p, "track_dt_stats", False)):
        cache = getattr(trainer, "_dt_cache", None)
        if cache is None:
            cache = {}
            setattr(trainer, "_dt_cache", cache)
        stats = [dt.abs().mean(), dt.abs().std(unbiased=False)]
        if beta is not None:
            stats.extend((beta.mean(), beta.min(), beta.max()))
        values = torch.stack(stats).detach().float().cpu().tolist()
        cache["dt_mean"], cache["dt_std"] = values[:2]
        if beta is not None:
            cache["beta_mean"], cache["beta_min"], cache["beta_max"] = values[2:]
        cache["dt_mode"] = base_mode
        cache["dt_beta_mode"] = beta_mode

    return out


@torch.no_grad()
def clip_accum_grads_(
    module,
    *,
    micro_idx: int,
    acc_steps: int =1,
    is_boundary: bool =True,
    mode: str = "end",
    clip_end: float = 1.0,
    clip_step: float | None = None,      # defaults to clip_end/sqrt(acc_steps)
    alpha: float = 3.0,                  # for delta_prophet
    clip_final: float | None = None,     # defaults to clip_end
    eps: float = 1e-12,
    state_attr: str = "_clip_accum_state",
):
    """
    Apply clipping to accumulated grads on `module` during grad-accumulation.

    Assumes your backward already scales by 1/acc_steps (i.e., you accumulate an average).
    Intended usage: call once per micro-step AFTER backward, BEFORE optimizer.step().

    Modes:
      - "none": do nothing
      - "end": only final clip at boundary (clip_final)
      - "buffer_const": clip full accumulated buffer each micro by clip_end
      - "buffer_sched": clip buffer each micro by clip_end*sqrt(j/acc_steps)
      - "delta": clip only the *increment* added this micro (preserves cancellation)
      - "delta_prophet": like delta, but sets cap from first delta norm in window: alpha*||delta_1||
    """
    mode = str(mode).lower()
    A = max(1, int(acc_steps))
    j = ((int(micro_idx) - 1) % A) + 1  # 1..A within window

    if clip_final is None:
        clip_final = float(clip_end)
    if clip_step is None:
        clip_step = float(clip_end) / math.sqrt(A)

    # collect params with grads
    params = [p for p in module.parameters() if p.grad is not None]
    if not params:
        return

    if mode == "none":
        return

    # ---------- buffer modes ----------
    if mode == "buffer_const":
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_end), error_if_nonfinite=False)

    elif mode == "buffer_sched":
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=float(clip_end) * math.sqrt(j / A),
            error_if_nonfinite=False,
        )

    # ---------- delta modes ----------
    elif mode in {"delta", "delta_prophet"}:
        st = getattr(module, state_attr, None)
        if st is None or j == 1:
            st = {"prev": None, "cap": None}
            setattr(module, state_attr, st)

        prev = st["prev"]
        if prev is None:
            # treat previous buffer as zeros (first micro in window)
            prev = [torch.zeros_like(p.grad) for p in params]

        # delta = current_buffer - prev_buffer
        d2 = 0.0
        deltas = []
        for p, pg in zip(params, prev):
            d = p.grad - pg
            deltas.append(d)
            d2 += float(d.detach().pow(2).sum())
        dnorm = math.sqrt(d2) + eps

        # choose cap
        if mode == "delta_prophet":
            if st["cap"] is None:          # set once per window
                st["cap"] = max(eps, float(alpha) * dnorm)
            cap = float(st["cap"])
        else:
            cap = float(clip_step)

        # apply clip to delta only: grad = prev + clipped(delta)
        if dnorm > cap:
            s = cap / dnorm
            for p, pg, d in zip(params, prev, deltas):
                p.grad.copy_(pg + d.mul(s))

        # snapshot current (post-clip) buffer as prev for next micro
        st["prev"] = [p.grad.detach().clone() for p in params]

    elif mode == "end":
        pass

    else:
        raise ValueError(f"clip_accum_grads_: unknown mode={mode!r}")

    # final safety clip at boundary (applies to buffer after all micros)
    if is_boundary and float(clip_final) > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip_final), error_if_nonfinite=False)
        # reset state so next window doesn't see stale prev
        st = getattr(module, state_attr, None)
        if isinstance(st, dict):
            st["prev"] = None
            st["cap"] = None


class FrozenGeneratorVJP(torch.autograd.Function):
    """Store latents, not synthesis activations; replay one chunk per VJP.

    Recognition still sees the entire B*K image batch, preserving its training
    BatchNorm statistics. Only frozen, deterministic synthesis is chunked.
    Backward receives dL/dimage and computes J_G(latent)^T dL/dimage without
    constructing a Jacobian. The trainer's detached latent bridge needs only
    this first derivative; the potential's higher derivatives remain separate.
    """

    @staticmethod
    def forward(ctx, latents, synthesize, chunk_size):
        ctx.save_for_backward(latents)
        ctx.synthesize, ctx.chunk_size = synthesize, chunk_size
        return torch.cat([synthesize(z) for z in latents.split(chunk_size)])

    @staticmethod
    @once_differentiable
    def backward(ctx, image_gradient):
        (latents,) = ctx.saved_tensors
        latent_gradient = torch.empty_like(latents)
        for start in range(0, len(latents), ctx.chunk_size):
            stop = start + ctx.chunk_size
            with torch.enable_grad():
                z = latents[start:stop].detach().requires_grad_(True)
                image = ctx.synthesize(z)
                (vjp,) = torch.autograd.grad(image, z, image_gradient[start:stop])
            latent_gradient[start:stop] = vjp
        return latent_gradient, None, None
