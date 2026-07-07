import os
import os.path as osp
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
import sys
import math
import time
from scipy.stats import truncnorm
from PIL import Image, ImageDraw



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

import math
from torch.optim.lr_scheduler import _LRScheduler

class PhasedCosineWithRestarts(_LRScheduler):
    """
    Cosine LR with warmup + SGDR-style restarts + phase offset (radians).

    For step s:
      if s < warmup_steps:
          lr = base_lr * (s+1)/warmup_steps
      else:
          let u = s - warmup_steps
          cycle_len = T0 * (Tmult ** cycle_idx)
          pos  = (u - sum_prev_cycles) / cycle_len   in [0,1)
          lr   = min_lr + 0.5*(base_lr-min_lr) * (1 + cos(2π*pos + phase))

    Notes:
      • base_lrs are captured from the optimizer's param_groups at construction.
      • min_lr can be scalar or per-group list (length == len(base_lrs)).
      • phase is a scalar in radians (e.g., 0 for support, π for recognizer).
      • last_epoch is the *number of steps already taken* (PyTorch convention).
    """
    def __init__(self, optimizer,
                 warmup_steps: int,
                 T0: int,
                 Tmult: float = 1.0,
                 min_lr=1e-6,
                 phase: float = 0.0,
                 last_epoch: int = -1):
        self.warmup_steps = int(max(0, warmup_steps))
        self.T0 = int(max(1, T0))
        self.Tmult = float(max(1.0, Tmult))
        self.phase = float(phase)
        self._cycle_boundaries = None  # built lazily
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

        if isinstance(min_lr, (list, tuple)):
            assert len(min_lr) == len(self.base_lrs), "min_lr list must match number of param groups"
            self.min_lrs = list(map(float, min_lr))
        else:
            self.min_lrs = [float(min_lr)] * len(self.base_lrs)

        super().__init__(optimizer, last_epoch=last_epoch)

    def _locate_cycle(self, u: int):
        # u = steps since warmup (>=0). Return (cycle_idx, pos_in_cycle[0,1), cycle_len)
        if self._cycle_boundaries is None:
            self._cycle_boundaries = []
        # Expand boundaries until u is within range
        total = 0
        c = 0
        while True:
            L = int(round(self.T0 * (self.Tmult ** c)))
            if u < total + L:
                pos = (u - total) / max(1, L)
                return c, pos, L
            total += L
            c += 1

    def get_lr(self):
        s = self.last_epoch  # steps completed
        # Warmup
        if s < self.warmup_steps:
            scale = (s + 1) / max(1, self.warmup_steps)
            return [base * scale for base in self.base_lrs]

        # After warmup
        u = s - self.warmup_steps
        _, pos, _ = self._locate_cycle(u)
        # Cosine with phase
        cos_arg = 2.0 * math.pi * pos + self.phase
        cval = 0.5 * (1.0 + math.cos(cos_arg))
        # Per-group LR
        lrs = []
        for base, minlr in zip(self.base_lrs, self.min_lrs):
            lrs.append(minlr + (base - minlr) * cval)
        return lrs


@torch.no_grad()
def sample_z(batch_size, generator, params, device = torch.device('cuda')):
    """
    Instead of sampling batch_size independent random vectors,
    sample one random vector and generate the rest as an orthonormal basis
    (Gram-Schmidt) to it. If batch_size > generator.dim_z, will pad with zeros.
    """
    dim_z = generator.dim_z if hasattr(generator, 'dim_z') else generator.latent_size

    # Draw one random vector
    z0 = torch.randn(dim_z, device=device)
    z0_norm = z0.norm()
    z0 = z0 / (z0_norm + 1e-8)

    # Create orthonormal basis (including z0 as the first vector)
    basis = [z0]
    for _ in range(1, min(batch_size, dim_z)):
        v = torch.randn(dim_z, device=device)
        # Gram-Schmidt orthogonalization
        for b in basis:
            v = v - (v @ b) * b
        v_norm = v.norm()
        if v_norm < 1e-8:
            # If degenerate, resample
            v = torch.randn(dim_z, device=device)
            for b in basis:
                v = v - (v @ b) * b
            v_norm = v.norm()
            if v_norm < 1e-8:
                v = torch.zeros_like(v)
        else:
            v = v / v_norm
        basis.append(v)
    # Stack basis vectors
    z = torch.stack(basis, dim=0)*z0_norm
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
       
def create_exp_dir(args, new_experiment=False):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json) and
    the given command (bash script).

    Experiment's directory name format:

        <gan_type>(-<stylegan2_resolution>)(-{Z,W})-<recognizer_type>-K<num_support_sets>-
            D<num_support_dipoles>(-LearnAlphas)(-LearnGammas)-eps<min_shift_magnitude>_<max_shift_magnitude>
    E.g.:

        experiments/wip/ProgGAN-ResNet-K200-N32-LearnGammas-eps0.35_0.5

    Args:
        args (argparse.Namespace): the namespace object returned by `parse_args()` for the current run

    """
    if new_experiment:
        print("Creating new experiment\n"+"-"*30+"\n"*2)
    exp_dir = "{}".format(args.gan_type)
    if args.gan_type == 'StyleGAN2':
        exp_dir += '-{}'.format(args.stylegan2_resolution)
        if args.shift_in_w_space:
            exp_dir += '-W'
        else:
            exp_dir += '-Z'
    if args.gan_type == 'BigGAN':
        biggan_classes = '-'
        for c in args.biggan_target_classes:
            biggan_classes += '{}'.format(c)
        exp_dir += '{}'.format(biggan_classes)
    exp_dir += "-{}".format(args.recognizer_type)
    exp_dir += "-K{}-D{}".format(args.num_support_sets, args.num_support_timesteps)
    if new_experiment:
        exp_dir += f"__{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        # grep all folders that start with exp_dir
        os.makedirs("experiments/wip", exist_ok=True)
        exp_dirs = [d for d in os.listdir("experiments/wip") if d.startswith(exp_dir)]
        # exclude folders that do not contain checkpoint.pt as a file in their recursive folder structure
        exp_dirs = [d for d in exp_dirs if osp.isfile(osp.join("experiments/wip", d, "models", "checkpoint.pt"))]
        # sort by last modified time
        exp_dirs.sort(key=lambda x: os.path.getmtime(osp.join("experiments/wip", x)))
        #  set exp_dir to the newest folder
        if exp_dirs:
            exp_dir = exp_dirs[-1]
        else:
            exp_dir = exp_dir + f"__{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Using existing experiment: {exp_dir}\n"+"-"*30+"\n"*2)
    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)
    # Save args namespace object in json format
    with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(wip_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    return exp_dir
import torch
from torch import nn
from typing import Iterable, Sequence, Union, Optional

# -----------------------------
# Hyperparameter helper function
# --------------------------------

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




# aux.py

def module_grad_norm(mod):
    total_sq = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().to('cpu').pow(2).sum().item())
    return math.sqrt(total_sq)


# ------------------------ per-k grad norms (stacked) ------------------------
@torch.no_grad()
def _per_k_grad_norms(support_sets) -> np.ndarray:
    K = support_sets.num_support_sets
    # Keep accumulation on the same device as gradients (prevents device sync/copies)
    try:
        dev = next(support_sets.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    g2 = torch.zeros(K, dtype=torch.float32, device=dev)

    for p in support_sets.parameters():
        g = p.grad
        if g is None:
            continue
        g = g.detach().float()
        if g.ndim == 0:
            continue
            
        # find which axis corresponds to K (don’t assume it’s dim 0)
        axes_with_K = [ax for ax, sz in enumerate(g.shape) if sz == K]
        if not axes_with_K:
            continue
        k_ax = axes_with_K[0]
        if k_ax != 0:
            g = g.movedim(k_ax, 0)  # put K in front
        g2 += g.reshape(K, -1).pow(2).sum(dim=1)
    return torch.sqrt(torch.clamp(g2, min=1e-12)).detach().cpu().numpy()

    
def _pack_BK(x_bk: torch.Tensor, *, return_b_idx: bool = True):
    """
    Flatten [B,K,...] → [B*K,...] in a *known* order and return the mapping + targets.
    Order: (b=0,k=0..K-1), (b=1,k=0..K-1), ...
    """
    assert x_bk.dim() >= 2, f"Expected [B,K,...], got {tuple(x_bk.shape)}"
    B, K = x_bk.shape[:2]

    # Make contiguous only if needed; use reshape to avoid copies when possible.
    x_bk_c = x_bk if x_bk.is_contiguous() else x_bk.contiguous()
    flat = x_bk_c.reshape(B * K, *x_bk_c.shape[2:])

    # Mapping & targets (same device as input)
    dev = x_bk.device
    k_idx = torch.arange(K, device=dev).repeat(B)                # [0..K-1, 0..K-1, ...]
    targets = k_idx                                              # class k for row (b,k)
    b_idx = torch.arange(B, device=dev).repeat_interleave(K) if return_b_idx else None  # [0,0,...,1,1,...]

    return flat, targets, (b_idx, k_idx), (B, K)








# ============================================================
# TrainingStatTracker (updated to support global_opt_step)
# ============================================================
class TrainingStatTracker(object):
    """
    Tracks metrics at two levels:
      - micro-step accumulation (within a grad-acc window)
      - optimizer-step aggregates (emitted once per window)

    Also tracks per-MLP analytics (EMA accuracy, EMA grad-norm, selection counts, confusion),
    optional histories for heatmaps, and learning rates.
    """

    def __init__(self, ema_decay: float = 0.9, ema_max_history: int = 200):
        # Window (micro-step) accumulators
        self._reset_window()

        # Global optimizer-step index (used by logging)
        # Convention: this marks the CURRENT step id used for logging;
        # it is incremented AFTER finalize_step() completes.
        self.global_opt_step: int = 0

        # LRs (latest seen per optimizer-step)
        self.last_support_lr = 0.0
        self.last_recognizer_lr = 0.0

        # Timing
        self.iter_times = np.array([])  # seconds per opt step

        # Per-MLP analytics (set after K is known)
        self.K = None
        self.ema_decay = float(ema_decay)
        self.ema_max_history = int(ema_max_history)
        self.per_k_ema_acc = None         # [K] float
        self.per_k_ema_grad = None        # [K] float
        self.per_k_select_counts = None   # [K] long
        self.confusion = None             # [K, K] long  (row=true, col=pred)
        self.ema_history = []             # list of np.array([K]) snapshots
        self.iter_history = []            # matching optimizer-step indices for heatmap

        # JSON-like store of per-step aggregates (string keys)
        self.stats_by_step = {}  # {step_idx: dict}

        # Pairwise (class x class) distance tracking (optional; filled only when computed)
        self.pairwise_avg_dist = None   # [K,K] float
        self.pairwise_w1 = None         # [K,K] float
        self.pairwise_w2 = None         # [K,K] float
        self.pairwise_steps = []        # list[int]
        self.pairwise_w1_mean_hist = [] # list[float]
        self.pairwise_w2_mean_hist = [] # list[float]

        # Information matrix / disentanglement tracking
        self.info_lambda = None            # [K,K] float (Lambda_ij = (dot)^2)
        self.info_conf_prob = None         # [K,K] float (row-stochastic p(pred|true))
        self.info_steps = []               # list[int]
        # histories (compute both metrics for both matrices)
        self.info_Q_lambda_hist = []       # list[float]  Q(Lambda)
        self.info_Q_conf_hist = []         # list[float]  Q(mathbf p)
        self.info_deff_lambda_hist = []    # list[float]  d_eff(Lambda)
        self.info_deff_conf_hist = []      # list[float]  d_eff(mathbf p^Tmathbf p)

    # ---------- window (micro-steps) ----------
    def _reset_window(self):
        self.win_count = 0
        self.win_sum =  dict()

    def _acc(self, key: str, val: float | None):
        if val is None:  # allow optional arguments
            return
        self.win_sum[key] = self.win_sum.get(key, 0.0) + float(val)

    def add_micro(
        self,
        *,
        acc: float,
        classification_loss: float,
        total_loss: float,
        entropy: float = 0.0,
        step1_norm: float = 0.0,
        step2_norm: float = 0.0,
        potential_std: float = 0.0,
        xf_now: float = 0.0,
        # ---- allow arbitrary extras without breaking ----
        **extras,
    ):
        """Accumulate values from a micro-step; all inputs are Python floats."""
        self.win_count += 1
        self._acc('accuracy_index', acc)
        self._acc('L_classification', classification_loss)
        self._acc('total_loss', total_loss)
        self._acc('entropy', entropy)
        self._acc('step1_norm', step1_norm)
        self._acc('step2_norm', step2_norm)
        self._acc('potential_std', potential_std)
        self._acc('xf_now', xf_now)
        # PDE components

        # Any extra scalar metrics can be merged automatically
        for k, v in extras.items():
            try:
                self._acc(k, v)
            except Exception:
                # ignore non-scalar or malformed extras
                pass

    def close_window(self):
        """Return window means and reset micro accumulators."""
        denom = max(1, self.win_count)
        means = {k: (v / denom) for k, v in self.win_sum.items()}
        self._reset_window()
        return means

    # ---------- per-MLP analytics ----------
    def init_per_k(self, K: int):
        """Call once when K is known."""
        self.K = int(K)
        self.per_k_ema_acc = np.zeros(self.K, dtype=np.float32)
        self.per_k_ema_grad = np.zeros(self.K, dtype=np.float32)
        self.per_k_select_counts = np.zeros(self.K, dtype=np.int64)
        self.confusion = np.zeros((self.K, self.K), dtype=np.int64)
        self.ema_history.clear()
        self.iter_history.clear()

        self.pairwise_avg_dist = None
        self.pairwise_w1 = None
        self.pairwise_w2 = None
        self.pairwise_steps.clear()
        self.pairwise_w1_mean_hist.clear()
        self.pairwise_w2_mean_hist.clear()

        self.info_lambda = None
        self.info_conf_prob = None
        self.info_steps.clear()
        self.info_Q_lambda_hist.clear()
        self.info_Q_conf_hist.clear()
        self.info_deff_lambda_hist.clear()
        self.info_deff_conf_hist.clear()

    def update_information_metrics(
        self,
        *,
        step_idx: int,
        lambda_kxk: np.ndarray,
        conf_prob_kxk: np.ndarray,
        max_history: int = 200,
        eps: float = 1e-8,
    ):
        """
        Store latest Lambda and confusion-prob matrix, and append scalar invariants histories.

        Lambda is expected to be symmetric and nonnegative (squared dot products).
        Confusion prob is expected row-stochastic (rows sum to ~1 when counts exist).
        For invariant comparisons, we use the symmetric Gram matrix G = Σ^T Σ.
        """
        if self.K is None:
            return
        K = int(self.K)
        L = np.array(lambda_kxk, dtype=np.float64).reshape(K, K)
        S = np.array(conf_prob_kxk, dtype=np.float64).reshape(K, K)
        self.info_lambda = L.astype(np.float32)
        self.info_conf_prob = S.astype(np.float32)

        def _Q(A: np.ndarray) -> float:
            # Q(A)=det(A)^2 / det(A⊙A) with small diagonal regularization.
            A = np.array(A, dtype=np.float64, copy=False)
            A_reg = A + float(eps) * np.eye(K, dtype=np.float64)
            AA_reg = (A * A) + float(eps) * np.eye(K, dtype=np.float64)
            s1, ld1 = np.linalg.slogdet(A_reg)
            s2, ld2 = np.linalg.slogdet(AA_reg)
            if s1 <= 0 or s2 <= 0:
                return 0.0
            return float(np.exp(2.0 * ld1 - ld2))

        def _deff(A: np.ndarray) -> float:
            # d_eff(A)=(tr A)^2 / tr(A^2) where A^2 is matrix product.
            A = np.array(A, dtype=np.float64, copy=False)
            tr = float(np.trace(A))
            tr2 = float(np.trace(A @ A))
            return float((tr * tr) / max(float(eps), tr2))

        # Use a symmetric Gram matrix for confusion: G = S^T S (PSD).
        G = S.T @ S
        Q_L = _Q(L)
        Q_G = _Q(G)
        dL = _deff(L)
        dG = _deff(G)

        self.info_steps.append(int(step_idx))
        self.info_Q_lambda_hist.append(Q_L)
        self.info_Q_conf_hist.append(Q_G)
        self.info_deff_lambda_hist.append(dL)
        self.info_deff_conf_hist.append(dG)

        if len(self.info_steps) > int(max_history):
            self.info_steps = self.info_steps[-int(max_history):]
            self.info_Q_lambda_hist = self.info_Q_lambda_hist[-int(max_history):]
            self.info_Q_conf_hist = self.info_Q_conf_hist[-int(max_history):]
            self.info_deff_lambda_hist = self.info_deff_lambda_hist[-int(max_history):]
            self.info_deff_conf_hist = self.info_deff_conf_hist[-int(max_history):]

    def update_pairwise_metrics(
        self,
        *,
        step_idx: int,
        avg_dist_kxk: np.ndarray,
        w1_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        max_history: int = 200,
    ):
        """
        Store the latest pairwise matrices, and append mean W1/W2 (off-diagonal) histories.
        """
        if self.K is None:
            return
        K = int(self.K)
        self.pairwise_avg_dist = np.array(avg_dist_kxk, dtype=np.float32).reshape(K, K)
        self.pairwise_w1 = np.array(w1_kxk, dtype=np.float32).reshape(K, K)
        self.pairwise_w2 = np.array(w2_kxk, dtype=np.float32).reshape(K, K)

        mask = ~np.eye(K, dtype=bool)
        w1_mean = float(self.pairwise_w1[mask].mean()) if mask.any() else float(self.pairwise_w1.mean())
        w2_mean = float(self.pairwise_w2[mask].mean()) if mask.any() else float(self.pairwise_w2.mean())
        self.pairwise_steps.append(int(step_idx))
        self.pairwise_w1_mean_hist.append(w1_mean)
        self.pairwise_w2_mean_hist.append(w2_mean)

        # cap history
        if len(self.pairwise_steps) > int(max_history):
            self.pairwise_steps = self.pairwise_steps[-int(max_history):]
            self.pairwise_w1_mean_hist = self.pairwise_w1_mean_hist[-int(max_history):]
            self.pairwise_w2_mean_hist = self.pairwise_w2_mean_hist[-int(max_history):]

    def update_per_k_after_micro(
        self,
        *,
        true_k: int,
        preds: np.ndarray,         # shape [B] int64 on CPU
        batch_size: int,
        grad_norm_selected_mlp: float | None = None,
    ):
        """
        Update EMA accuracy, selection counts, and confusion for the selected k of THIS micro-step.
        - Only the selected MLP's grad-norm is meaningful to track.
        """
        if self.K is None:
            return
        true_k = int(true_k)
        # acc for this micro-batch against the selected k
        acc = float((preds == true_k).mean())
        self.per_k_ema_acc[true_k] = self.per_k_ema_acc[true_k] * self.ema_decay + acc * (1.0 - self.ema_decay)
        self.per_k_select_counts[true_k] += int(batch_size)

        # confusion row update (count predicted classes)
        preds = np.asarray(preds).reshape(-1)
        if preds.dtype.kind not in "iu":
            preds = preds.astype(np.int64, copy=False)
        K = int(self.K)
        valid = (preds >= 0) & (preds < K)
        if not np.all(valid):
            preds = preds[valid]
        binc = np.bincount(preds, minlength=K).astype(np.int64)
        self.confusion[true_k, :] += binc

        # grad norm EMA (only for the MLP that received grads)
        if grad_norm_selected_mlp is not None:
            g = float(grad_norm_selected_mlp)
            self.per_k_ema_grad[true_k] = self.per_k_ema_grad[true_k] * self.ema_decay + g * (1.0 - self.ema_decay)

    def snapshot_per_k_history(self, step_idx: int):
        """Keep a thin history (capped) for heatmaps."""
        if self.K is None:
            return
        self.ema_history.append(self.per_k_ema_acc.copy())
        self.iter_history.append(int(step_idx))
        if len(self.ema_history) > self.ema_max_history:
            self.ema_history = self.ema_history[-self.ema_max_history:]
            self.iter_history = self.iter_history[-self.ema_max_history:]

    # ---------- LRs ----------
    def set_lrs(self, support_lr: float, recognizer_lr: float):
        self.last_support_lr = float(support_lr)
        self.last_recognizer_lr = float(recognizer_lr)

    # Allow trainer to sync starting index (resume)
    def set_opt_step(self, step_idx: int):
        self.global_opt_step = int(step_idx)

    # ---------- per-step finalize ----------
    def finalize_step(
        self,
        *,
        step_idx: int,
        window_means: dict,
        elapsed_from_start: float,
        mean_step_time: float,
        eta_seconds: float,
    ):
        """
        Called once per optimizer step to store a compact dictionary of metrics.
        - Stores under the provided step_idx
        - Exposes both legacy and new metric keys for backward compatibility
        - Increments global_opt_step AFTER storing
        """
        rec = dict(window_means)

        # Backward-compat aliases expected by some logs
        if 'classification_loss' not in rec and 'L_classification' in rec:
            rec['classification_loss'] = rec['L_classification']
        if 'kl_loss' not in rec and 'L_kl' in rec:
            rec['kl_loss'] = rec['L_kl']

        rec.update({
            'support_sets_lr': self.last_support_lr,
            'recognizer_lr': self.last_recognizer_lr,
            'mean_step_time_sec': float(mean_step_time),
            'elapsed_sec': float(elapsed_from_start),
            'eta_sec': float(eta_seconds),
        })
        self.stats_by_step[int(step_idx)] = rec

        # Advance the global step *after* storing
        self.global_opt_step = int(step_idx) + 1

    # ---------- time helpers ----------
    def push_step_time(self, dt_seconds: float):
        self.iter_times = np.append(self.iter_times, float(dt_seconds))

    def mean_step_time(self) -> float:
        return float(self.iter_times.mean()) if self.iter_times.size > 0 else 0.0


def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_stdout(num_lines):
    """Move cursor up and clear lines in terminal-friendly way."""
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[2K'
    for _ in range(num_lines):
        sys.stdout.write(cursor_up + erase_line + '\r')
    sys.stdout.flush()


def sec2dhms(t):
    """Convert seconds to 'DD days, HH hours, MM minutes, and SS seconds'."""
    t = int(t)
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)

def get_wh(img_paths):
    """Get width and height of images in given list of paths. Images are expected to have the same resolution.

    Args:
        img_paths (list): list of image paths

    Returns:
        width (int)  : the common images width
        height (int) : the common images height

    """
    img_widths = []
    img_heights = []
    for img in img_paths:
        img_ = Image.open(img)
        img_widths.append(img_.width)
        img_heights.append(img_.height)

    if len(set(img_widths)) == len(set(img_heights)) == 1:
        return img_widths[0], img_heights[1]
    else:
        raise ValueError("Inconsistent image resolutions in {}".format(img_paths))


def create_summarizing_gif(imgs_root, gif_filename, num_imgs=None, gif_size=None, gif_fps=30, gap=15, progress_bar_h=15,
                           progress_bar_color=(252, 186, 3)):
    """Create a summarizing GIF image given an images root directory (images generated across a certain latent path) and
    the number of images to appear as a static sequence. The resolution of the resulting GIF image will be
    ((num_imgs + 1) * gif_size, gif_size). That is, a static sequence of `num_imgs` images will be depicted in front of
    the animated GIF image (the latter will use all the available images in `imgs_root`).

    Args:
        imgs_root (str)            : directory of images (generated across a certain path)
        gif_filename (str)         : filename of the resulting GIF image
        num_imgs (int)             : number of images that will be used to build the static sequence before the
                                     animated part of the GIF
        gif_size (int)             : height of the GIF image (its width will be equal to (num_imgs + 1) * gif_size)
        gif_fps (int)              : GIF frames per second
        gap (int)                  : a gap between the static sequence and the animated path of the GIF
        progress_bar_h (int)       : height of the progress bar depicted to the bottom of the animated part of the GIF
                                     image. If a non-positive number is given, progress bar will be disabled.
        progress_bar_color (tuple) : color of the progress bar

    """
    # Check if given images root directory exists
    if not osp.isdir(imgs_root):
        raise NotADirectoryError("Invalid directory: {}".format(imgs_root))

    # Get all images under given root directory
    path_images = [osp.join(imgs_root, dI) for dI in os.listdir(imgs_root) if osp.isfile(osp.join(imgs_root, dI))]
    path_images.sort()

    # Set number of images to appear in the static sequence of the GIF
    num_images = len(path_images)
    if num_imgs is None:
        num_imgs = num_images
    elif num_imgs > num_images:
        num_imgs = num_images

    # Get paths of static images
    static_imgs = []
    for i in range(0, len(path_images), math.ceil(len(path_images) / num_imgs)):
        static_imgs.append(osp.join(imgs_root, '{:06}.jpg'.format(i)))
    num_imgs = len(static_imgs)

    # Get GIF image resolution
    if gif_size is not None:
        gif_w = gif_h = gif_size
    else:
        gif_w, gif_h = get_wh(static_imgs)

    # Create PIL static image
    static_img_pil = Image.new('RGB', size=(len(static_imgs) * gif_w, gif_h))
    for i in range(len(static_imgs)):
        static_img_pil.paste(Image.open(static_imgs[i]).resize((gif_w, gif_h)), (i * gif_w, 0))

    # Create PIL GIF frames
    gif_frames = []
    for i in range(len(path_images)):
        # Create new PIL frame
        gif_frame_pil = Image.new('RGB', size=((num_imgs + 1) * gif_w + gap, gif_h), color=(255, 255, 255))

        # Paste static image
        gif_frame_pil.paste(static_img_pil, (0, 0))

        # Paste current image
        gif_frame_pil.paste(Image.open(path_images[i]).resize((gif_w, gif_h)), (num_imgs * gif_w + gap, 0))

        # Draw progress bar
        if progress_bar_h > 0:
            gif_frame_pil_drawing = ImageDraw.Draw(gif_frame_pil)
            progress = (i / len(path_images)) * gif_w
            gif_frame_pil_drawing.rectangle(xy=[num_imgs * gif_w + gap, gif_h - progress_bar_h,
                                                num_imgs * gif_w + gap + progress, gif_h],
                                            fill=progress_bar_color)

        # Append to GIF frames list
        gif_frames.append(gif_frame_pil)

    # Save GIF file
    gif_frames[0].save(
        fp=gif_filename,
        append_images=gif_frames[1:],
        save_all=True,
        optimize=False,
        loop=0,
        duration=1000 // gif_fps)

from typing import Iterable, Sequence, Union, Optional
import math
from torch.optim.lr_scheduler import _LRScheduler

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
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    return CosineScheduleWithWarmup(optimizer, num_warmup_steps, num_training_steps, last_epoch)



# image_logging.py
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import re

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # optional; only used for GIF utilities


class ImageViz:
    """All image-creation helpers (moved out of Trainer)."""
    _manifold_state = {}

    @staticmethod
    def to_uint01(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().float().cpu()
        if torch.numel(x) == 0:
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if x.min() < 0.0:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    @staticmethod
    def _pick_first_batch_and_K(t: torch.Tensor, k_vis: int) -> torch.Tensor:
        if t.ndim == 5:            # [B,K,C,H,W]
            return t[0, :k_vis]
        elif t.ndim == 4:          # [B,C,H,W]
            return t[:k_vis]
        elif t.ndim == 3:          # [C,H,W]
            return t.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor ndim={t.ndim} for visualization")

    @staticmethod
    def _infer_k_vis(step1_src: torch.Tensor, n_vis: int) -> int:
        if step1_src.ndim == 5:
            return min(int(n_vis), int(step1_src.shape[1]))
        elif step1_src.ndim == 4:
            return min(int(n_vis), int(step1_src.shape[0]))
        else:
            return int(n_vis)

    @staticmethod
    def _maybe_downscale(t: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
        if scale is None or abs(scale - 1.0) < 1e-6:
            return t
        return F.interpolate(t.float(), scale_factor=scale, mode="area").type_as(t)

    @classmethod
    def make_triplet_grids(
        cls,
        x0: torch.Tensor,  # step1
        x1: torch.Tensor,  # step2
        x2: torch.Tensor,  # ref
        n_vis: int = 8,
        downscale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_vis = cls._infer_k_vis(x0, n_vis)
        s1 = cls._pick_first_batch_and_K(x0, k_vis)
        s2 = cls._pick_first_batch_and_K(x1, k_vis)

        # reference shape handling
        if x2.ndim == 5:
            ref = x2[0, :k_vis]
        elif x2.ndim == 4:
            ref = x2.repeat(k_vis, 1, 1, 1) if x2.shape[0] == 1 else x2[:k_vis]
        elif x2.ndim == 3:
            ref = x2.unsqueeze(0).repeat(k_vis, 1, 1, 1)
        elif x2.ndim == 2:
            # Allow latent-shaped ref [B,D] (e.g. identity toy generator).
            # Try to reshape it into an image using the shape of x0/x1.
            if x0.ndim == 5:
                C, H, W = int(x0.shape[2]), int(x0.shape[3]), int(x0.shape[4])
            elif x0.ndim == 4:
                C, H, W = int(x0.shape[1]), int(x0.shape[2]), int(x0.shape[3])
            else:
                raise ValueError(f"Cannot infer image shape from x0.ndim={x0.ndim} when x2 is [B,D]")

            D = int(x2.shape[1])
            if D != C * H * W:
                raise ValueError(
                    f"Reference is [B,D] with D={D}, but expected D=C*H*W={C}*{H}*{W}={C*H*W}. "
                    "Pass an image-shaped reference, or make generator(z) return [B,C,H,W]."
                )
            x2_img = x2.view(x2.shape[0], C, H, W)
            ref = x2_img.repeat(k_vis, 1, 1, 1) if x2_img.shape[0] == 1 else x2_img[:k_vis]
        else:
            raise ValueError(f"Unexpected reference tensor ndim={x2.ndim}")

        # optional downscale
        ref = cls._maybe_downscale(ref, downscale)
        s1  = cls._maybe_downscale(s1,  downscale)
        s2  = cls._maybe_downscale(s2,  downscale)

        ref_n = cls.to_uint01(ref)
        s1_n  = cls.to_uint01(s1)
        s2_n  = cls.to_uint01(s2)

        grid_ref = make_grid(ref_n, nrow=k_vis)
        grid_s1  = make_grid(s1_n,  nrow=k_vis)
        grid_s2  = make_grid(s2_n,  nrow=k_vis)
        stacked  = torch.cat([grid_ref, grid_s1, grid_s2], dim=1)

        diff1 = (s1_n - ref_n)
        diff2 = (s2_n - s1_n)
        grid_d1 = make_grid(diff1, nrow=k_vis)
        grid_d2 = make_grid(diff2, nrow=k_vis)
        stacked_diff = torch.cat([grid_d1, grid_d2], dim=1)
        return stacked, stacked_diff

    @staticmethod
    def plot_heatmap(mat_t_by_k, K: int, title: str, xlabel: str, ylabel: str):
        arr = np.array(mat_t_by_k)
        if arr.ndim == 2 and arr.shape[0] != K:
            arr = arr.T
        fig, ax = plt.subplots(figsize=(max(6, arr.shape[1] * 0.15), max(4, K * 0.15)))
        im = ax.imshow(arr, aspect='auto', origin='lower', interpolation='nearest')
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(K)); ax.set_yticklabels([str(i) for i in range(K)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Figure saving helpers
    # ----------------------------
    @staticmethod
    def save_fig_copy(fig, *, out_dir: str | Path, tag: str, step: int, dpi: int = 140) -> str:
        """
        Save a deterministic PNG copy of a Matplotlib figure for later collation (e.g., GIF).
        Returns the written path as a string.
        """
        out_dir = Path(out_dir) / str(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{int(step):08d}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        return str(path)

    @staticmethod
    def frames_to_gif(
        *,
        frames_dir: str | Path,
        out_gif: str | Path,
        fps: float = 6.0,
        glob_pattern: str = "*.png",
        resize: tuple[int, int] | None = None,
    ) -> str:
        """
        Collate PNG frames in `frames_dir` into a GIF.
        Frames are sorted by the leading integer in the filename, falling back to lexicographic.
        """
        if imageio is None:
            raise ImportError("imageio is required to write GIFs; install `imageio`.")

        frames_dir = Path(frames_dir)
        out_gif = Path(out_gif)
        files = sorted(frames_dir.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No frames found in {frames_dir} with glob {glob_pattern!r}")

        def key(p: Path):
            m = re.match(r"^(\d+)", p.stem)
            return (int(m.group(1)) if m else 10**18, p.name)

        files = sorted(files, key=key)
        duration = 1.0 / max(1e-6, float(fps))

        imgs = []
        for f in files:
            im = imageio.imread(f)
            if resize is not None:
                # simple nearest-neighbor resize via PIL if available
                try:
                    from PIL import Image
                    im = np.array(Image.fromarray(im).resize(resize, resample=Image.Resampling.BILINEAR))
                except Exception:
                    pass
            imgs.append(im)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_gif, imgs, duration=duration, loop=0)
        return str(out_gif)

    @staticmethod
    def plot_confusion(conf_mat_nd: np.ndarray, K: int):
        cm = torch.tensor(conf_mat_nd, dtype=torch.float32)
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        cm_norm = (cm / row_sums).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation='nearest', aspect='auto', origin='lower')
        ax.set_title("Classifier Confusion (row=true k, col=pred k)")
        ax.set_xlabel("predicted k"); ax.set_ylabel("true k")
        ax.set_xticks(np.arange(K)); ax.set_yticks(np.arange(K))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Pairwise distances + OT (class x class)
    # ----------------------------
    @staticmethod
    @torch.no_grad()
    def compute_pairwise_avg_and_wasserstein(
        z_bkd: torch.Tensor,
        *,
        solver: str = "sinkhorn",         # "sinkhorn" | "emd"
        reg: float = 3e-2,               # sinkhorn regularization
        ot_numItermax: int = 7_000,
        max_points_per_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-class pairwise:
        - avg L2 distance between samples (mean of all pairwise distances)
        - Wasserstein-1 distance (OT cost with L2 ground metric)
        - Wasserstein-2 distance (sqrt of OT cost with squared L2 ground metric)

        Args:
            z_bkd: [B,K,D] latent samples per class k
        Returns:
            avg_dist[K,K], w1[K,K], w2[K,K] as float32 numpy arrays
        """
        if z_bkd.dim() != 3:
            raise ValueError(f"Expected z_bkd [B,K,D], got {tuple(z_bkd.shape)}")
        B, K, D = map(int, z_bkd.shape)

        # optional subsampling for speed
        if max_points_per_class is not None and B > int(max_points_per_class):
            idx = torch.randperm(B, device=z_bkd.device)[: int(max_points_per_class)]
            z_bkd = z_bkd.index_select(0, idx)
            B = int(z_bkd.shape[0])

        # POT is CPU/numpy oriented; keep it local-imported
        try:
            import ot  # POT
        except Exception as e:
            raise ImportError("POT is required for OT distances; install `POT`.") from e

        z = z_bkd.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)   # [B,K,D]
        a = np.full((B,), 1.0 / float(B), dtype=np.float64)

        avg = np.zeros((K, K), dtype=np.float64)
        w1 = np.zeros((K, K), dtype=np.float64)
        w2 = np.zeros((K, K), dtype=np.float64)

        solver = str(solver).lower().strip()
        if solver not in {"sinkhorn", "emd"}:
            raise ValueError("solver must be one of {'sinkhorn','emd'}")

        for i in range(K):
            Xi = z[:, i, :]  # [B,D]
            for j in range(i, K):
                if i == j:
                    continue
                Xj = z[:, j, :]
                # cost matrices
                diff = Xi[:, None, :] - Xj[None, :, :]
                C2 = np.sum(diff * diff, axis=-1)              # squared L2
                C1 = np.sqrt(np.maximum(C2, 0.0))              # L2

                avg_ij = float(np.mean(C1))

                if solver == "emd":
                    w1_cost = float(ot.emd2(a, a, C1, numItermax=int(ot_numItermax)))
                    w2_cost = float(ot.emd2(a, a, C2, numItermax=int(ot_numItermax)))
                else:
                    w1_cost = float(ot.sinkhorn2(a, a, C1, reg=float(reg), numItermax=int(ot_numItermax)))
                    w2_cost = float(ot.sinkhorn2(a, a, C2, reg=float(reg), numItermax=int(ot_numItermax)))

                w2_val = math.sqrt(max(0.0, w2_cost))

                avg[i, j] = avg[j, i] = avg_ij
                w1[i, j] = w1[j, i] = w1_cost
                w2[i, j] = w2[j, i] = w2_val

        return avg.astype(np.float32), w1.astype(np.float32), w2.astype(np.float32)

    @staticmethod
    def plot_pairwise_distance_ot_panel(
        *,
        avg_dist_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        w1_mean_hist: list[float] | np.ndarray,
        w2_mean_hist: list[float] | np.ndarray,
        title: str = "Pairwise class distances (avg) + OT (W2) with mean W1/W2 histories",
    ):
        """
        2x2 panel:
        - (0,0) heatmap: average pairwise L2 distance
        - (0,1) heatmap: Wasserstein-2 distance
        - (1,0) line: mean Wasserstein-1 over time
        - (1,1) line: mean Wasserstein-2 over time
        """
        avg = np.array(avg_dist_kxk, dtype=np.float32)
        w2 = np.array(w2_kxk, dtype=np.float32)
        if avg.ndim != 2 or avg.shape[0] != avg.shape[1]:
            raise ValueError(f"avg_dist_kxk must be square [K,K], got {avg.shape}")
        if w2.shape != avg.shape:
            raise ValueError(f"w2_kxk must match avg_dist_kxk, got {w2.shape} vs {avg.shape}")
        K = int(avg.shape[0])

        # Keep the heatmap row taller (square panels) and make the time-series row flatter.
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 11),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax00, ax01 = axs[0, 0], axs[0, 1]
        ax10, ax11 = axs[1, 0], axs[1, 1]

        im0 = ax00.imshow(avg, interpolation="nearest", aspect="auto", origin="lower")
        ax00.set_title("Avg L2 distance (between class samples)")
        ax00.set_xlabel("class j"); ax00.set_ylabel("class i")
        ax00.set_xticks(np.arange(K)); ax00.set_yticks(np.arange(K))
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(w2, interpolation="nearest", aspect="auto", origin="lower")
        ax01.set_title("OT distance (W2)")
        ax01.set_xlabel("class j"); ax01.set_ylabel("class i")
        ax01.set_xticks(np.arange(K)); ax01.set_yticks(np.arange(K))
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        y_w1 = np.array(w1_mean_hist, dtype=np.float32).reshape(-1)
        y_w2 = np.array(w2_mean_hist, dtype=np.float32).reshape(-1)

        ax10.plot(xs, y_w1, linewidth=1.8)
        ax10.set_title("Mean Wasserstein-1 (off-diagonal) vs step")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("mean W1")
        ax10.grid(True, linewidth=0.3, alpha=0.4)

        ax11.plot(xs, y_w2, linewidth=1.8)
        ax11.set_title("Mean Wasserstein-2 (off-diagonal) vs step")
        ax11.set_xlabel("opt step"); ax11.set_ylabel("mean W2")
        ax11.grid(True, linewidth=0.3, alpha=0.4)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Bayesian/free-energy style panel (KL + KL/W2)
    # ----------------------------
    @staticmethod
    def plot_pairwise_kl_free_energy_panel(
        *,
        kl_kxk: np.ndarray,
        w2_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        kl_mean_hist: list[float] | np.ndarray,
        ratio_mean_hist: list[float] | np.ndarray,
        eps: float = 1e-8,
        title: str = "Pairwise KL (entropic) + KL/W2 ratio with mean histories",
    ):
        """
        2x2 panel:
        - (0,0) heatmap: KL_eps between entropically smoothed point clouds
        - (0,1) heatmap: KL_eps / W2_eps ratio
        - (1,0) line: mean KL_eps over time
        - (1,1) line: mean KL_eps / W2_eps over time
        """
        kl = np.array(kl_kxk, dtype=np.float32)
        w2 = np.array(w2_kxk, dtype=np.float32)
        if kl.ndim != 2 or kl.shape[0] != kl.shape[1]:
            raise ValueError(f"kl_kxk must be square [K,K], got {kl.shape}")
        if w2.shape != kl.shape:
            raise ValueError(f"w2_kxk must match kl_kxk, got {w2.shape} vs {kl.shape}")
        K = int(kl.shape[0])

        denom = np.maximum(w2, float(eps))
        ratio = kl / denom

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 4]},
        )
        ax00, ax01 = axs[0, 0], axs[0, 1]
        ax10, ax11 = axs[1, 0], axs[1, 1]

        im0 = ax00.imshow(kl, interpolation="nearest", aspect="auto", origin="lower")
        ax00.set_title("KL_eps (entropic)")
        ax00.set_xlabel("class j"); ax00.set_ylabel("class i")
        ax00.set_xticks(np.arange(K)); ax00.set_yticks(np.arange(K))
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(ratio, interpolation="nearest", aspect="auto", origin="lower")
        ax01.set_title("KL_eps / W2_eps")
        ax01.set_xlabel("class j"); ax01.set_ylabel("class i")
        ax01.set_xticks(np.arange(K)); ax01.set_yticks(np.arange(K))
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        y_kl = np.array(kl_mean_hist, dtype=np.float32).reshape(-1)
        y_ratio = np.array(ratio_mean_hist, dtype=np.float32).reshape(-1)

        ax10.plot(xs, y_kl, linewidth=1.8)
        ax10.set_title("Mean KL_eps (off-diagonal) vs step")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("mean KL")
        ax10.grid(True, linewidth=0.3, alpha=0.4)

        ax11.plot(xs, y_ratio, linewidth=1.8)
        ax11.set_title("Mean KL_eps / W2_eps (off-diagonal) vs step")
        ax11.set_xlabel("opt step"); ax11.set_ylabel("mean ratio")
        ax11.grid(True, linewidth=0.3, alpha=0.4)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Information matrix + confusion panel
    # ----------------------------
    @staticmethod
    def plot_information_matrix_panel(
        *,
        lambda_kxk: np.ndarray,
        conf_prob_kxk: np.ndarray,
        steps: list[int] | np.ndarray,
        Q_lambda_hist: list[float] | np.ndarray,
        Q_conf_hist: list[float] | np.ndarray,
        deff_lambda_hist: list[float] | np.ndarray,
        deff_conf_hist: list[float] | np.ndarray,
        title: str = "Information matrix Λ (squared dot products) + confusion with invariants",
    ):
        """
        2x2 panel (wider for big heatmaps):
        - (0,0) heatmap: Λ_ij = (v_i · v_j)^2
        - (0,1) heatmap: row-normalized confusion p(pred|true)
        - (1,0) line: Q(Λ) = det(Λ)^2 / det(Λ ⊙ Λ)
        - (1,1) line: d_eff(Λ) and d_eff(G) with G = Σ^T Σ
        """
        L = np.array(lambda_kxk, dtype=np.float32)
        C = np.array(conf_prob_kxk, dtype=np.float32)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"lambda_kxk must be square [K,K], got {L.shape}")
        if C.shape != L.shape:
            raise ValueError(f"conf_prob_kxk must match lambda_kxk, got {C.shape} vs {L.shape}")
        K = int(L.shape[0])

        # Set up gridspec with 2x2, upper row (row 0) subplots square, lower row (row 1) shorter
        from matplotlib import gridspec

        fig = plt.figure(figsize=(12, 8))
        # GridSpec: make top row height much bigger than bottom
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.38], figure=fig)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])

        # Ensure the upper row subplots are square
        ax00.set_box_aspect(1)
        ax01.set_box_aspect(1)
        # Lower row can be non-square ("shorter")

        # ticks: avoid unreadable plots when K is large
        stride = 1 if K <= 32 else max(1, K // 16)
        ticks = np.arange(0, K, stride, dtype=np.int64)

        im0 = ax00.imshow(L, interpolation="nearest", aspect="auto", origin="upper")
        ax00.set_title(r"$\Lambda_{ij} = (v_i \cdot v_j)^2$")
        ax00.set_xlabel("j (label)"); ax00.set_ylabel("i (label)")
        ax00.set_xticks(ticks); ax00.set_yticks(ticks)
        fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)
        ax00.set_box_aspect(1)

        im1 = ax01.imshow(C, interpolation="nearest", aspect="auto", origin="upper", vmin=0.0, vmax=1.0)
        ax01.set_title(r"$p(\mathrm{pred}\mid \mathrm{true})$ (row-normalized)")
        ax01.set_xlabel("pred"); ax01.set_ylabel("true")
        ax01.set_xticks(ticks); ax01.set_yticks(ticks)
        fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)
        ax01.set_box_aspect(1)

        xs = np.array(steps, dtype=np.int64).reshape(-1)
        yQ_L = np.array(Q_lambda_hist, dtype=np.float64).reshape(-1)
        yQ_S = np.array(Q_conf_hist, dtype=np.float64).reshape(-1)
        yD_L = np.array(deff_lambda_hist, dtype=np.float64).reshape(-1)
        yD_S = np.array(deff_conf_hist, dtype=np.float64).reshape(-1)

        ax10.plot(xs, yQ_L, linewidth=1.8, label="Q(Λ)")
        ax10.plot(xs, yQ_S, linewidth=1.8, label="Q(ΣᵀΣ)")
        ax10.set_title(r"$\mathcal{Q}(\Lambda)=\det(\Lambda)^2/\det(\Lambda\odot\Lambda)$")
        ax10.set_xlabel("opt step"); ax10.set_ylabel("Q(Λ)")
        ax10.grid(True, linewidth=0.3, alpha=0.4)
        ax10.legend(loc="best", fontsize=9, frameon=False)

        # Normalize effective dimension by ambient dimension D (here: matrix dimension K) so the scale is in [0,1].
        D_dim = float(K)
        yD_Ln = yD_L / max(1.0, D_dim)
        yD_Sn = yD_S / max(1.0, D_dim)
        ax11.plot(xs, yD_Ln, linewidth=1.8, label=r"$\frac{1}{D}d_{\mathrm{eff}}(\Lambda)$")
        ax11.plot(xs, yD_Sn, linewidth=1.8, label=r"$\frac{1}{D}d_{\mathrm{eff}}(\mathbf{p}^T\mathbf{p})$")
        ax11.set_title(r"Effective dimension/dimension: $\frac{1}{D}d_{\mathrm{eff}}(A)=(\mathrm{tr}\,A)^2/\mathrm{tr}(A^2)$")
        ax11.set_xlabel("opt step"); ax11.set_ylabel(r"$\frac{1}{D}d_{\mathrm{eff}}(\mathbf{p}^T\mathbf{p})$")
        ax11.grid(True, linewidth=0.3, alpha=0.4)
        ax11.set_ylim(0.0, 1.0)
        ax11.legend(loc="best", fontsize=9, frameon=False)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    @staticmethod
    @torch.no_grad()
    def compute_lambda_and_confusion(
        *,
        support_sets,
        recognizer,
        z_bd: torch.Tensor,   # [B,D]
        dt: torch.Tensor,     # [B,1] or [B,K]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute:
          - Λ_ij = E_b[(û_{b,i} · û_{b,j})^2] from per-class step vectors (squared cosine similarity)
          - Σ = p(pred|true) row-normalized confusion, using the recognizer on the same uv transform as training.
        """
        if z_bd.dim() != 2:
            raise ValueError(f"Expected z_bd [B,D], got {tuple(z_bd.shape)}")
        B, D = z_bd.shape
        K = int(getattr(support_sets, "num_support_sets", 1))

        # One-step per-k displacement field
        z_bkd, delta_bkd = support_sets.inference(z_bd, dt=dt, return_all=False)  # [B,K,D], [B,K,D]

        # ---- Lambda: average squared dot products between normalized deltas ----
        u = delta_bkd / delta_bkd.norm(dim=-1, keepdim=True).clamp_min_(1e-12)  # [B,K,D]
        dots_bkk = torch.einsum("bkd,bjd->bkj", u, u)  # [B,K,K]
        lam = (dots_bkk ** 2).mean(dim=0).to(dtype=torch.float32).detach().cpu().numpy()  # [K,K]

        # ---- Confusion: use training-time uv(a,b) = (2a-b, b) ----
        a = z_bkd.reshape(B * K, D)
        b = (z_bkd + delta_bkd).reshape(B * K, D)
        u_in = 2.0 * a - b
        v_in = b
        logits, _ = recognizer(u_in, v_in)  # [B*K,K]
        preds = logits.argmax(dim=-1)          # [B*K]
        targets = torch.arange(K, device=preds.device).repeat(B)  # [B*K]

        # Guard against logits with extra classes: drop out-of-range preds.
        valid = (preds >= 0) & (preds < K)
        if not torch.all(valid):
            preds = preds[valid]
            targets = targets[valid]

        if preds.numel() == 0:
            counts = torch.zeros((K, K), device=logits.device, dtype=torch.float32)
        else:
            idx = targets * K + preds
            counts = torch.bincount(idx, minlength=K * K).reshape(K, K).to(dtype=torch.float32)
        row_sums = counts.sum(dim=1, keepdim=True).clamp_min_(1.0)
        conf = (counts / row_sums).detach().cpu().numpy()  # [K,K]
        return lam, conf

    # ----------------------------
    # Nonlinear embeddings (final points)
    # ----------------------------
    @staticmethod
    def plot_final_points_nonlinear_embeddings(
        final_or_traj: torch.Tensor,
        *,
        max_points_per_class: int = 64,
        random_state: int = 0,
        state_key: str = "default",
        tsne_n_iter: int = 1000,
        tsne_n_iter_warm: int = 500,
        align_prev: bool = True,
        title: str = "Final points: nonlinear 2D embeddings (colored by class k)",
    ):
        """
        Make a 1x3 panel of nonlinear/manifold embeddings of the FINAL points, colored by class k.
        Uses a cached state per `state_key` to keep projections comparable across calls.
        Subsampling uses fixed indices per run; Isomap/Spectral fall back to PCA if the kNN graph is disconnected.

        Important: we do NOT subtract the initial position; we embed the raw final points.

        Args:
            final_or_traj:
              - [B,K,D] final points, or
              - [B,T+1,K,D] full trajectories (we take final timestep)
            max_points_per_class: sample up to this many points per class (default 64)
        """
        if final_or_traj.dim() == 4:
            z_bkd = final_or_traj[:, -1, :, :]  # [B,K,D]
        elif final_or_traj.dim() == 3:
            z_bkd = final_or_traj
        else:
            raise ValueError(f"Expected [B,K,D] or [B,T+1,K,D], got {tuple(final_or_traj.shape)}")

        if z_bkd.dim() != 3:
            raise ValueError(f"Expected z_bkd [B,K,D], got {tuple(z_bkd.shape)}")
        B, K, D = map(int, z_bkd.shape)

        # Stateful cache for stable, comparable projections across steps.
        state = ImageViz._manifold_state.setdefault(str(state_key), {})
        if state.get("B_raw") != B or state.get("K") != K or state.get("max_points") != int(max_points_per_class):
            state.clear()
            state["B_raw"] = B
            state["K"] = K
            state["max_points"] = int(max_points_per_class)

        # subsample to max_points_per_class per class (fixed indices for stability)
        if B > int(max_points_per_class):
            if state.get("subsample_idx") is None or state.get("subsample_B") != B:
                rng = np.random.RandomState(int(random_state))
                idx_np = rng.choice(B, size=int(max_points_per_class), replace=False)
                idx_np = np.sort(idx_np)
                state["subsample_idx"] = idx_np
                state["subsample_B"] = B
            idx = torch.as_tensor(state["subsample_idx"], device=z_bkd.device, dtype=torch.long)
            z_bkd = z_bkd.index_select(0, idx)
            B = int(z_bkd.shape[0])

        # Flatten to N x D and labels
        rng = np.random.RandomState(int(random_state))
        X = z_bkd.detach().to(device="cpu").reshape(B * K, D).numpy().astype(np.float64)
        X = X + rng.normal(loc=0.0, scale=1e-20, size=X.shape).astype(X.dtype, copy=False)
        labels = np.repeat(np.arange(K, dtype=np.int64), B)  # [K*B], blocks per k

        # local imports: keep core training lightweight unless plotting is enabled
        try:
            from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
        except Exception as e:
            raise ImportError("scikit-learn is required for nonlinear embeddings; install `scikit-learn`.") from e

        N = int(X.shape[0])
        n_neighbors = int(min(64, (N - 1)))
        perplexity = float(min(8.0, ((N - 1) / 3.0)))
        state["N"] = N

        def _is_valid_embedding(Y: np.ndarray) -> bool:
            if Y is None:
                return False
            if not np.isfinite(Y).all():
                return False
            return float(np.nanmax(np.std(Y, axis=0))) > 1e-8

        def _pca_fallback(X_in: np.ndarray) -> np.ndarray:
            if X_in.shape[1] >= 2:
                return X_in[:, :2].copy()
            return np.pad(X_in, ((0, 0), (0, 2 - X_in.shape[1])))

        def _align_embedding(Y: np.ndarray, Y_prev: np.ndarray) -> np.ndarray:
            if (Y_prev is None) or (Y_prev.shape != Y.shape):
                return Y
            if not _is_valid_embedding(Y) or not _is_valid_embedding(Y_prev):
                return Y
            Yc = Y - Y.mean(axis=0, keepdims=True)
            Pc = Y_prev - Y_prev.mean(axis=0, keepdims=True)
            denom = float(np.sum(Yc * Yc))
            if denom <= 1e-12:
                return Y
            M = Yc.T @ Pc
            if not np.isfinite(M).all():
                return Y
            try:
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
            except np.linalg.LinAlgError:
                return Y
            R = U @ Vt
            if np.linalg.det(R) < 0.0:
                Vt[-1, :] *= -1.0
                R = U @ Vt
            scale = float(np.trace((Yc @ R).T @ Pc)) / max(1e-12, denom)
            return (Yc @ R) * scale + Y_prev.mean(axis=0, keepdims=True)

        # optional PCA pre-reduction for speed/stability
        X_embed = X
        if D > 64:
            try:
                from sklearn.decomposition import PCA
                pca_dim = int(min(50, D, max(2, N - 1)))
                X_embed = PCA(
                    n_components=pca_dim,
                    svd_solver="randomized",
                    random_state=int(random_state),
                ).fit_transform(X_embed)
            except Exception:
                X_embed = X

        # pick a connected kNN graph size to avoid Isomap graph completion.
        try:
            from sklearn.neighbors import kneighbors_graph
            from scipy.sparse.csgraph import connected_components

            base_neighbors = int(min(N - 1, max(10, int(np.sqrt(N)))))
            max_neighbors = int(min(N - 1, max(64, int(np.sqrt(N) * 3))))
            k = max(2, base_neighbors)
            n_components = None
            graph_conn = None
            while True:
                graph_conn = kneighbors_graph(X_embed, n_neighbors=k, mode="connectivity", include_self=False)
                n_components = int(connected_components(graph_conn, directed=False, return_labels=False))
                if n_components <= 1 or k >= max_neighbors:
                    break
                k = min(max_neighbors, int(k * 1.5) + 1)
            n_neighbors = k
            can_graph = (n_components is not None) and (n_components <= 1)
        except Exception:
            graph_conn = None
            n_components = None
            can_graph = False

        # 3 embeddings
        prev_tsne = state.get("tsne") if align_prev else None
        tsne_iters = max(250, int(tsne_n_iter))

        tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=tsne_iters,
            random_state=int(random_state),
        )
        Y_tsne = tsne_model.fit_transform(X_embed)
        tsne_title = f"t-SNE (perplexity={perplexity:.1f})"
        if not _is_valid_embedding(Y_tsne):
            Y_tsne = _pca_fallback(X_embed)
            tsne_title = f"t-SNE (fallback PCA, perplexity={perplexity:.1f})"
        if align_prev and prev_tsne is not None:
            Y_tsne = _align_embedding(Y_tsne, prev_tsne)

        isomap_title = f"Isomap (n_neighbors={n_neighbors})"
        try:
            if not can_graph:
                raise RuntimeError("disconnected knn graph")
            Y_isomap = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X_embed)
            if not _is_valid_embedding(Y_isomap):
                raise RuntimeError("isomap produced degenerate embedding")
        except Exception:
            # fallback: PCA projection to keep plot stable if Isomap is ill-posed
            Y_isomap = _pca_fallback(X_embed)
            isomap_title = f"Isomap (fallback PCA, n_neighbors={n_neighbors})"

        spec_title = f"SpectralEmbedding (n_neighbors={n_neighbors})"
        try:
            if not can_graph:
                raise RuntimeError("disconnected knn graph")
            if graph_conn is not None:
                A = 0.5 * (graph_conn + graph_conn.T)
                Y_spec = SpectralEmbedding(
                    n_components=2,
                    affinity="precomputed",
                    random_state=int(random_state),
                ).fit_transform(A)
            else:
                Y_spec = SpectralEmbedding(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    random_state=int(random_state),
                ).fit_transform(X_embed)
            if not _is_valid_embedding(Y_spec):
                raise RuntimeError("spectral produced degenerate embedding")
        except Exception:
            Y_spec = _pca_fallback(X_embed)
            spec_title = f"SpectralEmbedding (fallback PCA, n_neighbors={n_neighbors})"

        if align_prev and state.get("isomap") is not None:
            Y_isomap = _align_embedding(Y_isomap, state.get("isomap"))
        if align_prev and state.get("spectral") is not None:
            Y_spec = _align_embedding(Y_spec, state.get("spectral"))

        state["tsne"] = Y_tsne.copy() if _is_valid_embedding(Y_tsne) else None
        state["isomap"] = Y_isomap.copy() if _is_valid_embedding(Y_isomap) else None
        state["spectral"] = Y_spec.copy() if _is_valid_embedding(Y_spec) else None

        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        cmap = plt.get_cmap("hsv", K)

        def _scatter(ax, Y, name: str):
            for k in range(K):
                m = (labels == k)
                ax.scatter(Y[m, 0], Y[m, 1], s=8, alpha=0.75, color=cmap(k), label=str(k))
            ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, linewidth=0.25, alpha=0.25)

        _scatter(axs[0], Y_tsne, tsne_title)
        _scatter(axs[1], Y_isomap, isomap_title)
        _scatter(axs[2], Y_spec, spec_title)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------
    # Path projections (latent-space)
    # ----------------------------
    @staticmethod
    def rollout_latent_paths(
        support_sets,
        z0: torch.Tensor,   # [1,D] or [B,D]
        dt: torch.Tensor,   # [B,1] or [B,K]
        *,
        steps: int | None = None,
        direction: int = +1,
    ) -> torch.Tensor:
        """
        Roll out full latent trajectories for visualization.

        Returns:
            traj: [B, T+1, K, D]
        """
        if z0.dim() != 2:
            raise ValueError(f"Expected z0 [B,D], got {tuple(z0.shape)}")
        B, D = z0.shape
        K = int(getattr(support_sets, "num_support_sets", 1))
        T = int(steps) if steps is not None else max(1, int(getattr(support_sets, "num_support_timesteps", 2)) - 1)

        # Expand to [B,K,D]
        z_curr = z0.unsqueeze(1).expand(B, K, D).contiguous()
        out = [z_curr]

        # Make viz deterministic: temporarily eval() to disable training-time latent noise.
        was_training = bool(getattr(support_sets, "training", False))
        support_sets.eval()

        # Preserve BN update flag if present.
        had_bn_flag = hasattr(getattr(support_sets, "F", None), "update_batchnorm")
        if had_bn_flag:
            prev_bn = bool(support_sets.F.update_batchnorm)
            support_sets.F.update_batchnorm = False
        try:
            for i in range(T):
                # ModelPDE implements _per_step(z_bkd, dt=..., direction=...)
                st, x_next, L_step, dt_used = support_sets._per_step(z_curr, dt=dt, direction=direction)  # noqa: SLF001
                z_curr = x_next
                out.append(z_curr)
        finally:
            if had_bn_flag:
                support_sets.F.update_batchnorm = prev_bn
            support_sets.train(was_training)

        traj = torch.stack(out, dim=1)  # [B, T+1, K, D]
        return traj

    @staticmethod
    def plot_spectral_projection_paths(traj_tkd: torch.Tensor, *, title: str = "Spectral (SVD/PCA) projection"):
        """
        Spectral projection: compute top-2 PCA directions (via SVD) of all points across (b,t,k),
        then plot each (b,k)-trajectory in that 2D space.

        Args:
            traj_tkd: [T+1, K, D] or [B, T+1, K, D]
        """
        
        if traj_tkd.dim() == 3:
            traj_btks = traj_tkd.unsqueeze(0)  # [1,T+1,K,D]
        elif traj_tkd.dim() == 4:
            traj_btks = traj_tkd
        else:
            raise ValueError(f"Expected traj [T+1,K,D] or [B,T+1,K,D], got {tuple(traj_tkd.shape)}")

        B, Tp1, K, D = traj_btks.shape
        # subtract per-batch starting point (t=0, any k is identical by construction)
        z0_bd = traj_btks[:, 0, 0, :]  # [B,D]
        disp = (traj_btks - z0_bd[:, None, None, :]).float()  # [B,T+1,K,D]

        X = disp.reshape(B * Tp1 * K, D)
        X = X - X.mean(dim=0, keepdim=True)
        # Low-rank SVD is efficient since N=B*K*(T+1) is usually manageable.
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        W = Vh[:2].T  # [D,2]
        Y = (X @ W).reshape(B, Tp1, K, 2).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("hsv", K)
        for k in range(K):
            alpha = float(min(0.9, 0.25 + 0.65 / max(1, B)))
            for b in range(B):
                ax.plot(Y[b, :, k, 0], Y[b, :, k, 1], color=cmap(k), linewidth=1.0, alpha=alpha)
            # emphasize the mean trajectory across batch
            Ym = Y[:, :, k, :].mean(axis=0)
            ax.plot(Ym[:, 0], Ym[:, 1], color=cmap(k), linewidth=2.0, alpha=0.95)
            ax.scatter(Ym[0, 0], Ym[0, 1], color=cmap(k), s=10, alpha=0.95)
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.axis("equal")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_sector_projection_paths(
        traj_tkd: torch.Tensor,
        *,
        title: str = "Sector (pie) projection: radial=grad, tangential=variance",
        sector_spread: float = 0.9,
        clamp_outward: bool = True,
    ):
        """
        Nonlinear projection into K pie sectors.

        - Each path k gets a base angle theta_k = 2*pi*k/K.
        - Radial coordinate comes from projection onto a "principal gradient" direction.
        - Tangential deviation within the sector comes from projection onto a principal variance direction
          (orthogonalized vs the gradient direction).
        """
        if traj_tkd.dim() == 3:
            traj_btkd = traj_tkd.unsqueeze(0)  # [1,T+1,K,D]
        elif traj_tkd.dim() == 4:
            traj_btkd = traj_tkd
        else:
            raise ValueError(f"Expected traj [T+1,K,D] or [B,T+1,K,D], got {tuple(traj_tkd.shape)}")

        B, Tp1, K, D = traj_btkd.shape
        z0_b1kd = traj_btkd[:, 0:1, :, :]  # [B,1,K,D]
        disp_btkd = (traj_btkd - z0_b1kd).float()  # [B,T+1,K,D]

        # --- principal gradient direction (use mean final displacement) ---
        # Average over batch+time to get a (K,D) summary, then average over K to get the global g (D,).
        df_kd = disp_btkd.mean(dim=(0, 1))  # [K,D]
        g = df_kd.mean(dim=0)               # [D]
        g = g / g.norm().clamp_min(1e-12)

        # --- principal variance direction orthogonal to g (use final displacements) ---
        # Use ALL displacements across (b,t,k) to estimate the dominant variance direction orthogonal to g.
        X = disp_btkd.reshape(B * Tp1 * K, D)
        X = X - X.mean(dim=0, keepdim=True)
        X_perp = X - (X @ g).unsqueeze(-1) * g.unsqueeze(0)
        if X_perp.norm() < 1e-10:
            v = torch.zeros_like(g)
            v[0] = 1.0
        else:
            _, _, Vh = torch.linalg.svd(X_perp, full_matrices=False)
            v = Vh[0]

        # scalar coords over (b,t,k)
        r_btk = torch.einsum("btkd,d->btk", disp_btkd, g)  # [B,T+1,K]
        s_btk = torch.einsum("btkd,d->btk", disp_btkd, v)  # [B,T+1,K]

        # Normalize to stable ranges for plotting.
        # Radius must start at 0 at t=0 (per b,k), and use a shared scale across all b/t/k.
        r_rel = r_btk - r_btk[:, 0:1, :]  # [B,T+1,K]
        r_out = r_rel.clamp_min(0.0) if clamp_outward else r_rel
        r_scaled = r_out / r_out.max().clamp_min(1e-12)  # [0,1]

        s_scaled = s_btk / s_btk.abs().max().clamp_min(1e-12)*3  # [-1,1]
        # sector geometry
        base = torch.linspace(0, 2 * math.pi, K + 1)[:-1]  # [K]
        half_sector = (math.pi / max(1, K)) * float(sector_spread)

        # coords: radius + within-sector angular deflection
        theta_btk = base.view(1, 1, K) + s_scaled * half_sector
        x = (r_scaled * torch.cos(theta_btk)).detach().cpu().numpy()
        y = (r_scaled * torch.sin(theta_btk)).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        # unit circle + sector lines
        circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0, alpha=0.5)
        ax.add_artist(circle)
        for k in range(K):
            th0 = 2 * math.pi * k / K
            ax.plot([0, math.cos(th0)], [0, math.sin(th0)], color="gray", linewidth=0.5, alpha=0.35)

        cmap = plt.get_cmap("hsv", K)
        for k in range(K):
            alpha = float(min(0.9, 0.25 + 0.65 / max(1, B)))
            for b in range(B):
                ax.plot(x[b, :, k], y[b, :, k], color=cmap(k), linewidth=1.0, alpha=alpha)
                ax.scatter(x[b, -1, k], y[b, -1, k], color=cmap(k), s=6, alpha=alpha)
            # emphasize mean trajectory across batch
            xm = x[:, :, k].mean(axis=0)
            ym = y[:, :, k].mean(axis=0)
            ax.plot(xm, ym, color=cmap(k), linewidth=2.0, alpha=0.95)

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        # Force symmetric limits so the unit circle is never clipped by autoscaling
        # (which can happen when trajectories occupy only a few quadrants).
        pad = 0.05
        ax.set_xlim(-(1.0 + pad), (1.0 + pad))
        ax.set_ylim(-(1.0 + pad), (1.0 + pad))
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout(pad=0.6)
        return fig



class ImageLogger:
    """
    Writes images into the SAME TensorBoard run directory as your main SummaryWriter.
    Keeps only the last `keep_last_images` image events by:
      • opening a short-lived SummaryWriter with a `.images.<step>` suffix,
      • logging the images for that step,
      • closing it,
      • pruning older `events.*.images*` files in the same run directory.
    """
    def __init__(self, writer: SummaryWriter, keep_last_images: int = 50, downscale: Optional[float] = None):
        self.writer = writer
        self.keep_last_images = int(keep_last_images)
        self.downscale = downscale
        # Ensure we can glob files reliably regardless of SummaryWriter implementation.
        self._log_dir = Path(getattr(writer, "log_dir", ""))

    def _list_image_eventfiles(self):
        # PyTorch appends filename_suffix to event filename, so match *.images*
        return sorted(
            [p for p in self._log_dir.glob("events.out.tfevents.*.images*") if p.is_file()],
            key=lambda p: p.stat().st_mtime
        )

    def _prune_old_images(self):
        files = self._list_image_eventfiles()
        if self.keep_last_images <= 0:
            to_delete = files  # keep none
        else:
            to_delete = files[:-self.keep_last_images]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass

    def log_triplet(self, tag_prefix: str, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                    step: int, n_vis: int = 8):
        # short-lived writer to the SAME run dir; one event file per image step
        triplet, diffs = ImageViz.make_triplet_grids(x0, x1, x2, n_vis=n_vis, downscale=self.downscale)
        self.writer.add_image(f"{tag_prefix}/triplet", triplet, step)
        self.writer.add_image(f"{tag_prefix}/diff_triplet_abs", diffs, step)
        self.writer.flush()
        #  writer.close()
        # self._prune_old_images()
        
    def close(self): self.writer.close()

# =========================
# TensorBoard server + ngrok
# =========================
def tb_start(exp_dir: str):
    """
    Starts TensorBoard programmatically and optionally opens an ngrok tunnel.
    Uses env vars:
      TB_HOST, TB_PORT,
      NGROK_AUTHTOKEN, NGROK_DOMAIN, NGROK_BASIC_AUTH, NGROK_REGION
    Returns: (tb_writer, tb_url, tb_obj, run_logdir)
    """
    from tensorboard import program
    from torch.utils.tensorboard import SummaryWriter
    import os

    exp_root, run_name = exp_dir.split("__")
    tb_dir = os.path.join("experiments", "tensorboard", "wip", exp_root)
    run_dir = os.path.join(tb_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_host = os.getenv("TB_HOST", "0.0.0.0")
    tb_port = int(os.getenv("TB_PORT", "6006"))

    tb = program.TensorBoard()
    tb.configure(argv=[
        None,
        "--logdir", tb_dir,
        "--host", tb_host,
        "--port", str(tb_port),
        "--reload_interval", "5",
    ])
    local_url = tb.launch()

    public_url = None
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok, conf
            cfg = conf.PyngrokConfig(auth_token=ngrok_token)
            conf.set_default(cfg)

            # Close any old tunnel on this port (useful for restarts)
            for t in ngrok.get_tunnels():
                if t.config.get("addr", "").endswith(f":{tb_port}"):
                    ngrok.disconnect(t.public_url)

            ngrok_hostname = os.getenv("NGROK_DOMAIN")   # reserved domain (premium)
            ngrok_auth = os.getenv("NGROK_BASIC_AUTH")   # "user:pass"
            ngrok_region = os.getenv("NGROK_REGION")     # e.g. "eu", "us"
            if ngrok_region:
                cfg.region = ngrok_region

            connect_kwargs = {"proto": "http", "addr": tb_port}
            if ngrok_hostname:
                connect_kwargs["hostname"] = ngrok_hostname
            if ngrok_auth:
                connect_kwargs["auth"] = ngrok_auth

            tunnel = ngrok.connect(**connect_kwargs)
            public_url = tunnel.public_url
        except Exception as e:
            print(f"[ngrok] Failed to create tunnel: {e}")

    tb_url = public_url or local_url
    print(f"#. TensorBoard local: {local_url}")
    if public_url:
        print(f"#. TensorBoard public: {public_url}", "\n" * 8)
    else:
        print("#. (No ngrok tunnel; set NGROK_AUTHTOKEN to expose publicly)")

    writer = SummaryWriter(log_dir=run_dir)
    return writer, tb_url, tb, run_dir

# aux.py
import math
import torch

# --------------------------
# helpers
# --------------------------
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


# --------------------------
# base dt samplers
# --------------------------
@torch.no_grad()
def _dt_legacy_uniform(B: int, device, half_range: int, low: int = 5000, high: int = 7500, dtype=torch.float32):
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


# --------------------------
# main entrypoint
# --------------------------
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

    K = int(getattr(trainer, "K", getattr(p, "num_support_sets", 1)))
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

    # ---- lightweight stats cache (for TB/debug) ----
    try:
        cache = getattr(trainer, "_dt_cache", None)
        if cache is None:
            cache = {}
            setattr(trainer, "_dt_cache", cache)

        cache["dt_mean"] = float(dt.abs().mean().item())
        cache["dt_std"] = float(dt.abs().std(unbiased=False).item())
        if beta is not None:
            cache["beta_mean"] = float(beta.mean().item())
            cache["beta_min"] = float(beta.min().item())
            cache["beta_max"] = float(beta.max().item())
        cache["dt_mode"] = base_mode
        cache["dt_beta_mode"] = beta_mode
    except Exception:
        pass

    return out

# =========================
# analytics helpers (short calls)
# =========================
@torch.no_grad()
def batch_acc_from_logits(logits: torch.Tensor, B: int, K: int, device) -> tuple[float, torch.Tensor]:
    preds = torch.argmax(logits, dim=1).view(B, K)
    true_2d = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
    acc = float((preds == true_2d).float().mean().item())
    return acc, preds


@torch.no_grad()
def dual_batch_acc_from_logits(
    logits0: torch.Tensor,
    logits: torch.Tensor,
    B: int,
    K: int,
    device,
) -> tuple[float, float, float, torch.Tensor]:
    step1_acc, _ = batch_acc_from_logits(logits0, B, K, device)
    step2_acc, preds = batch_acc_from_logits(logits, B, K, device)
    acc = 0.5 * (step1_acc + step2_acc)
    return acc, step1_acc, step2_acc, preds


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
    return float(ent.item())


@torch.no_grad()
def collect_wave_stats(support_sets, potential_preds: torch.Tensor) -> dict:
    wave_dict = support_sets.get_losses()
    wave_dict["potential_std"] = float(potential_preds.std().item())
    if "xf_now" in wave_dict and torch.is_tensor(wave_dict["xf_now"]):
        wave_dict["xf_now"] = float(wave_dict["xf_now"].norm(dim=-1).mean().item())
    # normalize to python floats
    out = {}
    for k, v in wave_dict.items():
        if torch.is_tensor(v):
            out[k] = float(v.item()) if v.numel() == 1 else v
        else:
            out[k] = float(v) if isinstance(v, (int, float)) else v
    return out


# =========================
# TB logging blocks (short names)
# =========================
def _tb_finite_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tb_finite_tensor(t: torch.Tensor) -> torch.Tensor | None:
    if t is None:
        return None
    t = t.detach().float().cpu().reshape(-1)
    finite = torch.isfinite(t)
    if not finite.any():
        return None
    return t[finite]


def tb_scalars(writer, step: int, win_means: dict, stat_tracker):
    for k, v in win_means.items():
        v = _tb_finite_float(v)
        if v is not None:
            writer.add_scalar(f"train/{k}", v, step)
    support_lr = _tb_finite_float(stat_tracker.last_support_lr)
    recognizer_lr = _tb_finite_float(stat_tracker.last_recognizer_lr)
    if support_lr is not None:
        writer.add_scalar("train/support_sets_lr", support_lr, step)
    if recognizer_lr is not None:
        writer.add_scalar("train/recognizer_lr", recognizer_lr, step)


def tb_grad_norms(writer, step: int, support_sets=None, recognizer=None, freq: int = 1):
    if step % freq != 0:
        return
    if support_sets is not None:
        gn_support = module_grad_norm(support_sets.F)
        gn_support = _tb_finite_float(gn_support)
        if gn_support is not None:
            writer.add_scalar("train/grad_norm/support_sets", gn_support, step)
    if recognizer is not None:
        gn_recon = module_grad_norm(recognizer)
        gn_recon = _tb_finite_float(gn_recon)
        if gn_recon is not None:
            writer.add_scalar("train/grad_norm/recognizer", gn_recon, step)


def tb_hists(writer, step: int, *, logits_det: torch.Tensor,
             potential_preds_det: torch.Tensor | None,
             K: int, log_potential: bool = True):
    logits_tb = _tb_finite_tensor(logits_det)
    if logits_tb is not None:
        writer.add_histogram("train/logits", logits_tb, step)
    if log_potential and potential_preds_det is not None:
        for k in range(K):
            potential_tb = _tb_finite_tensor(potential_preds_det[:, k])
            if potential_tb is not None:
                writer.add_histogram(f"potential_distribution/{k}", potential_tb, step)


def tb_figs(writer, step: int, stat_tracker, K: int, log_freq: int):
    # snapshot history only when we're logging figures
    stat_tracker.snapshot_per_k_history(step)

    if len(stat_tracker.ema_history) >= 2:
        hist_mat = np.stack(stat_tracker.ema_history, axis=1)[:, -30:]
        fig = ImageViz.plot_heatmap(
            hist_mat, K=K,
            title="Per-MLP EMA Accuracy over Time",
            xlabel="optimizer step snapshot",
            ylabel="MLP index k",
        )
        writer.add_figure("per_mlp/accuracy_heatmap", fig, global_step=step)
        plt.close(fig)

        fig_c = ImageViz.plot_confusion(stat_tracker.confusion, K=K)
        writer.add_figure("classifier/confusion_matrix", fig_c, global_step=step)
        plt.close(fig_c)


@torch.no_grad()
def decode_generator_output_for_viz(generator, output):
    """Decode latent generator outputs to RGB for TensorBoard / image logging."""
    base = generator.module if hasattr(generator, "module") else generator
    decode = getattr(base, "decode_with_vae", None)
    if decode is None:
        return output
    leading = output.shape[:-3]
    flat = output.reshape(-1, *output.shape[-3:])
    images = decode(flat)
    return images.reshape(*leading, *images.shape[1:])


@torch.no_grad()
def tb_images(img_logger, step: int, *, generator, z_first: torch.Tensor,
              img1_bk: torch.Tensor, img2_bk: torch.Tensor, n_vis: int):
    first_img = decode_generator_output_for_viz(generator, generator(z_first))
    img1_viz = decode_generator_output_for_viz(generator, img1_bk)
    img2_viz = decode_generator_output_for_viz(generator, img2_bk)
    img_logger.log_triplet(
        tag_prefix="images",
        x0=img1_viz, x1=img2_viz, x2=first_img,
        step=step,
        n_vis=n_vis,
    )


def tb_path_figs(
    writer,
    step: int,
    *,
    support_sets,
    z_first: torch.Tensor,
    dt_first: torch.Tensor,
    save_dir: str | Path | None = None,
):
    """
    Log latent-path projection figures for a single starting point z_first.
    """
    return
    traj_tkd = ImageViz.rollout_latent_paths(support_sets, z_first, dt_first)  # [T+1,K,D]

    fig1 = ImageViz.plot_spectral_projection_paths(traj_tkd)
    writer.add_figure("paths/spectral_projection", fig1, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig1, out_dir=save_dir, tag="paths__spectral_projection", step=step)
    plt.close(fig1)

    fig2 = ImageViz.plot_sector_projection_paths(traj_tkd)
    writer.add_figure("paths/sector_projection", fig2, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig2, out_dir=save_dir, tag="paths__sector_projection", step=step)
    plt.close(fig2)


def tb_pairwise_distance_figs(
    writer,
    step: int,
    *,
    stat_tracker,
    z_bkd: torch.Tensor,
    solver: str = "sinkhorn",
    reg: float = 2e-2,
    ot_numItermax: int = 5_000,
    max_points_per_class: int | None = None,
    save_dir: str | Path | None = None,
):
    """
    Compute + log a 2x2 panel of:
      - avg pairwise distance heatmap
      - OT/W2 heatmap
      - mean W1 history
      - mean W2 history

    Expects z_bkd shaped [B,K,D].
    """
    avg, w1, w2 = ImageViz.compute_pairwise_avg_and_wasserstein(
        z_bkd,
        solver=solver,
        reg=reg,
        ot_numItermax=ot_numItermax,
        max_points_per_class=max_points_per_class,
    )
    stat_tracker.update_pairwise_metrics(step_idx=int(step), avg_dist_kxk=avg, w1_kxk=w1, w2_kxk=w2)
    fig = ImageViz.plot_pairwise_distance_ot_panel(
        avg_dist_kxk=avg,
        w2_kxk=w2,
        steps=stat_tracker.pairwise_steps,
        w1_mean_hist=stat_tracker.pairwise_w1_mean_hist,
        w2_mean_hist=stat_tracker.pairwise_w2_mean_hist,
    )
    writer.add_figure("pairwise/dist_ot_panel", fig, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig, out_dir=save_dir, tag="pairwise__dist_ot_panel", step=step)
    plt.close(fig)


def tb_information_matrix_figs(
    writer,
    step: int,
    *,
    stat_tracker,
    support_sets,
    recognizer,
    z_bd: torch.Tensor,
    dt: torch.Tensor,
    save_dir: str | Path | None = None,
):
    """
    Compute + log a 2x2 information/confusion panel:
      - Λ heatmap (squared dot products between per-label step directions)
      - p(pred|true) confusion heatmap
      - Q(Λ) history
      - d_eff(Σ) history
    """
    lam, conf = ImageViz.compute_lambda_and_confusion(
        support_sets=support_sets,
        recognizer=recognizer,
        z_bd=z_bd,
        dt=dt,
    )
    stat_tracker.update_information_metrics(step_idx=int(step), lambda_kxk=lam, conf_prob_kxk=conf)

    fig = ImageViz.plot_information_matrix_panel(
        lambda_kxk=lam,
        conf_prob_kxk=conf,
        steps=stat_tracker.info_steps,
        Q_lambda_hist=stat_tracker.info_Q_lambda_hist,
        Q_conf_hist=stat_tracker.info_Q_conf_hist,
        deff_lambda_hist=stat_tracker.info_deff_lambda_hist,
        deff_conf_hist=stat_tracker.info_deff_conf_hist,
        title="Information/Confusion panel (batch-based) with disentanglement invariants",
    )
    writer.add_figure("info/info_matrix_panel", fig, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig, out_dir=save_dir, tag="info__info_matrix_panel", step=step)
    plt.close(fig)


def tb_final_point_embedding_figs(
    writer,
    step: int,
    *,
    z_bkd: torch.Tensor,
    max_points_per_class: int = 64,
    random_state: int = 0,
    save_dir: str | Path | None = None,
):
    """
    Log a 1x3 panel of nonlinear embeddings (t-SNE / Isomap / SpectralEmbedding)
    of final points [B,K,D], colored by class k.
    """
    fig = ImageViz.plot_final_points_nonlinear_embeddings(
        z_bkd,
        max_points_per_class=int(max_points_per_class),
        random_state=int(random_state),
        title=f"Final points embeddings @ step {int(step)} (B≤{int(max_points_per_class)} per class)",
    )
    writer.add_figure("manifold/final_points_embeddings", fig, global_step=step)
    if save_dir is not None:
        ImageViz.save_fig_copy(fig, out_dir=save_dir, tag="manifold__final_points_embeddings", step=step)
    plt.close(fig)


def json_append_by_step(path: str, step: int, rec: dict):
    """Load a JSON dict (if exists), set rec at key=step (int), and write back."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    if osp.isfile(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    data[str(int(step))] = rec
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def torch_append_by_step(path: str, step: int, payload: dict):
    """
    Load a torch-saved dict (if exists), set payload at key=step (int), and save back.
    NOTE: payload should already be moved to CPU + detached.
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    if osp.isfile(path):
        try:
            data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    else:
        data = {}
    data[int(step)] = payload
    torch.save(data, path)






# =========================
# prophet clipping
# =========================

        

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












# aux.py
import math
import numpy as np
import torch
import torch.nn.functional as F

class DualConfusionThermalizer:
    """
    Tracks two confusions:
      - step0: logits0 from (img0,img1)
      - step2: logits  from (img1,img2)  (the one that gives support_sets gradients)

    Maintains EMA confusion *probabilities* (row-stochastic), and produces beta_k.

    Beta rule (default):
      E0_k = -log(C0_kk + eps)
      E2_k = -log(C2_kk + eps)
      score_k = (E0_k - E2_k)  # >0 => step2 easier => increase beta to push out
      beta_k = exp(eta * (score_k - ema(score_k)))  (lag-based, centered)
      clamp + optional renorm to mean=1.
    """
    def __init__(
        self,
        K: int,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        use_soft: bool = True,
        eta: float = 0.5,
        beta_min: float = 0.5,
        beta_max: float = 2.0,
        renorm_mean1: bool = True,
        use_lag: bool = True,
    ):
        self.K = int(K)
        self.decay = float(ema_decay)
        self.eps = float(eps)
        self.use_soft = bool(use_soft)

        # confusion EMAs as float64 for stability (row=true, col=pred)
        self.C0 = np.eye(self.K, dtype=np.float64) / self.K  # harmless init
        self.C2 = np.eye(self.K, dtype=np.float64) / self.K

        # beta config
        self.eta = float(eta)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.renorm = bool(renorm_mean1)
        self.use_lag = bool(use_lag)

        # EMA of score for lag-based beta
        self.score_ema = np.zeros((self.K,), dtype=np.float64)

        # latest diagnostics
        self.last = {}

    @torch.no_grad()
    def update(self, logits0: torch.Tensor, logits2: torch.Tensor, B: int):
        """
        logits0/logits2: [B*K, K]
        """
        K = self.K
        device = logits2.device
        B = int(B)

        # reshape -> [B, K, K]  (dim1 = true_k, dim2 = pred_k)
        L0 = logits0.view(B, K, K)
        L2 = logits2.view(B, K, K)

        if self.use_soft:
            P0 = F.softmax(L0, dim=-1)   # [B,K,K]
            P2 = F.softmax(L2, dim=-1)
            # sum over batch -> mass per true_k/pred_k
            M0 = P0.sum(dim=0) / float(B)  # [K,K], row sums to 1
            M2 = P2.sum(dim=0) / float(B)
        else:
            # hard confusion counts turned into row-probs
            pred0 = L0.argmax(dim=-1)  # [B,K]
            pred2 = L2.argmax(dim=-1)
            oh0 = F.one_hot(pred0, num_classes=K).float().sum(dim=0) / float(B)  # [K,K]
            oh2 = F.one_hot(pred2, num_classes=K).float().sum(dim=0) / float(B)
            M0, M2 = oh0, oh2

        m0 = M0.detach().cpu().double().numpy()
        m2 = M2.detach().cpu().double().numpy()

        # EMA update
        d = self.decay
        self.C0 = d * self.C0 + (1.0 - d) * m0
        self.C2 = d * self.C2 + (1.0 - d) * m2

        # diag energies
        diag0 = np.clip(np.diag(self.C0), self.eps, 1.0)
        diag2 = np.clip(np.diag(self.C2), self.eps, 1.0)
        E0 = -np.log(diag0)
        E2 = -np.log(diag2)

        # score: >0 => step2 easier than step0 => boost beta to increase exploration/energy
        score = (E0 - E2)

        # lag-based center (prevents drift)
        if self.use_lag:
            self.score_ema = d * self.score_ema + (1.0 - d) * score
            score_used = (score - self.score_ema)
        else:
            score_used = (score - score.mean())

        beta = np.exp(self.eta * score_used)
        beta = np.clip(beta, self.beta_min, self.beta_max)

        if self.renorm:
            beta = beta * (float(K) / max(self.eps, beta.sum()))

        self.last = {
            "diag0_mean": float(diag0.mean()),
            "diag2_mean": float(diag2.mean()),
            "E0_mean": float(E0.mean()),
            "E2_mean": float(E2.mean()),
            "score_mean": float(score.mean()),
            "beta_mean": float(beta.mean()),
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
        }

        self._beta = beta  # store as numpy

    def beta_k(self, device=None, dtype=torch.float32) -> torch.Tensor:
        beta = getattr(self, "_beta", None)
        if beta is None:
            beta = np.ones((self.K,), dtype=np.float64)
        t = torch.tensor(beta, device=device, dtype=dtype)
        return t  # [K]


# --------------------------
# tiny convenience wrapper
# --------------------------
@torch.no_grad()
def update_dual_confusion_(
    stat_tracker,
    *,
    logits0: torch.Tensor,        # [B*K, K]
    logits2: torch.Tensor,        # [B*K, K]
    B: int,
    K: int,
    # defaults (can be overridden via params if you pass them in)
    ema_decay: float = 0.99,
    use_soft: bool = True,
    eta: float = 0.5,
    beta_min: float = 0.5,
    beta_max: float = 2.0,
    renorm_mean1: bool = True,
    use_lag: bool = True,
    eps: float = 1e-8,
):
    """
    Creates/updates stat_tracker.thermal (DualConfusionThermalizer).
    Also exposes stat_tracker.get_beta_k().
    """
    if not hasattr(stat_tracker, "thermal") or stat_tracker.thermal is None or getattr(stat_tracker.thermal, "K", None) != int(K):
        stat_tracker.thermal = DualConfusionThermalizer(
            K=int(K),
            ema_decay=float(ema_decay),
            eps=float(eps),
            use_soft=bool(use_soft),
            eta=float(eta),
            beta_min=float(beta_min),
            beta_max=float(beta_max),
            renorm_mean1=bool(renorm_mean1),
            use_lag=bool(use_lag),
        )

        # attach a getter (so your dt sampler can call stat_tracker.get_beta_k())
        def _get_beta_k():
            return stat_tracker.thermal.beta_k(device=logits2.device, dtype=logits2.dtype)
        stat_tracker.get_beta_k = _get_beta_k

    stat_tracker.thermal.update(logits0=logits0, logits2=logits2, B=int(B))
