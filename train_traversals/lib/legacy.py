"""Unused helpers retained for future review; not part of the active pipeline."""

import os
import os.path as osp
import json
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .aux import ImageViz
from .utils import CosineScheduleWithWarmup


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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    return CosineScheduleWithWarmup(optimizer, num_warmup_steps, num_training_steps, last_epoch)


@torch.no_grad()
def batch_acc_from_logits(logits: torch.Tensor, B: int, K: int, device) -> tuple[float, torch.Tensor]:
    preds = torch.argmax(logits, dim=1).view(B, K)
    true_2d = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
    acc = float((preds == true_2d).float().mean().item())
    return acc, preds


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
