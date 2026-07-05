# pde_losses.py
"""
Quick reference: PDEState (from lib.pde_ops) — shapes and helpers
-----------------------------------------------------------------
State wraps your two-potential PINN step with lazy, cached primitives.
Everything returns shape [B,K,1] for scalars and [B,K,D] for vectors,
unless noted. `when` is "now" or "next" (default "now").

Core tensors:
  - st.x()            -> [B,K,D]  current latent (leaf; grads flow through x)
  - st.dt()           -> [B,K,1]  signed timestep (+/- dt_value * direction)
  - st.x_next()       -> [B,K,D]  x + dt * v(now)
  - st.zeros()        -> [B,K,1]  zero tensor on state device/dtype
  - st.state["losses"]            dict accumulates detached mean of each loss

Semantic potential f and geometry:
  - st.f(when)        -> [B,K,1]  f(x) or f(x_next)
  - st.f_grad(when)   -> [B,K,D]  ∇_x f at x or x_next
  - st.f_laplace(when, probes=None) -> [B,K,1]  Hutchinson Laplacian of f
  - st.Xf(when)       -> [B,K,D]  ∇f / (||∇f||^2 + eps_norm2)

Slice energy ψ and spatial derivatives:
  - st.psi(when)      -> [B,K,1]  ψ(x, t_when), where t_now=f(x), t_next=t_now+dt
  - st.psi_grad(when) -> [B,K,D]  ∇_x ψ at x (or x_next)
  - st.psi_laplace(when, probes=None) -> [B,K,1]  Hutchinson Laplacian of ψ

Time derivatives of ψ (keep x fixed):
  - st.psi_t(when)    -> [B,K,1]  ∂ψ/∂t
  - st.psi_tt(when)   -> [B,K,1]  ∂²ψ/∂t²

Velocity and divergence:
  - st.v(when)        -> [B,K,D]  X_f(when) + ∇ψ(when)
  - st.v_div(when, probes=None) -> [B,K,1]  Hutchinson divergence of v(when)

Misc:
  - st.delta_y()      -> [B,K,1]  f(next) - f(now).detach()
  - Config keys (st.cfg): eps_norm2, laplace_probes, divergence_probes, rng,
    seed, dt_value, need_next, time_ad, detach_time_for_psi, etc.

Notes:
  • All primitives cache results; you can call them freely in losses.
  • Hutchinson traces/divergences are robust to affine/constant fields.
  • Prefer these helpers over manual autograd where possible for brevity and
    to inherit the project’s numerical safety choices.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, Type
import inspect, sys
import math
import torch
from torch import nn
from torch.autograd import grad

from lib.pde_ops import PDEState

# ---------------- Base ----------------

class PDELoss(nn.Module):
    name: str = "loss"
    needs_next: bool = False  # subclasses toggle when they require "next"

    def __init__(self, lam: float, **kwargs):
        super().__init__()
        self.lam = float(lam)
        self.ctx = kwargs

    def _loss(self, st: PDEState) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, st: PDEState) -> torch.Tensor:
        loss = self._loss(st) if self.lam != 0.0 else st.zeros()
        st.state["losses"][f"L_{self.name}"] = loss.detach().mean()
        return self.lam * loss


# ---------------- Losses ----------------

class OT(PDELoss):
    """Placeholder; define your optimal transport term here."""
    name = "ot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return st.zeros()

class BB(PDELoss):
    """Benamou–Brenier kinetic energy: ||v||^2."""
    name = "bb"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return 1/(st.f_grad("now").pow(2).sum(dim=-1, keepdim=True) + 1e-12)

class UnitSpeed(PDELoss):
    """(<∇f, v> - 1)^2, with ∇f detached to avoid batch-coupled grads."""
    name = "unitspeed"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return ((st.f_grad().detach() * st.v("now")).sum(dim=-1, keepdim=True) - 1.0).pow(2)

class SliceHJ(PDELoss):
    r"""Slice HJ residual at now: (ψ + 0.5||∇ψ||^2 - 0.5 ε^2 Δψ)^2."""
    name = "slicehj"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        H = 0.5 * st.psi_grad("now").pow(2).sum(dim=-1, keepdim=True)
        lap = st.psi_laplace("now") if eps > 0.0 else st.zeros()
        return (st.psi("now") + H - 0.5 * (eps**2) * lap).pow(2)

class HJ(PDELoss):
    r"""HJ residual at now: (ψ + 0.5||∇ψ||^2 - 0.5 ε^2 Δψ)^2."""
    name = "hj"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        H = 0.5 * st.psi_grad("now").pow(2).sum(dim=-1, keepdim=True)
        lap = st.psi_laplace("now") if eps > 0.0 else st.zeros()
        return (st.psi("now") + H - 0.5 * (eps**2) * lap).pow(2)

class Kinetic(PDELoss):
    r"""
    Wave-like kinetic term:
        ψ_tt(next) - ||∇ψ(next)|| / ||∇f(x + ∇ψ(next))||.
    """
    name = "kin"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(st.cfg["eps_norm2"])
        gpsi = st.psi_grad("next")
        x_hat = st.x() + gpsi
        # ∇f at x_hat (one grad call; PDEState doesn't cache this location)
        t_hat = st.f_m(x_hat)
        gf_hat = grad(t_hat.sum(), x_hat, create_graph=True)[0]
        return (st.psi_tt("next") - gpsi.norm(dim=-1, keepdim=True) /
                gf_hat.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt()).pow(2)

class Footpoint(PDELoss):
    r"""Footpoint: f(x + ∇ψ(now)) ≈ f(x) + dt."""
    name = "foot"
    def _loss(self, st: PDEState) -> torch.Tensor:
        t_next_from_hat = st.f_m(st.x() + st.psi_grad("now"))
        return (t_next_from_hat - (st.f() + st.dt())).pow(2)

class DivPrior(PDELoss):
    r"""
    Divergence prior: ( div v(now) + <s(x), v(now)> )^2, s(x)=-x by default.
    """
    name = "div"
    def _loss(self, st: PDEState) -> torch.Tensor:
        s = self.ctx.get("prior_score", None)
        score = s(st.x()) if callable(s) else -st.x()
        return (st.v_div("now") + (score * st.v("now")).sum(dim=-1, keepdim=True)).pow(2)

class Tangency(PDELoss):
    """(<∇f, ∇ψ>)^2."""
    name = "tan"
    def _loss(self, st: PDEState) -> torch.Tensor:
        return (st.f_grad() * st.psi_grad()).sum(dim=-1, keepdim=True).pow(2)

class FConvex(PDELoss):
    r"""
    Convexity via average curvature (Hutchinson trace of ∇²f):
      hinge( margin - tr(∇²f) )^2.
    """
    name = "fconvex"
    def _loss(self, st: PDEState) -> torch.Tensor:
        q = st.f_laplace(probes=int(self.ctx.get("probes", 1)))  # [B,K,1]
        margin = float(self.ctx.get("margin", 1e-3))
        return (margin - q).clamp_min(0.0).pow(2)

class DeltaY(PDELoss):
    """(f(next)-f(now).detach() - dt)^2 - (f(next)-f(now).detach())."""
    name = "deltay"
    needs_next = True
    def _loss(self, st: PDEState) -> torch.Tensor:
        dy = st.delta_y()
        return (dy - st.dt()).pow(2) - dy

# -------- Gaussian-output nudges (distributional + functional) --------
class GaussianKSD(PDELoss):
    r"""
    Kernel Stein discrepancy to N(0,1) for y=f(x). Optionally whiten per-k
    with detached stats to avoid batch-coupled gradients.

    Ctx:
      - mean_correction: bool (default True)
      - eps: float (default 1e-8)
      - bandwidth: Optional[float]  # if None, median heuristic per k
    """
    name = "ksd"

    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("eps", 1e-8))
        mean_correction = bool(self.ctx.get("mean_correction", True))
        bandwidth = self.ctx.get("bandwidth", None)
        truncation  = getattr(st, 'truncation', 1.0)

        y = st.f()  # [B,K,1]
        z = y
        if mean_correction:
            with torch.no_grad():
                current_mean = y.mean(dim=0, keepdim=True)
                # running_mean = self.ctx.get("running_mean", torch.zeros_like(y[:1]))
                # running_mean.lerp_(y.mean(dim=0, keepdim=True), 0.95)
                # self.ctx["running_mean"] = running_mean
            z = (y - current_mean)/st.f_m.running_output_std

        B, K, C = z.shape
        if B < 2:  # not enough pairs; return zeros
            return st.zeros()

        diff = z.unsqueeze(1) - z.unsqueeze(0)                     # [B,B,K,1]
        dist2 = (diff**2).sum(dim=-1)                              # [B,B,K]
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

        if bandwidth is None:
            med = dist2[eye_mask].view(B*(B-1), K).median(dim=0).values.clamp_min(eps)
            h2 = 0.5 * med
        else:
            h2 = torch.full((K,), float(bandwidth), device=z.device, dtype=z.dtype)**2

        Kmat = torch.exp(-dist2 / (2.0 * h2.view(1,1,K)))          # [B,B,K]
        s = -z                                                     # score of N(0,1)
        s_dot = torch.einsum("bkc,jkc->bjk", s, s)                 # [B,B,K]
        sx_diff = (s.unsqueeze(1) * diff).sum(-1)                  # [B,B,K]
        sy_diff = (s.unsqueeze(0) * (-diff)).sum(-1)               # [B,B,K]
        trace_xy = Kmat * (C / h2.view(1,1,K) - dist2 / (h2.view(1,1,K)**2))
        U = s_dot * Kmat + (sx_diff + sy_diff) * (Kmat / h2.view(1,1,K)) + trace_xy
        ksd2_k = U[eye_mask].view(B*(B-1), K).mean(dim=0)          # [K]
        return ksd2_k.view(1, K, 1).expand(B, K, 1)

class CurvatureCD(PDELoss):
    r"""
    Central-difference curvature penalty on f (near-affine prior):
      E_u [ ( f(x+εu) - 2f(x) + f(x-εu) ) / ε^2 ]^2.

    Ctx:
      - eps: float (default 1e-2)
      - n_dirs: int (default 2)
      - unit_dirs: bool (default True)
    """
    name = "curv"
    def _loss(self, st: PDEState) -> torch.Tensor:
        h = float(self.ctx.get("eps", 1e-2))
        n_dirs = int(self.ctx.get("n_dirs", 2))
        unit = bool(self.ctx.get("unit_dirs", True))
        x = st.x()
        f = st.f_m  # reuse module directly for perturbed queries
        y0 = st.f()
        acc = 0.0
        for _ in range(n_dirs):
            u = torch.randn_like(x)
            if unit:
                u = u / (u.flatten(-1).norm(dim=-1, keepdim=True).unsqueeze(-1) + 1e-8)
            d2 = (f(x + h*u) - 2*y0 + f(x - h*u)) / (h*h)
            acc = acc + d2.pow(2)
        return acc / float(n_dirs)

# ---------------- Other priors/constraints ----------------

class YMarginalScore(PDELoss):
    """
    1-D score matching on blurred marginal p_Y^ε to push y=f(x) toward N(0,1).
    """
    name = "ynormal"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon_y", 0.1))
        stopgrad = bool(self.ctx.get("stopgrad_in_weights", True))
        y = st.f()                                             # [B,K,1]
        ytil = y + math.sqrt(eps) * torch.randn_like(y)        # [B,K,1]

        y_s = y.squeeze(-1)                                    # [B,K]
        ytil_s = ytil.squeeze(-1)                              # [B,K]
        q = ( (ytil_s.detach() if stopgrad else ytil_s).unsqueeze(1) - y_s.unsqueeze(0) )  # [B,B,K]
        w = (-0.5 * q.pow(2) / eps).softmax(dim=1)             # [B,B,K]
        mu_hat = (w * y_s.unsqueeze(0)).sum(dim=1)             # [B,K]
        score_hat = (mu_hat - ytil_s) / eps                    # [B,K]
        return (score_hat + ytil_s).pow(2).unsqueeze(-1)       # [B,K,1]

class ContinuityKnownY(PDELoss):
    r"""
    Continuity residual with p_Y = N(0,1), y = f(x)+√ε ξ (detached in ψ):
      R = [(f - y)/ε + y] + div v + v · [ s_X(x) + ((y-f)/ε) ∇f ].
    """
    name = "ce_y"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("epsilon", 0.0))
        assert eps > 0.0, "ContinuityKnownY needs epsilon > 0"
        prior_score = self.ctx.get("prior_score", None)
        x = st.x()
        f = st.f()
        y = (f + math.sqrt(eps) * torch.randn_like(f)).detach()

        s_x = prior_score(x) if callable(prior_score) else -x
        s_y = s_x + ((y - f) / eps) * st.f_grad()
        R = (f - y) / eps + y + st.v_div("now") + (st.v("now") * s_y).sum(dim=-1, keepdim=True)
        return R.pow(2)

class VNormEMA(PDELoss):
    r"""
    Per-head EMA target for ||v|| with deviation penalty.

    Ctx (optional):
      - ema_beta: float in [0,1) (default 0.99)
      - relative: bool (default True)
      - eps: float (default 1e-8)
    """
    name = "vnorm"
    @torch.no_grad()
    def _update_ema(self, st: PDEState, vnorm_mean_k: torch.Tensor):
        # keep [1,K,1] EMA in ctx
        key = "ema_vnorm"
        beta = float(self.ctx.get("ema_beta", 0.99))
        if key not in self.ctx:
            self.ctx[key] = vnorm_mean_k.clone()
        else:
            self.ctx[key].mul_(beta).add_((1.0 - beta) * vnorm_mean_k)

    def _loss(self, st: PDEState) -> torch.Tensor:
        vnorm = st.v().norm(dim=-1, keepdim=True)                      # [B,K,1]
        with torch.no_grad():
            mean_k = vnorm.mean(dim=0, keepdim=True)                   # [1,K,1]
            self._update_ema(st, mean_k)
        ema = self.ctx["ema_vnorm"]
        if bool(self.ctx.get("relative", True)):
            diff = (vnorm - ema) / (ema.abs() + float(self.ctx.get("eps", 1e-8)))
        else:
            diff = vnorm - ema
        return diff.pow(2)

class GradGroupSecondMomentOrtho(PDELoss):
    r"""
    Group orthogonality via second-moment Gram off-diagonals.
    """
    name = "g2orth"
    def _loss(self, st: PDEState) -> torch.Tensor:
        field = str(self.ctx.get("field", "f"))
        when  = str(self.ctx.get("when", "now"))
        normalize = bool(self.ctx.get("normalize", True))
        eps = float(self.ctx.get("eps", st.cfg.get("eps_norm2", 1e-8)))

        if field == "psi":
            g = st.psi_grad(when)
        elif field == "v":
            g = st.v(when)
        else:
            g = st.f_grad(when)

        if normalize:
            g = g / (g.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt())

        gram = torch.einsum("bkd,bld->bkl", g, g)                      # [B,K,K]
        gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
        return gram.pow(2).sum(dim=-1, keepdim=True)                   # [B,K,1]

class SignedGradGroupSecondMomentOrtho(PDELoss):
    r"""
    Signed group orthogonality via positive second-moment Gram off-diagonals.
    """
    name = "signed_g2orth"
    def _loss(self, st: PDEState) -> torch.Tensor:
        field = str(self.ctx.get("field", "f"))
        when  = str(self.ctx.get("when", "now"))
        normalize = bool(self.ctx.get("normalize", True))
        eps = float(self.ctx.get("eps", st.cfg.get("eps_norm2", 1e-8)))

        if field == "psi":
            g = st.psi_grad(when)
        elif field == "v":
            g = st.v(when)
        else:
            g = st.f_grad(when)

        if normalize:
            g = g / (g.pow(2).sum(dim=-1, keepdim=True).add(eps).sqrt())

        gram = torch.einsum("bkd,bld->bkl", g, g)                      # [B,K,K]
        gram = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
        return gram.clamp_min(0.0).pow(2).sum(dim=-1, keepdim=True)     # [B,K,1]

class Poisson(PDELoss):
    r"""
    Weighted Poisson consistency using only state primitives:

      Δ_p ψ + div_p X_f
      = (Δψ + s·∇ψ) + (div X_f + s·X_f)
      = div v + s·(∇ψ + X_f)     (since div v = div X_f + Δψ)

    So the residual reduces to:  div v + s·v  (clean and cheap).
    """
    name = "poisson"
    def _loss(self, st: PDEState) -> torch.Tensor:
        s_fn = self.ctx.get("prior_score", None)
        s = s_fn(st.x()) if callable(s_fn) else -st.x()
        return (st.v_div("now") + (s * st.v("now")).sum(dim=-1, keepdim=True)).pow(2)

# ---------------- Losses for preventing convex trivial solutions ----------------
class FCurvUpper(PDELoss):
    r"""
    Upper hinge on average curvature (Laplacian) of f:
      L = max(0, tr(∇² f) - kappa_max)^2
    """
    name = "fcurvmax"
    def _loss(self, st: PDEState) -> torch.Tensor:
        kappa_max = float(self.ctx.get("kappa_max", 0.05))
        q = st.f_laplace(probes=int(self.ctx.get("probes", 1)))  # [B,K,1]
        return (q - kappa_max).clamp_min(0.0).pow(2)

class FAlongCurv(PDELoss):
    r"""
    Along-flow curvature control for f:
      q = v̂ᵀ (∇² f) v̂,   v̂ = normalized ∇f (or X_f direction)
    Penalize (q - target)^2 or hinge on q <= q_max.

    Ctx:
      - use_xf_dir: bool (default False)  # if True, use direction of X_f; else ∇f
      - target: Optional[float] (default None)  # if set, use (q - target)^2
      - q_max: Optional[float] (default None)   # if set, use max(0, q - q_max)^2
      - eps: float (default st.cfg["eps_norm2"])
    """
    name = "fcurvalong"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("eps", st.cfg.get("eps_norm2", 1e-8)))
        use_xf_dir = bool(self.ctx.get("use_xf_dir", False))

        # direction v̂
        if use_xf_dir:
            v = st.Xf()                                  # [B,K,D]
            vhat = v / (v.pow(2).sum(-1, keepdim=True).add(eps).sqrt())
        else:
            g = st.f_grad()                               # [B,K,D]
            vhat = g / (g.pow(2).sum(-1, keepdim=True).add(eps).sqrt())

        # HVP: (∇² f) v̂ via one autograd call
        g = st.f_grad()                                   # [B,K,D]
        s = (g * vhat).sum()
        Hv = grad(s, st.x(), create_graph=True)[0]        # [B,K,D]
        q = (Hv * vhat).sum(-1, keepdim=True)             # [B,K,1]

        if "q_max" in self.ctx and self.ctx["q_max"] is not None:
            return (q - float(self.ctx["q_max"])).clamp_min(0.0).pow(2)
        target = self.ctx.get("target", None)
        target = 0.0 if target is None else float(target)
        return (q - target).pow(2)

class StraightXf(PDELoss):
    r"""
    Straightness of the flow X_f via finite differences (no second-order AD):
      a ≈ (X_f(x + h v̂) - X_f(x)) / h,  v̂ = X_f / ||X_f||
      L = ||a||^2

    Ctx:
      - h: float (default 1e-2)
      - unit_dir: bool (default True)
    """
    name = "straightxf"
    def _loss(self, st: PDEState) -> torch.Tensor:
        h = float(self.ctx.get("h", 1e-2))
        unit_dir = bool(self.ctx.get("unit_dir", True))
        eps = float(st.cfg.get("eps_norm2", 1e-8))

        x = st.x()
        v0 = st.Xf()                                       # [B,K,D]
        if unit_dir:
            vhat = v0 / (v0.pow(2).sum(-1, keepdim=True).add(eps).sqrt())
        else:
            vhat = v0

        # evaluate X_f at displaced x using f_m
        x1 = (x + h * vhat).detach().clone().requires_grad_(True)
        f1 = st.f_m(x1)
        g1 = grad(f1.sum(), x1, create_graph=True)[0]
        Xf1 = g1 / g1.pow(2).sum(-1, keepdim=True).add(eps)

        a = (Xf1 - v0) / h                                 # [B,K,D]
        return a.pow(2).sum(-1, keepdim=True)
class ShiftMMD(PDELoss):
    r"""
    Two-sample match between f(next)-dt (A) and f(now) (B) via unbiased MMD^2
    with an RBF kernel. Encourages the endpoint distribution to be a shifted
    copy of the start.

    Ctx:
      - bandwidth: Optional[float]  # if None, median heuristic per k
      - whiten: bool (default True) # standardize jointly (detached stats)
      - eps: float (default 1e-8)
    """
    name = "shiftmmd"

    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("eps", 1e-8))
        whiten = bool(self.ctx.get("whiten", True))
        bw = self.ctx.get("bandwidth", None)

        A = st.f("next") - st.dt()    # [B,K,1]
        B = st.f("now")               # [B,K,1]
        Bsz, K, C = A.shape
        if Bsz < 2:
            return st.zeros()

        # Optional joint whitening for scale invariance (no grad through stats)
        if whiten:
            with torch.no_grad():
                Z = torch.cat([A, B], dim=0)             # [2B,K,1]
                mu = Z.mean(dim=0, keepdim=True)
                sd = Z.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
            A = (A - mu) / sd
            B = (B - mu) / sd

        # Helpers: within-set and cross-set squared distances -> [B,B,K]
        def pdist2(Y):               # Y: [B,K,C] -> [B,B,K]
            d = Y.unsqueeze(1) - Y.unsqueeze(0)
            return (d*d).sum(dim=-1)

        def cdist2(Y1, Y2):          # Y1,Y2: [B,K,C] -> [B,B,K]
            d = Y1.unsqueeze(1) - Y2.unsqueeze(0)
            return (d*d).sum(dim=-1)

        AA = pdist2(A)               # [B,B,K]
        BB = pdist2(B)               # [B,B,K]
        AB = cdist2(A, B)            # [B,B,K]  <-- cross distances (fixed)

        # Bandwidth per k (median heuristic over off-diags of AA,BB and all of AB)
        if bw is None:
            eye_mask = ~torch.eye(Bsz, dtype=torch.bool, device=A.device)
            AA_off = AA[eye_mask].view(Bsz*(Bsz-1), K)   # [(B^2-B),K]
            BB_off = BB[eye_mask].view(Bsz*(Bsz-1), K)   # [(B^2-B),K]
            AB_all = AB.view(Bsz*Bsz, K)                 # [B^2,K]
            pool = torch.cat([AA_off, BB_off, AB_all], dim=0)  # [(3B^2-2B),K]
            h2 = 0.5 * pool.median(dim=0).values.clamp_min(eps)  # [K]
        else:
            h2 = torch.full((K,), float(bw), device=A.device, dtype=A.dtype)**2

        # RBF kernels
        def krbf(d2): return torch.exp(-d2 / (2.0 * h2.view(1,1,K)))

        eye_mask = ~torch.eye(Bsz, dtype=torch.bool, device=A.device)
        KA = krbf(AA)[eye_mask].view(Bsz*(Bsz-1), K).mean(dim=0)  # E_{i!=j} k( Ai, Aj )
        KB = krbf(BB)[eye_mask].view(Bsz*(Bsz-1), K).mean(dim=0)  # E_{i!=j} k( Bi, Bj )
        KAB = krbf(AB).mean(dim=(0,1))                            # E_{i,j}   k( Ai, Bj )

        mmd2_k = KA + KB - 2.0 * KAB                              # [K]
        return mmd2_k.view(1, K, 1).expand(Bsz, K, 1)
        
        
        
class LevelSetSpread(PDELoss):
    r"""
    Level-set spread: for pairs with small |Δy|, penalize being close in x.
      L_k = mean_{i≠j} w_y(i,j) * exp(-||x_i-x_j||^2 / (2 τ^2)) / sum w_y
    Minimizing pushes equal-attribute level sets to have larger spatial support.

    Ctx:
      - delta: float (default 0.25)     # width of similarity window in y
      - tau: float (default 1.0)        # spatial scale in x
      - whiten_y: bool (default True)   # standardize y per k with detached stats
      - eps: float (default 1e-8)
    """
    name = "levelspread"
    def _loss(self, st: PDEState) -> torch.Tensor:
        eps = float(self.ctx.get("eps", 1e-8))
        delta = float(self.ctx.get("delta", 0.25))
        tau = float(self.ctx.get("tau", 1.0))
        whiten_y = bool(self.ctx.get("whiten_y", True))

        y = st.f()                                       # [B,K,1]
        if whiten_y:
            with torch.no_grad():
                mu = y.mean(0, keepdim=True)
                sd = y.std(0, unbiased=False, keepdim=True).clamp_min(eps)
            y = (y - mu) / sd

        x = st.x()                                       # [B,K,D]
        Bsz, K, _ = y.shape
        if Bsz < 2:
            return st.zeros()

        # pairwise over batch per k
        dy2 = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)      # [B,B,K]
        dx2 = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(-1)      # [B,B,K]

        wy = torch.exp(-dy2 / (2.0 * (delta**2)))                    # [B,B,K]
        kx = torch.exp(-dx2 / (2.0 * (tau**2)))                      # [B,B,K]

        mask = ~torch.eye(Bsz, dtype=torch.bool, device=y.device)
        wy_m = wy[mask].view(Bsz*(Bsz-1), K)
        kx_m = kx[mask].view(Bsz*(Bsz-1), K)

        num = (wy_m * kx_m).sum(0)                                   # [K]
        den = wy_m.sum(0).clamp_min(eps)                             # [K]
        val = (num / den)                                            # [K]
        return val.view(1, K, 1).expand(Bsz, K, 1)







# ---------------- Public registry API ----------------

def _normalize(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())

def loss_registry() -> Dict[str, Type[PDELoss]]:
    reg: Dict[str, Type[PDELoss]] = {}
    for _, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if issubclass(obj, PDELoss) and obj is not PDELoss:
            name = getattr(obj, "name", None)
            if isinstance(name, str) and name:
                reg[_normalize(name)] = obj
    return reg

def build_losses(lambda_dict: Dict[str, float], **ctx) -> Tuple[List[PDELoss], bool]:
    reg = loss_registry()
    losses: List[PDELoss] = []
    needs_next_any = False
    for raw_name, lam in lambda_dict.items():
        cls = reg.get(_normalize(raw_name), None)
        if cls is None:
            continue
        if isinstance(lam,dict):
            l = lam.pop("lam")
            ctx.update(lam)
            lam = l

        inst = cls(lam, **ctx)
        losses.append(inst)
        if getattr(cls, "needs_next", False):
            needs_next_any = True
    return losses, needs_next_any