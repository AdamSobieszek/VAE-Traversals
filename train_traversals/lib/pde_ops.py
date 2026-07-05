# pde_ops.py
# ---------------------------------------------------------------------
# Minimal, self-contained PDE ops for two-potential PINNs.
# Single class PDEState with lazy, cached primitives:
#   x, f, f_grad, f_laplace, psi, psi_grad, psi_laplace, psi_t, psi_tt, Xf, v, v_div
# Timepoint handled via when ∈ {"now","next"} (default "now"}.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Literal

import torch
from torch import nn
from torch.autograd import grad

When = Literal["now", "next"]
# --- RNG helpers (generator-safe across PyTorch versions) ---

def _make_gen(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def _randn_like(x: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    # Use shape-based API when a generator is provided (for compatibility)
    if gen is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)

def _rand_rademacher_like(x: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    # Draw 0/1 with randint, then map to {-1, +1}
    if gen is None:
        r = torch.randint(0, 2, x.shape, device=x.device)
    else:
        r = torch.randint(0, 2, x.shape, device=x.device, generator=gen)
    return r.to(dtype=x.dtype).mul_(2).sub_(1)

def _rand_vec_like(x: torch.Tensor, mode: str, gen: Optional[torch.Generator]) -> torch.Tensor:
    if mode == "gaussian":
        return _randn_like(x, gen)
    if mode == "rademacher":
        return _rand_rademacher_like(x, gen)
    raise ValueError(f"rng must be 'rademacher' or 'gaussian', got {mode!r}")

# ----------------------- helpers -----------------------

def _ensure_bkd(z: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return z as [B,K,D] and inferred K."""
    if z.dim() == 2:
        B, D = z.shape
        return z.unsqueeze(1), 1
    if z.dim() == 3:
        return z, z.shape[1]
    raise ValueError(f"z must be [B,D] or [B,K,D], got {tuple(z.shape)}")
# --- robust HVP / Laplacian that handle affine cases cleanly ---

def _hvp_from_grad(g: torch.Tensor,
                   x: torch.Tensor,
                   v: torch.Tensor,
                   create_graph: bool) -> torch.Tensor:
    """
    Hessian-vector product H_x(u) @ v using g = ∇_x u.
    If (g·v).sum() has no grad path to x (e.g., g is x-constant), return zeros.
    """
    s = (g * v).sum()
    if not s.requires_grad or not x.requires_grad:
        return torch.zeros_like(x)
    Hv = grad(s, x, create_graph=create_graph, allow_unused=True)[0]
    if Hv is None:
        Hv = torch.zeros_like(x)
    return Hv


def _laplacian_hutch(g: torch.Tensor,
                     x: torch.Tensor,
                     probes: int,
                     mode: str,
                     gen: Optional[torch.Generator],
                     create_graph: bool) -> torch.Tensor:
    if probes <= 0:
        raise ValueError("probes must be >= 1 for Laplacian.")
    acc = 0.0
    for _ in range(int(probes)):
        v = _rand_vec_like(x, mode, gen)
        Hv = _hvp_from_grad(g, x, v, create_graph=create_graph)
        acc = acc + (Hv * v).sum(dim=-1, keepdim=True)
    return acc / float(probes)


# --- robust divergence Hutchinson (handles x-independent fields) ---

def _divergence_hutch(v_field: torch.Tensor,
                      x: torch.Tensor,
                      probes: int,
                      mode: str,
                      gen: Optional[torch.Generator],
                      create_graph: bool) -> torch.Tensor:
    """
    Hutchinson divergence: trace(J_v).
    If (v·eps).sum() has no grad path to x (v independent of x), return zeros.
    """
    if probes <= 0:
        raise ValueError("probes must be >= 1 for divergence.")
    acc = 0.0
    for _ in range(int(probes)):
        eps = _rand_vec_like(x, mode, gen)
        s = (v_field * eps).sum()
        if not s.requires_grad or not x.requires_grad:
            Jv_eps = torch.zeros_like(x)
        else:
            Jv_eps = grad(s, x, create_graph=create_graph, allow_unused=True)[0]
            if Jv_eps is None:
                Jv_eps = torch.zeros_like(x)
        acc = acc + (Jv_eps * eps).sum(dim=-1, keepdim=True)
    return acc / float(probes)



# ----------------------- PDEState -----------------------

class PDEState:
    """
    Lazy, cached PDE primitives for a single step.

    Required:
      - f: nn.Module mapping x->[B,K,1]
      - psi: nn.Module mapping (x,t)->[B,K,1]
      - z: current latents [B,D] or [B,K,D]
      - direction: +1 or -1

    Config kwargs (all optional):
      detach_between_steps: bool = False
      dt_value: float = 1.0
      detach_time_for_psi: bool = True
      time_ad: str = "reverse"  # "reverse" or "forward" for psi_t / psi_tt
      track_param_through_xgrads: bool = True
      eps_norm2: float = 1e-8
      laplace_probes: int = 1
      divergence_probes: int = 1
      rng: str = "rademacher"   # or "gaussian"
      seed: Optional[int] = None
      need_next: bool = False   # if True, makes "next" timepoint available
    """

    def __init__(self,
                 f: nn.Module,
                 psi: nn.Module,
                 z: torch.Tensor,
                 direction: int = +1,
                 **config):
        self.f_m = f
        self.psi_m = psi

        z_bk, K = _ensure_bkd(z)
        self.B, self.K, self.D = z_bk.shape
        self.device, self.dtype = z_bk.device, z_bk.dtype
        self.direction = direction

        # defaults
        self.cfg: Dict[str, Any] = {
            "detach_between_steps": False,
            "dt_value": 1.0,
            "detach_time_for_psi": True,
            "time_ad": "reverse",
            "track_param_through_xgrads": True,
            "eps_norm2": 1e-8,
            "laplace_probes": 1,
            "divergence_probes": 1,
            "rng": "rademacher",
            "seed": None,
            "need_next": False,
        }
        self.cfg.update(config)

        # state dict (cache)
        self.state: Dict[Any, Any] = {}

        # core tensors
        if self.cfg["detach_between_steps"]:
            x_leaf = z_bk.detach().clone().requires_grad_(True)
        else:
            x_leaf = z_bk.requires_grad_(True)

        if not isinstance(self.direction, torch.Tensor) and not isinstance(self.cfg["dt_value"], torch.Tensor):
            dt = torch.full((self.B, self.K, 1),
                            float(self.cfg["dt_value"]) * float(self.direction),
                            device=self.device, dtype=self.dtype)
        else:
            dt = (self.direction * self.cfg["dt_value"]).reshape(self.B, -1, 1)
            if self.K>dt.shape[1]:
                dt = dt.repeat(1, self.K//dt.shape[1], 1)
        self.state["x"] = x_leaf           # [B,K,D]
        self.state["dt"] = dt              # [B,K,1]
        self.state["losses"] = {}

        # optional RNG for Hutchinson
        self._gen = _make_gen(self.cfg["seed"], self.device)

    # ------------- basics -------------

    def x(self) -> torch.Tensor:
        return self.state["x"]

    def dt(self) -> torch.Tensor:
        return self.state["dt"]

    def delta_y(self) -> torch.Tensor:
        if "delta_y" not in self.state:
            self.state["delta_y"] = self.f("next") - self.f("now").detach()
        return self.state["delta_y"]

    # ------------- f and geometry -------------

    def f(self, when: When = "now") -> torch.Tensor:
        key = ("f", when)
        if key not in self.state:
            if when == "now":
                self.state[key] = self.f_m(self.x())
            elif when == "next":
                self.state[key] = self.f_m(self.x_next())
        return self.state[key]


    def f_grad(self, when: When = "now") -> torch.Tensor:
        key = ("f_grad", when)
        if key not in self.state:
            create_graph = bool(self.cfg["track_param_through_xgrads"])
            self.state[key] = grad(self.f(when).sum(), self.x() if when == "now" else self.x_next(), create_graph=create_graph)[0]
        return self.state[key]

    def f_laplace(self, when: When = "now", probes: Optional[int] = None) -> torch.Tensor:
        key = ("f_laplace", probes, when)
        if key not in self.state:
            p = int(self.cfg["laplace_probes"] if probes is None else probes)
            self.state[key] = _laplacian_hutch(
                g=self.f_grad(when),
                x=self.x() if when == "now" else self.x_next(),
                probes=p,
                mode=self.cfg["rng"],
                gen=self._gen,
                create_graph=True
            )
        return self.state[key]

    def Xf(self, when: When = "now") -> torch.Tensor:
        key = ("Xf", when)
        if key not in self.state:
            g = self.f_grad(when)
            norm2 = g.pow(2).sum(dim=-1, keepdim=True).add_(float(self.cfg["eps_norm2"]))
            self.state[key] = g / norm2
        return self.state[key]


    # def Xf_div(self) -> torch.Tensor:

    # ------------- f-pseudotime -------------
    def _t(self, when: When = "now") -> torch.Tensor:
        key = ("t", when)
        if key in self.state and self.state[key] is not None:
            return self.state[key]
        if when == "now":
            t = self.f(when)
            if self.cfg["detach_time_for_psi"]:
                t = t.detach()
        elif when == "next":
            if not self.cfg["need_next"]:
                raise ValueError("NEXT requested but need_next=False; set need_next=True in config.")
            t = self._t("now") + self.dt()
        else:
            raise ValueError("when must be 'now' or 'next'")
        self.state[key] = t
        return t

    # ------------- psi and spatial derivatives -------------

    def psi(self, when: When = "now") -> torch.Tensor:
        key = ("psi", when)
        if key not in self.state:
            self.state[key] = self.psi_m(self.x() if when == "now" else self.x_next(), self._t(when))

        return self.state[key]

    def psi_grad(self, when: When = "now") -> torch.Tensor:
        key = ("psi_grad", when)
        if key not in self.state:
            create_graph = bool(self.cfg["track_param_through_xgrads"])
            self.state[key] = grad(self.psi(when).sum(), self.x(), create_graph=create_graph)[0]
        return self.state[key]

    def psi_laplace(self, when: When = "now", probes: Optional[int] = None) -> torch.Tensor:
        key = ("psi_laplace", when, probes)
        if key not in self.state:
            p = int(self.cfg["laplace_probes"] if probes is None else probes)
            self.state[key] = _laplacian_hutch(
                g=self.psi_grad(when),
                x=self.x(),
                probes=p,
                mode=self.cfg["rng"],
                gen=self._gen,
                create_graph=True
            )
        return self.state[key]

    # ------------- time derivatives of psi -------------

    def psi_t(self, when: When = "now") -> torch.Tensor:
        key = ("psi_t", when)
        if key in self.state:
            return self.state[key]

        # hold x fixed when differentiating wrt t
        x_fixed = self.x().detach()
        t_in = self._t(when)
        if self.cfg["detach_time_for_psi"]:
            t_in = t_in.detach()

        mode = self.cfg["time_ad"]
        if mode == "reverse":
            t_leaf = t_in.clone().requires_grad_(True)
            u = self.psi_m(x_fixed, t_leaf)
            ut = grad(u.sum(), t_leaf, create_graph=True)[0]
        elif mode == "forward":
            try:
                from torch.func import jvp
            except Exception as e:
                raise RuntimeError("time_ad='forward' requires torch.func.jvp") from e
            _, ut = jvp(lambda tt: self.psi_m(x_fixed, tt), (t_in,), (torch.ones_like(t_in),))
        else:
            raise ValueError("time_ad must be 'reverse' or 'forward'")

        self.state[key] = ut
        return ut

    def psi_tt(self, when: When = "now") -> torch.Tensor:
        key = ("psi_tt", when)
        if key in self.state:
            return self.state[key]

        mode = self.cfg["time_ad"]
        if mode == "reverse":
            # Recompute with a time leaf so we can differentiate ut w.r.t. t
            x_fixed = self.x().detach()
            t_in = self._t(when)
            if self.cfg["detach_time_for_psi"]:
                t_in = t_in.detach()
            t_leaf = t_in.clone().requires_grad_(True)
            u = self.psi_m(x_fixed, t_leaf)
            ut = grad(u.sum(), t_leaf, create_graph=True)[0]
            utt = grad(ut.sum(), t_leaf)[0]
        elif mode == "forward":
            try:
                from torch.func import jvp
            except Exception as e:
                raise RuntimeError("time_ad='forward' requires torch.func.jvp") from e
            x_fixed = self.x().detach()
            t_in = self._t(when)
            if self.cfg["detach_time_for_psi"]:
                t_in = t_in.detach()
            def ut_fun(tt):
                return jvp(lambda s: self.psi_m(x_fixed, s),
                        (tt,), (torch.ones_like(tt),))[1]
            _, utt = jvp(ut_fun, (t_in,), (torch.ones_like(t_in),))
        else:
            raise ValueError("time_ad must be 'reverse' or 'forward'")

        self.state[key] = utt
        return utt


    # ------------- velocity and divergence -------------

    def v(self, when: When = "now") -> torch.Tensor:
        key = ("v", when)
        if key not in self.state:
            self.state[key] = self.Xf(when) + self.psi_grad(when)
        return self.state[key]

    def v_div(self, when: When = "now", probes: Optional[int] = None) -> torch.Tensor:
        key = ("v_div", when, probes)
        if key not in self.state:
            p = int(self.cfg["divergence_probes"] if probes is None else probes)
            self.state[key] = _divergence_hutch(
                v_field=self.v(when),
                x=self.x(),
                probes=p,
                mode=self.cfg["rng"],
                gen=self._gen,
                create_graph=True
            )
        return self.state[key]

    # ------------- optional convenience -------------

    def x_next(self) -> torch.Tensor:
        if "x_next" not in self.state:
            self.state["x_next"] = self.x() + self.dt() * self.v("now")
        return self.state["x_next"]

    def zeros(self) -> torch.Tensor:
        """A [B,K,1] zero tensor on the state's device/dtype."""
        return torch.zeros((self.B, self.K, 1), device=self.device, dtype=self.dtype)


