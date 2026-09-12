import math
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from lib.pde_ops import PDEState, broadcast_bk
from lib.pde_losses import build_losses


@torch.no_grad()
def cone_noise(delta: torch.Tensor, aperture: float = 0.2,
                        gaussian: torch.Tensor = None) -> torch.Tensor:
    """Sample the transverse part of a fixed-angle Gaussian-cone step.

    Project isotropic Gaussian noise onto delta's orthogonal complement and
    normalize it to radius aperture*||delta||. Thus the step delta+noise has
    tan(angle)=aperture away from the numerical clamps: its azimuth is uniform,
    its angle is fixed. This is
    the existing normalized-Gaussian kernel, not additive Gaussian diffusion.
    Noise is stop-gradient; only the deterministic step trains the potential.
    ``gaussian`` permits reproducible kernel comparisons without changing RNG.
    """
    noise = torch.randn_like(delta) if gaussian is None else gaussian.clone()
    radius2 = delta.square().sum(-1, keepdim=True)
    projection = (noise * delta).sum(-1, keepdim=True) / radius2.clamp_min(1e-12)
    noise.addcmul_(delta, projection, value=-1)
    return noise.mul_(radius2.sqrt().mul_(aperture) / noise.norm(dim=-1, keepdim=True).clamp_min_(1e-12))

@torch.no_grad()
def gaussian_cone_noise(delta: torch.Tensor, aperture: float = 0.5,
                        gaussian: torch.Tensor = None) -> torch.Tensor:
    """Sample the transverse part of a fixed-angle Gaussian-cone step.
    """
    dim_correction = 1/delta.shape[-1]**0.5
    noise = torch.randn_like(delta) if gaussian is None else gaussian.clone()
    radius2 = delta.norm(dim=-1, keepdim=True)
    return noise.mul_(radius2.mul_(aperture * dim_correction))


# ================================================================
# Core stacked layers (vectorized over traversal-set axis K)
# ================================================================
class StackedLinear(nn.Module):
    """
    Per-k linear layers evaluated in parallel.
    Weight: [K, out, in], Bias: [K, out]
    Forward expects x: [B, K, in] -> y: [B, K, out]
    """
    def __init__(self, K: int, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.K = int(K)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # [K, out, in]
        self.weight = nn.Parameter(
            torch.empty(self.K, self.out_features, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.K, self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # K independent Kaiming-uniform initializations
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weight.data[k], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, in]
        # y[b,k,o] = sum_i x[b,k,i] * W[k,o,i] + b[k,o]
        y = torch.einsum("bki,koi->bko", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)  # [1,K,out]
        return y


class StackedSemanticPotential(nn.Module):
    """
    Stacked scalar potentials F^k(x):
      - x: [B, K, D_in]
      - outputs: [B, K, n_out] (often n_out=1)

    Only the final outputs are mean-centered per-k via running EMA.
    """
    def __init__(
        self,
        K: int,
        n_in: int,
        n_out: int = 1,
        n_hidden: int = 128,
        activation: nn.Module = nn.Softplus(),
        final_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_hidden = int(n_hidden)

        # Shared shape convention: StackedLinear expects input [B, K, *]
        self.fc1 = StackedLinear(self.K, self.n_in, self.n_hidden)
        self.act1 = activation

        # self.fc2 = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        # self.act2 = activation

        self.fc3 = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        self.act3 = activation

        self.fc4 = StackedLinear(self.K, self.n_hidden, self.n_out)

        # Running mean (per k, per output channel), registered as buffer
        self.register_buffer("running_mean", torch.zeros(self.K, self.n_out))

        # Additional linear component from input to output, initialized
        # to small coordinate-like directions (per k).
        self.dir_linear = StackedLinear(self.K, self.n_in, self.n_out, bias=False)
        with torch.no_grad():
            W = self.dir_linear.weight  # [K, n_out, n_in]
            W.zero_()
            for k in range(self.K):
                sign = 1 if ((k-(k % self.n_in)) // self.n_in) % 2 == 0 else -1
                W[k, 0, k % self.n_in] = sign * 0.1
            W.add_(torch.randn_like(W) * 1e-7)

        # Scalar per-k gain on the fc4 branch
        self.c = nn.Parameter(torch.full((self.K, 1), 1.0))

        self.final_activation = final_activation
        self.update_batchnorm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, D_in]
        returns: [B, K, n_out]
        """
        return self._evaluate(x)[0]

    def value_and_grad(self, x: torch.Tensor):
        """F and its input gradient, with a differentiable explicit chain rule.

        No dense Jacobian or autograd engine invocation is needed for the
        standard scalar Softplus potential. Other architectures keep autograd.
        Parameters, weighted coordinates and higher input derivatives retain
        their graph, including differentiation through the gradient norm.
        """
        supported = (type(self) is StackedSemanticPotential and self.n_out == 1
                     and not hasattr(self, "fc2") and type(self.act1) is nn.Softplus
                     and type(self.act3) is nn.Softplus and type(self.final_activation) is nn.Identity)
        if not supported:
            value = self(x)
            return value, torch.autograd.grad(value.sum(), x, create_graph=True)[0]
        return self._evaluate(x, input_gradient=True)

    def _evaluate(self, x, input_gradient=False, update_stats=True):
        # Stacked MLP body
        h1 = self.fc1(x)
        h = self.act1(h1) + h1
        # h = self.fc2(h)
        # h = self.act2(h) + h
        h3 = self.fc3(h)
        h = self.act3(h3)

        # Base potential + direct linear term
        out_mlp = self.fc4(h) * self.c         # [B, K, n_out]
        out_dir = self.dir_linear(x)           # [B, K, n_out]
        out = out_mlp + out_dir                # [B, K, n_out]

        # EMA mean update over batch (per k, per output channel)
        if update_stats and self.training and self.update_batchnorm:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)   # [K, n_out]
                self.running_mean.lerp_(batch_mean, 0.1)

        # Zero-mean per k
        out_centered = out - self.running_mean  # broadcasts [K, n_out] over batch

        # Optional nonlinearity on final potentials
        value = self.final_activation(out_centered)
        if not input_gradient:
            return value, None

        def slope(activation, preactivation):
            scaled = preactivation * activation.beta
            return torch.where(scaled > activation.threshold, 1.0, scaled.sigmoid())

        adjoint = slope(self.act3, h3) * (self.fc4.weight[:, 0] * self.c)
        adjoint = torch.einsum("bko,koi->bki", adjoint, self.fc3.weight)
        adjoint = adjoint * (1 + slope(self.act1, h1))
        gradient = torch.einsum("bko,koi->bki", adjoint, self.fc1.weight)
        return value, gradient + self.dir_linear.weight[:, 0]


def _base_velocity(potential, x, eps):
    _, g = potential._evaluate(x, input_gradient=True, update_stats=False)
    return g / (g.square().sum(-1, keepdim=True) + eps)


def _fixed_point(function, initial, max_iter, atol, rtol, label):
    """Check the equation residual, never silently accept an unsolved step."""
    current = initial
    for _ in range(max_iter):
        updated = function(current)
        error = (updated - current).abs().amax(dim=-1)
        scale = current.abs().amax(dim=-1)
        if bool(torch.all(error <= atol + rtol * scale)):
            return current
        current = updated
    raise RuntimeError(
        f"{label} did not converge in {max_iter} iterations. "
        "Reduce dt or use potential_jet; increasing max_iter only helps a contracting solve."
    )


class _ImplicitMidpoint(torch.autograd.Function):
    """Solve m=x+a*v(m), retaining one potential graph, not solver history.

    Backward solves (I-a*J_v)^T lambda = upstream. The explicit potential
    gradient below means training needs only this first derivative of m.
    Higher-order differentiation THROUGH this custom backward is unsupported.
    """
    @staticmethod
    def forward(ctx, x, a, potential, eps, max_iter, atol, rtol, *parameters):
        m = _fixed_point(lambda m: x + a * _base_velocity(potential, m, eps),
                         x, max_iter, atol, rtol, "Midpoint forward solve")
        ctx.potential, ctx.eps = potential, eps
        ctx.settings = max_iter, atol, rtol
        ctx.save_for_backward(m, a, *parameters)
        return m.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        midpoint, a, *parameters = ctx.saved_tensors
        with torch.enable_grad():
            m = midpoint.detach().requires_grad_(True)
            v = _base_velocity(ctx.potential, m, ctx.eps)

            def adjoint(lam):
                jtv = torch.autograd.grad(v, m, a * lam, retain_graph=True)[0]
                return upstream + jtv

            lam = _fixed_point(adjoint, upstream, *ctx.settings, "Midpoint backward solve")
            parameter_grads = (torch.autograd.grad(v, parameters, a * lam, allow_unused=True)
                               if parameters else ())
        grad_a = (lam * v.detach()).sum(-1, keepdim=True)
        return (lam, grad_a, None, None, None, None, None, *parameter_grads)


class DirectionalSemanticPotential:
    """Parameter-free directional view of one shared scalar network.

    midpoint: m=x+s*h/2*I_eps(grad F(m)); grad F_s(x)=s*grad F(m).
    potential_jet: F_s(x)=s*F(x)+h/4*log(||grad F(x)||²+eps).
    See docs/bidirectional_traversals.md for the generating-potential proof.
    """
    prefer_explicit_values = True

    def __init__(self, base, direction, dt, architecture, eps, solver_settings):
        self.base, self.direction, self.dt = base, direction, dt
        self.architecture, self.eps = architecture, eps
        self.solver_settings = solver_settings

    def __call__(self, x):
        return self.value_and_grad(x)[0]

    def value_and_grad(self, x):
        s, h = self.direction, self.dt
        if self.architecture == "midpoint":
            parameters = tuple(p for p in self.base.parameters() if p.requires_grad)
            m = _ImplicitMidpoint.apply(x, s * h / 2, self.base, self.eps,
                                        *self.solver_settings, *parameters)
            value, g = self.base._evaluate(m, input_gradient=True)
            q = g.square().sum(-1, keepdim=True)
            # The rational term is constant only when eps=0. Keeping it
            # makes the scalar identity exact for our regularized inversion.
            value = s * value + h / 4 * (q + self.eps).log() - h / 2 * q / (q + self.eps)
            return value, s * g

        # One scalar correction, no directional heads or detached gates.
        # Differentiating log(q) is a Hessian-vector product, not a Hessian.
        with torch.enable_grad():
            x = x.requires_grad_(True)
            value, g = self.base._evaluate(x, input_gradient=True)
            log_q = (g.square().sum(-1, keepdim=True) + self.eps).log()
            correction = torch.autograd.grad(log_q.sum(), x, create_graph=True)[0]
            return s * value + h / 4 * log_q, s * g + h / 4 * correction


def add_traversal_arguments(parser):
    parser.add_argument("--potential-hidden", type=int, default=32,
                        help="potential hidden width, saved for checkpoint reconstruction")
    parser.add_argument("--traversal-architecture", choices=("euler", "potential_jet", "midpoint"),
                        default="euler", help="shared directional potential / step architecture")
    parser.add_argument("--bidirectional", action="store_true", help="sample a sign per sample and traversal")
    parser.add_argument("--midpoint-max-iter", type=int, default=80)
    parser.add_argument("--midpoint-atol", type=float, default=1e-6)
    parser.add_argument("--midpoint-rtol", type=float, default=1e-6)
    parser.add_argument("--traversal-noise", type=float, default=0.5,
                        help="training Gaussian aperture; 0 for deterministic reversal")


def traversal_options(args):
    """Shared by training and checkpoint reconstruction; old args use Euler."""
    return dict(n_hidden=getattr(args, "potential_hidden", 128),
                architecture=getattr(args, "traversal_architecture", "euler"),
                midpoint_max_iter=getattr(args, "midpoint_max_iter", 80),
                midpoint_atol=getattr(args, "midpoint_atol", 1e-6),
                midpoint_rtol=getattr(args, "midpoint_rtol", 1e-6),
                noise_aperture=getattr(args, "traversal_noise", 0.5))


class TraversalPDE(nn.Module):
    """
    K-parallel Traversals powered by PDEState and modular PDE losses.
    """
    def __init__(
        self,
        num_traversal_sets: int,
        num_traversal_timesteps: int,
        traversal_vectors_dim: int,
        n_hidden: int = 128,
        final_activation: nn.Module = nn.Identity(),
        lambdas: Optional[Dict[str, float]] = None,          # ONLY what you want active
        n_laplace_probes: int = 1,
        # PDEState config
        time_ad: str = "reverse",
        detach_between_steps: bool = False,
        eps_norm2: float = 1e-8,
        divergence_probes: int = 1,
        rng: str = "rademacher",
        seed: Optional[int] = None,
        # optional: prior score function for DivPrior/Poisson (defaults to Gaussian score -x)
        prior_score: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        architecture: str = "euler",
        midpoint_max_iter: int = 80,
        midpoint_atol: float = 1e-6,
        midpoint_rtol: float = 1e-6,
        noise_aperture: float = 0.5,
    ):
        super().__init__()
        if lambdas is None:
            lambdas = {}

        if architecture not in ("euler", "potential_jet", "midpoint"):
            raise ValueError(f"Unknown traversal architecture: {architecture}")
        if architecture != "euler" and type(final_activation) is not nn.Identity:
            raise ValueError("Directional architectures require the standard scalar Softplus/Identity potential")
        if midpoint_max_iter < 1 or midpoint_atol <= 0 or midpoint_rtol < 0 or noise_aperture < 0:
            raise ValueError("Invalid midpoint solver settings or noise aperture")
        if eps_norm2 < 0:
            raise ValueError("eps_norm2 must be nonnegative")
        if architecture == "midpoint":
            # These losses use only explicit first input gradients. Other
            # PINNs may differentiate through the implicit backward twice.
            supported = {"bb", "g2orth", "signed_g2orth", "epsilon"}
            unsupported = [key for key, weight in lambdas.items() if weight and key.lower() not in supported]
            if unsupported:
                raise ValueError(f"midpoint supports BB/g2orth/signed_g2orth losses; got {unsupported}")
        self.architecture = architecture
        self.midpoint_settings = (int(midpoint_max_iter), float(midpoint_atol), float(midpoint_rtol))
        self.noise_aperture = float(noise_aperture)

        self.num_traversal_sets = int(num_traversal_sets)
        self.num_traversal_timesteps = int(num_traversal_timesteps)
        self.traversal_vectors_dim = int(traversal_vectors_dim)

        # Learnable per-k scale (kept for parity; not wired to dt by default)
        self.c = nn.Parameter(torch.full((self.num_traversal_sets, 1), 1.0))

        # Stacked semantic potential
        self.F = StackedSemanticPotential(
            K=self.num_traversal_sets,
            n_in=self.traversal_vectors_dim,
            n_out=1,
            n_hidden=n_hidden,
            final_activation=final_activation,
        )

        # PDEState config for each step
        self._pde_cfg = dict(
            time_ad=time_ad,
            detach_between_steps=detach_between_steps,
            eps_norm2=eps_norm2,
            laplace_probes=int(n_laplace_probes),
            divergence_probes=int(divergence_probes),
            rng=rng,
            seed=seed,
        )

        # Build modular loss list from registry.
        # Pass modules/params needed by certain losses through ctx.
        epsilon = float(lambdas.get("epsilon", 0.0))  # scalar param, not a loss
        self.losses, self._needs_next = build_losses(
            lambdas,
            F=self.F,
            epsilon=epsilon,
            prior_score=prior_score,
            dim_correction=1/(self.traversal_vectors_dim**0.5),
        )

        # for telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    # ---- one step ----
    def _per_step(
        self,
        z_bkd: torch.Tensor,
        dt: torch.Tensor = 1.0,
        direction: int = +1,
        *,
        compute_losses: bool = True,
    ):
        sign = broadcast_bk(direction, z_bkd, "direction")
        if not bool(torch.all((sign == 1) | (sign == -1))):
            raise ValueError("direction must contain only -1 or +1")
        step = broadcast_bk(dt, z_bkd, "dt")
        if not bool(torch.isfinite(step).all()):
            raise ValueError("dt must be finite")
        # Preserve signed-dt inference used by existing visualizers.
        sign = sign * torch.where(step < 0, -1.0, 1.0)
        step = step.abs()
        potential = self.F
        if self.architecture != "euler":
            potential = DirectionalSemanticPotential(self.F, sign, step, self.architecture,
                                                     self._pde_cfg["eps_norm2"], self.midpoint_settings)
        st = PDEState(
            f=potential,
            z=z_bkd,
            direction=sign if self.architecture == "euler" else 1,
            need_next=self._needs_next,
            dt_value=step,
            semantic_direction=sign if self.architecture != "euler" else 1,
            **self._pde_cfg,
        )

        # compute & sum selected losses [B,K,1]
        L_sum = sum((loss(st) for loss in self.losses), st.zeros()) if compute_losses else st.zeros()

        # next latent: x_next = x + dt * v(now)
        x_next = st.x_next()

        # optional small step noise
        if self.training and self.noise_aperture:
            x_next = x_next + gaussian_cone_noise(x_next.detach() - st.x().detach(), self.noise_aperture)
        # Do not feed the signed state timestep back into the next rollout
        # iteration: direction remains fixed for the whole trajectory.
        return st, x_next, L_sum, dt

    # ---- unrolled training ----
    def forward(
        self,
        z: torch.Tensor,
        t_index: torch.Tensor,
        dt: torch.Tensor,
        direction: int = +1,
        w_avg: torch.Tensor = None,
    ):
        """
        Returns:
          potential_preds: [B,K,T+1] detached potential predictions
          latent1_bk: [B,K,D] latent at t_index
          latent2_bk: [B,K,D] next latent after t_index
          L_total_mean: scalar
          delta_y: [B,K,1] detached final f(next)-f(now).detach()
        """
        B, D = z.shape
        K = self.num_traversal_sets
        T = max(1, int(self.num_traversal_timesteps) - 1)

        if t_index.ndim == 1:
            t_index = t_index.unsqueeze(-1)
        i_target = torch.clamp(t_index, 0, T - 1).long().squeeze(-1)

        # expand once to K stacks
        if w_avg is not None:
            z = z - w_avg.reshape(1, D)
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()

        potential_preds = []
        path = [z_curr]
        L_accum = z.new_zeros(B, K, 1)
        self.F.update_batchnorm = True

        for _ in range(T):
            st, x_next, L_step, dt = self._per_step(
                z_curr,
                dt=dt,
                direction=direction,
            )

            potential_preds.append(st.f().detach())

            # accumulate loss per step
            L_accum = L_accum + L_step

            # capture (latent1, latent2) at the requested index
            path.append(x_next)

            # advance
            z_curr = x_next
            last_st = st

            # do not move the running mean at subsequent steps
            self.F.update_batchnorm = False

        # average over steps
        L_total_mean = (L_accum / T).mean()

        # for predicting the attribute
        potential_preds.append(last_st.f("next").detach())
        potential_preds = torch.cat(potential_preds, dim=-1)

        self._acc = {
            "xf_now": last_st.Xf().detach(),
            "L_mean": L_total_mean.detach(),
            **last_st.state["losses"],
        }

        path = torch.stack(path)
        batch = torch.arange(B, device=z.device)
        latent1_bk, latent2_bk = path[i_target, batch], path[i_target + 1, batch]

        if w_avg is not None:
            latent1_bk = latent1_bk + w_avg.reshape(1, 1, D)
            latent2_bk = latent2_bk + w_avg.reshape(1, 1, D)

        return potential_preds, latent1_bk, latent2_bk, L_total_mean, last_st.delta_y().detach()

    def get_losses(self) -> Dict[str, torch.Tensor]:
        return self._acc

    @torch.enable_grad()
    def inference(
        self,
        z: torch.Tensor,
        t_index: torch.Tensor = None,
        dt: torch.Tensor = 1.0,
        direction: int = +1,
    ) -> List[torch.Tensor]:
        """One differentiable step; eval mode omits unused training penalties.

        Training mode retains loss evaluation and its RNG draws, preserving
        the stochastic kernel for callers that intentionally infer in train().
        """
        if len(z.shape) == 3:
            B, K, D = z.shape
            z_curr = z
        else:
            B, D = z.shape
            K = self.num_traversal_sets
            z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()

        st, x_next, L_step, dt = self._per_step(
            z_curr,
            dt=dt,
            direction=direction,
            compute_losses=self.training,
        )
        return z_curr, x_next - z_curr
