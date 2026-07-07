import math
from typing import Callable, Dict, List, Optional

import torch
from torch import nn

from lib.pde_ops import PDEState
from lib.pde_losses import build_losses


# ================================================================
# Core stacked layers (vectorized over support-set axis K)
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
        # Stacked MLP body
        h = self.fc1(x)
        h = self.act1(h) + h
        # h = self.fc2(h)
        # h = self.act2(h) + h
        h = self.fc3(h)
        h = self.act3(h)

        # Base potential + direct linear term
        out_mlp = self.fc4(h) * self.c         # [B, K, n_out]
        out_dir = self.dir_linear(x)           # [B, K, n_out]
        out = out_mlp + out_dir                # [B, K, n_out]

        # EMA mean update over batch (per k, per output channel)
        if self.training and self.update_batchnorm:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)   # [K, n_out]
                self.running_mean.lerp_(batch_mean, 0.1)

        # Zero-mean per k
        out_centered = out - self.running_mean  # broadcasts [K, n_out] over batch

        # Optional nonlinearity on final potentials
        return self.final_activation(out_centered)


class TraversalPDE(nn.Module):
    """
    K-parallel TraversalPDE powered by PDEState and modular PDE losses.
    """
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
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
    ):
        super().__init__()
        if lambdas is None:
            lambdas = {}

        self.num_support_sets = int(num_support_sets)
        self.num_support_timesteps = int(num_support_timesteps)
        self.support_vectors_dim = int(support_vectors_dim)

        # Learnable per-k scale (kept for parity; not wired to dt by default)
        self.c = nn.Parameter(torch.full((self.num_support_sets, 1), 1.0))

        # Stacked semantic potential
        self.F = StackedSemanticPotential(
            K=self.num_support_sets,
            n_in=self.support_vectors_dim,
            n_out=1,
            final_activation=nn.Identity(),
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
            dim_correction=1/(self.support_vectors_dim**0.5),
        )

        # for telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    # ---- one step ----
    def _per_step(
        self,
        z_bkd: torch.Tensor,
        dt: torch.Tensor = 1.0,
        direction: int = +1,
    ):
        st = PDEState(
            f=self.F,
            z=z_bkd,
            direction=torch.ones(1,),
            need_next=self._needs_next,
            dt_value=dt,
            **self._pde_cfg,
        )

        # compute & sum selected losses [B,K,1]
        per_bk = [L(st) for L in self.losses]
        L_sum = sum(per_bk) if per_bk else st.zeros()

        # next latent: x_next = x + dt * v(now)
        x_next = st.x_next()

        # optional small step noise
        if self.training:
            with torch.no_grad():
                step_delta = x_next - st.x()
                step_delta_sq_norms = step_delta.pow(2).sum(dim=-1, keepdim=True)
                step_delta_norms = step_delta_sq_norms.sqrt()
                latent_noise = torch.randn_like(x_next)
                latent_noise = latent_noise - step_delta * (
                    (latent_noise * step_delta).sum(dim=-1, keepdim=True) / step_delta_sq_norms.clamp_min(1e-12)
                )
                latent_noise = latent_noise / latent_noise.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
                latent_noise = latent_noise * (step_delta_norms.clamp_min(step_delta_norms.mean().item()/3) / 5.0)
            x_next_noisy = x_next + latent_noise
        else:
            x_next_noisy = x_next

        return st, x_next_noisy, L_sum, st.dt()

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
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps) - 1)

        if t_index.ndim == 1:
            t_index = t_index.unsqueeze(-1)
        i_target = torch.clamp(t_index, 0, T - 1).long().squeeze(-1)

        # expand once to K stacks
        if w_avg is not None:
            z = z - w_avg.reshape(1, D)
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()

        potential_preds = []
        latent1_bk = None
        latent2_bk = None
        last_st: Optional[PDEState] = None

        L_accum = None  # accumulate per-[B,K,1]

        step_iter = range(T)
        self.F.update_batchnorm = True

        for i in step_iter:
            st, x_next, L_step, dt = self._per_step(
                z_curr,
                dt=dt,
                direction=direction,
            )

            potential_preds.append(st.f().detach())

            # accumulate loss per step
            L_accum = L_step if L_accum is None else L_accum + L_step

            # capture (latent1, latent2) at the requested index
            mask_b = (i_target == i).view(B, 1, 1)
            if latent1_bk is None:
                latent1_bk = torch.where(mask_b, z_curr, torch.zeros_like(z_curr))
                latent2_bk = torch.where(mask_b, x_next, torch.zeros_like(x_next))
            else:
                latent1_bk = torch.where(mask_b, z_curr, latent1_bk)
                latent2_bk = torch.where(mask_b, x_next, latent2_bk)

            # advance
            z_curr = x_next
            last_st = st

            # do not move the running mean at subsequent steps
            self.F.update_batchnorm = False

        # average over steps
        L_total_per_bk = L_accum / float(T) if L_accum is not None else last_st.zeros()
        L_total_mean = L_total_per_bk.mean()

        # for predicting the attribute
        potential_preds.append(last_st.f("next").detach())
        potential_preds = torch.cat(potential_preds, dim=-1)

        self._acc = {
            "xf_now": last_st.Xf()
            if last_st is not None
            else torch.zeros(B, K, D, device=z.device, dtype=z.dtype),
            "L_mean": L_total_mean.detach(),
            **last_st.state["losses"],
        }

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
        if len(z.shape) == 3:
            B, K, D = z.shape
            z_curr = z
        else:
            B, D = z.shape
            K = self.num_support_sets
            z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()

        st, x_next, L_step, dt = self._per_step(
            z_curr,
            dt=dt,
            direction=direction,
        )
        return z_curr, x_next - z_curr