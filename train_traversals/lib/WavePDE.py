import math
from typing import Dict, Tuple, Optional, List

import torch
from torch import nn
from torch.autograd import grad
from torch.func import jvp as jvp_fwd

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



class OrthonormalRotationK(nn.Module):
    """
    Learnable orthonormal rotation acting along the K dimension.

    Input:  x [B, K, C]
    Output: y [B, K, C]

    Internal parameter 'weight' is projected onto the set of orthonormal
    matrices via a simple Gram–Schmidt procedure (no torch.linalg.qr).
    """
    def __init__(self, K: int, eps: float = 1e-8):
        super().__init__()
        self.K = int(K)
        self.eps = eps
        # Start at identity (no rotation initially)
        self.weight = nn.Parameter(torch.eye(self.K))

    @torch.no_grad()
    def _orthonormalize_(self):
        """
        Classical Gram–Schmidt on columns of self.weight:
        produces Q with Q^T Q = I (within numerical tolerance).
        """
        W = self.weight.data  # [K, K]
        K = self.K

        # We'll build Q column-by-column
        Q = torch.zeros_like(W)

        for i in range(K):
            # Take current column
            v = W[:, i]

            if i > 0:
                # Project v onto span of previous Q columns and subtract
                # proj_coeffs shape: [i]
                proj_coeffs = torch.matmul(Q[:, :i].T, v)          # [i]
                v = v - torch.matmul(Q[:, :i], proj_coeffs)        # [K]

            # Normalize
            norm = v.norm(p=2)
            if norm < self.eps:
                # Degenerate direction: fall back to a standard basis vector
                v = torch.zeros_like(v)
                v[i] = 1.0
                norm = 1.0

            Q[:, i] = v / norm

        self.weight.data.copy_(Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K, C]
        returns: [B, K, C]
        """
        # Re-orthonormalize before using the weight
        self._orthonormalize_()

        # Rotate along K: treat last dim C as 'channels', rotate K per channel.
        # x: [B, K, C] -> [B, C, K]
        x_bcK = x.transpose(1, 2)      # [B, C, K]
        # Apply rotation: for each (B, C): x[b, c, :] @ R^T
        y_bcK = x_bcK @ self.weight.T  # [B, C, K]
        # Restore [B, K, C]
        y = y_bcK.transpose(1, 2)
        return y


class StackedSemanticPotential(nn.Module):
    """
    Stacked scalar potentials F^k(x):
      - x: [B, K, D_in]
      - outputs: [B, K, n_out] (often n_out=1)
    Only the final outputs are mean-centered per-k via running EMA.
    After mean subtraction, we apply a learnable orthonormal rotation
    along K to enable cross-potential gradients.
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

        # self.fc3 = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        self.act3 = activation

        self.fc4 = StackedLinear(self.K, self.n_hidden, self.n_out)

        # Running mean (per k, per output channel), registered as buffer
        self.register_buffer("running_mean", torch.zeros(self.K, self.n_out))
        # If you ever want std again, register similarly:
        # self.register_buffer("running_std", torch.ones(self.K, self.n_out))

        # Additional linear component from input to output, initialized
        # to small "coordinate-like" directions (per k).
        self.dir_linear = StackedLinear(self.K, self.n_in, self.n_out, bias=False)
        with torch.no_grad():
            W = self.dir_linear.weight  # [K, n_out, n_in]
            W.zero_()
            for k in range(self.K):
                W[k, 0, k % self.n_in] = 0.1
            W.add_(torch.randn_like(W) * 1e-3)

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
        # h = self.act1(h) + h
        # h = self.fc2(h)
        # h = self.act2(h) + h
        # h = self.fc3(h)
        h = self.act3(h)

        # Base potential + direct linear term
        out_mlp = self.fc4(h) * self.c         # [B, K, n_out]
        out_dir = self.dir_linear(x)           # [B, K, n_out]
        out = out_mlp + out_dir                # [B, K, n_out]

        # EMA mean update over batch (per k, per output channel)
        if self.training and self.update_batchnorm:
            with torch.no_grad():
                batch_mean = out.mean(dim=0)   # [K, n_out]
                # EMA: running_mean <- (1 - alpha)*running_mean + alpha*batch_mean
                self.running_mean.lerp_(batch_mean, 0.1)  # use 0.1 or 0.9 as you like

        # Zero-mean per k
        out_centered = out - self.running_mean  # broadcasts [K, n_out] over batch

        # Optional nonlinearity on final potentials
        return self.final_activation(out_centered)



class StackedSinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding broadcast over K.
    Input: t [B, K, 1] -> emb [B, K, E]; E must be even.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "time embedding dim should be even"
        self.emb_dim = int(emb_dim)
        half_dim = emb_dim // 2
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,K,1]
        angles = t * self.freqs.view(1, 1, -1).to(t.dtype)  # [B,K,half]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,K,E]

class StackedSliceEnergy(nn.Module):
    def __init__(self, K: int, n_in: int, n_out: int = 1, n_hidden: int = 64, final_activation: nn.Module = nn.Identity(),
                 apply_output_bn: bool = False):
        super().__init__()
        self.K = int(K)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_hidden = int(n_hidden)


        # x pathway
        self.layer_x = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation1 = nn.Tanh()

        # time pathway (uses sinusoidal embeddings of size n_in)
        self.layer_pos = StackedSinusoidalPositionEmbeddings(self.n_hidden)
        self.layer_time = StackedLinear(self.K, self.n_hidden, self.n_hidden)
        self.activation2 = nn.GELU()
        self.layer_time2 = StackedLinear(self.K, self.n_hidden, self.n_in)

        # fusion + output
        self.layer_fusion = StackedLinear(self.K, self.n_in, self.n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = StackedLinear(self.K, self.n_in, self.n_out)
        self.activation4 = nn.Tanh()

        self.apply_output_bn = bool(apply_output_bn)
        self.out_bn = OutputBatchNormPerK(self.K, self.n_out) if self.apply_output_bn else None

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # x: [B,K,D], time: [B,K,1]
        xh = self.activation1(self.layer_x(x))
        t_feat = self.layer_pos(time)
        t_feat = self.layer_time(t_feat)
        t_feat = self.activation2(t_feat)
        t_feat = self.layer_time2(t_feat)

        h = self.activation3(self.layer_fusion(xh + t_feat))
        out = self.activation4(self.layer_out(h))  # [B,K,1]
        # if self.apply_output_bn:
        #     out = self.out_bn(out)
        return out

class SkipSliceEnergy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return (torch.zeros_like(x)*x).sum(dim=-1, keepdim=True)

class WavePDE(nn.Module):
    """
    K-parallel WavePDE powered by PDEState and modular PDE losses.
    """
    def __init__(
        self,
        num_support_sets: int,
        num_support_timesteps: int,
        support_vectors_dim: int,
        lambdas: Dict[str, float] = {},                   # ONLY what you want active (may include 0.0)
        n_laplace_probes: int = 1,
        apply_bn_on_psi_output: bool = False,
        # PDEState config
        time_ad: str = "reverse",
        detach_between_steps: bool = False,
        eps_norm2: float = 1e-8,
        divergence_probes: int = 1,
        rng: str = "rademacher",
        seed: Optional[int] = None,
        # optional: prior score function for DivPrior (defaults to Gaussian score -x)
        prior_score: Optional[callable] = None,
        only_potential: bool = False,
    ):
        super().__init__()
        self.num_support_sets = int(num_support_sets)
        self.num_support_timesteps = int(num_support_timesteps)
        self.support_vectors_dim = int(support_vectors_dim)

        # Learnable per-k scale (kept for parity; not wired to dt by default)
        self.c = nn.Parameter(torch.full((self.num_support_sets, 1), 1.0))

        # Stacked potentials
        
        self.PSI = StackedSliceEnergy(
            K=self.num_support_sets,
            n_in=self.support_vectors_dim,
            n_out=1,
            final_activation=nn.Identity(),
            apply_output_bn=apply_bn_on_psi_output,
        ) if not only_potential else SkipSliceEnergy()

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

        # Build modular loss list from registry (only specified keys are included).
        # Pass modules/params needed by certain losses through ctx.
        epsilon = float(lambdas.get("epsilon", 0.0))  # a scalar param (not a loss)
        self.losses, self._needs_next = build_losses(
            lambdas,
            F=self.F,
            epsilon=epsilon,
            prior_score=prior_score,
        )

        # for telemetry
        self._acc: Dict[str, torch.Tensor] = {}

    # ---- one step ----
    def _per_step(self, z_bkd: torch.Tensor, dt: torch.Tensor=1.0, direction: int = +1):
        st = PDEState(
            f=self.F,
            psi=self.PSI,
            z=z_bkd,
            need_next=self._needs_next,
            dt_value=dt,  # use ±1 step; wire self.c here if desired
            **self._pde_cfg,
        )

        # compute & sum selected losses [B,K,1]
        per_bk = [L(st) for L in self.losses]
        L_sum = sum(per_bk) if per_bk else st.zeros()  # if no losses, zero tensor

        # next latent (semi-implicit Euler @ now)
        x_next = st.x_next()

        # (optional) same small step noise as before
        if self.training:
            with torch.no_grad():
                step_delta_norms = (x_next - st.x()).norm(dim=-1, keepdim=True)
                latent_noise = torch.randn_like(x_next)
                latent_noise = latent_noise / latent_noise.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
                latent_noise = latent_noise * (step_delta_norms.clamp_min(step_delta_norms.mean().item()/3) / 5.0)
            x_next_noisy = x_next + latent_noise
        else:
            x_next_noisy = x_next
        return st, x_next_noisy, L_sum, st.dt()

    # ---- unrolled training ----
    def forward(self, z: torch.Tensor, t_index: torch.Tensor, dt: torch.Tensor, direction: int = +1, w_avg: torch.Tensor = None):
        """
        Returns:
          potential_preds: [B,K,1] (detached)
          latent1_bk, latent2_bk: [B,K,D] (pair captured at t_index)
          L_total_mean: scalar
        """
        B, D = z.shape
        K = self.num_support_sets
        T = max(1, int(self.num_support_timesteps)-1)

        if t_index.ndim == 1:
            t_index = t_index.unsqueeze(-1)
        i_target = torch.clamp(t_index, 0, T - 1).long().squeeze(-1)
        
        # expand once to K stacks
        if w_avg is not None:
            z = z - w_avg.reshape(1,D)
        z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()
        potential_preds = []
        latent1_bk = None
        latent2_bk = None
        last_st: Optional[PDEState] = None

        L_accum = None  # accumulate per-[B,K,1]

        step_iter = range(T)# if  else reversed(range(T))
        self.F.update_batchnorm = True
        for i in step_iter:
            st, x_next, L_step, dt = self._per_step(z_curr, dt=dt, direction=direction)


            potential_preds.append(st.f().detach())
            # accumulate loss per step
            L_accum = (L_step if L_accum is None else L_accum + L_step)

            # capture (latent1, latent2) at the requested index
            mask_b = (i_target == i).reshape(B, 1, 1)
            if latent1_bk is None:
                latent1_bk = torch.where(mask_b, z_curr, torch.zeros_like(z_curr))
                latent2_bk = torch.where(mask_b, x_next, torch.zeros_like(x_next))
            else:
                latent1_bk = torch.where(mask_b, z_curr, latent1_bk)
                latent2_bk = torch.where(mask_b, x_next, latent2_bk)

            # advance
            z_curr = x_next
            last_st = st
            # dont move the batchnorm at subsequent steps
            self.F.update_batchnorm = False


        # average over steps
        L_total_per_bk = L_accum / float(T) if L_accum is not None else last_st.zeros()
        L_total_mean = L_total_per_bk.mean()

        # for predicting the attribute
        potential_preds.append(last_st.f("next").detach())
        potential_preds = torch.cat(potential_preds, dim=-1)

        self._acc = {
            "xf_now": last_st.Xf() if last_st is not None else torch.zeros(B, K, D, device=z.device, dtype=z.dtype),
            "L_mean": L_total_mean.detach(),
            **last_st.state["losses"],
        }

        if w_avg is not None:
            latent1_bk = latent1_bk + w_avg.reshape(1,1,D)
            latent2_bk = latent2_bk + w_avg.reshape(1,1,D)
        return potential_preds, latent1_bk, latent2_bk, L_total_mean, last_st.delta_y().detach()

    def get_losses(self) -> Dict[str, torch.Tensor]:
        return self._acc

    @torch.enable_grad()
    def inference(self, z: torch.Tensor, t_index: torch.Tensor = None, dt: torch.Tensor = 1.0, direction: int = +1) -> List[torch.Tensor]:
        if len(z.shape) == 3:
            B, K, D = z.shape
            z_curr = z
        else:
            B, D = z.shape
            K = self.num_support_sets
            z_curr = z.unsqueeze(1).expand(B, K, D).contiguous()


        st, x_next, L_step, dt = self._per_step(z_curr, dt=dt, direction=direction)
        return z_curr, x_next-z_curr
