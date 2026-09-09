"""Activation-bounded input gradients for a frozen deterministic generator."""
import torch
from torch.autograd.function import once_differentiable


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
