# Traversal optimization follow-up — 2026-09-09

All model execution, numerical derivative checks, and synthesis benchmarks ran
on MPS, outside the sandbox, using `/opt/anaconda3/envs/manip311/bin/python`
and PyTorch `2.15.0.dev20260817`. TensorBoard's normal PNG/event serialization
runs on the host; there is no CPU model benchmark or CUDA performance claim.

The comparison baseline is the worktree **after the preceding optimization
pass**, not the original repository. Its source snapshot is currently at
`/tmp/traversal-baseline.YZjFvX`. Unrelated pre-existing edits were preserved.

## Results

SNGAN AnimeFaces, B=6, K=32, D=128, 20 configured timesteps (19 steps), potential
hidden width 32, LeNet recognizer; BB=0.25 and signed_g2orth=1.0. Timings are
warmed, alternating-order medians with MPS synchronization, not unsynchronized
Python dispatch timings. Saved-storage measurements are separate from timings.

| Operation | Before | After | Interpretation |
|---|---:|---:|---|
| PDE rollout + backward | 33.49 ms | 28.41 ms | 15% faster |
| Differentiable eval inference step | 0.879 ms | 0.546 ms | 38% faster |
| Cone kernel, same Gaussian draw | 0.161 ms | 0.140 ms | Small, noisy microbenchmark improvement |
| Complete image logging + SNGAN B=6 inference | 172.61 ms | 147.34 ms | 15% faster by overlapping encoding; includes final flush |
| `loss_allK`, synthesis and backward | 2,727.93 ms | 2,696.89 ms | Essentially unchanged; another paired run was 2,671 vs 2,672 ms |
| Peak live autograd-saved storage in `loss_allK` | 3,266.31 MiB | 3,096.35 MiB | 170 MiB less, without recomputation |

Logging alone remains about 144 ms: asynchronous encoding does not eliminate
that work. The benefit comes from overlapping it with accelerator work. PNGs
were byte-identical and both existing image tags remained readable after close.

Optional `--generator-recompute-chunk 32` trades time for much lower activation
memory. On the B*K=192 differentiable generator batch:

| Generator VJP | Time | Peak live autograd-saved storage |
|---|---:|---:|
| Ordinary synthesis + backward (default) | 1,531 ms | 2,913.14 MiB |
| Replay chunks of 32 | 2,466 ms | 485.64 MiB |
| Replay chunks of 64 | 2,428 ms | 971.14 MiB |

These memory numbers count distinct live storages saved by autograd, excluding
parameter storage. They are **not total VRAM or allocator/workspace peaks**.

## What changed, and why it is equivalent

- `loss_allK`: 147 to 56 lines, with small synthesis, pair-scoring, VJP and
  telemetry helpers. It backpropagates the historical classifier pair before
  constructing the differentiable endpoint graph. No optimizer step occurs
  between the two pairs. The detached latent bridge, classification masking,
  accumulation denominator, eight return values and metric names remain.
  The whole helper block is also shorter than the previous loss implementation.
- The scalar stacked Softplus potential exposes a differentiable explicit
  input-gradient chain rule. This avoids invoking autograd's reverse engine to
  obtain that gradient at every step; autograd still differentiates the formula
  for training and higher derivatives. Weighted coordinates, the residual
  connection, gains, centering, parameter names and shapes are unchanged.
  Other activations, final nonlinearities and subclasses fall back to autograd.
  Re-enabling the commented `fc2` also disables this specialization. Set
  `S._pde_cfg['explicit_input_grad'] = False` to compare the generic path.
- Endpoint selection stacks the trajectory once and gathers each batch item's
  two requested states. All training steps and their PDE losses still run;
  there is no biased truncation at the maximum sampled timestep.
- `gaussian_cone_noise` names and documents the existing stochastic kernel:
  project an isotropic Gaussian onto the transverse subspace, then normalize
  it to 0.2 times the deterministic step length. It has a fixed cone angle,
  not a Gaussian radial distribution. Existing clamps and stop-gradient behavior
  remain; zero/tiny steps are checked against the old expression.
- Eval inference skips unused PDE penalties while retaining differentiation.
  Training-mode inference retains loss evaluation and its RNG draws.
- Optional frozen-generator replay retains only latents in forward, then
  recomputes one chunk per input VJP. Recognizer batches are never chunked, so
  BatchNorm training semantics remain. It requires frozen, eval, deterministic,
  batch-independent synthesis and is disabled by default. Only SNGAN was tested.
- TensorBoard image encoding uses one bounded worker. It receives detached host
  grids only; pending work is drained before accepting another batch. `flush()`
  and `close()` wait and propagate errors. `asynchronous=False` restores the
  synchronous path. Existing tags and the commented image-pruning experiments
  are retained. Earlier histogram-transfer and statistics optimizations remain.
- Antisymmetric scoring now goes through DataParallel's forward entry point
  rather than bypassing scatter/gather via a bound module method. CUDA behavior
  has not been benchmarked on this machine.

## Rejected recognizer candidates

1. Group both pair orders along a grouped-convolution channel axis, sharing
   weights and keeping each BatchNorm's original N-sized batches: 102 vs 96 ms.
2. Factor the first convolution into center/offset filters `(A+B)` and `(A-B)`,
   halving its convolution arithmetic: 100 vs 97 ms. MPS overhead erased the gain.

Neither candidate was retained. Concatenating orders into one 2N training batch
was not substituted: that would change BatchNorm statistics. The earlier
`AntisymmetricLeNet` architecture remains available, but is explicitly documented
as a different, difference-only model that discards image-center context—not an
equivalent optimization of the existing classifier.

## Verification and reproduction

19 regression tests pass, including input/Hessian/mixed-parameter derivatives,
Softplus thresholds and activation fallbacks, Gaussian-cone degeneracies,
endpoint selection with clamping/offset/detached steps, differentiable inference,
existing trainer/logging contracts, and asynchronous encoding failure handling.

The paired full SNGAN check holds the rollout fixed to isolate trainer algebra:
logits, images and traversal gradients are identical; recognizer gradient error
is at most 7.5e-9, and BatchNorm buffers match. The explicit rollout is checked
separately: endpoint error <2e-6, parameter-gradient error <5e-7. Tiny changes to
images can flip MaxPool near-ties, so bitwise full-network gradient agreement is
not claimed for the analytic-gradient rollout. Generator replay produces
identical images and input-VJP error <2.1e-11 for the tested cotangent.

Run these commands outside the sandbox; every benchmark refuses a missing MPS
backend. Do not run accelerator benchmarks concurrently.

```sh
/opt/anaconda3/envs/manip311/bin/python -c 'import torch; assert torch.backends.mps.is_available(); torch.set_default_device("mps"); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_traversal_revision.py", "tests/test_traversal_optimizations.py"]))'
/opt/anaconda3/envs/manip311/bin/python tests/benchmark_traversal_revision.py --baseline /tmp/traversal-baseline.YZjFvX --full
/opt/anaconda3/envs/manip311/bin/python tests/benchmark_generator_vjp.py
/opt/anaconda3/envs/manip311/bin/python tests/benchmark_image_logging_overlap.py
```

The first, third and fourth commands are self-contained. The second requires
the pre-edit snapshot (copy it before `/tmp` is cleared if retaining that exact
historical comparison). No long training/convergence run, large GAN, non-SNGAN
wrapper, or CUDA/multi-GPU benchmark was performed in this follow-up.
