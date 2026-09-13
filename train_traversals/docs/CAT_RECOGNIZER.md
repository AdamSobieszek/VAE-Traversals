# CAT recognizer

Select it with `--recognizer-type CAT` in `train.py`, `train_GAT.py`, or
`train_StyleGAN.py`. Existing script defaults and the LeNet/ResNet backbones are
unchanged. The normal Python API remains:

```python
from lib.recognizer import Recognizer

R = Recognizer('CAT', dim_index=32, channels=3, pool_size=1)
logits, aux = R(img0, img1)  # [B, 32], None
reward_logits = R.whitened_antisymmetric_logits(img0, img1)
```

There is no class-label input. `dim_index` is the output dimension; the traversal
index is supplied only to cross-entropy by the existing trainer.

## Representation and rationale

GAT's discriminator uses a ViT representation with RoPE, qk normalization,
SwiGLU and residual LayerScale initialized to 0.1. These are retained, along
with CAT's RMSNorm, fixed 2D sine/cosine positions, learned scale embeddings
and separate CLS tokens. See [GAT, Appendix A.1](https://arxiv.org/html/2509.24935v3#A1).

CAT deliberately isolates attention by resolution. Its scale-aggregated
diagnostic removes both the discriminator attention mask and generator
consistency regularization and performs worse, with strong cross-scale
interaction. This is evidence motivating the retained separation, not a
controlled demonstration that the same choice must win on Traversals.
See [CAT, sections 3.1 and 4.2](https://arxiv.org/html/2605.26449v1).

The computation is:

1. Apply the existing `pool_size` average pooling to each input independently.
2. Concatenate the ordered pair along channels, producing `[B, 2C, H, H]`.
3. Build the pyramid internally using repeated differentiable 2x2 average
   pooling with stride two. This is box-filtered factor-two downsampling,
   supported natively on MPS. Nothing is detached or clamped, including the
   extrapolated images used by whitening.
4. At each scale, apply the **same** strided convolutional patch projection.
   Add fixed spatial positions and that scale's learned embedding, and prepend
   its learned CLS token plus scale embedding. CAT's RoPE rotates spatial query
   and key tokens and leaves CLS unrotated.
5. Pass each sequence independently through the **same** Transformer blocks.
   Neither patches nor CLS tokens attend across resolutions. Processing a
   complete scale before the next is equivalent to CAT's block-first loop
   because these blocks have no cross-scale state or stochastic dropout.
6. Stack the CLS features as `[B, scales, width]`, apply a shared RMSNorm and
   linear K-class head, and average scale logits. This adds little capacity,
   sends supervision to every scale, and keeps logit magnitude independent of
   the number of scales. The mean combines evidence without cross-scale token
   interactions. It does not impose any separate per-scale training loss.

`R.pair_encoder.forward_pyramid(paired_scales, return_per_scale=True)` returns
`(logits, per_scale_logits)` for diagnostics; each supplied scale is already
channel-concatenated. No activation graph is retained on the module for logging.
This internal boundary can later accept generator-provided image pyramids.

The recognizer has no label embedding, class conditioning/dropout, projection
discriminator dot product, real/fake head or GAN loss. It also has no R1/R2
penalty, DINO/REPA alignment, generator consistency loss, MNG, pretrained encoder,
or additional change to the traversal potentials or training objective.

## Controls, size and input resolution

| Python architecture kwarg | Default | CLI override |
| --- | ---: | --- |
| `hidden_size` | 192 | `--recognizer-hidden-size` |
| `depth` | 4 | `--recognizer-depth` |
| `num_heads` | 3 | `--recognizer-num-heads` |
| `patch_size` | 4 | `--recognizer-patch-size` |
| `num_scales` | 2 | `--recognizer-num-scales` |
| `mlp_ratio` | 4.0 | Python only |
| `layerscale` | 0.1 | Python only |
| `gradient_checkpointing` | False | `--recognizer-checkpointing` |
| `fused_attn` | True | `--recognizer-unfused-attention` disables |

For example, append `--recognizer-type CAT --recognizer-num-scales 3` to the
existing training command. `--recognizer-pool-size` explicitly overrides the
script's existing initial pooling rule; omitting it preserves that rule.
Architecture flags are used only for CAT. All positional tables are fixed,
cached per resolution/device/dtype, and excluded from state dictionaries.
Input resolution may change between calls without adding trainable parameters.

The original square-grid RoPE implementation is retained. Inputs must be square
and the **working resolution after initial pooling** must be divisible by
`patch_size * 2**(num_scales-1)`. Invalid sizes produce an error instead of
discarding border pixels. The head dimension must be divisible by four.

At 32x32, the default token lengths are 65 and 17; at 64x64 they are 257 and 65.
Attention cost still grows quadratically with token count. For 256–1024 pixel
images, set a larger patch size and/or explicit initial pooling as appropriate
for the desired working resolution. The model does not silently resize all
inputs to a fixed absolute resolution. For example, 256x256 working images with
`patch_size=16` have the same token counts as 64x64 images with `patch_size=4`.

For RGB pairs, default architecture parameter count is exactly
`1,800,576 + 193*K`: **1,806,752** at K=32, about **53 times smaller** than the
paper's approximately 96M Base discriminator. Counting the attached code gives
104,448,768 trainable parameters with its default auxiliary projection, or
97,103,872 with `z_dims=()`. Including its fixed positional Parameters gives
104,709,888 and 97,364,992 respectively. These implementation counts were made
using meta tensors, without allocating or executing Base on MPS.

SDPA is enabled for the trainer's ordinary reverse-mode gradient path.
Checkpointing uses `use_reentrant=False` and is off by default to avoid
recomputation overhead for this small backbone. Explicit attention is available
for higher-order derivative experiments whose accelerator SDPA kernel may not
support double backward; it passed the included second-derivative check.

## Exact score contracts

Raw `forward` returns unconstrained directional scores. The outer wrappers are
responsible for odd-under-swap supervision:

```text
A(x,y) = R(x,y) - R(y,x)
W(x,y) = R(2*x-y,y) - R(2*y-x,x)
```

In particular, whitening is **not** replaced by `A(2*x-y,y)`.
Both `A(y,x)=-A(x,y)` and `W(y,x)=-W(x,y)`. The existing `torch.lerp` operations,
`unflatten`, `unsqueeze`, `expand`, and late materialization of shared centers
are preserved. The wrappers accept `[B,...]` centers with `[B*K,...]` endpoints
and return `[B*K,K]`; raw `forward` continues to take matched batches.

The pre-existing `forward(..., antisymmetric=True)` call to the undefined
`antisymmetric_pair_logits` was repaired to call `antisymmetric_logits`, the
ordinary coordinate helper. The trainer continues to call the whitened helper
explicitly. Integer and one-hot CE targets are unchanged.

## Reuse and changed files

- `../CAT/models/transformer.py`: exact extraction of CAT's TransformerBlock
  and sine/cosine position helpers, reusing its existing attention, RoPE, and
  SwiGLU modules.
- `../CAT/models/discriminator.py`: import those extracted components.
- `lib/cat_recognizer.py`: import shared CAT code through a private package
  namespace, avoiding conflicting `models` names; pair pyramid, positional caches, scale-wise backbone,
  shared classification head and optional checkpointing.
- `lib/recognizer.py`: CAT dispatch, optional architecture kwargs, shared CLI
  helpers, and the repaired forward antisymmetry dispatch.
- `lib/config.py`: add CAT to the existing type registry.
- `train.py`, `train_GAT.py`, `train_StyleGAN.py`: expose architecture and pooling
  overrides using the existing recognizer selection mechanism.
- `requirements.txt`: declare timm and einops used by the reused components.
- `tests/test_cat_recognizer.py`: focused architecture/contract/gradient tests.
- `tests/test_bidirectional_sngan.py`: include CAT in the existing tiny frozen
  SNGAN supervision test.
- `docs/CAT_RECOGNIZER.md`: this design and validation record.

## Validation (2026-09-12)

Run with `/opt/anaconda3/envs/manip311/bin/python` outside the sandbox for MPS:

```sh
/opt/anaconda3/envs/manip311/bin/python -m pytest tests/test_cat_recognizer.py tests/test_bidirectional_sngan.py -q -s
```

- 40 focused recognizer tests passed: LeNet/ResNet forward/backward, standard CAT
  construction, configuration, channels 1/3/4, initial pooling, scales 1–4,
  matched/shared-center batches, both swap identities, identical-image zero
  scores, integer/one-hot CE, scale isolation, checkpoint parity, image and
  parameter gradients, second derivatives with explicit attention, and
  resolution-independent state dictionaries, and safe inference-to-training
  reuse of positional caches.
- Default CAT MPS backward passed in FP32, FP16 and BF16, with checkpointed FP32
  also passing. For the FP32 B=1/K=3 test, image gradient norms were approximately
  0.02590 and 0.06843. Observed memory after backward was about 14.6 MB of live
  tensors and 70–71.2 MB of total driver allocation; these are allocator
  snapshots, not a peak-memory profile.
- Six real frozen-SNGAN backward tests passed: LeNet and CAT each paired with
  euler, potential_jet, and midpoint Traversal implementations. No optimizer
  updates or warm-up. CAT classification losses were 1.3554–1.3556, with finite
  nonzero potential parameter gradient norms (linear 1.51–1.56, nonlinear
  0.356–0.361) and no gradients on frozen generator parameters.
- A saved pre-edit CAT discriminator and the factored version had **zero**
  difference on CPU and MPS for logits, auxiliary features, every input gradient
  and every trainable parameter gradient, with strict state-dict loading.
- All three training CLIs passed `--help`; no GAN training was launched.

These checks establish compatibility and gradient flow, not learned metric
quality or final traversal performance. Architecture comparisons remain for
the planned CUDA training experiments.
