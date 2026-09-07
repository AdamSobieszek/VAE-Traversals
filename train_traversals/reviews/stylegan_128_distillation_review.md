**Historical review:** the preferred changes have since been implemented. The launcher now runs four architectural ablations from the specified SOTA checkpoint; its old diagnostic profiles described below have been replaced. See [current usage](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/reviews/stylegan128_ablation_usage.md).

**The highest-value changes are trustworthy validation, numerically stable perceptual weighting, a measurable source-weight anchor, and more nonlinear spatial capacity in the decoder.** I would address those before searching dozens of loss-weight combinations. The current pairing and optimizer design are largely sound; several important defaults and measurement choices make it difficult to identify what is actually helping.

This is an implementation review and experiment plan, dated 2026-09-06. Training code and pretrained weights have not been modified. Numerical probes used `/opt/anaconda3/envs/manip311/bin/python`. Your supplied run used a large CUDA GPU, and all batch-12 training recommendations below target that hardware. Local image-level checks should use MPS and batch 1, not reproduce the CUDA workload. The available Python process reports neither CUDA nor MPS available, so I ran only small CPU numerical probes and checkpoint comparisons; there are no GPU timings, training comparisons, or demonstrated improvements in image quality. The specific best-run metrics and checkpoints described in the request were not available in an experiments directory. I inspected the bundled teacher and early-output checkpoint instead. Recommendations below are hypotheses with explicit tests, not claims of established optimal hyperparameters.

**1. What the pipeline actually does**

The teacher maps Z into 18 W slots, runs the full 1024 synthesis, and area-downsamples its raw RGB output to 128. In finetuning, the student receives the leading 12 slots, bypasses its frozen mapping, runs b4–b128 with all toRGB branches skipped, and decodes the final 256-channel activation. Because the b128 toRGB is skipped, only slots 0–10 affect student output; slot 11 is accepted but unused. The teacher still uses slots 11–17.

The decoder is a linear 256→3 spatial 3×3 convolution plus a nonlinear 256→16→3 pointwise SiLU branch. It has spatial processing and nonlinear processing, but the nonlinear branch cannot combine neighboring feature locations. The synthesis prefix itself still provides a large receptive field; this is a limitation of the head, not a claim that the entire model sees only 3×3 pixels.

In the supplied command, 50% of samples are ordinary and each augmentation gets approximately 16.7%. These probabilities come from the implicit `w_augment_percent=0.5` default. At 10,000 steps and batch 12, that is about 60,000 ordinary examples and 20,000 per augmentation. `crops_per_sample=1` produces fresh pairs every step.

For the bundled checkpoints, I verified that student mapping weights match the teacher, all copied synthesis-prefix parameters match the teacher exactly, all prefix constant-noise buffers match, and the actual head width is 16. This bundled student is an initial converted decoder model, not evidence about the later finetuned winner.

**2. Confirmed findings, ordered by practical importance**

| Finding | Evidence in the current code | Consequence and proposed change |
|---|---|---|
| Validation is only five ordinary examples in the supplied run | `StyleGAN_SkipNetwork128.py:1738` sets validation count from `plot_count`; `make_validation_pairs()` never augments | Best-checkpoint selection cannot establish performance on any augmentation and has high sampling variance. Separate validation count from plot count and validate all four distributions. |
| Source decay is extremely weak at the supplied scale | `AdamWWithReferenceDecay.step()`, line 675, uses `scheduled_lr * coefficient` | Preserve its decoupled semantics but calibrate the coefficient using cumulative shrinkage and measured update/drift norms. |
| Tiny saliency products can lose precision under AMP | `image_reconstruction_loss()`, line 856, calls LPIPS inside autocast; `stylegan_feature_codec.py:617` casts weights to the map dtype before multiplication | Explicitly compute weighted products and reductions in FP32; first validate full FP32 LPIPS, then optimize its trunk separately. |
| Warmup does not protect the synthesis network | `_finetuning_parameter_groups()`, line 1303, assigns zero warmup to the synthesis and decoder-linear groups | On step 1, prefix LR is nearly 1e-4 while the residual-head LR is only 5e-7. Add explicit warmup for every trainable group or a head-only adaptation stage. |
| Training truncation differs from the requested value | `map_z()`, line 319, multiplies psi by `0.95*(1+0.1*N(0,1))` during training | `--truncation-psi=1` actually trains with batch-shared psi having mean 0.95 and standard deviation 0.095. Make jitter explicit; start the main comparison with exact psi=1. |
| Head width is ignored during finetuning | `train()`, line 1619 onward, loads the checkpoint architecture | `--hidden-channels=32` alone cannot enlarge this student. Implement explicit checkpoint migration or reject conflicting width arguments. |
| EMA is neither disabled nor independently assessed | `ModelEMA`, line 631; validation uses EMA, plots use live weights | Decay 0.5 averages recent updates and retains the whole copy/update cost. Add a true no-EMA path and make selection, plots, and export use the same selected state. |
| Augmentation plots change both the examples and training RNG | `save_w_augmentation_grid()`, line 1023, fixes base identities but samples displacements from the live generator | Comparisons across steps are confounded; changing plot cadence changes future training samples. Cache fixed augmented Ws and targets with private RNGs. |
| Reported PSNR is saliency-weighted | `evaluate()`, line 1102, derives it from weighted MSE | Keep it as a named weighted metric; also report conventional global RGB MSE/PSNR. |
| Changing saliency mass may silently do nothing | `face_saliency_mask()`, line 561 in `stylegan_feature_codec.py`, omits `inner_mass` from the cache key | Fix the key before exposing/tuning this value. A probe requesting 0.5 then 0.8 returned identical masks. |

Source links: [trainer](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/models/StyleGAN_SkipNetwork128.py), [integrated student](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/models/StyleGAN2_mps/early_output_model.py), [StyleGAN blocks](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/models/StyleGAN2_mps/model.py:370), [saliency and reduction](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/models/stylegan_feature_codec.py:561).

**3. What I would retain**

Keep the frozen teacher, frozen matching mapping, constant matched noise, raw area-downsampled 128 targets, and native [-1,1] pixel-space convention. Downsampling the 1024 teacher is the correct supervision for the stated target; do not substitute the teacher's own b128 toRGB output. Retain pixel supervision alongside LPIPS and a moderate detail objective. The existing teacher-reference lookup aliases already-loaded teacher tensors, avoiding another full source-weight copy, and reference decay skips parameters with no gradient. These are good choices.

Your full-frame crop settings are efficient: `crop_size=128` does not call LPIPS twice or run an extra teacher pass on `fullres_every` steps. That flag only selects the spatial extent of a single loss pass; the teacher already runs to 1024 for every fresh pair. `fullres_every=0` simply makes the intent clearer. Keep `crops_per_sample=1`: increasing it to four at this crop size repeats optimization on the same complete images, rather than supplying four different crops.

Saliency already affects L1, MSE, fine-detail loss, and spatial VGG LPIPS. Adding it to more of these losses would duplicate existing behavior. VGG is a reasonable perceptual training network; the [official LPIPS repository](https://github.com/richzhang/PerceptualSimilarity) explicitly distinguishes its usefulness for backpropagation from AlexNet's use as a forward metric.

**4. Source-weight decay: preserve the implementation idea, change how it is calibrated**

The implemented recurrence is, schematically:

`theta[t+1] - theta0 = (1 - lr[t]*lambda)*(theta[t] - theta0) - lr[t]*AdamDirection[t]`.

The reference pull is outside the Adam moments, applied before the task update, and ordinary synthesis weight decay is zero. This is a coherent AdamW-style source anchor. It is not equivalent to adding an L2-SP penalty to an Adam loss: that penalty would enter the adaptive moments. The distinction follows [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101); the rationale for retaining a pretrained reference is supported by [L2-SP](https://proceedings.mlr.press/v80/li18a.html). Neither paper determines the right coefficient for this generator.

For the exact 10,000-step synthesis schedule, `sum(lr[t]) = 0.549955`. In exact arithmetic, an offset present at the beginning, with no subsequent task updates, is multiplied by `product(1-lr[t]*lambda)`:

| Coefficient at synthesis peak LR 1e-4 | Peak per-step pull | Offset removed over 10k steps |
|---|---:|---:|
| 0.01, current run | 1e-6 | 0.548% |
| 0.1 | 1e-5 | 5.351% |
| 1 | 1e-4 | 42.304% |
| 2 | 2e-4 | 66.713% |
| 5 | 5e-4 | 93.609% |

These are schedule diagnostics, not predicted final drift: task updates create offsets throughout training, and late offsets receive much less cumulative pull. At initialization equal to the teacher, the reference term is zero. Floating-point resolution makes the tiny current pull even less effective: an actual optimizer probe with FP32 `theta=1.01`, reference 1.0, LR 1e-4, coefficient 0.01, and a zero gradient produced no representable parameter change.

Start with an isolated 0.01→1.0 coefficient ablation at the existing LR. Do not jump straight to 5 at 1e-4 as the default. For a more conservative prefix LR of 2e-5, coefficient 5 gives the same peak pull of 1e-4 and the same contraction over an otherwise identical schedule as coefficient 1 at 1e-4. The task updates are still different. Changing run length or warmup changes cumulative anchoring; print the resulting contraction rather than reusing coefficients blindly.

The implementation upgrade should:

1. Retain the original full teacher as the reference, even when starting from an already-finetuned student. Log drift from both the original teacher and the starting student to distinguish retention from adaptation.
2. Group reference-decayed tensors by device/dtype, cache the parameter/reference lists, and use a tested foreach interpolation path where supported. Preserve missing-gradient and skipped-AMP-step behavior, with the scalar implementation as fallback. Avoid allocating a full `theta-reference` tensor per parameter every step.
3. Log per-block relative drift, absolute RMS drift for near-zero reference tensors such as biases/noise strengths, and sampled reference-pull/task-update ratios every 200 steps. Raw parameter drift alone is insufficient because StyleGAN's demodulation creates scale invariances.
4. Use low or zero task LRs for early identity/geometry blocks rather than relying on very large decay to undo destructive updates. Keep a no-reference-decay control. If desired drift still cannot be controlled without harming fidelity, test a small early-block activation penalty as a separate ablation, not simultaneously with stronger decay.
5. Retain source checkpoint hashes and optimizer-decay semantics in exports; do not silently reinterpret older per-step coefficients as LR-scaled ones.

**5. Precision and loss reductions**

The mask is normalized to sum 1 across 16,384 pixels. Measured mask values range from approximately 9.9e-6 to 4.0e-4. In a controlled FP16 reduction probe using the real mask, a constant perceptual map of approximately 0.001 yielded 0.001033, while about 35.7% of individual products rounded to zero. At approximately 0.0001, the result was 0.0000572 and 94.1% of products were zero. These are CPU half-arithmetic probes of the vulnerable operation, not measurements of real CUDA training loss. CUDA reduction promotion does not recover precision already lost in a half-precision product. A rounded-zero forward product does not itself prove a zero gradient: multiplication's derivative can remain nonzero, especially with GradScaler. Treat this as a confirmed loss-value precision issue and test gradients before attributing an accuracy plateau to it.

Make prediction/target differences, squared errors, masks, focal maps, and their products/reductions FP32 explicitly. Initially run LPIPS entirely in a disabled-autocast region with FP32 inputs. That gives a reliable reference. If profiling justifies it, allow its convolutions under autocast while converting features to FP32 for normalization/differences and calibrated maps to FP32 before weighting. Validate values and prediction gradients against full FP32 on ordinary and all three augmented groups, including nearly identical images. PyTorch documents [nested disabled-autocast regions](https://docs.pytorch.org/docs/2.14/amp.html) for controlling precision.

Normalize and reduce each image independently, then average samples. The current batch-global denominator with sample-dependent focal weights also changes relative sample influence; an image with greater total focal mass contributes more. Keep the intended 50/16.7/16.7/16.7 distribution explicit rather than letting focal normalization alter it invisibly.

Keep the existing fine-detail FP32 protection. Avoid hard-clamping the whole student output or adding a terminal tanh: raw StyleGAN RGB can exceed [-1,1], especially under augmentation. The current LPIPS clamp has zero gradient outside the range, but L1/MSE still pull those pixels toward the target, so it is not a complete training dead zone. Log out-of-range fractions by group before changing this contract. Any alternative to clipping should be a separately validated perceptual-loss choice.

Omitting `--amp` currently does not force the synthesis network into FP32: its b128 block selects native FP16 on CUDA independently. It also disables GradScaler. Add explicit synthesis/loss precision settings rather than using the AMP switch as a full-FP32 diagnostic.

**6. A concrete loss recipe and how to tune it**

Do not infer dominance from scalar coefficients alone. At an absolute pixel residual of 0.05, the current MSE derivative magnitude is `2*0.1*0.05=0.01` versus `0.75` for L1, before common spatial weighting. MSE mainly adds pressure on larger errors; it is unlikely to be the principal detail driver. LPIPS and focal-gradient scales depend on their feature and filtering operations.

First keep the current coefficients after fixing numerics and validation. Then test this candidate, with loss definitions held fixed:

`L = 1.0*L1 + 0.1*MSE + 0.15*detail + 0.75*LPIPS`.

It modestly strengthens exact pixel fidelity and reduces overlapping perceptual/focal pressure. This is an informed starting hypothesis, not an analytically optimal setting. Compare it with the existing `(0.75, 0.1, 0.25, 1.0)` recipe before introducing extra losses. A candidate that looks smoother but erases teacher eyelashes fails the detail criteria even if its global MSE improves.

For saliency, retain the current spatial layout initially but mix it with a uniform distribution:

`pixel_mask = 0.5*uniform + 0.5*current_face_mask`;

`detail_mask = 0.25*uniform + 0.75*current_face_mask`;

`LPIPS = 0.5*global_LPIPS + 0.5*face_weighted_LPIPS`.

The existing T-zone occupies about 9.62% of pixels and gets approximately 50% of mask mass. The proposed pixel mixture still gives it about 29.8% of mass, over three times its area share, while making errors in hair, silhouette, and background more visible to training. Test the global/face mixture independently of loss coefficients. Preserve an unweighted floor for strongly augmented faces whose features move away from the fixed FFHQ coordinates.

For LPIPS, start with the existing spatial-map implementation plus FP32 weighting. As a later optimization, obtain calibrated per-layer distance maps and apply a separately normalized resized mask at each feature resolution before summing. This avoids upsampling five maps to 128 and allows shallower layers to retain more facial emphasis while deeper layers remain predominantly global. It changes the objective slightly, so compare against the original spatial implementation; do not call it a bitwise-equivalent optimization. The [LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py) provides the underlying per-layer computation.

I would not make tiny eye/mouth LPIPS crops the default. Full-frame VGG already includes shallow detail features; small crops remove context, and deeper receptive fields can be dominated by padding. If fixed-mask alignment is visibly poor, first test a detached teacher-derived landmark mask with a fixed-mask fallback. If a face crop is still useful, use one shared teacher-defined crop of roughly 80–96 native pixels, resized identically to 128, and test a small additional weight such as 0.1. Upsampling changes feature scale; it does not create missing detail. Account for its additional perceptual pass.

Measure weighted loss gradient RMS with respect to the prediction and selected decoder/prefix parameters every 200–400 steps on a fixed diagnostic batch. Record cosine similarities between major objectives and per-group gradient norms. Use these measurements to investigate conflicts or vanishing terms; do not automatically force every objective to have equal gradients. Keep global clipping at 1 initially, log its pre-clip norm and clipping frequency, then test separate group clipping only if one group consistently suppresses others.

The current detail loss emphasizes target texture and remaining error, which is reasonable. Its high-pass-plus-Scharr filtering deemphasizes broader edge placement. If face contours or eyelid locations remain wrong, test a plain first-derivative Charbonnier term mixed into the existing detail term, rather than merely increasing the focal coefficient. Cache fixed filters, skip detail computation entirely when its weight is zero, and keep the detached focal floor/cap until an ablation shows a benefit from changing them.

**7. Preserve the three augmentation tasks, but make their sampling deliberate**

Current augmentation is `w + mapped_w2` on all slots, slots 0–7, or odd zero-based slots. A mapped W vector is not a zero-mean displacement: the augmented slots have expected mean near `2*w_avg`, and their covariance increases under independent draws. This is not standard replacement-style mixing. It may be exactly the intervention distribution wanted for your traversals, so I would retain it in the principal training and test sets. Replacing it wholesale with `w2-w_avg` could improve natural-looking images while failing the requested augmentation behavior.

At batch 12, use six ordinary examples and two examples of each augmentation, randomly shuffled. Keep base/displacement draws independent across samples. This preserves the existing expected objective while preventing a group from disappearing from a batch. If all four groups deserve equal importance, test 3/3/3/3 separately and make the evaluation weighting match that choice. Do not silently trade ordinary fidelity for augmentation improvements.

Make truncation jitter a separate, logged parameter. Main comparison: exact psi=1 for both ordinary and augmented base Ws; untruncated mapped displacements as now. Keep the original jitter as an ablation. If robustness across augmentation magnitude is needed, retain substantial mass at the exact current strength 1.0 while adding intermediate strengths; validate strength 1.0 explicitly. Do not introduce a strength curriculum and a new displacement definition together.

An important limit: arbitrary independently edited late W+ slots are impossible for the current feature-only student to reproduce exactly when they do not affect its prefix. However, that impossibility does **not** establish a floor for these three particular augmentations. For ordinary mapped bases and mapped displacements, every slot 11–17 is recoverable from slot 9 for odd indices and slot 10 for even indices, for all three masks. I verified this relation directly. Exposing those W values to the head may still simplify learning; it supplies a shortcut to information encoded indirectly in features, not necessarily new information in this restricted distribution.

**8. Architecture changes with a practical migration path**

My first architecture ablation is a wider pointwise branch, 16→32, with the existing linear branch unchanged. Copy the first 16 hidden channels and their output weights, initialize new input channels normally, and zero only their output weights. This preserves the old prediction at migration while allowing the new channels to learn. Do not zero both ends of a new branch.

My preferred larger candidate retains the complete old decoder and adds a separate nonlinear spatial correction:

`residual = Conv1x1(32→3)(SiLU(Conv3x3(32→32)(SiLU(Conv1x1(256→32)(features)))))`.

Zero-initialize the final projection only; add this residual to the old prediction. That makes checkpoint migration function-preserving and gives the head a learned nonlinear combination of neighboring features. It avoids forcing all new local detail processing into the pretrained synthesis weights. Do not simultaneously widen the old pointwise branch in the first spatial-branch test, so the architectural comparison stays interpretable.

At 128×128, the present width-16 decoder uses approximately 0.181 GMAC/image, excluding activations/bias operations. Widening only to 32 gives approximately 0.249 GMAC. The additional width-32 spatial branch costs approximately 0.287 GMAC, making the combined candidate approximately 0.468 GMAC. These are operation counts, not measured latency. Measure end-to-end batch-1 and batch-12 latency on the deployment backend; choose by fidelity at the actual speed budget. Test width 64 only if width 32 helps enough to justify further cost.

Next, separately test a small style-conditioned modulation of the new hidden branch. Concatenate W slots 9 and 10, normalize using fixed teacher-distribution statistics, and use a small MLP to predict hidden-channel scale and bias. Initialize modulation around identity. If the intended inference contract later permits arbitrary W+, expose the full teacher-layout W tensor and explicitly condition on late slots instead. Version the generator configuration and converter, update both training/deployment decoder definitions, and preserve loading of old v2 checkpoints. Do not silently fabricate missing styles for arbitrary W+ inputs.

Restoring an RGB skip path is another useful, lower-priority ablation. The current folded average-W b128 toRGB contains neither sample-specific modulation nor the earlier RGB skip sum. Recomputing the real prefix RGB gives a strong low-frequency/color representation, but reinstating it adds operations and changes the learned decomposition. For migration, add it through an initially zero-influence correction or train a new residual-on-prefix-RGB head; do not directly add the full skip image to an already-corrected decoder and double-count its reconstruction. Retained toRGB weights may also be mismatched to already-finetuned prefix activations, so assess that branch empirically.

**9. Recommended staged training after the fixes exist**

Use the best available compatible student as the starting point for a production refinement, but start every ablation from the exact same checkpoint hash. Keep the original teacher as the reference. All examples remain full 128 frames, batch 12, constant noise, six ordinary plus two per augmentation, and explicit psi=1.

| Stage | Successful optimizer updates | Trainable groups and peak LRs | Source decay and schedule |
|---|---:|---|---|
| Fit migrated head | 1,000 | Old head 1e-4; new spatial branch 2e-4; prefix frozen | 200-step head warmup; no source updates |
| Adapt later synthesis | 9,000 | Decoder 1e-4; b64/b128 2e-5; b32 5e-6; b4–b16 frozen | 200-step warmup for newly unfrozen groups; reference coefficients 5 for b64/b128 and 20 for b32 as initial matched-pull candidates; cosine to 0.1× peak |
| Polish if held-out groups still improve | Up to 10,000 | Decoder 2e-5; b64/b128 5e-6; b32 and earlier frozen initially | Continue with measured source decay, initially coefficient 5; stop on held-out saturation |

The stronger b32 coefficient is a testable retention prior, not a known optimum. Include weaker-anchor and all-prefix-unfrozen controls. A frozen early block cannot forget; this is more reliable than making it learn aggressively and hoping decay cancels the change. Conversely, if early layers need to adapt for these interventions, persistent group-specific validation error is a reason to unfreeze b16/b32 at a small LR. Do not reset frozen layers to original weights when starting from an already-finetuned student.

Begin with the original loss coefficients during head migration; evaluate the candidate loss recipe and saliency mixtures separately before adopting them for polishing. Default to live-weight evaluation/export and no EMA allocation. An EMA experiment can be revisited with the improved benchmark, but it is not a prerequisite for this plan.

When unfreezing, create/enable the intended optimizer groups with fresh moments for newly trainable tensors and explicit stage schedules. Do not reuse a global step index in a cosine schedule intended to restart locally without defining the behavior. Log attempted steps and successful updates separately when GradScaler skips a step; advance update-based schedules and EMA only on successful updates.

**10. Validation and checkpoint selection needed to judge these changes**

Create a deterministic validation bank with 128 independent identities, each evaluated normally and under each of the three fixed augmentation patterns: 512 image pairs. Use a separate fixed mapped displacement per identity, shared across its three patterns to make the intervention comparison meaningful. Bank sampling and training use independent RNG streams. Keep a final untouched test bank of 512 identities × four conditions, and group bootstrap confidence intervals by base identity. Smaller 32-per-group validation can serve frequent checks; evaluate the full bank every 800 updates and the final test bank only for finalists.

Report per condition: global L1/MSE/PSNR, global VGG LPIPS, face-weighted LPIPS, T-zone pixel/detail error, non-T-zone error, and 90th-percentile per-image error. Ordinary and augmented teacher targets must use identical noise, downsampling, and precision contracts. Save fixed PNG panels with live/selected student state and a common absolute-error display scale; JPEG compression can obscure the 128-pixel details being inspected.

Use a fixed evaluation score independent of the current training coefficients. One concrete selection rule is to normalize global LPIPS and global L1 to the starting checkpoint's per-group means, average those two ratios within each group, then combine groups with weights 0.5/1⁄6/1⁄6/1⁄6. Save the best aggregate and the best per-group checkpoints, plus raw constituent metrics. Do not accept an aggregate winner if an individual group has a statistically clear regression on identity/detail. Define an initial practical target of at least 5% lower face LPIPS and T-zone error, with no material global fidelity regression, as an experiment acceptance criterion rather than a promised gain.

Do not compare the existing logged `loss` across runs with different coefficients or masks: a smaller weighted sum may just reflect different weights. Likewise, do not call the current weighted PSNR conventional PSNR. Keep the old five-image grid for visual continuity, independent of validation sample count.

**11. Efficiency work worth doing, and its limits**

Profile teacher mapping/synthesis, student forward/backward, VGG, detail filtering, optimizer/reference decay, and validation separately on the actual GPU. Full 1024 target synthesis is a likely major cost, but relative cost and fused-modconv performance must be measured. The local StyleGAN block defaults to fused modulation even during training; benchmark fused versus nonfused at batch 12 for both peak memory and throughput. Do not blindly assume one is faster.

Use a deterministic CPU/disk cache of W inputs and 128 teacher targets when repeating ablations. Store compact mapped base/displacement vectors, pattern metadata, noise/teacher hashes, and raw FP16 or FP32 targets; avoid JPEG or uint8 training targets. Raw FP16 target storage is 96 KiB/image, about 11 GiB for 120,000 targets, plus latent metadata. It is a candidate storage format whose quantization must first be checked against the FP32 target. Cache b128 features only for decoder-only training, never as a replacement for a trainable prefix forward pass.

A shuffled replay pool with partial refresh can amortize teacher cost without repeating identical batches consecutively, but it trades fresh target diversity for speed. Record unique examples and teacher calls, and compare quality at equal teacher-generation and wall-clock budgets. Precomputing targets moves cost; it does not eliminate it. Caching LPIPS features for a huge dataset can exceed target-storage cost substantially; restrict it to validation or a bounded reuse pool.

Other low-risk work: batch validation instead of evaluating one image at a time; remove the unused EMA copy; cache augmentation masks and avoid per-batch device-to-host `bool`/`nonzero` checks where possible; prebuild fixed detail kernels; skip zero-weight loss branches; keep teacher tensors detached. Do not add the redundant `no_grad()` claim that teacher VGG features currently retain a graph: with frozen VGG and non-gradient targets, ordinary autograd already avoids that graph.

If later stages freeze b4–b32, run that prefix under no-grad and train the suffix normally. This can reduce activation-memory cost. Do not cache outputs of changing blocks. Compile and memory-format experiments come after correctness, with matching gradients and fixed-noise outputs checked for all four distributions.

**12. Resume/export and smaller correctness fixes**

`--reset-steps` has no effect in the supplied command because it has no `--resume`. Without an explicit early-output checkpoint path, each new process starts from the bundled student, not from the previous experiment's best output. The bundled file inspected here has an untouched teacher prefix.

When resuming with reset steps, `best_loss` is still copied from the old checkpoint. Reset or recompute it under the new evaluation definition, especially when weights or masks change. When resuming without resetting, optimizer `base_lr` and warmup fields can come from the saved state and override new CLI intentions. Distinguish exact continuation from weights-only finetuning, print resolved group settings, and save/restore GradScaler and CPU/CUDA/NumPy RNG state for real continuation. If batch reuse remains supported, also preserve its cached examples/refresh position or document a non-exact restart.

Fail on missing learned teacher tensors rather than merely printing a count and continuing with randomly initialized weights. Ignore only known-compatible duplicated entries or reconstructible buffers, and verify noise-buffer compatibility. The inspected bundled checkpoint passed mapping/prefix/noise comparisons, so this is hardening rather than an observed corruption.

Fix the misleading baseline CLI: the docstring advertises `--train-baseline`, which does not exist, and `--frozen-torgb` has help text describing the opposite action. CLI and dataclass defaults also differ in several places. Keep one source of truth and report resolved configuration plus actual loaded architecture. Those issues mostly affect decoder-only runs, but can invalidate architecture experiments.

**13. Runnable experiments on the unchanged trainer**

The accompanying [launcher](/Users/adamsobieszek/PycharmProjects/VAE-Traversals/train_traversals/scripts/StyleGAN128_review_ablation.sh) uses only existing flags. It offers four profiles:

| Profile | Decoder LR | Synthesis LR | Reference coefficient | Loss weights L1/MSE/detail/LPIPS |
|---|---:|---:|---:|---|
| `control` | 1e-4 | 1e-4 | 0.01 | 0.75 / 0.1 / 0.25 / 1.0 |
| `anchor` | 1e-4 | 1e-4 | 1 | Same as control |
| `conservative` | 2e-4 | 2e-5 | 5 | Same as control |
| `loss` | 1e-4 | 1e-4 | 0.01 | 1.0 / 0.1 / 0.15 / 0.75 |

Run `control` and `anchor` first; they isolate the reference coefficient. `conservative` is a combined allocation-of-learning experiment and cannot isolate head LR from synthesis LR. `loss` isolates the proposed coefficient recipe relative to control. These are diagnostic profiles, not the fully improved implementation described above.

Common settings retain batch 12, 10k steps, constant noise, full 128 images, fresh pairs, saliency, and AMP. EMA decay is zero so validation matches current parameters, but the unchanged implementation still allocates and copies the EMA model. Plotting is disabled to eliminate its random-stream side effect; `plot_count=128` supplies 128 ordinary validation identities as a temporary workaround. It still does not validate augmentations, fix FP16 saliency, remove truncation jitter, or warm up synthesis. Results cannot establish a four-distribution winner until the validation patch exists.

From the repository root:

```bash
bash train_traversals/scripts/StyleGAN128_review_ablation.sh --dry-run anchor
bash train_traversals/scripts/StyleGAN128_review_ablation.sh control
bash train_traversals/scripts/StyleGAN128_review_ablation.sh anchor
```

To start from your current best public integrated checkpoint, pass its path as the second positional argument, identically for every profile. The script defaults to the bundled checkpoint only because that verified file exists locally. It refuses to reuse an output directory and requires CUDA for actual training, so these batch-12 profiles cannot silently fall back to a local CPU/MPS run. Dry runs work locally. `PYTHON_BIN` may select a different interpreter in a remote VM; locally it defaults to the required manip311 environment. No training job was launched in this review.

**14. Implementation order and stopping rules**

1. Implement deterministic four-group validation, live-state best export, explicit truncation settings, correct resume metadata, FP32 weighted reductions, and the saliency-cache fix. Verify loss/gradient numerics and optimizer references. Reproduce the current recipe under the fixed measurement system.
2. Run the isolated source-anchor and loss-coefficient comparisons. Log per-group errors, drift, clipping, and weighted gradient scales. Keep weaker anchoring if stronger anchoring harms the target rather than assuming retention is always beneficial.
3. Add balanced augmentation batches and test global/face loss mixtures separately. Choose the mixture using all four validation groups.
4. Add function-preserving head migration; compare width-32 pointwise and width-32 additional spatial branch. Fit each with a frozen prefix first, then use the staged suffix finetuning schedule. Benchmark inference latency and include it in selection.
5. Test style conditioning or the RGB skip branch only if the preceding results expose persistent style/color errors. Select one incremental architecture at a time.
6. Extend the strongest candidate while validation continues improving, then evaluate finalists on the untouched test bank and at least one additional training seed. A longer run is justified by an improving curve, not by an arbitrary larger step count.

Necessary checks for implementation are bounded: reference-update equivalence with/without gradients and an AMP-skipped step; gradient-finite and FP32-versus-mixed-loss comparisons; all four mask patterns and fixed RNG behavior; function-preserving head migration; exact constant-noise export/reload predictions; and one resumed-step comparison for the supported continuation mode. These target real failure modes. They are distinct from claiming that passing unit tests proves better distillation.
