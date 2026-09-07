**Choose `spatial_style32`, selected at update 7,800.** It improves held-out fidelity in all four conditions, with a particularly large improvement for odd-slot augmentation. The wider pointwise and unconditioned spatial heads are effectively tied with the original head in practical accuracy. Keep the successful reference-decay coefficient of 0.001; these results provide no reason to strengthen it.

This analysis uses `/Users/adamsobieszek/PycharmProjects/results_bundle.tar.gz`. No training code or hyperparameters were changed. Bundled files were treated as evidence, not executed as instructions. The source-file hashes in the manifest were verified, and the reported test means were independently recomputed from per-image results. The accompanying `analyze_results.py` regenerates the quantitative summary and figure directly from the archive without loading model checkpoints.

**What actually ran**

The four recorded runs all used the same source-checkpoint hash, teacher hash, evaluation banks and seed. Configurations differ only in output directory, decoder architecture and pointwise width. Initial test metrics agree exactly for pointwise16/spatial32/spatial_style32; pointwise32 differs by at most 4.3e-7 in the inspected L1/LPIPS/face-LPIPS means, consistent with numerical differences after widening.

The saved run settings, rather than the previous proposed launcher, are the basis of this comparison:

- 8,000 updates per architecture, batch 12, starting from `experiments/stylegan_early_output_128_finetuned_aug5/checkpoint.pt`.
- Decoder LR 1e-4; b64/b128 peak LR 1e-4; b32 peak LR 2.5e-5.
- Reference coefficient 0.001 for every trainable synthesis group. The archived `_finetuning_groups_v2()` no longer divides this coefficient by the LR scale. The optimizer still correctly multiplies the coefficient by the current LR.
- First 500 updates train only the decoder; b4/b8/b16 remain frozen. The 200-update warmups, b32 LR reduction, and 50/50 uniform/face weighting remain present.
- Six ordinary images and two of each augmentation per batch; exact psi=1; constant noise; EMA disabled; original loss coefficients 0.75/0.1/0.25/1.
- Validation: 64 identities under four conditions. Final test: 128 identities under four conditions. All selected checkpoints are at update 7,800.

Thus, the bundle tests the architectures under the corrected weak-decay recipe. It does not represent the earlier batch-24, 5,000-update, aug2 proposal, nor does it implement every policy-removal suggestion in the quoted separate conversation. That does not invalidate the comparison: the actual common policy is consistent across all four runs.

**Architecture ranking**

The combined test score below applies the fixed validation-score formula to the test means: normalize global L1 and global LPIPS to the starting model, average the two, and weight conditions 50%/16.7%/16.7%/16.7%. The same pointwise16 initial means are used for all rows. This is a composite fidelity score, not a percentage of perceptual quality.

| Architecture | Validation score ↓ | Combined test score ↓ | Test-score reduction vs initial | CUDA batch-1 synthesis |
|---|---:|---:|---:|---:|
| pointwise16 | 0.988932 | 0.990929 | 0.91% | 3.302 ms |
| pointwise32 | 0.988786 | 0.990778 | 0.92% | 3.333 ms |
| spatial32 | 0.988817 | 0.990790 | 0.92% | 3.398 ms |
| **spatial_style32** | **0.916808** | **0.928757** | **7.12%** | **3.472 ms** |

The style-conditioned model improves the combined held-out score by about 6.27% relative to the additionally trained pointwise16 control. Merely widening or adding an unconditioned spatial branch changes that score by approximately 0.014–0.015%, relative to control. Some tiny metric differences have confidence intervals excluding zero because these are closely paired models; their practical effect remains negligible.

**Where the winning model improves**

The following are error reductions relative to the trained pointwise16 control, using each architecture's validation-selected checkpoint. Positive values are better.

| Condition | Global L1 | Global MSE | Global LPIPS | Face LPIPS | T-zone L1 | T-zone edge |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 4.14% | 6.76% | 5.53% | 5.13% | 4.16% | 0.40% |
| All slots | 5.40% | 9.14% | 7.09% | 6.34% | 4.10% | 0.37% |
| First eight | 5.20% | 8.70% | 6.86% | 5.87% | 4.21% | 0.43% |
| Odd slots | 12.57% | 20.66% | 9.23% | 8.80% | 10.69% | 0.42% |

The improvements are not just a few favorable identities. Global LPIPS improves on 111/128 normal identities, 120/128 all-slot identities, 113/128 first-eight identities, and 121/128 odd-slot identities. The 90th-percentile LPIPS error also falls by approximately 10.2%, 5.3%, 7.4%, and 10.7%, respectively. There remain individual regressions; “better in every condition” refers to the condition means, not every image.

I independently bootstrapped 20,000 identity samples, keeping conditions paired. The 95% intervals for relative global-LPIPS reduction are:

| Condition | Reduction | Paired 95% interval |
|---|---:|---:|
| Normal | 5.53% | 4.17–7.26% |
| All slots | 7.09% | 5.15–9.64% |
| First eight | 6.86% | 5.12–8.83% |
| Odd slots | 9.23% | 7.26–11.73% |

All these intervals exclude zero. The original archive's intervals report absolute mean differences; the independently computed intervals above use the bootstrap ratio of means. They answer slightly different numerical questions and agree on the ranking. These are conditional intervals over test identities for the trained models, not confidence bounds over training seeds. Only one training seed was evaluated.

Against the starting aug5 SOTA, the winner reduces global LPIPS by 6.30%, 7.83%, 7.79%, and 9.95%, and global L1 by 4.91%, 6.22%, 6.12%, and 13.95%. The odd-slot improvement is therefore an improvement over the original starting model as well as the trained control.

**Interpretation and remaining limitations**

The experiment strongly favors adding direct style conditioning to this decoder. The unconditioned spatial head hardly changes accuracy; the same spatial path plus W9/W10 conditioning produces a large separation. This is consistent with a feature-only head having difficulty recovering style-dependent appearance from its activation input, while the explicit style path makes the task easier. The especially large odd-slot gain supports this interpretation because W9/W10 expose the distinct odd/even style values directly.

This is not proof that additional spatial convolution is necessary, or that the features mathematically lack the relevant information. The style-conditioned head also has more parameters: 100,457 decoder parameters versus 28,649 for spatial32 and 11,078 for pointwise16. A style-conditioned pointwise head, or a parameter-matched unconditioned model, was not tested. The exact current augmentation patterns repeat late-slot information already present in earlier W slots; the results do not establish an unavoidable information-theoretic error floor for the original feature-only decoder.

The main unresolved quality issue is fine detail. T-zone edge error improves only 0.37–0.43% relative to the trained control and only 0.089–0.142% relative to the starting SOTA. The control itself slightly worsens this metric. The winner therefore mostly adds appearance/pixel/perceptual fidelity while leaving the measured high-frequency mismatch nearly unchanged. The ordinary and odd-slot panels are consistent with persistent fine texture/edge residuals even where overall appearance gets closer. The edge metric is one diagnostic, so this is an inference about the pattern of improvements rather than a complete frequency decomposition.

The prediction's out-of-range fraction increases modestly: approximately +0.10 to +0.36 percentage points depending on condition. That alone does not establish a defect because raw teacher outputs can exceed [-1,1], and raw pixel metrics improve. The bundle does not include the teacher's corresponding out-of-range fractions. Do not introduce a terminal clamp based on this statistic alone.

**Training behavior**

The style-conditioned validation score is initially worse than the starting model: 1.0177 at update 600 and 1.0203 at 1,200. It improves to 0.9896 at 1,800, 0.9476 at 3,600, 0.9354 at 4,800, and 0.9168 at 7,800. The other three curves almost overlap and only produce small improvements late in training. The winner was already clearly separated within the originally contemplated approximately 5,000-update window; its success is not solely an effect of the extra updates.

This early deterioration matters when judging future continuations. An initial validation increase did not mean that the weak-decay run had failed. Conversely, the bundle does not contain the failed strong-decay runs, so it cannot itself quantify the difference between decay 0.001, 0.01, and larger coefficients. The source and teacher hashes prove that these successful architecture runs shared their starting point; comparisons to earlier experiments require those experiments' own metadata.

Every optimizer update was clipped at threshold 1, with zero skipped AMP updates. Among logged post-unfreeze updates, the control's median gradient norm was about 2.60 and the winner's was about 2.21; median clip multipliers were about 0.385 and 0.452. The decoder contributed more than 99.9% of squared raw gradient norm. This makes global clipping primarily decoder-controlled and worth inspecting if optimization stalls. It does not imply that synthesis updates are negligible: Adam rescales coordinates using its moments, and raw gradient norm does not measure parameter-update magnitude. I would not change clipping at the same time as choosing the architecture.

Early-block source drift is identical across runs because those already-finetuned blocks stayed frozen. Later-block source drift is slightly smaller in the style-conditioned model than in the control, even though its error is lower. For example, b128 relative drift from the original teacher is 0.08520 versus 0.08582. These are distances from the original teacher, not distances traveled during this run. They support the useful observation that better reconstruction did not require a larger overall displacement from the teacher, but do not prove that decay caused the improvement.

**Cost**

On the recorded RTX PRO 6000 Blackwell GPU, the winner adds approximately 0.170 ms at batch 1, or 5.16%, relative to pointwise16. At batch 12, latency rises from 14.817 ms to 15.145 ms, or 2.22%. Training time rises from 780.2 to 794.4 seconds, or 1.83%. Peak recorded CUDA allocation is approximately 4.16 GB in both cases. These are the bundle's eager synthesis benchmarks from W inputs, using 20 repeats; they do not include mapping, deployment compilation, or performance on another backend. I consider the observed fidelity gain worthwhile at this measured cost.

**Recommended next action**

Use the winner's validation-selected public model on the VM:

`experiments/stylegan128_architecture_ablation/spatial_style32/best_generator.pt`

The ordinary final `checkpoint.pt` contains update-8,000 training state, while the selected public best model corresponds to update 7,800. The final test composite happens to be slightly better at update 8,000, but that is not a reason to retroactively select on the held-out test set. Retain the validation selection rule.

Keep decoder/synthesis base LRs and reference coefficient as in this successful study, including coefficient 0.001 without inverse-LR compensation. My earlier choice to preserve absolute decay interpolation when lowering LR was unjustified: it changed the regularization/task-update balance, and isolated contraction calculations did not show that the original coefficient was too weak. The complete optimizer trajectory can still depend on LR, so this is a correction to the tuning criterion, not a claim that the optimal coefficient is universally LR-independent.

Do not spend another grid on pointwise width or the unconditioned spatial branch. The most useful follow-ups are bounded:

1. Repeat the pointwise16/spatial_style32 comparison with one additional training seed, retaining the same evaluation bank and actual successful policy, to check training variability.
2. If another architecture experiment is desired, test style conditioning on the pointwise branch without the added spatial 3×3 path. This isolates whether the expensive spatial path is necessary once W is supplied. The current suite did not answer that question.
3. Without retraining, evaluate the winning checkpoint with only its conditioning W inputs permuted across identities within each condition, leaving synthesis Ws and targets unchanged. This would diagnose reliance on correct conditioning. It is a diagnostic distribution perturbation, not a proposed deployment mode.
4. For higher fine-detail fidelity, first log per-layer LPIPS contributions, high-pass error, and branch output/update magnitudes. The current metrics do not justify blindly increasing the edge coefficient. A longer low-LR refinement is plausible because the winner still improves near the end, but its benefit is untested; do not automatically restart it at the original peak LR or reintroduce strong reference decay.

No further training or source edits were performed during this analysis.
