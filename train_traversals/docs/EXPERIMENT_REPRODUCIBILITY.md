# Experiment records

`train.py`, `train_StyleGAN.py`, and `train_GAT.py` automatically produce:

```text
experiments/wip/<experiment>/
  lib/                       # source files copied at experiment creation
  train*.py                  # the invoked entry point, when available
  config.yaml                # first launch's resolved configuration
  runs/<UTC timestamp>/      # each subsequent trainer launch
    lib/
    train*.py
    config.yaml
```

The `lib` snapshot includes recognizers, PDEs, losses, trainer, utilities and
configuration files. Files are copied without modification; Python bytecode
caches are excluded. Existing snapshots and YAML files are never overwritten.
The existing `args.json` and `command.sh` continue to describe the latest launch.
For an older experiment without a manifest, the first new record describes the
current launch; it cannot recover the code or configuration of earlier training.

The YAML is written after checkpoint restoration and optimizer/scheduler resets,
before the training loop. It contains:

- All parsed arguments (including parser defaults) and the invocation.
- Resolved trainer settings, starting steps, precision and validation settings.
- Model and traversal classes, public settings, nested layer configuration,
  parameter counts, devices and dtypes; PDE configuration and active loss weights.
- Optimizer defaults and **actual parameter-group settings**, including learning
  rates, weight decay, betas, epsilon, and the parameter names in each group.
- Scheduler state, including warmup/total steps, decay settings and current position.
- Python/PyTorch/platform versions, initial Torch seed and determinism settings.

Parameter-group values are authoritative when they differ from optimizer defaults.
On resume, learning rates and scheduler position are the restored/reset values.

For other trainers, call `lib.aux.save_experiment_config(directory, args,
models={...}, traversals={...}, optimizers={...}, schedulers={...}, training={...})`
with named mappings of instantiated objects after initialization and restoration.
The optional `training` mapping records resolved settings held in local variables.
The function returns the saved YAML path.

This is a source/configuration record, not a standalone or bitwise replay bundle.
Weights and optimizer moments remain in existing checkpoints. It does not snapshot
external dependencies, generator code outside `lib`, pretrained weights, datasets,
or full RNG states. Arbitrary objects become class identifiers, tensors become
shape/dtype metadata (scalar non-parameter tensors become values), and private
attributes are omitted except configuration dictionaries ending in `_cfg`.
Constants only used inside functions remain visible in the copied source.

## VP pair generation

Run `python lib/val_utils.py --exp /path/to/experiment` with the same flags and
values accepted by `gen_pairs.py`. The standalone script retains its own implementation for
comparison. Both write `vp_pairs/pair_000000.jpg`, subsequent numbered side-by-side
JPEGs, and one-hot labels in `vp_pairs/labels.npy`.

The new pipeline uses `load_pair_experiment`, `generate_pairs`, and
`save_pair_images`. Its `unroll_traversal` iterator is also used by validation.
Pair generation preserves independent starts per head, checkpoint fallback order,
and `shift_leap * 2 / (T//2 - 1)` step sizes. By default it generates forward pairs.
Add `--bidirectional` to independently choose +1 or -1 with equal probability for
each pair, keeping that sign throughout its rollout. Negative pairs save the
endpoint first and the original image second; traversal labels stay unchanged.
Bidirectional runs also save `directions.npy`: one +1 or -1 per saved pair, in
the same order as `labels.npy`, recording the rollout direction before image swapping.

Pass `--seed 123` to both commands for reproducible sampling and direction choices.
Seeding occurs after model loading. Compare with the same batch size, device,
weights and arguments; this does not enforce deterministic backend kernels.
Omitting the seed retains unseeded behavior. Use separate copies of an experiment
when comparing outputs, since both entry points write the same output names.
`bash scripts/SNGAN_genpairs.sh` runs both implementations with seed 123 in isolated
folders under the configured experiment and compares all output files byte for byte.
