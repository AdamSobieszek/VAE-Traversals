"""Fixed-cohort, noise-free traversal validation and compact TensorBoard logging."""

from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image
from tensorboard.compat.proto.summary_pb2 import Summary
from torchvision.utils import make_grid

from .aux import ImageViz, decode_generator_output_for_viz, sec2dhms, update_stdout
from .utils import sample_z


def add_validation_arguments(parser):
    parser.add_argument('--val-freq', type=int, default=10,
                        help='validate every N optimizer steps; 0 disables validation')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='initial positions per validation batch; generator/recognizer process batch_size * K pairs')
    parser.add_argument('--val-num-positions', type=int, default=32,
                        help='number of fixed starting positions, each evaluated across all K paths')
    parser.add_argument('--val-seed', type=int, default=12345)
    parser.add_argument('--val-dt', type=float, default=None,
                        help='fixed validation step size; defaults to 2 / max(1, T//2 - 1)')


@contextmanager
def _fixed_rng(seed, device):
    """Do not let cohort sampling or stochastic generators advance training RNGs."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps = torch.mps.get_rng_state() if torch.device(device).type == 'mps' else None
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


@contextmanager
def _evaluation(*models):
    modes = {module: module.training for model in models for module in model.modules()}
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for module, training in modes.items():
            module.training = training


@lru_cache(maxsize=4)
def _bootstrap_weights(n, k, seed, samples):
    """Reuse the fixed cohort's crossed-bootstrap draws as multiplicities."""
    rng = np.random.default_rng(seed)
    positions, heads = np.zeros((samples, n)), np.zeros((samples, k))
    for i in range(samples):
        positions[i] = np.bincount(rng.integers(n, size=n), minlength=n)
        heads[i] = np.bincount(rng.integers(k, size=k), minlength=k)
    positions.setflags(write=False)
    heads.setflags(write=False)
    return positions, heads


def accuracy_curve(correct, seed, bootstrap_samples=1000):
    """Mean and pointwise 95% crossed-bootstrap intervals for [position, K, step].

    Resample positions and traversal heads independently, retaining every step
    of each selected trajectory. Signed steps remain separate and ordered.
    """
    correct = np.asarray(correct, dtype=np.float64)
    n, k, _ = correct.shape
    positions, heads = _bootstrap_weights(n, k, seed, bootstrap_samples)
    # Same draws and crossed resampling, without materializing 1,000 indexed cohorts.
    means = np.einsum('rn,nkt,rk->rt', positions, correct, heads, optimize=True) / (n * k)
    low, high = np.quantile(means, [0.025, 0.975], axis=0)
    return correct.mean(axis=(0, 1)), low, high


class TraversalValidation:
    """Own the cohort, rollout, statistics and output; trainer only schedules run()."""

    @torch.no_grad()
    def __init__(self, trainer, generator):
        p = trainer.params
        self.frequency = int(getattr(p, 'val_freq', 10))
        self.batch_size = int(getattr(p, 'val_batch_size', getattr(p, 'batch_size', 8)))
        self.num_positions = int(getattr(p, 'val_num_positions', 32))
        self.seed = int(getattr(p, 'val_seed', 12345))
        # randint(0, T//2-1) selects pairs through (x[steps-1], x[steps]).
        self.steps = int(p.num_traversal_timesteps) // 2 - 1
        self.dt = getattr(p, 'val_dt', None)
        if self.dt is None:
            self.dt = 2.0 / max(1, int(p.num_traversal_timesteps) // 2 - 1)
        self.directions = (1, -1) if getattr(p, 'bidirectional', False) else (1,)
        self.center = self.steps if len(self.directions) == 2 else 0
        self.path_indices = np.arange(-self.center, self.steps + 1)
        self.step_indices = self.path_indices[self.path_indices != 0]
        if self.frequency and self.steps < 1:
            raise ValueError('validation requires T//2 - 1 > 0, matching training randint')
        if self.frequency < 0 or self.batch_size < 1 or self.num_positions < 1:
            raise ValueError('val_freq must be nonnegative; validation batch size and positions must be positive')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('val_dt must be finite and positive')
        self.positions = None
        if self.frequency:
            base = trainer._generator_module(generator)
            # sample_z pads batches above dim_z with zeros; sample small cohorts
            # so every requested position is a real sample, even when N > D.
            dim = int(base.dim_z if hasattr(base, 'dim_z') else base.latent_size)
            with _fixed_rng(self.seed, trainer.device), _evaluation(generator):
                self.positions = torch.cat([
                    sample_z(min(dim, self.num_positions - i), base, p, trainer.device)
                    for i in range(0, self.num_positions, dim)
                ]).detach().cpu()

    def log_progress(self, done, wipe=True):
        if wipe:
            update_stdout(3)
        elapsed = time.time() - self._val_t0
        eta = (self._val_total - done) * elapsed / done if done else 0.
        print('\\__.Validation [batch-steps: {:03d}/{:03d}]'.format(done, self._val_total))
        print('   \\__Elapsed time   : {}'.format(sec2dhms(elapsed)[:-6]))
        print('   \\__ETA            : {}'.format(sec2dhms(eta)[:-6]))

    def _rollout(self, trainer, traversal, z):
        """Roll out together, storing vertices directly in [-S, ..., 0, ..., S] order."""
        b, d = z.shape
        k, directions = traversal.num_traversal_sets, len(self.directions)
        current = z[None, :, None].expand(directions, b, k, d).reshape(directions * b, k, d).contiguous()
        signs = z.new_tensor(self.directions).repeat_interleave(b)
        path = z.new_empty(directions * self.steps + 1, b, k, d)
        path[self.center].copy_(current[:b])
        for index in range(self.steps):
            with trainer.fp32_context():
                origin, delta = traversal.inference(current, dt=self.dt, direction=signs)
            following = (origin.detach() + delta.detach()).reshape(directions, b, k, d)
            path[self.center + index + 1].copy_(following[0])
            if self.center:
                path[self.center - index - 1].copy_(following[1])
            current = following.view(directions * b, k, d)
            del origin, delta
        return path.view(-1, b * k, d)

    def _capture_path_images(self, generator, images, vertex, start):
        """Keep only five already-synthesized vertices of position zero."""
        if start or vertex not in self._image_rows:
            return
        images = decode_generator_output_for_viz(generator, images[:self._image_columns]).detach()
        if max(images.shape[-2:]) > 64:
            images = torch.nn.functional.interpolate(images.float(), size=(64, 64), mode='area')
        if self._path_images is None:
            self._path_images = images.new_empty(5, self._image_columns, *images.shape[1:])
        for row in self._image_rows[vertex]:
            self._path_images[row].copy_(images)

    def _save_path_images(self, output, writer, step):
        rows = self._path_images.float().cpu()
        self._path_images = None
        # Match triplet normalization, with five rows and traversal heads as columns.
        images = torch.cat([ImageViz.to_uint01(row) for row in rows])
        grid = make_grid(images, nrow=self._image_columns, padding=2)
        pixels = grid.mul(255).clamp_(0, 255).byte().permute(1, 2, 0).numpy()
        with BytesIO() as buffer:
            Image.fromarray(pixels).save(buffer, format='PNG', optimize=True, compress_level=6)
            encoded = buffer.getvalue()
        (output / f'path_{int(step):08d}.png').write_bytes(encoded)
        if writer is not None:
            # Reuse the compressed PNG rather than encoding it again in add_image.
            summary = Summary(value=[Summary.Value(tag='validation/path_images', image=Summary.Image(
                height=pixels.shape[0], width=pixels.shape[1], colorspace=pixels.shape[2],
                encoded_image_string=encoded))])
            writer._get_file_writer().add_summary(summary, step)

    def _correct(self, trainer, generator, traversal, recognizer):
        """Stream each image trajectory with one rolling endpoint batch.

        Initial images are shared across directions, and across heads for
        generators declaring deterministic shared outputs. Stochastic generators
        keep distinct initial draws per head, but reuse each realized endpoint:
        both pairs adjoining a trajectory vertex must see the same image.
        """
        k = traversal.num_traversal_sets
        self._image_columns = min(k, 32)
        self._image_vertices = np.rint(np.linspace(0, len(self.path_indices) - 1, 5)).astype(int)
        self._image_rows = {}
        for row, vertex in enumerate(self._image_vertices):
            self._image_rows.setdefault(int(vertex), []).append(row)
        self._path_images = None
        metrics = torch.empty(2, self.num_positions * k, len(self.step_indices), device=trainer.device,
                              dtype=torch.float32)
        shared = getattr(trainer._generator_module(generator), 'share_initial_output', False)
        for batch, start in enumerate(range(0, self.num_positions, self.batch_size)):
            z = self.positions[start:start + self.batch_size].to(trainer.device)
            path = self._rollout(trainer, traversal, z)
            initial = trainer._synthesize(generator, z).repeat_interleave(k, dim=0) if shared else None
            labels = torch.arange(k, device=z.device).repeat(len(z))
            previous = initial if shared and not self.center else trainer._synthesize(generator, path[0])
            self._capture_path_images(generator, previous, 0, start)
            hits, losses = metrics[:, start * k:(start + len(z)) * k]
            for index in range(1, len(path)):
                following = (initial if shared and index == self.center else
                             trainer._synthesize(generator, path[index]))
                self._capture_path_images(generator, following, index, start)
                direction = -1 if index <= self.center else 1
                # Negative vertices are visited inward; keep the training pair
                # outward-facing so whitening still uses its original center.
                pair = (following, previous) if direction < 0 else (previous, following)
                logits = trainer._pair_logits(recognizer, *pair).float() * direction
                hits[:, index - 1].copy_(logits.argmax(-1) == labels)
                losses[:, index - 1].copy_(torch.nn.functional.cross_entropy(logits, labels, reduction='none'))
                previous = following
                self.log_progress(batch * len(self.step_indices) + index)
            del path
        # One accelerator/host synchronization for the entire accuracy tensor.
        return metrics.cpu().numpy().astype(np.float64).reshape(2, self.num_positions, k, -1)

    @torch.no_grad()
    def run(self, trainer, generator, traversal_sets, recognizer, step):
        if self.positions is None:
            return None
        traversal = getattr(traversal_sets, 'module', traversal_sets)
        k = traversal.num_traversal_sets
        self._val_t0 = time.time()
        self._val_total = ((self.num_positions + self.batch_size - 1) // self.batch_size) * len(self.step_indices)
        self.log_progress(0, wipe=False)
        try:
            with _fixed_rng(self.seed, trainer.device), _evaluation(generator, traversal_sets, recognizer):
                correct, cross_entropy = self._correct(trainer, generator, traversal, recognizer)
            mean, low, high = accuracy_curve(correct, self.seed)
            mean_ce = cross_entropy.mean(axis=(0, 1))
            record = dict(step=int(step), accuracy=float(mean.mean()),
                          cross_entropy=float(mean_ce.mean()), step_cross_entropy=mean_ce.tolist(),
                          step_indices=self.step_indices.tolist(), path_indices=self.path_indices.tolist(),
                          image_step_indices=self.path_indices[self._image_vertices].tolist(),
                          step_accuracy=mean.tolist(), ci_low=low.tolist(), ci_high=high.tolist(),
                          num_positions=self.num_positions, num_paths=k, seed=self.seed,
                          dt=float(self.dt), directions=list(self.directions),
                          ci_method='95% pointwise crossed bootstrap over positions and traversal heads')
            output = Path(trainer.wip_dir) / 'validation'
            output.mkdir(parents=True, exist_ok=True)
            with (output / 'stats.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            np.savez_compressed(output / 'latest.npz', accuracy_by_position_path_step=correct,
                                cross_entropy_by_position_path_step=cross_entropy,
                                step_indices=self.step_indices, path_indices=self.path_indices)
            writer = trainer.tb_writer
            self._save_path_images(output, writer, step)
            if writer is not None:
                writer.add_scalar('validation/accuracy', record['accuracy'], step)
                writer.add_scalar('validation/cross_entropy', record['cross_entropy'], step)
                fig, (ax, ce_ax) = plt.subplots(2, 1, figsize=(max(8, .65 * len(self.path_indices)), 7), sharex=True)
                try:
                    # Zero is a shared vertex, not a classification pair: leave a
                    # gap there instead of inventing an accuracy or CE at zero.
                    x = self.path_indices
                    valid = x != 0
                    plotted = np.full((4, len(x)), np.nan)
                    plotted[:, valid] = np.stack((mean, low, high, mean_ce))
                    mean_plot, low_plot, high_plot, ce_plot = plotted
                    ax.plot(x, mean_plot, marker='o', label='Mean accuracy')
                    ax.fill_between(x, low_plot, high_plot, alpha=0.25, label='95% CI (positions and paths)')
                    ax.axhline(1 / k, color='gray', linestyle='--', label='Chance')
                    # Log of the error rate, reversed: more space near 100%.
                    # A small offset keeps perfect accuracy finite and visible.
                    epsilon = 1e-4
                    ax.set_yscale('function', functions=(
                        lambda y: -np.log10(np.maximum(1 - y + epsilon, np.finfo(float).tiny)),
                        lambda y: 1 + epsilon - np.power(10., -y)))
                    ax.set_yticks([0, .5, .9, .99, .999, .9999, 1])
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
                    for level in np.arange(0, 1.01, .1):
                        ax.axhline(level, color='gray', linestyle='--', linewidth=.5, alpha=.18, zorder=0)
                    for position, value in zip(self.step_indices, mean):
                        ax.annotate(f'{100 * value:.4f}%', (position, value), xytext=(0, -9),
                                    textcoords='offset points', ha='center', va='top', fontsize=7)
                    ax.set(ylabel='Accuracy (log-scaled error rate)',
                           ylim=(0, 1), title='Noise-free validation')
                    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.14), ncol=3, fontsize=8)
                    ce_ax.plot(x, ce_plot, marker='o')
                    ce_ax.set(xlabel='Signed outward step (shared origin at 0)', ylabel='Cross-entropy',
                              xticks=x)
                    fig.tight_layout()
                    writer.add_figure('validation/accuracy_by_step', fig, step)
                finally:
                    plt.close(fig)
            return record
        finally:
            update_stdout(3)
