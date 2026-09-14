"""Post-hoc cohort AGOP and within-anchor conditional cone diagnostics.

Legacy adjacent/pooled/diagonal protocols keep their deterministic pair banks.
Example: python -m lib.explainability --exp EXP --heads 0 --protocol cone
         --positions 1 --noise-rays 32 --probe log-evidence
Cone defaults render sensitivity, cumulative signed evidence and class response;
add --cone-figures sensitivity evidence confidence response for all four families.
--dt specifies ONE full inference step, subsequently interpolated with fixed noise.
Numerical banks/results persist only with --save-data. Nonpersistent covariance
defaults to the loader's identity geometry; --cone-covariance supplies it explicitly.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from .aux import ImageViz, decode_generator_output_for_viz
from .config import GAN_RESOLUTIONS
from .recognizer import Recognizer, recognizer_options
from .val_utils import TraversalValidation, _evaluation, _fixed_rng, load_pair_experiment


def load_experiment(directory, device):
    """Rebuild the existing inference stack and add its separately saved recognizer."""
    directory = Path(directory)
    args, generator, traversal = load_pair_experiment(directory, device)
    pool = getattr(args, 'recognizer_pool_size', None)
    pool = pool or (2 if args.recognizer_type.endswith('LeNet')
                    and args.gan_type != 'SNGAN_AnimeFaces' else 1)
    if getattr(args, 'recognizer_pool_size', None) is None \
            and args.gan_type == 'StyleGAN2' and args.stylegan2_resolution == 1024:
        pool = 4
    channels = 1 if args.gan_type == 'SNGAN_MNIST' else (4 if args.gan_type == 'GAT' else 3)
    resolution = (args.stylegan2_resolution if args.gan_type == 'StyleGAN2'
                  else GAN_RESOLUTIONS[args.gan_type])
    recognizer = Recognizer(args.recognizer_type, traversal.num_traversal_sets, channels, pool,
                            **recognizer_options(args, resolution)).to(device)
    checkpoint = directory / 'models' / 'checkpoint.pt'
    state = torch.load(checkpoint if checkpoint.is_file() else
                       directory / 'models' / 'recognizer.pt', map_location=device)
    state = state.get('recognizer', state)
    if state and all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    recognizer.load_state_dict(state, strict=True)
    return args, generator.eval(), traversal.eval(), recognizer.eval()


def _operator(rows, vectors, batch=256):
    result = torch.zeros(rows.shape[1], vectors.shape[1], device='cpu')
    for start in range(0, len(rows), batch):
        gradient = torch.from_numpy(np.asarray(rows[start:start + batch]))
        result.add_(gradient.mT @ (gradient @ vectors))
    return result / len(rows)


def _split_top(rows, groups, parity, iterations, batch=256):
    """Matrix-free top mode on one anchor half, retaining all groups per anchor."""
    width, anchors = rows.shape[1], len(rows) // groups
    vector = torch.randn(width, 1, generator=torch.Generator().manual_seed(parity), device='cpu')
    vector /= torch.linalg.vector_norm(vector)
    for _ in range(iterations):
        result, count = torch.zeros_like(vector), 0
        for group in range(groups):
            block = rows[group * anchors:(group + 1) * anchors]
            for start in range(parity, anchors, 2 * batch):
                gradient = torch.from_numpy(np.asarray(block[start:start + 2 * batch:2]))
                result.add_(gradient.mT @ (gradient @ vector)); count += len(gradient)
        norm = torch.linalg.vector_norm(result)
        vector = result / norm.clamp_min(1e-30)
    return vector[:, 0]


def _exact_top(rows):
    gradient = torch.from_numpy(np.asarray(rows))
    values, vectors = torch.linalg.eigh(gradient @ gradient.mT / len(rows))
    value, left = values[-1].clamp_min(0), vectors[:, -1]
    return gradient.mT @ left / torch.sqrt(len(rows) * value.clamp_min(1e-30))


def covariance_modes(rows, shape, num_modes, exact_max, iterations, split_groups=1, components=2): # Dont know why someone added components arg because this used to be hard coded to two because there are two images in an input
    """Return joint modes; large banks use matrix-free subspace iteration and Rayleigh values."""
    count, rank = len(rows), min(len(rows), rows.shape[1], num_modes)
    if count <= exact_max and count * rows.shape[1] * 4 <= 512 * 2**20:
        gradient = torch.from_numpy(np.asarray(rows))
        values, left = torch.linalg.eigh(gradient @ gradient.mT / count)
        values, left = values[-rank:].flip(0).clamp_min(0), left[:, -rank:].flip(1)
        modes = (left.mT @ gradient) / torch.sqrt(count * values.clamp_min(1e-30))[:, None]
        residual = torch.linalg.vector_norm(
            gradient.mT @ (gradient @ modes.mT) / count - modes.mT * values, dim=0
        ) / values.abs().clamp_min(1e-30)
        grouped = np.asarray(rows).reshape(split_groups, -1, rows.shape[1])
        split = (abs(torch.dot(
            _exact_top(grouped[:, ::2].reshape(-1, rows.shape[1])),
            _exact_top(grouped[:, 1::2].reshape(-1, rows.shape[1]))))
            if grouped.shape[1] >= 2 else torch.tensor(float('nan'), device='cpu'))
        method = 'exact dual-Gram'
    else:
        rng = torch.Generator().manual_seed(0)
        basis = torch.linalg.qr(torch.randn(
            rows.shape[1], rank, generator=rng, device='cpu')).Q
        for _ in range(iterations):
            basis = torch.linalg.qr(_operator(rows, basis)).Q
        action = _operator(rows, basis)
        values, rotation = torch.linalg.eigh(basis.mT @ action)
        values, rotation = values.flip(0).clamp_min(0), rotation[:, -rank:].flip(1)
        modes = (basis @ rotation).mT
        residual = torch.linalg.vector_norm(
            _operator(rows, modes.mT) - modes.mT * values, dim=0
        ) / values.abs().clamp_min(1e-30)
        split = (abs(torch.dot(_split_top(rows, split_groups, 0, iterations),
                               _split_top(rows, split_groups, 1, iterations)))
                 if count // split_groups >= 2
                 else torch.tensor(float('nan'), device='cpu'))
        method = f'matrix-free ({iterations} iterations)'
    pivot = modes.gather(1, modes.abs().argmax(1, keepdim=True)).sign()
    modes.mul_(pivot.masked_fill(pivot == 0, 1))
    return (modes.reshape(rank, components, *shape).numpy(), values.numpy(),
            residual.numpy(), float(split), method)


def _gradients(trainer, recognizer, head, direction, first, second):
    """Centered head contrast and independent endpoint/common/relative derivatives."""
    with torch.enable_grad():
        endpoints = [x.detach().float().requires_grad_(True) for x in (first, second)]
        logits = trainer._pair_logits(recognizer, *endpoints).float() * direction
        score = logits[:, head] - logits.mean(-1)
        gradients = list(torch.autograd.grad(score.sum(), endpoints))
    gradients += [(gradients[0] + gradients[1]) / np.sqrt(2),
                  (gradients[1] - gradients[0]) / np.sqrt(2)]
    return score.detach(), [gradient.detach().float() for gradient in gradients]


def _image(images):
    image = ImageViz.to_uint01(images[0].detach().float().cpu())
    image = image.movedim(0, -1).numpy()
    return np.repeat(image, 3, -1) if image.shape[-1] == 1 else image


def _signed(data, scale=None):
    data = np.asarray(data)
    scale = max(float(scale or np.quantile(abs(data), .995)), np.finfo('float32').eps)
    if data.ndim == 3 and data.shape[0] == 3:
        return np.clip(.5 + data.transpose(1, 2, 0) / (2 * scale), 0, 1), {}
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    return data, dict(cmap='seismic', vmin=-scale, vmax=scale)


def render(frames, destination, head, protocol, fps, native_shape):
    """Render fixed-layout cohort summaries; frames are steps or joint-mode ranks."""
    difference_scale = max(np.quantile(abs(frame['difference']), .995) for frame in frames)
    energy_scale = max(np.quantile(np.stack(frame['energy']), .995) for frame in frames)
    cross_scale = max(np.quantile(abs(frame['cross']), .995) for frame in frames)
    native_rgb = native_shape[0] == 3

    def mode_panel(mode):
        if native_rgb or native_shape[0] == 1:
            return _signed(mode)
        magnitude = np.linalg.norm(mode, axis=0)
        return magnitude, dict(cmap='viridis', vmin=0,
                               vmax=max(np.quantile(magnitude, .995), 1e-12))

    def panels(frame):
        mode = frame['mode']
        common = (mode[0] + mode[1]) / np.sqrt(2)
        result = [(frame['images'][0], {}), (frame['images'][1], {}),
                  _signed(frame['difference'], difference_scale)]
        result += [mode_panel(value) for value in (mode[0], mode[1], common)]
        result += [(value, dict(cmap='magma', vmin=0, vmax=max(energy_scale, 1e-12)))
                   for value in frame['energy']]
        return result + [(frame['cross'], dict(
            cmap='seismic', vmin=-max(cross_scale, 1e-12), vmax=max(cross_scale, 1e-12)))]

    qualifier = 'signed mode' if native_rgb else 'native channel magnitude'
    names = ('Example x', 'Example y', 'Example RGB difference',
             f'Joint x: {qualifier}', f'Joint y: {qualifier}', f'Common: {qualifier}',
             'Cohort common energy', 'Cohort relative energy', 'Cohort endpoint cross')
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), dpi=120)
    fig.subplots_adjust(left=.03, right=.99, bottom=.03, top=.86, wspace=.18, hspace=.32)
    artists, heading = [], fig.suptitle('')
    for axis, (data, options), name in zip(axes.flat, panels(frames[0]), names):
        artists.append(axis.imshow(data, **options))
        axis.set_title(name); axis.set_axis_off()
    writer = FFMpegWriter(fps=fps, codec='libvpx-vp9',
                          extra_args=['-crf', '30', '-b:v', '0', '-pix_fmt', 'yuv420p'])
    try:
        with writer.saving(fig, str(destination), dpi=120):
            for frame in frames:
                for artist, (data, options) in zip(artists, panels(frame)):
                    artist.set_data(data)
                    if 'vmin' in options:
                        artist.set_clim(options['vmin'], options['vmax'])
                heading.set_text(
                    f'{protocol} | centered head {head} | fixed pair bank | {frame["label"]}\n'
                    f'N={frame["count"]} | native {native_shape} | {frame["method"]} | '
                    f'λ={frame["value"]:.3g} ({frame["explained"]:.1%} trace) | '
                    f'score={frame["score_mean"]:+.3g}±{frame["score_std"]:.2g} | '
                    f'‖y-x‖={frame["difference_norm"]:.3g}')
                fig.canvas.draw(); writer.grab_frame()
    finally:
        plt.close(fig)


class PairAGOP:
    """Post-hoc analysis of one fixed, reusable native-input pair bank."""

    def __init__(self, trainer, generator, traversal, recognizer, validation, args):
        self.trainer, self.generator, self.traversal, self.recognizer = (
            trainer, generator, traversal, recognizer)
        self.validation, self.args = validation, args
        self.positions, self.seed = validation.positions[:args.positions], validation.seed
        self.steps = args.steps if args.steps is not None else validation.steps
        self.dt, self.directions = args.dt if args.dt is not None else validation.dt, validation.directions
        self.center = self.steps if len(self.directions) == 2 else 0
        self.path_indices = np.arange(-self.center, self.steps + 1)
        self.step_indices = self.path_indices[self.path_indices != 0]
        self.temporary_bank = None
        self.bank, self.bank_meta = self._pair_bank()

    def _rollout(self, z):
        return TraversalValidation._rollout(self, self.trainer, self.traversal, z)

    def _pair_bank(self):
        producers = [0] if self.args.protocol == 'diagonal' else self.args.producers
        edges = 1 if self.args.protocol == 'diagonal' else len(self.step_indices)
        checkpoint = Path(self.args.exp) / 'models' / 'checkpoint.pt'
        if not checkpoint.is_file():
            base = checkpoint.parent / 'traversal_sets.pt'
            candidates = ([base] if base.is_file() else []) + sorted(
                checkpoint.parent.glob('traversal_sets-*'))
            checkpoint = candidates[-1] if candidates else checkpoint
        stamp = checkpoint.stat() if checkpoint.is_file() else None
        arguments = Path(self.args.exp) / 'args.json'
        args_stamp = arguments.stat() if arguments.is_file() else None
        config = dict(version=3, anchors=len(self.positions), producers=producers, steps=self.steps,
                      dt=self.dt, seed=self.seed, batch_size=self.args.batch_size,
                      directions=list(self.directions),
                      checkpoint=([stamp.st_mtime_ns, stamp.st_size] if stamp else None),
                      arguments=([args_stamp.st_mtime_ns, args_stamp.st_size] if args_stamp else None),
                      source='diagonal' if self.args.protocol == 'diagonal' else 'adjacent')
        identifier = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        output = Path(self.args.exp) / 'validation' / 'explainability'
        metadata_path = output / f'pair_bank_{identifier}.json'
        if self.args.save_data:
            path = output / f'pair_bank_{identifier}.npy'
        else:
            handle = tempfile.NamedTemporaryFile(
                dir=output, suffix='.pair-bank.tmp', delete=False)
            path, self.temporary_bank = Path(handle.name), Path(handle.name)
            handle.close()
        if self.args.save_data and path.is_file() and metadata_path.is_file():
            return np.lib.format.open_memmap(path, mode='r'), json.loads(metadata_path.read_text())

        bank = None
        with _fixed_rng(self.seed, self.trainer.device), _evaluation(
                self.generator, self.traversal):
            for start in range(0, len(self.positions), self.args.batch_size):
                z = self.positions[start:start + self.args.batch_size].to(self.trainer.device)
                if self.args.protocol == 'diagonal':
                    with torch.no_grad():
                        image = self.trainer._synthesize(self.generator, z)
                    batches = [(0, 0, (image, image))]
                else:
                    path_batch = self._rollout(z).view(
                        len(self.path_indices), len(z), self.traversal.num_traversal_sets, -1)
                    shared = getattr(self.trainer._generator_module(
                        self.generator), 'share_initial_output', False)
                    with torch.no_grad():
                        initial = self.trainer._synthesize(self.generator, z) if shared else None
                    batches = []
                    for producer_index, producer in enumerate(producers):
                        with torch.no_grad():
                            previous = (initial if shared and not self.center else
                                        self.trainer._synthesize(
                                            self.generator, path_batch[0, :, producer]))
                        for edge in range(edges):
                            with torch.no_grad():
                                following = (initial if shared and edge + 1 == self.center else
                                             self.trainer._synthesize(
                                                 self.generator, path_batch[edge + 1, :, producer]))
                            batches.append((producer_index, edge, (previous, following)))
                            previous = following
                for producer_index, edge, pair in batches:
                    if bank is None:
                        shape = (len(producers), edges, len(self.positions), 2, *pair[0].shape[1:])
                        bank = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
                    bank[producer_index, edge, start:start + len(z)] = torch.stack(pair, 1).float().cpu()
        bank.flush()
        metadata = {**config, 'id': identifier, 'file': path.name, 'shape': list(bank.shape),
                    'directions': [1] * edges,
                    'path_directions': ([0] if self.args.protocol == 'diagonal'
                                        else [-1 if i + 1 <= self.center else 1
                                              for i in range(edges)]),
                    'labels': (['diagonal pair'] if self.args.protocol == 'diagonal'
                               else [f'signed step {value:+d}' for value in self.step_indices]),
                    'nuisance_policy': ('deterministic shared-output generator' if getattr(
                        self.trainer._generator_module(self.generator),
                        'share_initial_output', False)
                        else 'fixed realized native tensors; conditional on seed and batch size')}
        if self.args.save_data:
            metadata_path.write_text(json.dumps(metadata, indent=2))
        return bank, metadata

    def close(self):
        if self.temporary_bank is not None:
            del self.bank
            self.temporary_bank.unlink(missing_ok=True)
            self.temporary_bank = None

    def stem(self, head):
        return f'agop_{self.args.protocol}_head_{head:03d}_bank_{self.bank_meta["id"]}'

    def _samples(self, group):
        if self.args.protocol == 'adjacent':
            samples = self.bank[:, group].reshape(-1, *self.bank.shape[3:])
            directions = np.full(len(samples), self.bank_meta['directions'][group])
            label = self.bank_meta['labels'][group]
        else:
            samples = self.bank.reshape(-1, *self.bank.shape[3:])
            directions = np.repeat(
                np.tile(self.bank_meta['directions'], len(self.bank_meta['producers'])),
                len(self.positions))
            label = 'diagonal pair' if self.args.protocol == 'diagonal' else 'uniform pooled bank'
        return samples, directions, label

    def _group(self, head, group):
        samples, directions, label = self._samples(group)
        rows = temporary = stat = None
        try:
            for start in range(0, len(samples), self.args.batch_size):
                native = torch.from_numpy(np.array(
                    samples[start:start + self.args.batch_size], copy=True)).to(self.trainer.device)
                pair = native[:, 0], native[:, 1]
                direction = torch.as_tensor(
                    directions[start:start + len(native)], device=self.trainer.device)[:, None]
                score, gradients = _gradients(
                    self.trainer, self.recognizer, head, direction, *pair)
                if stat is None:
                    stat = self._new_stat(pair, gradients, label)
                    handle = tempfile.NamedTemporaryFile(
                        dir=Path(self.args.exp) / 'validation', suffix='.agop.tmp', delete=False)
                    temporary = Path(handle.name); handle.close()
                    rows = np.lib.format.open_memmap(
                        temporary, mode='w+', dtype='float32',
                        shape=(len(samples), 2 * gradients[0][0].numel()))
                rows[start:start + len(native)] = torch.cat(
                    [value.flatten(1) for value in gradients[:2]], 1).cpu()
                self._add(stat, score, gradients, pair)
            modes, values, residuals, split, method = covariance_modes(
                rows, stat.pop('shape'), min(self.args.modes, len(rows)),
                self.args.exact_max_rows, self.args.power_iterations,
                len(self.args.producers) * (
                    len(self.step_indices) if self.args.protocol == 'pooled' else 1)
                if self.args.protocol != 'diagonal' else 1)
            mean, square = stat['score'] / stat['count']
            difference, difference_square = stat['difference_norm'] / stat['count']
            stat.update(score_mean=mean, score_std=np.sqrt(max(0, square - mean ** 2)),
                        difference_norm=difference, difference_std=np.sqrt(
                            max(0, difference_square - difference ** 2)),
                        traces=stat['traces'] / stat['count'],
                        energy=np.stack([value.numpy() / stat['count'] for value in stat['energy']]),
                        cross=stat['cross'].numpy() / stat['count'], modes=modes, values=values,
                        residuals=residuals, split_agreement=split, method=method)
            stat['difference'] = stat['difference'].numpy()
            return stat
        finally:
            if temporary is not None:
                del rows
                temporary.unlink(missing_ok=True)

    def _new_stat(self, pair, gradients, label):
        with torch.no_grad():
            decoded = [decode_generator_output_for_viz(self.generator, value[:4]) for value in pair]
        shape = tuple(gradients[0].shape[1:])
        return dict(label=label, count=0, score=np.zeros(2), difference_norm=np.zeros(2),
                    traces=np.zeros(2),
                    energy=[torch.zeros(shape[1:], device='cpu') for _ in range(2)],
                    cross=torch.zeros(shape[1:], device='cpu'),
                    images=[_image(value) for value in decoded],
                    difference=(decoded[1][0] - decoded[0][0]).float().cpu(), shape=shape)

    @staticmethod
    def _add(stat, score, gradients, pair):
        stat['count'] += len(score)
        stat['score'] += [float(score.sum()), float(score.square().sum())]
        delta = (pair[1] - pair[0]).float().flatten(1).norm(dim=1)
        stat['difference_norm'] += [float(delta.sum()), float(delta.square().sum())]
        stat['traces'] += [float(value.square().sum()) for value in gradients[2:]]
        for total, value in zip(stat['energy'], gradients[2:]):
            total.add_(value.square().sum(1).sum(0).cpu())
        stat['cross'].add_((gradients[0] * gradients[1]).sum(1).sum(0).cpu())

    def analyze(self, head):
        groups = len(self.step_indices) if self.args.protocol == 'adjacent' else 1
        spectra = [self._group(head, group) for group in range(groups)]
        selections = ([(group, 0) for group in range(groups)] if groups > 1 else
                      [(0, rank) for rank in range(len(spectra[0]['values']))])
        frames = []
        for group, rank in selections:
            stat, trace = spectra[group], sum(spectra[group]['traces'])
            frames.append({**stat, 'mode': stat['modes'][rank], 'value': stat['values'][rank],
                           'explained': stat['values'][rank] / max(trace, 1e-30),
                           'label': stat['label'] if groups > 1 else f'joint mode rank {rank + 1}'})

        output, stem = Path(self.args.exp) / 'validation' / 'explainability', self.stem(head)
        render(frames, output / f'{stem}.webm', head, self.args.protocol,
               self.args.fps, spectra[0]['modes'].shape[2:])
        if self.args.save_data:
            np.savez_compressed(
                output / f'{stem}.npz',
                modes=np.stack([stat['modes'] for stat in spectra]),
                eigenvalues=np.stack([stat['values'] for stat in spectra]),
                explained_trace=np.stack([
                    stat['values'] / max(sum(stat['traces']), 1e-30) for stat in spectra]),
                # This orthonormal change of basis preserves the joint trace.
                relative_sensitivity=[
                    stat['traces'][1] / max(sum(stat['traces']), 1e-30) for stat in spectra],
                residuals=np.stack([stat['residuals'] for stat in spectra]),
                traces=np.stack([stat['traces'] for stat in spectra]),
                common_energy=np.stack([stat['energy'][0] for stat in spectra]),
                relative_energy=np.stack([stat['energy'][1] for stat in spectra]),
                endpoint_cross=np.stack([stat['cross'] for stat in spectra]),
                score_mean=[stat['score_mean'] for stat in spectra],
                score_std=[stat['score_std'] for stat in spectra],
                difference_norm=[stat['difference_norm'] for stat in spectra],
                split_agreement=[stat['split_agreement'] for stat in spectra],
                labels=[stat['label'] for stat in spectra])
            metadata = dict(
                protocol=self.args.protocol, probe=f'centered head {head}',
                pair_population=('diagonal base-image cohort' if self.args.protocol == 'diagonal'
                                 else dict(producers=self.bank_meta['producers'],
                                           pair_bank=self.bank_meta['file'])),
                weighting='uniform over anchors, producers, directions, and included edges',
                anchors=len(self.positions), rows=[stat['count'] for stat in spectra],
                native_shape=list(spectra[0]['modes'].shape[2:]),
                spectral_method=[stat['method'] for stat in spectra],
                steps=0 if self.args.protocol == 'diagonal' else self.steps, dt=self.dt,
                nuisance_policy=self.bank_meta['nuisance_policy'],
                preview=len(self.positions) < 64, clipping='display only: 99.5% absolute quantile',
                mode_display='unit joint mode shape; native non-RGB modes use channel magnitude',
                gradient_coordinates='native recognizer input tensor values; contact sheets are decoded',
                diagonal_operator=('joint [gx; gy], with b=(gy-gx)/2 eigenvalues equal to joint/2'
                                   if self.args.protocol == 'diagonal' else None),
                animation=('top joint mode at each edge' if self.args.protocol == 'adjacent'
                           else 'joint eigenmodes in descending rank'),
                numerical_arrays=f'{stem}.npz')
            (output / f'{stem}.json').write_text(json.dumps(metadata, indent=2))
        print(f'Saved {output / f"{stem}.webm"}')


def _cone_probe(trainer, recognizer, first, second, head, direction, probe,
                baseline, gradients=True):
    """Differentiate independent native endpoints; the diagonal baseline is frozen."""
    with torch.enable_grad() if gradients else torch.no_grad():
        x, y = [v.detach().float().requires_grad_(gradients) for v in (first, second)]
        logits = trainer._pair_logits(recognizer, x, y).float() * direction
        logp = logits.log_softmax(-1)
        score = (logits[:, head] - logits.mean(-1) if probe == 'centered-logit'
                 else logp[:, head] - baseline[head])
        derivative = torch.autograd.grad(score.sum(), (x, y)) if gradients else ()
    return score.detach(), logp.detach(), tuple(v.detach() for v in derivative)


def _cone_integral(previous_gradient, gradient, previous_image, image):
    """One sampled curved-path trapezoid, contracting native channels only."""
    return ((previous_gradient + gradient) * .5 * (image - previous_image)).sum(1)


def _cone_signed(array, scale):
    """Non-RGB native channels are separate signed tiles, never decoded saliency."""
    if array.ndim == 3 and array.shape[0] not in (1, 3):
        array = np.concatenate(list(array), axis=1)
    return _signed(array, scale)


def _cone_movie(destination, count, fps, draw, figsize=(14, 8)):
    fig = plt.figure(figsize=figsize, dpi=100)
    writer = FFMpegWriter(fps=fps, codec='libvpx-vp9',
                          extra_args=['-crf', '30', '-b:v', '0', '-pix_fmt', 'yuv420p'])
    try:
        with writer.saving(fig, str(destination), dpi=100):
            for index in range(count):
                fig.clear()
                draw(fig, index)
                fig.tight_layout(rect=(0, 0, 1, .92))
                writer.grab_frame()
    finally:
        plt.close(fig)
    print(f'Saved {destination}')


def _cone_show(axis, data, title, **options):
    axis.imshow(data, **options)
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


class ConditionalCone:
    """Within-anchor diagnostics with compact fixed latent rays and streamed images.

    No population mixes anchors or producers. Spectral splits separate whole noise
    rays, and no path-averaged eigensystem is inferred from per-node eigenvectors.
    Negative directions use the training/validation outward-pair score convention.
    """

    def __init__(self, trainer, generator, traversal, recognizer, validation, args):
        self.trainer, self.generator, self.traversal, self.recognizer = (
            trainer, generator, traversal, recognizer)
        self.args, self.validation = args, validation
        self.output = Path(args.exp) / 'validation' / 'explainability'
        self.dt = validation.dt if args.dt is None else args.dt
        self.directions = args.cone_directions or validation.directions
        self.aperture = traversal.noise_aperture if args.aperture is None else args.aperture
        self.coarse_tau = np.asarray(args.tau_grid, dtype=np.float64)
        r = args.quadrature_refinement
        self.tau = np.concatenate([np.linspace(a, b, r + 1)[:-1]
                                   for a, b in zip(self.coarse_tau[:-1], self.coarse_tau[1:])]
                                  + [self.coarse_tau[-1:]])
        self.weights = np.zeros_like(self.tau)
        self.weights[:-1] += np.diff(self.tau) / 2
        self.weights[1:] += np.diff(self.tau) / 2
        self.shared = getattr(trainer._generator_module(generator), 'share_initial_output', False)
        self.geometry = getattr(traversal, 'cov_matrix', None)
        self.identity = self._identity()

    @staticmethod
    def _hash_tensor(value):
        return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def _identity(self):
        # Include effective runtime geometry; it is absent from traversal state_dict.
        models = Path(self.args.exp) / 'models'
        sources = ([Path(self.args.exp) / 'args.json'] + sorted(models.glob('*.pt'))
                   + sorted(models.glob('traversal_sets-*')))
        stamps = {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns]
                  for p in sources if p.is_file()}
        config = dict(version=1, sources=stamps,
                      runtime_covariance=(self._hash_tensor(self.geometry)
                                          if self.geometry is not None else 'identity (cov_matrix=None)'),
                      runtime_architecture=getattr(self.traversal, 'architecture', None),
                      dt=float(self.dt), aperture=float(self.aperture), tau=self.tau.tolist(),
                      noise_rays=self.args.noise_rays, seed=self.validation.seed,
                      probe=self.args.probe, producers=self.args.producers, heads=self.args.heads,
                      modes=self.args.modes, exact_max=self.args.exact_max_rows,
                      iterations=self.args.power_iterations, figures=self.args.cone_figures,
                      thresholds=self.args.confidence_thresholds,
                      success_threshold=self.args.success_threshold,
                      crossing_refinements=self.args.crossing_refinements,
                      normal_epsilon=self.args.normal_epsilon,
                      quadrature_refinement=self.args.quadrature_refinement,
                      joint=self.args.cone_joint, fps=self.args.fps,
                      directions=list(self.directions),
                      generator_config=vars(self.trainer.params),
                      nuisance_policy=('deterministic batch-independent synthesis' if self.shared else
                                       'same RNG seed reset for every singleton synthesis; shared across rays and tau'))
        return config

    def _synthesize(self, z):
        # Singleton replay for stochastic generators makes nuisance independent of
        # chunk size, ray, producer and tau, including x == y at the diagonal.
        with torch.no_grad():
            if self.shared:
                with _fixed_rng(self.nuisance_seed, self.trainer.device):
                    return self.trainer._synthesize(self.generator, z).detach().float()
            result = []
            for latent in z:
                with _fixed_rng(self.nuisance_seed, self.trainer.device):
                    result.append(self.trainer._synthesize(self.generator, latent[None]).detach().float())
            return torch.cat(result)

    def _preview(self, image):
        with torch.no_grad(), _fixed_rng(self.nuisance_seed, self.trainer.device):
            decoded = decode_generator_output_for_viz(self.generator, image[:1])
            if max(decoded.shape[-2:]) > 160:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(160, 160), mode='area')
            return _image(decoded)

    def _spectrum(self, rows, shape, previous=None, components=1):
        modes, values, residuals, split, method = covariance_modes(
            rows, shape, self.args.modes, self.args.exact_max_rows,
            self.args.power_iterations, components=components)
        if previous is not None:
            for mode, old in zip(modes, previous):
                if np.sum(mode * old) < 0:
                    mode *= -1
        block = max(1, (32 * 2**20) // (rows.shape[1] * 8))
        trace = sum(float(np.square(rows[i:i + block], dtype=np.float64).sum())
                    for i in range(0, len(rows), block)) / len(rows)
        if trace == 0:
            modes[:] = 0
            residuals[:] = np.nan
            split = float('nan')
        gap = float((values[0] - values[1]) / max(values[0], 1e-30)) if len(values) > 1 else None
        return dict(modes=modes, values=values, residuals=residuals, split=split,
                    trace=trace, explained=values / trace if trace else np.full_like(values, np.nan),
                    eigengap=gap, method=method)

    def _path(self, anchor, delta, head, direction, root, need_gradients):
        """Only rolling native images/gradients and current AGOP rows live on disk."""
        n, ts = len(delta), len(self.tau)
        x = self._synthesize(anchor[None])
        shape = tuple(x.shape[1:])
        with torch.no_grad():
            diagonal = (self.trainer._pair_logits(self.recognizer, x, x).float() * direction)[0]
            baseline = diagonal.log_softmax(-1)
        logp = np.empty((ts, n, len(baseline)), dtype=np.float32)
        scores = np.empty((ts, n), dtype=np.float32)
        frames, previous_modes = [], None

        def bank(name, bank_shape):
            return np.lib.format.open_memmap(root / (name + '.tmp'), mode='w+',
                                            dtype='float32', shape=bank_shape)

        if need_gradients:
            previous_y = bank('previous_y', (n, *shape))
            previous_g = bank('previous_g', (n, *shape))
            coarse_y, coarse_g = bank('coarse_y', (n, *shape)), bank('coarse_g', (n, *shape))
            attribution = bank('attribution', (n, *shape[1:]))
            coarse_attribution = bank('coarse_attribution', (n, *shape[1:]))
            attribution[:] = 0
            coarse_attribution[:] = 0
            rows = bank('rows', (n, int(np.prod(shape))))
            joint_rows = bank('joint', (n, 2 * int(np.prod(shape)))) if self.args.cone_joint else None
            # Result maps stay disk-backed as well; render/export without a giant RAM bank.
            result_modes = bank('modes', (ts, min(self.args.modes, n, int(np.prod(shape))), 1, *shape))
            result_mean = bank('mean', (ts, *shape))
            result_energy = bank('energy', (ts, *shape[1:]))
            result_attr = bank('attribution_mean', (ts, *shape[1:]))
            result_std = bank('attribution_std', (ts, *shape[1:]))
            totals, errors = np.zeros((ts, n)), np.zeros((ts, n))
            coarse_errors = np.full((ts, n), np.nan)
            if joint_rows is not None:
                joint_modes = bank('joint_modes', (ts, min(self.args.modes, n, 2 * int(np.prod(shape))), 2, *shape))
                common_relative = bank('common_relative', (ts, 2, *shape[1:]))
        for ti, tau in enumerate(self.tau):
            previews = []
            if need_gradients:
                mean = np.zeros(shape, dtype=np.float64)
                energy = np.zeros(shape[1:], dtype=np.float64)
                cr = np.zeros((2, *shape[1:]), dtype=np.float64)
            for start in range(0, n, self.args.batch_size):
                end = min(n, start + self.args.batch_size)
                y = self._synthesize(anchor[None] + float(tau) * delta[start:end])
                score, lp, derivatives = _cone_probe(
                    self.trainer, self.recognizer, x.expand(len(y), *shape), y, head,
                    direction, self.args.probe, baseline, need_gradients)
                logp[ti, start:end], scores[ti, start:end] = lp.cpu(), score.cpu()
                for j in range(start, min(end, 2)):
                    previews.append(self._preview(y[j-start:j-start+1]))
                if not need_gradients:
                    continue
                gx, gy = [v.cpu().numpy() for v in derivatives]
                native = y.cpu().numpy()
                rows[start:end] = gy.reshape(len(y), -1)
                mean += gy.sum(0, dtype=np.float64)
                energy += np.square(gy, dtype=np.float64).sum((0, 1))
                if joint_rows is not None:
                    joint_rows[start:end] = np.concatenate([gx.reshape(len(y), -1), gy.reshape(len(y), -1)], 1)
                    cr += np.stack([np.square((gx + gy) / np.sqrt(2)).sum((0, 1)),
                                    np.square((gy - gx) / np.sqrt(2)).sum((0, 1))])
                if ti:
                    attribution[start:end] += _cone_integral(previous_g[start:end], gy,
                                                            previous_y[start:end], native)
                if ti % self.args.quadrature_refinement == 0:
                    if ti:
                        coarse_attribution[start:end] += _cone_integral(
                            coarse_g[start:end], gy, coarse_y[start:end], native)
                    coarse_g[start:end], coarse_y[start:end] = gy, native
                previous_g[start:end], previous_y[start:end] = gy, native
            if need_gradients:
                stat = self._spectrum(rows, shape, previous_modes)
                result_modes[ti] = stat.pop('modes')
                previous_modes = result_modes[ti]
                result_mean[ti], result_energy[ti] = mean / n, energy / n
                result_attr[ti] = attribution.mean(0)
                result_std[ti] = attribution.std(0)
                totals[ti] = attribution.sum((1, 2), dtype=np.float64)
                errors[ti] = totals[ti] - (scores[ti] - scores[0])
                if ti % self.args.quadrature_refinement == 0:
                    coarse_errors[ti] = coarse_attribution.sum((1, 2), dtype=np.float64) - (scores[ti] - scores[0])
                coherence = float(np.square(mean / n).sum() / stat['trace']) if stat['trace'] else None
                stat.update(mode=result_modes[ti, 0, 0], mean=result_mean[ti], energy=result_energy[ti],
                            attribution=result_attr[ti], attribution_std=result_std[ti], coherence=coherence)
                if joint_rows is not None:
                    joint = self._spectrum(joint_rows, shape, joint_modes[ti-1] if ti else None, components=2)
                    joint_modes[ti] = joint.pop('modes')
                    common_relative[ti] = cr / n
                    stat.update(joint=joint, common_relative=common_relative[ti])
            else:
                stat = {}
            central = self._synthesize(anchor[None] + float(tau) * self.deterministic[None])
            frames.append(dict(stat, tau=float(tau), central=self._preview(central), examples=previews))
        result = dict(frames=frames, logp=logp, scores=scores, diagonal_logits=diagonal.cpu().numpy(),
                      diagonal_probabilities=baseline.exp().cpu().numpy(), anchor=self._preview(x),
                      native_shape=shape)
        if need_gradients:
            result.update(modes=result_modes, mean=result_mean, energy=result_energy,
                          attribution=result_attr, attribution_std=result_std,
                          totals=totals, errors=errors, coarse_errors=coarse_errors)
            if joint_rows is not None:
                result.update(joint_modes=joint_modes, common_relative=common_relative)
        return result

    def _render_path(self, data, destination, producer, head, title, evidence=False):
        frames, tau = data['frames'], self.tau
        keys = ('attribution', 'attribution_std') if evidence else ('energy', 'mode', 'mean')
        scales = {key: max(max(float(np.quantile(abs(f[key]), .995)) for f in frames), 1e-12)
                  for key in keys}
        units = 'centered-logit change' if self.args.probe == 'centered-logit' else 'log-probability change (nats)'
        probabilities = np.exp(data['logp'])

        def draw(fig, i):
            frame = frames[i]
            axes = fig.subplots(2, 4).ravel()
            fig.suptitle(f'{title} | τ={tau[i]:.4f} | {self.args.probe}\n'
                         f'fixed anchor; {self.args.noise_rays} Gaussian rays only; native {data["native_shape"]}')
            _cone_show(axes[0], data['anchor'], 'Fixed anchor (decoded preview)')
            _cone_show(axes[1], frame['central'], 'Deterministic central path (reference only)')
            _cone_show(axes[2], np.concatenate(frame['examples'], axis=1), 'Fixed noisy ray IDs: ' + ', '.join(map(str, range(len(frame['examples'])))))
            if evidence:
                signed, options = _signed(frame['attribution'], scales['attribution'])
                _cone_show(axes[3], signed, f'Mean cumulative attribution\n{units}; native locations', **options)
                _cone_show(axes[4], frame['attribution_std'], 'Across-ray attribution standard deviation',
                           cmap='magma', vmin=0, vmax=scales['attribution_std'])
                axes[5].plot(tau, (data['scores'] - data['scores'][0]).mean(1), label='Measured score change')
                axes[5].plot(tau, data['totals'].mean(1), '--', label='Attribution total')
                axes[5].set_ylabel(units)
                axes[5].legend(fontsize=7)
                axes[6].plot(tau, np.abs(data['errors']).mean(1), label='Mean |per-ray residual|')
                axes[6].plot(tau, np.abs(data['errors']).max(1), label='Max |per-ray residual|')
                nodes = np.arange(0, len(tau), self.args.quadrature_refinement)
                axes[6].plot(tau[nodes], np.abs(data['coarse_errors'][nodes]).mean(1), ':', label='Coarse mean |residual|')
                axes[6].legend(fontsize=7)
                axes[6].set_title('Sampled curved-path quadrature completeness', fontsize=9)
                axes[7].hist(data['errors'][i], bins=min(12, self.args.noise_rays))
                limit = max(float(abs(data['errors']).max()), 1e-12)
                axes[7].set(xlim=(-limit, limit), ylim=(0, self.args.noise_rays), title='Per-ray signed residuals')
            else:
                _cone_show(axes[3], frame['energy'], 'Conditional endpoint energy (native locations)',
                           cmap='magma', vmin=0, vmax=scales['energy'])
                for axis, key, name in ((axes[4], 'mode', 'Leading endpoint mode'),
                                        (axes[5], 'mean', 'Mean endpoint derivative')):
                    signed, options = _cone_signed(frame[key], scales[key])
                    _cone_show(axis, signed, name + '\nnon-RGB channels shown separately', **options)
                for selected in dict.fromkeys((producer, head)):
                    values = probabilities[:, :, selected]
                    line, = axes[6].plot(tau, values.mean(1), label=f'p{selected}')
                    low, high = np.quantile(values, [.1, .9], axis=1)
                    axes[6].fill_between(tau, low, high, color=line.get_color(), alpha=.2)
                axes[6].set(ylim=(0, 1), title='Recognition: mean and 10–90% rays')
                axes[6].legend()
                axes[7].bar(np.arange(len(frame['values'])) + 1, frame['values'])
                axes[7].set_ylim(0, max(max(f['values'][0] for f in frames), 1e-12))
                coherence = 'undefined (zero trace)' if frame['coherence'] is None else f'{frame["coherence"]:.3f}'
                axes[7].set_title(f'λ; shown trace={np.nansum(frame["explained"]):.1%}\n'
                                  f'Coherence={coherence}; ray split={frame["split"]:.3f}\n'
                                  f'Modes unstable near ties; gap={frame["eigengap"]}', fontsize=8)
            for axis in (axes[5], axes[6]) if evidence else (axes[6],):
                axis.axvline(tau[i], color='gray', linestyle=':')
                axis.set_xlabel('τ')
        _cone_movie(destination, len(tau), self.args.fps, draw)

    def _confidence(self, anchor, delta, path, producer, direction, stem, metadata):
        probabilities = np.exp(path['logp'][:, :, producer])
        endpoint = probabilities[-1]
        accepted = (endpoint >= self.args.success_threshold if self.args.success_threshold is not None
                    else np.ones(len(delta), dtype=bool))
        x = self._synthesize(anchor[None])
        shape = tuple(x.shape[1:])
        baseline = torch.from_numpy(path['diagonal_logits']).to(x.device).log_softmax(-1)
        frames, old_modes = [], None
        with tempfile.TemporaryDirectory(dir=self.output, prefix='.confidence-') as temporary:
            rows = np.lib.format.open_memmap(Path(temporary) / 'normals.tmp', mode='w+', dtype='float32',
                                            shape=(len(delta), int(np.prod(shape))))
            for threshold in self.args.confidence_thresholds:
                crossings = np.full(len(delta), np.nan)
                brackets = np.full((len(delta), 2), np.nan)
                valid_ids, previews, undefined = [], [], 0
                eligible = (path['diagonal_probabilities'][producer] < threshold and
                            (self.args.success_threshold is None or threshold < self.args.success_threshold))
                if eligible:
                    for ray in np.flatnonzero(accepted):
                        candidates = np.flatnonzero((probabilities[:-1, ray] < threshold) &
                                                   (probabilities[1:, ray] >= threshold))
                        if not len(candidates):
                            continue
                        index = candidates[0]
                        low, high = self.tau[index:index+2]
                        for _ in range(self.args.crossing_refinements):
                            mid = float((low + high) / 2)
                            y = self._synthesize(anchor[None] + mid * delta[ray:ray+1])
                            _, logp, _ = _cone_probe(self.trainer, self.recognizer, x, y, producer,
                                                    direction, 'log-evidence', baseline, False)
                            if float(logp[0, producer].exp()) >= threshold:
                                high = mid
                            else:
                                low = mid
                        location = float((low + high) / 2)
                        crossings[ray], brackets[ray] = location, [low, high]
                        y = self._synthesize(anchor[None] + location * delta[ray:ray+1])
                        _, _, (_, gy) = _cone_probe(self.trainer, self.recognizer, x, y, producer,
                                                    direction, 'log-evidence', baseline)
                        normal = gy[0].cpu().numpy().reshape(-1)
                        norm = float(np.linalg.norm(normal))
                        if not np.isfinite(norm) or norm <= self.args.normal_epsilon:
                            undefined += 1
                            continue
                        rows[len(valid_ids)] = normal / norm
                        valid_ids.append(int(ray))
                        if len(previews) < 2:
                            previews.append(self._preview(y))
                if valid_ids:
                    stat = self._spectrum(rows[:len(valid_ids)], shape, old_modes)
                    old_modes = stat['modes']
                    energy = np.zeros(shape[1:], dtype=np.float64)
                    for row in rows[:len(valid_ids)]:
                        energy += np.square(row.reshape(shape)).sum(0) / len(valid_ids)
                    mode = stat['modes'][0, 0]
                else:
                    stat = dict(values=np.array([]), explained=np.array([]), residuals=np.array([]),
                                split=float('nan'), trace=0., eigengap=None, method='no valid normals')
                    energy, mode = np.zeros(shape[1:]), np.zeros(shape)
                frames.append(dict(stat, threshold=threshold, energy=energy, mode=mode, crossings=crossings,
                                   brackets=brackets, valid_ids=valid_ids, undefined=undefined,
                                   previews=previews, eligible=bool(eligible)))
        scale = max(max(float(f['energy'].max()) for f in frames), 1e-12)
        signed_scale = max(max(float(abs(f['mode']).max()) for f in frames), 1e-12)

        def draw(fig, i):
            f = frames[i]
            axes = fig.subplots(2, 3).ravel()
            fig.suptitle(f'{metadata["label"]} | confidence surface η={f["threshold"]:g}\n'
                         f'Endpoint acceptance={accepted.mean():.1%}; valid normals={len(f["valid_ids"])}/{accepted.sum()}; '
                         f'undefined={f["undefined"]}; eligible threshold={f["eligible"]}')
            _cone_show(axes[0], path['anchor'], 'Fixed anchor (decoded preview)')
            if f['previews']:
                _cone_show(axes[1], np.concatenate(f['previews'], axis=1), f'Crossing ray IDs {f["valid_ids"][:2]}')
            else:
                axes[1].text(.1, .5, 'No valid resolved crossings')
                axes[1].set_axis_off()
            _cone_show(axes[2], f['energy'], 'Unit-normal endpoint AGOP energy', cmap='magma', vmin=0, vmax=scale)
            signed, options = _cone_signed(f['mode'], signed_scale)
            _cone_show(axes[3], signed, 'Leading native normal mode; channels separate if non-RGB', **options)
            axes[4].hist(f['crossings'][np.isfinite(f['crossings'])], bins=np.linspace(0, 1, 11))
            axes[4].set(xlim=(0, 1), ylim=(0, len(delta)), xlabel='τ crossing', title='Earliest grid-resolved bracket, refined')
            axes[5].bar(np.arange(len(f['values'])) + 1, f['values'])
            axes[5].set(ylim=(0, 1), title=f'Spectrum; trace={f["trace"]:.3g}')
        destination = self.output / f'{stem}_confidence.webm'
        if self.args.overwrite or not destination.exists():
            _cone_movie(destination, len(frames), self.args.fps, draw)
        if self.args.save_data:
            for index, frame in enumerate(frames):
                np.savez_compressed(self.output / f'{stem}_confidence_{index:02d}.npz',
                                    **{key: frame[key] for key in ('energy', 'mode', 'values', 'residuals', 'explained',
                                                                   'crossings', 'brackets', 'valid_ids', 'undefined')},
                                    accepted=accepted, threshold=frame['threshold'],
                                    modes=frame.get('modes', np.empty((0, 1, *shape))))
            (self.output / f'{stem}_confidence.json').write_text(json.dumps(dict(
                metadata, probe='log-probability of producer', thresholds=self.args.confidence_thresholds,
                accepted_fraction=float(accepted.mean()), surface_population='accepted rays with resolved crossings and defined normals',
                crossing_policy='earliest upward bracket on sampled grid; bisection may miss unresolved sub-grid oscillations',
                coverage=[len(f['valid_ids']) for f in frames], undefined=[f['undefined'] for f in frames]), indent=2))

    def _response(self, responses, evidence, anchor_preview, stem, metadata):
        scale = max(float(np.quantile(abs(evidence), .995)), 1e-12)

        def draw(fig, i):
            axes = fig.subplots(1, 3)
            fig.suptitle(f'{metadata["label"]} | τ={self.tau[i]:.4f}\n'
                         'Within-anchor producer subset; probabilities normalized over ALL recognizer classes')
            _cone_show(axes[0], anchor_preview, 'Fixed anchor (decoded preview)')
            for axis, matrix, title, options in (
                    (axes[1], responses[i], 'Mean probabilities', dict(vmin=0, vmax=1, cmap='viridis')),
                    (axes[2], evidence[i], 'Mean log evidence (nats)', dict(vmin=-scale, vmax=scale, cmap='seismic'))):
                artist = axis.imshow(matrix, aspect='auto', **options)
                axis.set(title=title, xlabel='Predicted class (full normalization)', ylabel='Producer')
                axis.set_yticks(range(len(self.args.producers)), self.args.producers)
                fig.colorbar(artist, ax=axis, fraction=.045)
        destination = self.output / f'{stem}_response.webm'
        if self.args.overwrite or not destination.exists():
            _cone_movie(destination, len(self.tau), self.args.fps, draw, (14, 5))
        if self.args.save_data:
            np.savez_compressed(self.output / f'{stem}_response.npz', tau=self.tau,
                                probabilities=responses, log_evidence=evidence, producers=self.args.producers)
            (self.output / f'{stem}_response.json').write_text(json.dumps(metadata, indent=2))

    def run(self):
        from .TraversalPDE import gaussian_cone_noise

        figures = set(self.args.cone_figures)
        need_gradients = bool(figures & {'sensitivity', 'evidence'})
        with _fixed_rng(self.validation.seed, self.trainer.device), _evaluation(
                self.generator, self.traversal, self.recognizer):
            for ai, position in enumerate(self.validation.positions[:self.args.positions]):
                anchor = position.to(self.trainer.device).detach()
                self.nuisance_seed = self.validation.seed + 104729 * (ai + 1)
                for direction in self.directions:
                    with torch.no_grad(), self.trainer.fp32_context():
                        origin, displacement = self.traversal.inference(
                            anchor[None], dt=anchor.new_full((1, 1), self.dt), direction=direction)
                    origins, displacements = origin[0].detach(), displacement[0].detach()
                    # Current inference returns the original uncentered validation anchor.
                    if not torch.equal(origins, anchor[None].expand_as(origins)):
                        raise ValueError('Traversal inference changed its origin; cannot label producers with one shared anchor.')
                    metadata = dict(self.identity, anchor_index=ai, anchor_hash=self._hash_tensor(anchor),
                                    orientation=int(direction), score_multiplier=int(direction),
                                    nuisance_seed=self.nuisance_seed, tau_weights=self.weights.tolist(),
                                    coarse_tau=self.coarse_tau.tolist(), baseline_derivatives='diagonal held fixed',
                                    coordinates='native recognizer inputs; decoded previews are not attribution coordinates',
                                    spectral_population='uncentered second-endpoint gradients, Gaussian rays only at fixed tau',
                                    split_policy='independent noise ray halves at each tau',
                                    path_rule='one full deterministic inference; fixed isotropic Gaussian displacement; linear latent interpolation',
                                    integral='per-ray sampled curved image-path trapezoids; native-channel contraction',
                                    units='centered logits' if self.args.probe == 'centered-logit' else 'nats',
                                    label=f'Anchor {ai}; outward direction {direction:+d}')
                    identifier = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()[:12]
                    stem = f'cone_anchor_{ai:04d}_dir_{direction:+d}_{identifier}'
                    responses, evidence = [], []
                    for producer in self.args.producers:
                        self.deterministic = displacements[producer]
                        noise_seed = self.validation.seed + 1000003 * (ai + 1) + 1009 * (producer + 1) + (direction < 0)
                        rng = torch.Generator(device='cpu').manual_seed(noise_seed)
                        gaussian = torch.randn(self.args.noise_rays, anchor.numel(),
                                               generator=rng, device='cpu',
                                               dtype=self.deterministic.dtype).to(anchor.device)
                        delta = (self.deterministic[None] + gaussian_cone_noise(
                            self.deterministic[None].expand_as(gaussian), self.aperture, gaussian)).detach()
                        producer_stem = f'{stem}_producer_{producer:03d}'
                        producer_meta = dict(metadata, producer=producer, noise_seed=int(noise_seed),
                                             ray_ids=list(range(self.args.noise_rays)),
                                             displacement_hash=self._hash_tensor(delta),
                                             label=f'{metadata["label"]}; producer {producer}')
                        if self.args.save_data:
                            np.savez_compressed(self.output / f'{producer_stem}_rays.npz', anchor=anchor.cpu(),
                                                deterministic=self.deterministic.cpu(), gaussian=gaussian.cpu(),
                                                displacements=delta.cpu(), tau=self.tau, tau_weights=self.weights,
                                                runtime_covariance=(self.geometry.cpu() if self.geometry is not None else np.empty(0)))
                        heads = list(dict.fromkeys(self.args.heads)) if need_gradients else [producer]
                        for hi, head in enumerate(heads):
                            head_stem = f'{producer_stem}_head_{head:03d}_{self.args.probe}'
                            with tempfile.TemporaryDirectory(dir=self.output, prefix='.cone-') as temporary:
                                path = self._path(anchor, delta, head, direction, Path(temporary), need_gradients)
                                if hi == 0:
                                    responses.append(np.exp(path['logp']).mean(1))
                                    baseline = path['diagonal_logits']
                                    baseline = baseline - np.logaddexp.reduce(baseline)
                                    evidence.append((path['logp'] - baseline).mean(1))
                                    anchor_preview = path['anchor']
                                    if 'confidence' in figures:
                                        self._confidence(anchor, delta, path, producer, direction, producer_stem, producer_meta)
                                title = f'{producer_meta["label"]}; probe head {head}'
                                for figure in ('sensitivity', 'evidence'):
                                    destination = self.output / f'{head_stem}_{figure}.webm'
                                    if figure in figures and (self.args.overwrite or not destination.exists()):
                                        self._render_path(path, destination, producer, head, title, figure == 'evidence')
                                if self.args.save_data:
                                    arrays = {key: value for key, value in path.items()
                                              if isinstance(value, np.ndarray) and key != 'anchor'}
                                    if need_gradients:
                                        for key in ('values', 'residuals', 'explained', 'trace', 'split', 'eigengap'):
                                            arrays[key] = np.asarray([f[key] if f[key] is not None else np.nan for f in path['frames']])
                                        arrays['coherence'] = np.asarray([f['coherence'] if f['coherence'] is not None else np.nan for f in path['frames']])
                                        if self.args.cone_joint:
                                            for key in ('values', 'residuals', 'explained', 'trace', 'split'):
                                                arrays['joint_' + key] = np.asarray([f['joint'][key] for f in path['frames']])
                                    np.savez_compressed(self.output / f'{head_stem}.npz', **arrays, tau=self.tau)
                                    (self.output / f'{head_stem}.json').write_text(json.dumps(dict(
                                        producer_meta, head=head, native_shape=list(path['native_shape']),
                                        diagonal_logits=path['diagonal_logits'].tolist(),
                                        diagonal_probabilities=path['diagonal_probabilities'].tolist(),
                                        mode_display='global sign aligned to preceding frame; near ties have unstable identities',
                                        spectral_methods=[f.get('method') for f in path['frames']]), indent=2, default=str))
                                del path
                    if 'response' in figures:
                        self._response(np.stack(responses, axis=1), np.stack(evidence, axis=1),
                                       anchor_preview, stem, metadata)


def main(argv=None):
    from .trainer import TraversalTrainer
    from .utils import choose_device

    parser = argparse.ArgumentParser(description='Post-hoc joint-AGOP analysis')
    parser.add_argument('--exp', required=True)
    parser.add_argument('--protocol', choices=('pooled', 'adjacent', 'diagonal', 'cone'), default='adjacent')
    parser.add_argument('--heads', type=int, nargs='+', required=True)
    parser.add_argument('--producers', type=int, nargs='+',
                        help='pair-producing heads (default: selected --heads)')
    parser.add_argument('--positions', '--explainability-positions', type=int, default=512)
    parser.add_argument('--batch-size', '--explainability-batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--exact-max-rows', type=int, default=512)
    parser.add_argument('--power-iterations', type=int, default=8)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-data', action='store_true',
                        help='persist pair-bank .npy and result .npz/.json files')
    parser.add_argument('--overwrite', action='store_true')
    cone = parser.add_argument_group('Within-anchor conditional cone figures (--protocol cone)')
    cone.add_argument('--cone-figures', nargs='+', default=['sensitivity', 'evidence', 'response'],
                      choices=('sensitivity', 'evidence', 'confidence', 'response'))
    cone.add_argument('--probe', choices=('centered-logit', 'log-evidence'), default='centered-logit')
    cone.add_argument('--noise-rays', type=int, default=256)
    cone.add_argument('--aperture', type=float, default=None,
                      help='isotropic cone aperture; defaults to traversal.noise_aperture')
    cone.add_argument('--tau-grid', type=float, nargs='+', default=np.linspace(0, 1, 17).tolist(),
                      help='strictly increasing integration nodes including 0 and 1')
    cone.add_argument('--quadrature-refinement', type=int, default=3,
                      help='subdivide each tau interval; report completeness against the coarse grid')
    cone.add_argument('--cone-directions', type=int, choices=(-1,1), nargs='+',
                      help='outward path orientations; defaults to validation directions; scores use this sign')
    cone.add_argument('--cone-joint', action='store_true',
                      help='also export joint modes and conditional common/relative energy with --save-data')
    cone.add_argument('--cone-covariance', type=Path,
                      help='optional .npy runtime SPD covariance (not stored in traversal checkpoints)')
    cone.add_argument('--confidence-thresholds', type=float, nargs='+', default=[.5, .75, .9])
    cone.add_argument('--success-threshold', type=float, default=None,
                      help='optional endpoint producer probability filter, fixed for the entire path')
    cone.add_argument('--crossing-refinements', type=int, default=8)
    cone.add_argument('--normal-epsilon', type=float, default=1e-12)
    args = parser.parse_args(argv)
    if min(args.positions, args.batch_size, args.modes, args.exact_max_rows,
           args.power_iterations) < 1 or (args.steps is not None and args.steps < 1):
        parser.error('counts and --steps must be positive')
    if args.protocol == 'cone':
        if args.steps is not None:
            parser.error('--protocol cone uses one full step specified by --dt, not rollout --steps')
        if args.noise_rays < 1 or args.quadrature_refinement < 1 or args.crossing_refinements < 0:
            parser.error('rays/refinement must be positive; crossing refinements must be nonnegative')
        grid = np.asarray(args.tau_grid)
        if len(grid) < 2 or not np.isfinite(grid).all() or grid[0] != 0 or grid[-1] != 1 or np.any(np.diff(grid) <= 0):
            parser.error('--tau-grid must increase strictly from 0 to 1')
        if args.aperture is not None and (not np.isfinite(args.aperture) or args.aperture < 0):
            parser.error('--aperture must be finite and nonnegative')
        if args.dt is not None and (not np.isfinite(args.dt) or args.dt <= 0):
            parser.error('--dt must be finite and positive; choose path signs with --cone-directions')
        if not np.isfinite(args.fps) or args.fps <= 0 or not np.isfinite(args.normal_epsilon) or args.normal_epsilon <= 0:
            parser.error('fps and normal epsilon must be finite and positive')
        thresholds = args.confidence_thresholds + ([] if args.success_threshold is None else [args.success_threshold])
        if any(not np.isfinite(t) or not 0 < t < 1 for t in thresholds):
            parser.error('confidence and success thresholds must lie strictly between 0 and 1')
        args.confidence_thresholds = sorted(set(args.confidence_thresholds))
        if args.cone_directions:
            args.cone_directions = list(dict.fromkeys(args.cone_directions))
        if args.cone_joint and not args.save_data:
            parser.error('--cone-joint requires --save-data (additional numerical export)')

    torch.set_float32_matmul_precision('high')
    device = choose_device()
    params, generator, traversal, recognizer = load_experiment(args.exp, device)
    if args.protocol == 'cone' and args.cone_covariance is not None:
        covariance = torch.from_numpy(np.load(args.cone_covariance, allow_pickle=False)).float()
        dimension = traversal.dim_z if hasattr(traversal, 'dim_z') else generator.dim_z
        if covariance.shape != (dimension, dimension) or not torch.isfinite(covariance).all() \
                or not torch.allclose(covariance, covariance.T):
            parser.error('--cone-covariance must be a finite symmetric latent-dimension square matrix')
        if int(torch.linalg.cholesky_ex(covariance).info) != 0:
            parser.error('--cone-covariance must be positive definite')
        traversal.cov_matrix = covariance.to(device)
    args.producers = list(dict.fromkeys(args.producers or args.heads))
    # Override dt to 2.0 for a full step
    args.dt = 2.0
    selected = set(args.heads + args.producers)
    if any(head < 0 or head >= traversal.num_traversal_sets for head in selected):
        parser.error(f'heads and producers must be in [0, {traversal.num_traversal_sets - 1}]')
    params.val_freq, params.val_num_positions, params.val_batch_size = (
        1, args.positions, args.batch_size)
    if args.seed is not None:
        params.val_seed = args.seed
    trainer = TraversalTrainer.__new__(TraversalTrainer)
    trainer.params, trainer.wip_dir, trainer.device = params, args.exp, device
    trainer.mixed_precision = getattr(params, 'mixed_precision', 'no') or 'no'
    trainer.amp_dtype = torch.bfloat16 if trainer.mixed_precision == 'bf16' else None
    trainer.amp_enabled = trainer.amp_dtype is not None and \
        torch.amp.autocast_mode.is_autocast_available(device.type)
    if hasattr(generator, 'set_mixed_precision'):
        generator.set_mixed_precision(trainer.mixed_precision)
    validation = TraversalValidation(trainer, generator)
    output = Path(args.exp) / 'validation' / 'explainability'
    output.mkdir(parents=True, exist_ok=True)
    if args.save_data:
        np.save(output / 'anchors.npy', validation.positions.numpy())
    if args.protocol == 'cone':
        ConditionalCone(trainer, generator, traversal, recognizer, validation, args).run()
        return
    analysis = PairAGOP(trainer, generator, traversal, recognizer, validation, args)
    try:
        for head in dict.fromkeys(args.heads):
            suffixes = ('webm', 'npz', 'json') if args.save_data else ('webm',)
            artifacts = [output / f'{analysis.stem(head)}.{suffix}' for suffix in suffixes]
            if args.overwrite or not all(path.exists() for path in artifacts):
                analysis.analyze(head)
    finally:
        analysis.close()


if __name__ == '__main__':
    main()
