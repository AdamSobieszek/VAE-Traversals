"""Validation batching, deterministic evaluation and state restoration."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lib.val_utils import TraversalValidation, accuracy_curve
from lib.TraversalPDE import TraversalPDE


class Generator(torch.nn.Module):
    dim_z = 4

    def forward(self, z):
        assert not torch.is_grad_enabled()
        return z


class Recognizer(torch.nn.Module):
    pass


class Trainer:
    device = torch.device('cpu')
    tb_writer = None

    def __init__(self, path, batch=2):
        self.params = SimpleNamespace(val_freq=1, val_batch_size=batch,
            val_num_positions=5, val_seed=81, val_dt=0.05, num_traversal_timesteps=4,
            z_truncation=1., bidirectional=True)
        self.wip_dir = str(path)
        self.sizes = []

    def _generator_module(self, model):
        return model

    def fp32_context(self):
        return nullcontext()

    def _synthesize(self, model, z):
        self.sizes.append(len(z))
        return model(z)

    def _pair_logits(self, model, a, b):
        assert not model.training and not torch.is_grad_enabled()
        return b - a


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
@pytest.mark.parametrize('architecture', ['euler', 'potential_jet', 'midpoint'])
def test_validation_preserves_state_and_batches(tmp_path, architecture, device):
    trainer = Trainer(tmp_path)
    trainer.device = torch.device(device)
    torch.manual_seed(42)
    generator, recognizer = Generator().to(device).train(), Recognizer().to(device).train()
    traversal = TraversalPDE(4, 4, 4, n_hidden=8, architecture=architecture,
                            noise_aperture=10).to(device).train()
    traversal.F.eval()  # mixed child modes must also survive validation
    before = {key: value.clone() for key, value in traversal.state_dict().items()}
    rng = torch.get_rng_state().clone()
    validation = TraversalValidation(trainer, generator)
    assert torch.equal(rng, torch.get_rng_state())
    assert not (validation.positions == 0).all(dim=1).any()
    one = validation.run(trainer, generator, traversal, recognizer, 0)
    two = validation.run(trainer, generator, traversal, recognizer, 0)
    assert one == two
    assert len(one['step_accuracy']) == 3
    assert max(trainer.sizes) <= 2
    assert generator.training and recognizer.training and traversal.training
    assert not traversal.F.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in traversal.parameters())
    for key, value in traversal.state_dict().items():
        torch.testing.assert_close(before[key], value)
    # Images/classification must give the same results with a different chunk size.
    validation.batch_size = 7
    assert validation.run(trainer, generator, traversal, recognizer, 0) == one


def test_intervals_capture_both_axes():
    for data in (np.array([0., 1.])[:, None, None].repeat(8, axis=1),
                 np.array([0., 1.])[None, :, None].repeat(8, axis=0)):
        mean, low, high = accuracy_curve(data, 7)
        assert mean[0] == .5
        assert low[0] < mean[0] < high[0]
    mean, low, high = accuracy_curve(np.ones((1, 1, 3)), 7)
    np.testing.assert_array_equal(mean, low)
    np.testing.assert_array_equal(mean, high)


def test_disabled_and_exception_restore(tmp_path):
    trainer = Trainer(tmp_path)
    generator = Generator()
    trainer.params.val_freq = 0
    validation = TraversalValidation(trainer, generator)
    assert validation.positions is None
    trainer.params.val_freq = 1
    validation = TraversalValidation(trainer, generator)
    traversal = TraversalPDE(4, 4, 4, n_hidden=8).train()
    recognizer = Recognizer().train()
    rng = torch.get_rng_state().clone()
    def fail(*args):
        raise RuntimeError('classifier failed')
    trainer._pair_logits = fail
    with pytest.raises(RuntimeError, match='classifier failed'):
        validation.run(trainer, generator, traversal, recognizer, 0)
    assert traversal.training and recognizer.training and generator.training
    assert torch.equal(rng, torch.get_rng_state())


def test_known_labels_and_tensorboard(tmp_path):
    class Paths(torch.nn.Module):
        num_traversal_sets = 4

        def inference(self, z, dt, direction):
            assert not self.training
            sign = direction.reshape(-1, *([1] * (z.ndim - 1)))
            return z, torch.eye(4, device=z.device)[None].expand_as(z) * sign

    class Writer:
        def __init__(self):
            self.scalars, self.figures = [], []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def add_figure(self, tag, fig, step):
            self.figures.append((tag, step))

    trainer = Trainer(tmp_path, batch=3)  # chunks cross path and position boundaries
    trainer.tb_writer = Writer()
    generator = Generator()
    validation = TraversalValidation(trainer, generator)
    record = validation.run(trainer, generator, Paths(), Recognizer(), 5)
    assert record['step_accuracy'] == [1., 1., 1.]
    assert record['ci_low'] == record['ci_high'] == [1., 1., 1.]
    assert trainer.tb_writer.scalars == [('validation/accuracy', 1., 5)]
    assert trainer.tb_writer.figures == [('validation/accuracy_by_step', 5)]
    data = np.load(tmp_path / 'validation' / 'latest.npz')
    assert data['accuracy_by_position_path_step'].shape == (5, 4, 3)
