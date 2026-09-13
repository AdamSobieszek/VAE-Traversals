"""Compare the shared pipeline directly with the standalone gen_pairs script."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch

from lib.TraversalPDE import TraversalPDE
from lib.val_utils import pairs_main


class TinyGenerator(torch.nn.Module):
    dim_z = 4
    shift_in_w_space = False

    def forward(self, z):
        return z.tanh().reshape(-1, 1, 2, 2).repeat(1, 3, 1, 1)


@pytest.mark.parametrize('architecture,layout,filename', [
    ('euler', 'traversal_sets', 'checkpoint.pt'),
    ('potential_jet', 'support_sets', 'traversal_sets.pt'),
    ('midpoint', 'state_dict', 'traversal_sets-000010.pt'),
    ('euler', None, 'checkpoint.pt'),
])
@pytest.mark.parametrize('bidirectional', [False, True])
def test_pairs_match_original(tmp_path, monkeypatch, architecture, layout, filename, bidirectional):
    import lib.utils
    from models import gan_load

    monkeypatch.setattr(lib.utils, 'choose_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(gan_load, 'build_sngan', lambda **kwargs: TinyGenerator())
    params = dict(gan_type='SNGAN_MNIST', num_traversal_sets=2, num_traversal_timesteps=6,
                  traversal_architecture=architecture, z_truncation=.8)
    # Use the same argument resolver as both entry points.
    from argparse import Namespace
    from lib.TraversalPDE import traversal_options
    model = TraversalPDE(2, 6, 4, **traversal_options(Namespace(**params)))
    state = model.state_dict()
    for name in ('old', 'new'):
        root = tmp_path / name
        (root / 'models').mkdir(parents=True)
        (root / 'args.json').write_text(json.dumps(params))
        torch.save({layout: state} if layout else state, root / 'models' / filename)
    # Five pairs forces a partial second batch; non-default dt/resize/quality matter.
    options = ['--n-samples', '5', '--batch-size', '2', '--shift-leap', '.3',
               '--img-size', '7', '--img-quality', '81', '--gif',
               '--gif-size', '128', '--gif-fps', '12', '--seed', '123']
    if bidirectional:
        options.append('--bidirectional')
    script = Path(__file__).resolve().parents[1] / 'gen_pairs.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--exp', str(tmp_path / 'old'), *options])
    torch.manual_seed(123)
    runpy.run_path(str(script), run_name='__main__')
    torch.manual_seed(999)
    pairs_main(['--exp', str(tmp_path / 'new'), *options])
    old, new = (tmp_path / name / 'vp_pairs' for name in ('old', 'new'))
    assert sorted(p.name for p in old.iterdir()) == sorted(p.name for p in new.iterdir())
    for path in old.iterdir():
        assert path.read_bytes() == (new / path.name).read_bytes(), path.name
    assert np.load(new / 'labels.npy').shape == (5, 2)
    if bidirectional:
        directions = np.load(new / 'directions.npy')
        assert directions.shape == (5,)
        assert set(directions) == {-1, 1}
    else:
        assert not (new / 'directions.npy').exists()
    previous = {p.name: p.read_bytes() for p in new.iterdir()}
    torch.manual_seed(456)
    pairs_main(['--exp', str(tmp_path / 'new'), *options])
    assert previous == {p.name: p.read_bytes() for p in new.iterdir()}


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
@pytest.mark.parametrize('directions', [(1,), (1, -1)])
def test_validation_rollout_matches_previous(device, directions):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from lib.val_utils import TraversalValidation

    traversal = TraversalPDE(2, 8, 4, n_hidden=8).to(device).eval()
    starts = torch.randn(3, 4, device=device)
    steps, dt = 3, .2
    center = steps if len(directions) == 2 else 0
    current = starts[None, :, None].expand(len(directions), 3, 2, 4).reshape(-1, 2, 4).contiguous()
    signs = starts.new_tensor(directions).repeat_interleave(3)
    expected = starts.new_empty(len(directions) * steps + 1, 3, 2, 4)
    expected[center].copy_(current[:3])
    for index in range(steps):
        origin, delta = traversal.inference(current, dt=dt, direction=signs)
        following = (origin.detach() + delta.detach()).reshape(len(directions), 3, 2, 4)
        expected[center + index + 1].copy_(following[0])
        if center:
            expected[center - index - 1].copy_(following[1])
        current = following.reshape(-1, 2, 4)
    validation = SimpleNamespace(steps=steps, dt=dt, directions=directions, center=center)
    actual = TraversalValidation._rollout(validation, SimpleNamespace(fp32_context=nullcontext), traversal, starts)
    torch.testing.assert_close(actual, expected.reshape(-1, 6, 4), rtol=0, atol=0)


def test_negative_pairs_are_reordered(tmp_path, monkeypatch):
    from argparse import Namespace
    import lib.val_utils as validation

    class LinearTraversal:
        num_traversal_sets = 2
        num_traversal_timesteps = 6

        def inference(self, z, dt, direction):
            return z, (dt * direction)[..., None].expand_as(z)

    pairs = []
    monkeypatch.setattr(validation, 'sample_z',
                        lambda batch, *args: torch.zeros(batch, 4))
    monkeypatch.setattr(validation, 'save_pair_images',
                        lambda directory, first, second, *args: pairs.append((first.clone(), second.clone())))
    args = Namespace(exp=str(tmp_path), batch_size=2, n_samples=7, shift_leap=1.,
                     img_size=2, img_quality=75, seed=123, bidirectional=True)
    validation.generate_pairs(args, Namespace(), TinyGenerator(), LinearTraversal(), torch.device('cpu'))
    first, second = (torch.cat(images) for images in zip(*pairs))
    assert len(first) == 7
    assert torch.all(second > first)
    assert (first < 0).any() and (first == 0).any()  # both sampled directions
    assert torch.all((first == 0) | (second == 0))  # same original latent is retained
