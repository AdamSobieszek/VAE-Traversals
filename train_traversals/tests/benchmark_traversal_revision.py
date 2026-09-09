"""Paired MPS correctness/performance checks against a pre-edit source snapshot.

Run with --baseline /tmp/traversal-baseline.YZjFvX [--full]. Models, derivative
checks and synthesis run exclusively on MPS. No automatic CPU fallback.
"""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_old(path, name):
    spec = importlib.util.spec_from_file_location('lib.baseline_' + name, path / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(functions, rounds=5, inner=1):
    for fn in functions.values():
        fn()
    results = {key: [] for key in functions}
    for i in range(rounds):
        for name in list(functions)[::1 if i % 2 == 0 else -1]:
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(inner):
                functions[name]()
            torch.mps.synchronize()
            results[name].append((time.perf_counter() - start) * 1000 / inner)
    print({key: {'median_ms': round(statistics.median(v), 3), 'samples_ms': [round(x, 3) for x in v]}
           for key, v in results.items()}, flush=True)


def check(actual, expected, label, atol=3e-5, rtol=3e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(label, 'max_abs_error', (actual - expected).abs().max().item(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), 'MPS required'
    torch.set_default_device('mps')
    torch.manual_seed(71)
    from lib.TraversalPDE import TraversalPDE, gaussian_cone_noise
    from lib.trainer_potential_nue import TrainerPotential
    from lib.recognizer import Recognizer
    from models.gan_load import build_sngan
    old_pde = load_old(args.baseline, 'TraversalPDE')
    old_trainer = load_old(args.baseline, 'trainer_potential_nue').TrainerPotential
    cfg = dict(num_traversal_sets=32, num_traversal_timesteps=20, traversal_vectors_dim=128,
               n_hidden=32, lambdas={'BB': .25, 'signed_g2orth': 1.})
    before, after = old_pde.TraversalPDE(**cfg), TraversalPDE(**cfg)
    after.load_state_dict(before.state_dict(), strict=True)
    z = torch.randn(6, 128)
    t = torch.tensor([[0], [2], [4], [8], [10], [17]])
    dt = torch.full((6, 1), .2)

    x = torch.randn(6, 32, 128, requires_grad=True)
    before.eval(); after.eval()
    y = before.F(x)
    g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    value, explicit = after.F.value_and_grad(x)
    check(value, y, 'potential')
    check(explicit, g, 'input gradient')
    probe = torch.randn_like(g)
    h0 = torch.autograd.grad((g * probe).sum(), x, create_graph=True)[0]
    h1 = torch.autograd.grad((explicit * probe).sum(), x, create_graph=True)[0]
    check(h1, h0, 'Hessian-vector product')
    for f, grad_value in ((before.F, g), (after.F, explicit)):
        f.zero_grad(set_to_none=True)
        grad_value.square().sum().backward()
    for (name, p), (_, q) in zip(before.F.named_parameters(), after.F.named_parameters()):
        if p.grad is not None or q.grad is not None:
            check(q.grad if q.grad is not None else torch.zeros_like(q),
                  p.grad if p.grad is not None else torch.zeros_like(p), 'mixed gradient ' + name)

    delta = torch.randn(6, 32, 128)
    noise = torch.randn_like(delta)
    def legacy_cone():
        norm2 = delta.square().sum(-1, keepdim=True)
        orth = noise - delta * ((noise * delta).sum(-1, keepdim=True) / norm2.clamp_min(1e-12))
        return orth / orth.norm(dim=-1, keepdim=True).clamp_min(1e-12) * (norm2.sqrt() / 5.)
    cone = gaussian_cone_noise(delta, gaussian=noise)
    check(cone, legacy_cone(), 'cone same Gaussian draw')
    check((cone * delta).sum(-1), torch.zeros(6, 32), 'cone orthogonality')
    check(cone.norm(dim=-1), .2 * delta.norm(dim=-1), 'cone aperture')
    paired({'cone old': legacy_cone, 'cone new': lambda: gaussian_cone_noise(delta, gaussian=noise)}, inner=100)

    def pde_run(model):
        model.zero_grad(set_to_none=True)
        result = model(z, t, dt)
        (result[3] + result[2].square().mean()).backward()
        return result
    result0, result1 = pde_run(before), pde_run(after)
    for i in range(5):
        check(result1[i], result0[i], 'rollout output ' + str(i), atol=2e-4)
    for (name, p), (_, q) in zip(before.named_parameters(), after.named_parameters()):
        if p.grad is not None:
            check(q.grad, p.grad, 'rollout gradient ' + name, atol=2e-4)
    paired({'rollout autograd': lambda: pde_run(before), 'rollout explicit': lambda: pde_run(after)}, inner=4)
    check(after.inference(z, dt=dt)[1], before.inference(z, dt=dt)[1], 'inference step')
    paired({'inference previous': lambda: before.inference(z, dt=dt),
            'inference revised': lambda: after.inference(z, dt=dt)}, inner=30)

    if not args.full:
        return
    generator = build_sngan(str(ROOT / 'models/pretrained/generators/SNGAN_AnimeFaces/generator.pt'),
                            'SNGAN_AnimeFaces').eval().requires_grad_(False)
    recognizer = Recognizer('LeNet', 32).train()
    initial_r = {k: v.detach().clone() for k, v in recognizer.state_dict().items()}
    def trainer(cls):
        obj = cls.__new__(cls)
        obj.params = argparse.Namespace(lambda_cls=1., lambda_pde=1.)
        obj.device = torch.device('mps')
        obj.use_cuda = obj.amp_enabled = False
        obj.cross_entropy = torch.nn.CrossEntropyLoss()
        return obj
    old, new = trainer(old_trainer), trainer(TrainerPotential)
    def run(tr, model):
        recognizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        return tr.loss_allK(model, generator, recognizer, z, t, dt, 2, need_images=True)
    out0 = run(old, before)
    sgrads = [p.grad.clone() if p.grad is not None else None for p in before.parameters()]
    rgrads = [p.grad.clone() if p.grad is not None else None for p in recognizer.parameters()]
    rstate = {k: v.detach().clone() for k, v in recognizer.state_dict().items()}
    recognizer.load_state_dict(initial_r)
    # Hold the rollout bit-for-bit fixed when testing trainer algebra: tiny
    # analytic-gradient roundoff can flip near-ties in R's MaxPool backward.
    out1 = run(new, before)
    for i in (1, 2, 4, 5, 6, 7):
        check(out1[i], out0[i], 'full output ' + str(i), atol=5e-4, rtol=1e-3)
    for (name, p), q in zip(recognizer.named_parameters(), rgrads):
        if q is not None:
            check(p.grad, q, 'recognizer gradient ' + name, atol=1e-3, rtol=1e-3)
    for (name, p), expected in zip(before.named_parameters(), sgrads):
        if expected is not None:
            check(p.grad, expected, 'full traversal gradient ' + name, atol=1e-5, rtol=1e-3)
    for name, value in recognizer.state_dict().items():
        if 'running' in name or 'tracked' in name:
            torch.testing.assert_close(value, rstate[name], atol=2e-5, rtol=1e-4)
    print('BatchNorm running states match', flush=True)
    del out0, out1, rgrads
    from benchmark_generator_vjp import SavedStorage
    for name, tr, model in [('previous', old, before), ('revised', new, after)]:
        saved = SavedStorage([generator, recognizer, model])
        with torch.autograd.graph.saved_tensors_hooks(saved.pack, saved.unpack):
            run(tr, model)
        print(name, 'peak live saved activation MiB', round(saved.peak / 2**20, 2), flush=True)
    paired({'full previous': lambda: run(old, before), 'full revised': lambda: run(new, after)}, rounds=3)


if __name__ == '__main__':
    main()
