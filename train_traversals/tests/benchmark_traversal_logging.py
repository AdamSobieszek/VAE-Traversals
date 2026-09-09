"""MPS-only benchmarks for accelerator synchronization in training telemetry."""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.TraversalPDE import TraversalPDE
from lib.aux import ImageViz, module_grad_norm
from lib.recognizer import Recognizer


DEVICE = torch.device("mps")


def timed(name, fn, *, repeats=7):
    samples = []
    for _ in range(repeats):
        gc.collect()
        torch.mps.synchronize()
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append(1e3 * (time.perf_counter() - start))
    print(f"{name:34s} median={statistics.median(samples):8.3f} ms  samples={samples}")


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required; refusing to run on CPU")

    k = 32
    traversal = TraversalPDE(k, 20, 128, n_hidden=32, lambdas={"BB": 0.25}).to(DEVICE)
    recognizer = Recognizer("LeNet", k).to(DEVICE)
    for module in (traversal, recognizer):
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)

    timed("grad norm traversal", lambda: module_grad_norm(traversal))
    timed("grad norm recognizer", lambda: module_grad_norm(recognizer))

    potentials = torch.randn(6, k, 20, device=DEVICE)

    def legacy_hist_transfer():
        return [potentials[:, index].detach().float().cpu() for index in range(k)]

    def current_hist_transfer():
        tensor = potentials.detach().float().cpu()
        return [tensor[:, index] for index in range(k)]

    timed("hist transfer legacy (K copies)", legacy_hist_transfer)
    timed("hist transfer current (one copy)", current_hist_transfer)

    ref = torch.randn(1, 3, 64, 64, device=DEVICE)
    step1 = torch.randn(1, k, 3, 64, 64, device=DEVICE)
    step2 = torch.randn_like(step1)

    timed("image grids current", lambda: ImageViz.make_triplet_grids(step1, step2, ref, n_vis=k))


if __name__ == "__main__":
    main()
