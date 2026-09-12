"""MPS SNGAN VJP: paired timing and peak live saved-storage (not allocator peak)."""
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_traversal_revision import paired, check


class SavedStorage:
    """Count distinct live autograd-saved storages, excluding model weights."""
    def __init__(self, modules):
        self.excluded = {p.untyped_storage().data_ptr() for m in modules for p in m.parameters()}
        self.live = {}
        self.bytes = self.peak = 0

    def pack(self, tensor):
        owner = self
        storage = tensor.untyped_storage()
        key, size = storage.data_ptr(), storage.nbytes()
        if key in self.excluded:
            return tensor
        if key not in self.live:
            self.live[key] = 0
            self.bytes += size
            self.peak = max(self.peak, self.bytes)
        self.live[key] += 1

        class Saved:
            def __init__(self):
                self.tensor = tensor

            def __del__(self):
                owner.live[key] -= 1
                if owner.live[key] == 0:
                    del owner.live[key]
                    owner.bytes -= size

        return Saved()

    @staticmethod
    def unpack(saved):
        return saved if isinstance(saved, torch.Tensor) else saved.tensor


def main():
    assert torch.backends.mps.is_available(), "MPS required; no CPU fallback"
    torch.set_default_device("mps")
    torch.manual_seed(81)
    from models.gan_load import build_sngan
    from lib.utils import FrozenGeneratorVJP

    generator = build_sngan(str(ROOT / "models/pretrained/generators/SNGAN_AnimeFaces/generator.pt"),
                            "SNGAN_AnimeFaces").eval().requires_grad_(False)
    z = torch.randn(192, 128, requires_grad=True)
    cotangent = torch.randn(192, 3, 64, 64) / (192 * 3 * 64 * 64)

    def run(chunk):
        image = FrozenGeneratorVJP.apply(z, generator, chunk) if chunk else generator(z)
        gradient = torch.autograd.grad(image, z, cotangent)[0]
        return image.detach(), gradient

    reference = run(0)
    for chunk in (0, 32, 64):
        gc.collect()
        storage = SavedStorage([generator])
        with torch.autograd.graph.saved_tensors_hooks(storage.pack, storage.unpack):
            output = run(chunk)
        check(output[0], reference[0], f"chunk={chunk} images", atol=2e-5)
        check(output[1], reference[1], f"chunk={chunk} latent VJP", atol=2e-7)
        print(f"chunk={chunk}: peak live saved storage {storage.peak / 2**20:.2f} MiB", flush=True)
        del output
    paired({str(chunk): lambda c=chunk: run(c) for chunk in (0, 32, 64)}, rounds=3)


if __name__ == "__main__":
    main()
