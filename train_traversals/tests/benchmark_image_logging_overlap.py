"""Real TensorBoard PNG encoding overlapped with MPS SNGAN synthesis.

No CPU model path; TensorBoard's normal image encoding necessarily uses host
data. Temporary event files are closed and removed when the benchmark exits.
"""
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_traversal_revision import paired


def main():
    assert torch.backends.mps.is_available(), "MPS required"
    torch.set_default_device("mps")
    from lib.aux import ImageLogger
    from models.gan_load import build_sngan
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    generator = build_sngan(str(ROOT / "models/pretrained/generators/SNGAN_AnimeFaces/generator.pt"),
                            "SNGAN_AnimeFaces").eval().requires_grad_(False)
    z = torch.randn(6, 128)
    with torch.no_grad():
        images = generator(torch.randn(64, 128)).unflatten(0, (2, 32))
    step = 0
    with tempfile.TemporaryDirectory(prefix="traversal-tb-") as directory:
        loggers = {name: ImageLogger(SummaryWriter(str(Path(directory) / name)), asynchronous=async_)
                   for name, async_ in (("synchronous", False), ("overlapped", True))}

        def run(logger, synthesize):
            nonlocal step
            step += 1
            logger.log_triplet("images", images[:1], images[1:], images[0, :1], step, n_vis=32)
            if synthesize:
                with torch.no_grad():
                    generator(z)
                torch.mps.synchronize()
            logger.flush()  # Count all encoding, not just submission latency.

        print("Complete image event pair:", flush=True)
        paired({name: lambda logger=logger: run(logger, False) for name, logger in loggers.items()}, rounds=5)
        print("Image event pair plus SNGAN B=6, including drain:", flush=True)
        paired({name: lambda logger=logger: run(logger, True) for name, logger in loggers.items()}, rounds=5)
        encoded = []
        for name, logger in loggers.items():
            logger.close()
            events = EventAccumulator(str(Path(directory) / name)).Reload()
            assert set(events.Tags()["images"]) == {"images/triplet", "images/diff_triplet_abs"}
            encoded.append([events.Images(tag)[-1].encoded_image_string for tag in sorted(events.Tags()["images"])])
        assert encoded[0] == encoded[1]
        print("Both image tags written and readable after close", flush=True)


if __name__ == "__main__":
    main()
