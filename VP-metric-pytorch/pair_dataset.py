"""Image-pair datasets and an accuracy-preserving decoded-image cache."""

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def load_image(filename):
    """Decode an image with Pillow, matching the historical input pipeline."""
    with Image.open(filename) as image:
        return np.array(image.convert("RGB"))


def _image_tensor(image):
    """Return a CHW uint8 view; DataLoader performs the contiguous batch copy."""
    return torch.from_numpy(image).permute(2, 0, 1)


def downsample_chw_uint8(image, factor):
    """Max-pool a CHW uint8 image with non-overlapping ``factor`` x ``factor`` windows.

    Odd trailing rows/columns that do not fill a full window are dropped, matching
    ``torch.nn.functional.max_pool2d`` with ``kernel_size=stride=factor``.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("downsample factor must be >= 1")
    if factor == 1:
        return image
    channels, height, width = image.shape
    out_h = height // factor
    out_w = width // factor
    if out_h < 1 or out_w < 1:
        raise ValueError(
            "downsample factor {} leaves an empty spatial size for shape {}".format(
                factor, tuple(image.shape)
            )
        )
    cropped = np.asarray(image)[:, : out_h * factor, : out_w * factor]
    blocked = cropped.reshape(channels, out_h, factor, out_w, factor)
    return blocked.max(axis=(2, 4))


def _to_training_tensor(image_hwc, downsample=1):
    """Decode path: HWC uint8 -> optional max-pool -> CHW uint8 tensor."""
    image_tensor = _image_tensor(image_hwc)
    if downsample > 1:
        image_tensor = torch.from_numpy(
            downsample_chw_uint8(image_tensor.numpy(), downsample)
        )
    return image_tensor


class PairImageDataset(data.Dataset):
    """Load pair images without labels, primarily for cache construction."""

    def __init__(
        self,
        data_dir,
        length,
        image_tmpl="pair_{:06d}.jpg",
        downsample=1,
    ):
        self.data_dir = os.fspath(data_dir)
        self.length = int(length)
        self.image_tmpl = image_tmpl
        self.downsample = int(downsample)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.image_tmpl.format(idx))
        return _to_training_tensor(load_image(path), self.downsample)

    def __len__(self):
        return self.length


class PairDataset(data.Dataset):
    """Return pair images and precomputed integer class targets."""

    def __init__(
        self,
        data_dir,
        idx_list,
        image_tmpl="pair_{:06d}.jpg",
        transform=None,
        targets=None,
        image_cache=None,
        normalize_on_cpu=False,
        downsample=1,
    ):
        self.data_dir = os.fspath(data_dir)
        self.idx_list = np.asarray(idx_list, dtype=np.int64)
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.normalize_on_cpu = normalize_on_cpu
        self.image_cache = os.fspath(image_cache) if image_cache else None
        self.downsample = int(downsample)
        self._cached_images = None

        if targets is None:
            labels = np.load(os.path.join(self.data_dir, "labels.npy"))
            targets = np.argmax(np.abs(labels), axis=1)
        self.targets = np.asarray(targets, dtype=np.int64)

    def _cache(self):
        if self._cached_images is None:
            # Copy-on-write mode makes NumPy expose writable views to PyTorch
            # without copying any image pages. The dataset never mutates them.
            self._cached_images = np.load(self.image_cache, mmap_mode="c")
        return self._cached_images

    def __getitem__(self, hyper_idx):
        idx = int(self.idx_list[hyper_idx])
        if self.image_cache:
            # Cache already stores downsampled uint8 CHW pixels.
            image = self._cache()[idx]
            image_tensor = torch.from_numpy(image)
        else:
            path = os.path.join(self.data_dir, self.image_tmpl.format(idx))
            image_tensor = _to_training_tensor(load_image(path), self.downsample)
            image = image_tensor.numpy()

        if self.transform is not None:
            # Retain compatibility with callers of the original dataset class.
            image = np.transpose(image, (1, 2, 0))
            image_tensor = self.transform(image)
        elif self.normalize_on_cpu:
            # Preserve the historical ToTensor + Normalize path inside loader
            # workers, independent of the selected accelerator backend.
            image_tensor = image_tensor.float()
            image_tensor.div_(255.0).sub_(0.5).div_(0.5)
        return image_tensor, int(self.targets[idx])

    def __len__(self):
        return len(self.idx_list)


def _source_signature(data_dir, n_data, image_tmpl):
    """Fingerprint source file metadata so a regenerated dataset is recached."""
    digest = hashlib.blake2b(digest_size=20)
    for idx in range(n_data):
        stat = os.stat(os.path.join(data_dir, image_tmpl.format(idx)))
        digest.update(struct.pack("<QQ", stat.st_size, stat.st_mtime_ns))
    return digest.hexdigest()


def _metadata_path(cache_path):
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _cache_is_valid(cache_path, n_data, image_shape, source_signature, downsample):
    try:
        cached = np.load(cache_path, mmap_mode="r")
        metadata = json.loads(
            _metadata_path(cache_path).read_text(encoding="utf-8")
        )
        array_valid = (
            cached.dtype == np.uint8
            and cached.shape == (n_data,) + tuple(image_shape)
        )
        return (
            array_valid
            and metadata.get("schema_version") == 1
            and metadata.get("source_signature") == source_signature
            and int(metadata.get("downsample", 1)) == int(downsample)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def prepare_image_cache(
    data_dir,
    cache_path,
    n_data,
    workers,
    batch_size,
    rebuild=False,
    image_tmpl="pair_{:06d}.jpg",
    downsample=1,
):
    """Build or reuse a memory-mapped array of Pillow-decoded uint8 pixels.

    When ``downsample`` is greater than 1, non-overlapping max-pooling is applied
    before writing the cache so later epochs load the reduced resolution directly.

    Returns a metadata dictionary. If there is not enough free disk space, the
    returned ``path`` is ``None`` and callers can transparently use JPEG files.
    """
    cache_path = Path(cache_path)
    downsample = int(downsample)
    first_path = Path(data_dir) / image_tmpl.format(0)
    first = _to_training_tensor(load_image(first_path), downsample)
    image_shape = tuple(first.shape)
    required_bytes = int(n_data * first.numel())
    source_signature = _source_signature(data_dir, n_data, image_tmpl)

    if not rebuild and _cache_is_valid(
        cache_path, n_data, image_shape, source_signature, downsample
    ):
        return {
            "path": os.fspath(cache_path),
            "bytes": required_bytes,
            "image_shape": list(image_shape),
            "downsample": downsample,
            "built": False,
            "elapsed_seconds": 0.0,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_path.parent).free
    # Leave a little headroom so an almost-full filesystem is not exhausted.
    if free_bytes < required_bytes + max(required_bytes // 20, 64 * 1024**2):
        print(
            "Decoded image cache disabled: requires {:.2f} GiB but only "
            "{:.2f} GiB is free.".format(
                required_bytes / 1024**3, free_bytes / 1024**3
            )
        )
        return {
            "path": None,
            "bytes": required_bytes,
            "image_shape": list(image_shape),
            "downsample": downsample,
            "built": False,
            "elapsed_seconds": 0.0,
            "reason": "insufficient_disk_space",
        }

    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    metadata_path = _metadata_path(cache_path)
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    if metadata_temporary.exists():
        metadata_temporary.unlink()

    print(
        "Building decoded image cache ({:.2f} GiB, downsample={}) at {}".format(
            required_bytes / 1024**3, downsample, cache_path
        )
    )
    started = time.perf_counter()
    destination = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(n_data,) + image_shape,
    )
    source = PairImageDataset(
        data_dir, n_data, image_tmpl=image_tmpl, downsample=downsample
    )
    loader_kwargs = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "num_workers": int(workers),
        "pin_memory": False,
    }
    if workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = data.DataLoader(source, **loader_kwargs)

    offset = 0
    next_progress = 10
    try:
        for images in loader:
            stop = offset + len(images)
            if tuple(images.shape[1:]) != image_shape:
                raise ValueError(
                    "all pair images must have the same shape; expected {}, "
                    "received {}".format(image_shape, tuple(images.shape[1:]))
                )
            destination[offset:stop] = images.numpy()
            offset = stop
            progress = (100 * offset) // n_data
            if progress >= next_progress:
                print("  decoded image cache: {}%".format(progress))
                next_progress = progress + 10
        destination.flush()
        del destination
        destination = None
        with metadata_temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "source_signature": source_signature,
                    "samples": n_data,
                    "image_shape": list(image_shape),
                    "downsample": downsample,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        os.replace(temporary, cache_path)
        os.replace(metadata_temporary, metadata_path)
    except Exception:
        if destination is not None:
            del destination
        if temporary.exists():
            temporary.unlink()
        if metadata_temporary.exists():
            metadata_temporary.unlink()
        raise

    elapsed = time.perf_counter() - started
    print("Decoded image cache ready in {:.1f}s".format(elapsed))
    return {
        "path": os.fspath(cache_path),
        "bytes": required_bytes,
        "image_shape": list(image_shape),
        "downsample": downsample,
        "built": True,
        "elapsed_seconds": elapsed,
    }
