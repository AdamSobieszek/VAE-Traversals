#!/usr/bin/env python3
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm


CSV_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def stream_rows(split):
    request = Request(CSV_URLS[split], headers={"User-Agent": "eqvae-openimages-downloader"})
    with urlopen(request, timeout=60) as response:
        lines = (line.decode("utf-8", "replace") for line in response)
        yield from csv.DictReader(lines)


def choose_url(row, prefer_thumbnail=True):
    if prefer_thumbnail and row.get("Thumbnail300KURL"):
        return row["Thumbnail300KURL"]
    return row.get("OriginalURL") or row.get("Thumbnail300KURL")


def collect_items(split, count, skip_ids, prefer_thumbnail=True):
    items = []
    for row in stream_rows(split):
        image_id = row.get("ImageID")
        image_url = choose_url(row, prefer_thumbnail=prefer_thumbnail)
        if not image_id or not image_url or image_id in skip_ids:
            continue
        items.append((image_id, image_url))
        if len(items) >= count:
            break
    return items


def download_one(item, output_dir, timeout):
    image_id, image_url = item
    output_path = output_dir / f"{image_id}.jpg"
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, image_id, "exists"

    tmp_path = output_path.with_suffix(".tmp")
    try:
        request = Request(image_url, headers={"User-Agent": "eqvae-openimages-downloader"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
        tmp_path.write_bytes(data)

        # Verify the file is an image and normalize it to RGB JPEG for the training loader.
        with Image.open(tmp_path) as image:
            image.convert("RGB").save(output_path, "JPEG", quality=95)
        tmp_path.unlink(missing_ok=True)
        return True, image_id, "downloaded"
    except (OSError, URLError, TimeoutError) as exc:
        tmp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return False, image_id, str(exc)


def download_split(split, output_dir, count, workers, timeout, prefer_thumbnail):
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_ids = set()

    while True:
        existing_ids = {path.stem for path in output_dir.glob("*.jpg")}
        remaining = max(count - len(existing_ids), 0)
        if remaining == 0:
            print(f"{split}: {count} images available in {output_dir}")
            return

        # Some OpenImages source URLs are stale; over-sample candidates and skip failures.
        skip_ids = existing_ids | failed_ids
        items = collect_items(split, max(remaining * 3, workers), skip_ids, prefer_thumbnail=prefer_thumbnail)
        if not items:
            raise RuntimeError(f"Could only download {len(existing_ids)} of {count} requested {split} images")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_one, item, output_dir, timeout) for item in items]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {split}"):
                ok, image_id, _message = future.result()
                if not ok:
                    failed_ids.add(image_id)


def main():
    parser = argparse.ArgumentParser(description="Download a small OpenImages subset for EQ-VAE fine-tuning.")
    parser.add_argument("--output-root", type=Path, default=Path("data/openimages"))
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--full-res", action="store_true", help="Use OriginalURL instead of Thumbnail300KURL when possible.")
    args = parser.parse_args()

    if args.train_count < 1 or args.val_count < 1:
        raise ValueError("Both --train-count and --val-count must be positive")

    os.makedirs(args.output_root, exist_ok=True)
    prefer_thumbnail = not args.full_res
    download_split("train", args.output_root / "train", args.train_count, args.workers, args.timeout, prefer_thumbnail)
    download_split("validation", args.output_root / "validation", args.val_count, args.workers, args.timeout, prefer_thumbnail)


if __name__ == "__main__":
    main()
