"""Differentiable and high-throughput FairFace/CelebA age raters.

`rank_interpretable_paths.py` only consumes precomputed `age.npy` /
`celeba_age.npy` scores. Those scores are produced in
`traverse_attribute_space.py` by two pretrained classifiers:

* FairFace ResNet-34 (9 age bins)
* Talk-to-Edit CelebA attribute ResNet-50, `Young` head (6 age bins)

This file downloads those two checkpoints, applies the same ImageNet-style
preprocessing with differentiable tensor ops, and returns soft ordinal age
scores so gradients flow from the ratings back to the input pixels.

Face-detector crops and argmax labels from the original script are omitted
because they are not differentiable. A blank `[0, 255]` RGB tensor is used as
the placeholder input, matching `PathImages` in `traverse_attribute_space.py`.
"""

import hashlib
import argparse
import csv
import json
import os
import os.path as osp
import platform
import re
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# Dropbox tarballs used by `download_models.py` / `lib/config.py`.
_FAIRFACE_URL = 'https://www.dropbox.com/s/xnxd2wnfgzt3og1/fairface.tar?dl=1'
_FAIRFACE_SHA256 = '0e78ff8b79612e52e226461fb67f6cff43cef0959d1ab2b520acdcc9105d065e'
_FAIRFACE_WEIGHTS = 'fairface/fairface_alldata_4race_20191111.pt'

_CELEBA_URL = 'https://www.dropbox.com/s/ulyu428dw620vhi/celeba_attributes.tar?dl=1'
_CELEBA_SHA256 = '45276f2df865112c7488fe128d8c79527da252aad30fc541417b9961dfdd9bbc'
_CELEBA_WEIGHTS = 'celeba_attributes/eval_predictor.pth.tar'

# Talk-to-Edit 5-attribute layout; only the `Young` head is used for age.
_CELEBA_ATTR_INFO = {
    '6': {'name': 'Bangs', 'value': [0, 1, 2, 3, 4, 5]},
    '16': {'name': 'Eyeglasses', 'value': [0, 1, 2, 3, 4, 5]},
    '25': {'name': 'No_Beard', 'value': [0, 1, 2, 3, 4, 5]},
    '32': {'name': 'Smiling', 'value': [0, 1, 2, 3, 4, 5]},
    '40': {'name': 'Young', 'value': [0, 1, 2, 3, 4, 5]},
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_DEFAULT_CACHE = osp.join(_THIS_DIR, 'models', 'pretrained')


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _download_tar(url, sha256sum, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    tmp_tar = osp.join(dest_dir, '.tmp.tar')
    if osp.isfile(tmp_tar):
        os.remove(tmp_tar)

    print('Downloading {}'.format(url))
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request) as response, open(tmp_tar, 'wb') as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    digest = hashlib.sha256()
    with open(tmp_tar, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4096), b''):
            digest.update(chunk)
    hexdigest = digest.hexdigest()
    if hexdigest != sha256sum:
        print('Warning: sha256 mismatch (got {}); extracting anyway if the tar is valid'.format(hexdigest))

    try:
        with tarfile.open(tmp_tar, mode='r') as archive:
            archive.extractall(dest_dir)
    except tarfile.TarError as exc:
        os.remove(tmp_tar)
        raise ValueError('Download from {} was not a valid tar archive'.format(url)) from exc
    os.remove(tmp_tar)


def ensure_age_checkpoints(cache_dir=None):
    """Download FairFace and CelebA attribute checkpoints if they are missing."""
    cache_dir = _DEFAULT_CACHE if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    fairface_path = osp.join(cache_dir, _FAIRFACE_WEIGHTS)
    if not osp.isfile(fairface_path):
        _download_tar(_FAIRFACE_URL, _FAIRFACE_SHA256, cache_dir)
    if not osp.isfile(fairface_path):
        raise FileNotFoundError('FairFace weights missing after download: {}'.format(fairface_path))

    celeba_path = osp.join(cache_dir, _CELEBA_WEIGHTS)
    if not osp.isfile(celeba_path):
        _download_tar(_CELEBA_URL, _CELEBA_SHA256, cache_dir)
    if not osp.isfile(celeba_path):
        raise FileNotFoundError('CelebA weights missing after download: {}'.format(celeba_path))

    return fairface_path, celeba_path


def _interpolate(images, size):
    kwargs = dict(size=size, mode='bilinear', align_corners=False)
    try:
        return F.interpolate(images, antialias=True, **kwargs)
    except TypeError:
        return F.interpolate(images, **kwargs)


def resize_short_edge(images, size):
    """Match torchvision `Resize(size)` on a tensor: scale the shorter side."""
    height, width = images.shape[-2:]
    if height < width:
        new_height = size
        new_width = int(round(width * size / float(height)))
    elif width < height:
        new_width = size
        new_height = int(round(height * size / float(width)))
    else:
        new_height = new_width = size
    return _interpolate(images, (new_height, new_width))


def center_crop(images, size):
    height, width = images.shape[-2:]
    if height < size or width < size:
        raise ValueError('Input {}x{} is smaller than crop size {}'.format(height, width, size))
    top = (height - size) // 2
    left = (width - size) // 2
    return images[..., top:top + size, left:left + size]


def imagenet_normalize(images):
    mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def to_unit_interval(images, image_range):
    """Map RGB images to `[0, 1]` without breaking autograd."""
    if image_range == '255':
        return images / 255.0
    if image_range == '01':
        return images
    if image_range == '-11':
        return images.mul(0.5).add(0.5)
    raise ValueError("image_range must be one of {'255', '01', '-11'}")


def soft_ordinal_score(logits):
    """Differentiable stand-in for `(argmax + 1) / n_bins`.

    The original traversal script stores `(argmax + max_softmax) / n_bins`.
    When the classifier is confident those two formulas agree; the expected
    bin index used here stays smooth for every input.
    """
    num_bins = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    bins = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    expected = (probs * bins).sum(dim=-1)
    return (expected + 1.0) / float(num_bins), probs


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FCBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.fc(x)))


class CelebAAttrResNet(nn.Module):
    """Talk-to-Edit CelebA attribute predictor (ResNet-50 + 5 classifier heads)."""

    def __init__(self, attr_info):
        super(CelebAAttrResNet, self).__init__()
        self.inplanes = 64
        self.attr_info = attr_info
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.stem = FCBlock(512 * Bottleneck.expansion, 512)
        for key, val in attr_info.items():
            name = 'classifier' + str(key).zfill(2) + val['name']
            setattr(self, name, nn.Sequential(FCBlock(512, 256), nn.Linear(256, len(val['value']))))

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.stem(self.avgpool(x).flatten(1))

    def forward_young(self, x):
        """Run only the age head, avoiding the other four attribute heads."""
        x = self.forward_features(x)
        return self.classifier40Young(x)

    def forward(self, x):
        x = self.forward_features(x)
        predictions = {}
        for key, val in self.attr_info.items():
            classifier = getattr(self, 'classifier' + str(key).zfill(2) + val['name'])
            predictions[val['name']] = classifier(x)
        return predictions


def _build_fairface(weights_path, device):
    try:
        model = torchvision.models.resnet34(weights=None)
    except TypeError:
        model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(_torch_load(weights_path, map_location=device))
    return model.to(device)


def _build_celeba(weights_path, device):
    model = CelebAAttrResNet(_CELEBA_ATTR_INFO)
    checkpoint = _torch_load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device)


def _freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class DifferentiableAgePredictors(nn.Module):
    """Frozen FairFace + CelebA age raters with differentiable preprocessing."""

    def __init__(self, device=None, cache_dir=None, image_range='255'):
        super(DifferentiableAgePredictors, self).__init__()
        if image_range not in ('255', '01', '-11'):
            raise ValueError("image_range must be one of {'255', '01', '-11'}")
        self.image_range = image_range
        self.device = torch.device('cpu' if device is None else device)
        fairface_path, celeba_path = ensure_age_checkpoints(cache_dir)
        self.fairface = _freeze(_build_fairface(fairface_path, self.device))
        self.celeba = _freeze(_build_celeba(celeba_path, self.device))

    def train(self, mode=True):
        # Keep BN/dropout in eval so ratings stay deterministic while remaining differentiable.
        del mode
        return super(DifferentiableAgePredictors, self).train(False)

    def _fairface_input(self, images):
        # Original: detect/crop face, /255, Resize(224), CenterCrop(224), ImageNet norm.
        # The crop is replaced by a differentiable resize + center crop of the full frame.
        unit = to_unit_interval(images, self.image_range)
        face_like = center_crop(resize_short_edge(unit, 256), 256)
        return imagenet_normalize(center_crop(resize_short_edge(face_like, 224), 224))

    def _celeba_input(self, images):
        # StyleGAN2 branch in traverse_attribute_space.py: [0, 255] -> [-1, 1], then
        # Resize(224), CenterCrop(224), ImageNet norm. ImageNet stats are applied to
        # the [-1, 1] tensor, matching the original script.
        unit = to_unit_interval(images, self.image_range)
        scaled = unit.mul(2.0).add(-1.0)
        return imagenet_normalize(center_crop(resize_short_edge(scaled, 224), 224))

    def fairface_age(self, images):
        logits = self.fairface(self._fairface_input(images))[:, 9:18]
        scores, probs = soft_ordinal_score(logits)
        return scores, probs

    def celeba_age(self, images):
        logits = self.celeba.forward_young(self._celeba_input(images))
        scores, probs = soft_ordinal_score(logits)
        return scores, probs

    def forward(self, images):
        fairface_age, fairface_probs = self.fairface_age(images)
        celeba_age, celeba_probs = self.celeba_age(images)
        return {
            'fairface_age': fairface_age,
            'fairface_age_probs': fairface_probs,
            'celeba_age': celeba_age,
            'celeba_age_probs': celeba_probs,
        }


def blank_image_tensor(batch_size=1, height=256, width=256, device=None, requires_grad=True):
    """Placeholder RGB tensor in the `[0, 255]` range used by `PathImages`."""
    return torch.zeros(batch_size, 3, height, width, device=device, requires_grad=requires_grad)


class FastAgeScorer(nn.Module):
    """Inference-only scorer for already resized `[0, 1]` RGB batches."""

    def __init__(self, predictors, celeba_stylegan_scaling=True):
        super(FastAgeScorer, self).__init__()
        self.fairface = predictors.fairface
        self.celeba = predictors.celeba
        self.celeba_stylegan_scaling = celeba_stylegan_scaling
        self.register_buffer(
            'mean',
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'std',
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images):
        fairface_input = (images - self.mean) / self.std
        celeba_images = (
            images.mul(2.0).sub(1.0)
            if self.celeba_stylegan_scaling
            else images
        )
        celeba_input = (celeba_images - self.mean) / self.std
        fairface_logits = self.fairface(fairface_input)[:, 9:18]
        celeba_logits = self.celeba.forward_young(celeba_input)
        fairface_scores, _ = soft_ordinal_score(fairface_logits)
        celeba_scores, _ = soft_ordinal_score(celeba_logits)
        return fairface_scores, celeba_scores


class FaceImageDataset(Dataset):
    """Decode and resize images on CPU before batched accelerator inference."""

    def __init__(
            self,
            image_paths,
            image_size=224,
            jpeg_draft=True,
            intermediate_size=None,
    ):
        self.image_paths = tuple(image_paths)
        self.image_size = image_size
        self.jpeg_draft = jpeg_draft
        self.intermediate_size = intermediate_size

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _resize_center_crop(image, size):
        width, height = image.size
        scale = size / float(min(width, height))
        resized = (
            max(size, int(round(width * scale))),
            max(size, int(round(height * scale))),
        )
        if image.size != resized:
            image = image.resize(resized, Image.Resampling.BILINEAR)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        return image.crop((left, top, left + size, top + size))

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            # JPEG draft decoding reduces 1024x1024 test images to 256x256 in
            # libjpeg before the final resize, greatly reducing CPU work.
            if self.jpeg_draft and image.format == 'JPEG':
                image.draft('RGB', (self.image_size, self.image_size))
            image = image.convert('RGB')
            if self.intermediate_size is not None:
                image = self._resize_center_crop(image, self.intermediate_size)
            image = self._resize_center_crop(image, self.image_size)
            tensor = torchvision.transforms.functional.pil_to_tensor(image)
        return tensor, path.name


def _natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', path.name)
    ]


def discover_images(input_dir, limit=None):
    """Find supported images in deterministic natural filename order."""
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError('Image directory not found: {}'.format(input_dir))
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    paths = sorted(
        (path for path in input_dir.iterdir() if path.suffix.lower() in extensions),
        key=_natural_sort_key,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError('No supported images found in {}'.format(input_dir))
    return paths


def resolve_device(device='auto'):
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def build_fast_scorer(
        device='auto',
        cache_dir=None,
        channels_last=True,
        compile_model=False,
        celeba_stylegan_scaling=True,
):
    """Load frozen weights and prepare the streamlined inference graph."""
    device = resolve_device(device)
    predictors = DifferentiableAgePredictors(
        device=device,
        cache_dir=cache_dir,
        image_range='01',
    )
    scorer = FastAgeScorer(
        predictors,
        celeba_stylegan_scaling=celeba_stylegan_scaling,
    ).to(device).eval()
    if channels_last:
        scorer = scorer.to(memory_format=torch.channels_last)
    if compile_model:
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is unavailable in this PyTorch build')
        scorer = torch.compile(scorer, mode='reduce-overhead', dynamic=True)
    return scorer, device


def rate_image_directory_fast(
        scorer,
        device,
        image_paths,
        batch_size=32,
        workers=4,
        amp=True,
        channels_last=True,
        jpeg_draft=True,
        intermediate_size=None,
        warmup=True,
):
    """Rate a directory in batches under `torch.inference_mode()`.

    Returns one prediction dictionary per image and timing statistics. Model
    loading and optional compilation happen in `build_fast_scorer`, outside
    the measured dataset pass.
    """
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if workers < 0:
        raise ValueError('workers cannot be negative')

    dataset = FaceImageDataset(
        image_paths,
        jpeg_draft=jpeg_draft,
        intermediate_size=intermediate_size,
    )
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': workers > 0,
    }
    if workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    amp_enabled = amp and device.type in ('cuda', 'mps')
    amp_dtype = torch.float16
    results = []

    with torch.inference_mode():
        if warmup:
            sample_size = batch_size
            sample = torch.zeros(
                sample_size,
                3,
                224,
                224,
                device=device,
                dtype=torch.float32,
            )
            if channels_last:
                sample = sample.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
            ):
                scorer(sample)
            _synchronize(device)
            del sample

        _synchronize(device)
        started = time.perf_counter()
        for uint8_images, filenames in loader:
            images = uint8_images.to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == 'cuda',
            ).div_(255.0)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
            ):
                fairface_scores, celeba_scores = scorer(images)
            fairface_scores = fairface_scores.float().cpu().tolist()
            celeba_scores = celeba_scores.float().cpu().tolist()
            results.extend(
                {
                    'filename': filename,
                    'fairface_age': fairface_age,
                    'celeba_age': celeba_age,
                }
                for filename, fairface_age, celeba_age in zip(
                    filenames,
                    fairface_scores,
                    celeba_scores,
                )
            )
        _synchronize(device)
        elapsed = time.perf_counter() - started

    return results, {
        'images': len(results),
        'inference_seconds': elapsed,
        'images_per_second': len(results) / elapsed,
        'milliseconds_per_image': elapsed * 1000.0 / len(results),
    }


def save_experiment(results, metadata, output_root):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    experiment_dir = Path(output_root).expanduser().resolve() / ('age_inference_' + timestamp)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = experiment_dir / 'predictions.csv'
    with predictions_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=('filename', 'fairface_age', 'celeba_age'),
        )
        writer.writeheader()
        writer.writerows(results)

    with (experiment_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)
    return experiment_dir


def differentiable_demo(device='auto'):
    """Keep the original autograd example available explicitly."""
    device = resolve_device(device)
    raters = DifferentiableAgePredictors(device=device, image_range='255')
    images = blank_image_tensor(device=device, requires_grad=True)
    outputs = raters(images)
    (outputs['fairface_age'].sum() + outputs['celeba_age'].sum()).backward()
    print('fairface_age:', outputs['fairface_age'].detach().cpu())
    print('celeba_age:', outputs['celeba_age'].detach().cpu())
    print('input grad norm:', float(images.grad.norm()))


def main():
    parser = argparse.ArgumentParser(
        description='Differentiable demo or fast batched age inference',
    )
    parser.add_argument(
        '--input-dir',
        default='/Users/adamsobieszek/PycharmProjects/psychGAN/omi/images',
    )
    parser.add_argument(
        '--output-root',
        default=osp.join(_THIS_DIR, 'experiments'),
    )
    parser.add_argument('--cache-dir', default=_DEFAULT_CACHE)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument('--amp', dest='amp', action='store_true')
    amp_group.add_argument('--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=None)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no-channels-last', action='store_true')
    parser.add_argument('--exact-jpeg-decode', action='store_true')
    parser.add_argument('--no-warmup', action='store_true')
    parser.add_argument('--differentiable-demo', action='store_true')
    args = parser.parse_args()

    if args.differentiable_demo:
        differentiable_demo(args.device)
        return

    paths = discover_images(args.input_dir, args.limit)
    setup_started = time.perf_counter()
    scorer, device = build_fast_scorer(
        device=args.device,
        cache_dir=args.cache_dir,
        channels_last=not args.no_channels_last,
        compile_model=args.compile,
    )
    # FP16 autocast helps CUDA, but benchmarks slower than FP32 on Apple MPS.
    amp = device.type == 'cuda' if args.amp is None else args.amp
    setup_seconds = time.perf_counter() - setup_started
    results, timing = rate_image_directory_fast(
        scorer=scorer,
        device=device,
        image_paths=paths,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=amp,
        channels_last=not args.no_channels_last,
        jpeg_draft=not args.exact_jpeg_decode,
        warmup=not args.no_warmup,
    )
    metadata = {
        'created_at': datetime.now().astimezone().isoformat(),
        'input_dir': str(Path(args.input_dir).expanduser().resolve()),
        'output_scale': (
            'soft ordinal age score in (0, 1]; '
            '(expected zero-based bin + 1) / number of bins'
        ),
        'device': str(device),
        'machine': platform.machine(),
        'torch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'amp': amp and device.type in ('cuda', 'mps'),
        'compile': args.compile,
        'channels_last': not args.no_channels_last,
        'jpeg_draft': not args.exact_jpeg_decode,
        'setup_seconds': setup_seconds,
        **timing,
    }
    experiment_dir = save_experiment(results, metadata, args.output_root)
    print('Rated {} images with both models in {:.3f}s ({:.2f} images/s)'.format(
        timing['images'],
        timing['inference_seconds'],
        timing['images_per_second'],
    ))
    print('Saved experiment:', experiment_dir)


if __name__ == '__main__':
    main()
