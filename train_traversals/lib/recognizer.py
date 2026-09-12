import torch
from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


def add_recognizer_arguments(parser):
    """Optional CAT controls shared by the existing training entry points."""
    group = parser.add_argument_group('CAT recognizer architecture')
    group.add_argument('--recognizer-preset', default='auto',
                       choices=('auto', 'legacy', '32', '64', '128', '256', '512', '1024'),
                       help='CAT capacity by generated resolution before pooling; legacy preserves the original defaults')
    for name in ('hidden-size', 'depth', 'num-heads', 'patch-size', 'num-scales'):
        group.add_argument(f'--recognizer-{name}', type=int, default=None,
                           help=f'CAT {name}; omitted uses the model default')
    group.add_argument('--recognizer-checkpointing', action='store_true',
                       help='recompute CAT blocks in backward to save activation memory')
    group.add_argument('--recognizer-unfused-attention', action='store_true',
                       help='use CAT explicit attention (e.g. for higher-order derivatives)')
    group.add_argument('--recognizer-pool-size', type=int, default=None,
                       help='override initial average pooling before the backbone/pyramid')


def recognizer_options(args, input_resolution=None):
    """Architecture-only kwargs; existing convolutional constructors stay unchanged."""
    if args.recognizer_type != 'CAT':
        return {}
    preset = getattr(args, 'recognizer_preset', 'auto')
    if preset == 'auto':
        # Choose the next resolution tier for non-power-of-two generator outputs.
        preset = (next((r for r in (32, 64, 128, 256, 512, 1024) if r >= input_resolution), 1024)
                  if input_resolution is not None else None)
    else:
        preset = None if preset == 'legacy' else int(preset)
    options = {name: getattr(args, f'recognizer_{name}', None)
               for name in ('hidden_size', 'depth', 'num_heads', 'patch_size', 'num_scales')}
    return {**{name: value for name, value in options.items() if value is not None},
            'preset': preset,
            'gradient_checkpointing': getattr(args, 'recognizer_checkpointing', False),
            'fused_attn': not getattr(args, 'recognizer_unfused_attention', False)}


class Recognizer(nn.Module):
    def __init__(self, recognizer_type, dim_index, channels=3, pool_size=1, **architecture_kwargs):
        super(Recognizer, self).__init__()
        self.recognizer_type = recognizer_type
        self.dim_index = dim_index
        self.channels = channels
        self.pool_size = pool_size
        if architecture_kwargs and recognizer_type != 'CAT':
            raise TypeError("Architecture kwargs are only supported for the CAT recognizer.")
        if self.pool_size > 1:
            self.avg_pool = nn.AvgPool2d(kernel_size=(self.pool_size, self.pool_size), stride=self.pool_size)

        # === LeNet ===
        if self.recognizer_type == 'LeNet':
            # Define LeNet backbone for feature extraction
            self.lenet_width = 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.channels * 2, 3 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(3 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(3 * self.lenet_width, 8 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(8 * self.lenet_width),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(8 * self.lenet_width, 60 * self.lenet_width, kernel_size=(5, 5)),
                nn.BatchNorm2d(60 * self.lenet_width),
                nn.ReLU()
            )

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, self.dim_index)
            )


        # === ResNet ===
        elif self.recognizer_type == 'ResNet':
            # Define ResNet18 backbone for feature extraction
            self.features_extractor = resnet18(pretrained=False)
            # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
            self.features_extractor.conv1 = nn.Conv2d(in_channels=self.channels * 2,
                                                      out_channels=64,
                                                      kernel_size=(7, 7),
                                                      stride=(2, 2),
                                                      padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.features = self.features_extractor.avgpool
            self.features.register_forward_hook(save_hook)

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Linear(512, self.dim_index)

        elif self.recognizer_type == 'CAT':
            from .cat_recognizer import CATPairEncoder
            self.pair_encoder = CATPairEncoder(dim_index, channels, **architecture_kwargs)

        else:
            raise ValueError(f"Unsupported recognizer type: {self.recognizer_type!r}")

    def forward(self, img0, img1, *, antisymmetric=False):
        if antisymmetric:
            return self.whitened_antisymmetric_logits(img0, img1), None
        if self.pool_size > 1:
            img0 = self.avg_pool(img0)
            img1 = self.avg_pool(img1)
        if self.recognizer_type == 'LeNet':
            features = self.feature_extractor(torch.cat([img0, img1], dim=1))
            features = features.mean(dim=[-1, -2]).view(img0.shape[0], -1)
            logits = self.path_indices(features).view(img0.shape[0], -1)
            return logits, None
        elif self.recognizer_type == 'ResNet':
            self.features_extractor(torch.cat([img0, img1], dim=1))
            features = self.features.output.view([img0.shape[0], -1])
            logits = self.path_indices(features).view(img0.shape[0], -1)
            return logits, None
        elif self.recognizer_type == 'CAT':
            return self.pair_encoder(torch.cat([img0, img1], dim=1)), None
        raise ValueError(f"Unsupported recognizer type: {self.recognizer_type!r}")

    def antisymmetric_logits(self, img0, img1):
        """Antisymmetrize the recognizer in original image coordinates. For these convolutional models, we find this works much worse than the whitened version."""
        if img0.shape[0] != img1.shape[0]:
            B = img0.shape[0]
            K = img1.shape[0] // B
            C, H, W = img0.shape[1:]
            img0 = img0.unsqueeze(1).expand(B, K, C, H, W).flatten(0, 1)
        return self(img0, img1)[0] - self(img1, img0)[0]

    def whitened_antisymmetric_logits(self, img0, img1):
        """Antisymmetrize the recognizer in whitened coordinates = (2*img0-img1, img1).
        We use 2*img0-img1 instead of img0-img1 to make it positive and for both inputs to be in the same range.
        Let delta = img1-img0, then the whitened coordinates are (img0-delta, img0+delta)."""

        B = img0.shape[0]

        # Views: also handles equal batch sizes, where K = 1.
        x = img0.unsqueeze(1)              # [B, 1, ...]
        y = img1.unflatten(0, (B, -1))      # [B, K, ...]

        # We calculate the 2*x-y term using lerp for faster computation.
        reflected0 = torch.lerp(x, y, -1.0).flatten(0, 1)
        reflected1 = torch.lerp(y, x, -1.0).flatten(0, 1)

        # Materialize the repetition only where the classifier needs it.
        img0_full = x.expand_as(y).flatten(0, 1)

        return self(reflected0, img1)[0] - self(reflected1, img0_full)[0]
