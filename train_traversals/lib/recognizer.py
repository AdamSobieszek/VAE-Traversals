import torch
from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Recognizer(nn.Module):
    def __init__(self, recognizer_type, dim_index, channels=3, pool_size=1):
        super(Recognizer, self).__init__()
        self.recognizer_type = recognizer_type
        self.dim_index = dim_index
        self.channels = channels
        self.pool_size = pool_size
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

        else:
            raise ValueError(f"Unsupported recognizer type: {self.recognizer_type!r}")

    def forward(self, x1, x2, *, antisymmetric=False):
        if antisymmetric:
            return self.antisymmetric_pair_logits(x1, x2), None
        if self.pool_size > 1:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        if self.recognizer_type == 'LeNet':
            features = self.feature_extractor(torch.cat([x1, x2], dim=1))
            features = features.mean(dim=[-1, -2]).view(x1.shape[0], -1)
            logits = self.path_indices(features).view(x1.shape[0], -1)
            return logits, None
        elif self.recognizer_type == 'ResNet':
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([x1.shape[0], -1])
            logits = self.path_indices(features).view(x1.shape[0], -1)
            return logits, None
        raise ValueError(f"Unsupported recognizer type: {self.recognizer_type!r}")

    def antisymmetric_logits(self, center, endpoint):
        """Antisymmetrize the recognizer in whitened endpoint coordinates."""
        reflected = 2 * center - endpoint
        return self.antisymmetric_pair_logits(reflected, endpoint)

    def antisymmetric_pair_logits(self, first, second):
        return self(first, second)[0] - self(second, first)[0]


class AntisymmetricRecognizer(nn.Module):
    """Recognizer whose pair score is antisymmetric by construction.

    The LeNet variant is an odd network of the endpoint difference. The ResNet
    variant uses ``phi(x1) - phi(x2)`` with a shared image encoder and an odd,
    bias-free head. Consequently swapping the pair negates the logits exactly,
    so the whitened score needs only one pair evaluation rather than two. This
    is an additional architecture; the legacy :class:`Recognizer` and its
    checkpoint keys remain unchanged. In particular, the difference-only LeNet
    discards image-center context: it is not a drop-in numerical optimization
    of the legacy pair classifier and needs a separate training experiment.
    """

    def __init__(self, recognizer_type, dim_index, channels=3, pool_size=1):
        super().__init__()
        self.recognizer_type = recognizer_type
        self.dim_index = dim_index
        self.channels = channels
        self.pool_size = pool_size
        if self.pool_size > 1:
            self.avg_pool = nn.AvgPool2d(kernel_size=self.pool_size, stride=self.pool_size)

        if self.recognizer_type == "LeNet":
            self.lenet_width = 2
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(self.channels, 3 * self.lenet_width, kernel_size=5, bias=False),
                nn.GroupNorm(3, 3 * self.lenet_width, affine=False),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(3 * self.lenet_width, 8 * self.lenet_width, kernel_size=5, bias=False),
                nn.GroupNorm(8, 8 * self.lenet_width, affine=False),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(8 * self.lenet_width, 60 * self.lenet_width, kernel_size=5, bias=False),
                nn.GroupNorm(12, 60 * self.lenet_width, affine=False),
                nn.Tanh(),
            )
            feature_dim = 60 * self.lenet_width
        elif self.recognizer_type == "ResNet":
            self.feature_extractor = resnet18(weights=None)
            self.feature_extractor.conv1 = nn.Conv2d(
                self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(self.feature_extractor.conv1.weight, mode="fan_out", nonlinearity="relu")
            self.feature_extractor.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported recognizer type: {self.recognizer_type!r}")

        hidden = 42 * getattr(self, "lenet_width", 2)
        self.path_indices = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, self.dim_index, bias=False),
        )

    def _encode(self, x):
        features = self.feature_extractor(x)
        if self.recognizer_type == "LeNet":
            features = features.mean(dim=(-1, -2))
        return features.flatten(1)

    def forward(self, x1, x2, *, antisymmetric=False):
        if antisymmetric:
            return self.antisymmetric_pair_logits(x1, x2), None
        if self.pool_size > 1:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        if self.recognizer_type == "LeNet":
            return self.path_indices(self._encode(x1 - x2)), None
        # On MPS two N-sized convolutions are faster and use less peak memory
        # than one 2N-sized convolution for the training batch sizes used here.
        features1 = self._encode(x1)
        features2 = self._encode(x2)
        return self.path_indices(features1 - features2), None

    def antisymmetric_logits(self, center, endpoint):
        reflected = 2 * center - endpoint
        return self.antisymmetric_pair_logits(reflected, endpoint)

    def antisymmetric_pair_logits(self, first, second):
        # forward(y, x) == -forward(x, y), hence their difference is 2*forward(x, y).
        return 2 * self(first, second)[0]



# Legacy recognizer
class Reconstructor(nn.Module):
    def __init__(self, recognizer_type, dim_index, dim_time, channels=3, pool_size=1):
        super(Reconstructor, self).__init__()
        self.recognizer_type = recognizer_type
        self.dim_index = dim_index
        self.dim_time = dim_time
        self.channels = channels
        self.pool_size = pool_size
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


            # Define regression head (for predicting shift magnitudes)
            self.shift_magnitudes = nn.Sequential(
                nn.Linear(60 * self.lenet_width, 42 * self.lenet_width),
                nn.BatchNorm1d(42 * self.lenet_width),
                nn.ReLU(),
                nn.Linear(42 * self.lenet_width, 2)
            )

        # === ResNet ===
        elif self.recognizer_type == 'ResNet':
            # Define ResNet18 backbone for feature extraction
            self.features_extractor = resnet18(pretrained=False)
            # Modify ResNet18 first conv layer so as to get 2 rgb images (concatenated as a 6-channel tensor)
            self.features_extractor.conv1 = nn.Conv2d(in_channels=6,
                                                      out_channels=64,
                                                      kernel_size=(7, 7),
                                                      stride=(2, 2),
                                                      padding=(3, 3), bias=False)
            nn.init.kaiming_normal_(self.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.features = self.features_extractor.avgpool
            self.features.register_forward_hook(save_hook)

            # Define classification head (for predicting warping functions (paths) indices)
            self.path_indices = nn.Linear(512, self.dim_index)

            self.shift_magnitudes = nn.Linear(512, 2)

    def forward(self, x1, x2):
        if self.pool_size > 1:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        if self.recognizer_type == 'LeNet':
            features = self.feature_extractor(torch.cat([x1, x2], dim=1))
            features = features.mean(dim=[-1, -2]).view(x1.shape[0], -1)
            return self.path_indices(features).view(x1.shape[0], -1), self.shift_magnitudes(features).view(x1.shape[0], -1)
        elif self.recognizer_type == 'ResNet':
            self.features_extractor(torch.cat([x1, x2], dim=1))
            features = self.features.output.view([x1.shape[0], -1])
            return self.path_indices(features).view(x1.shape[0], -1), self.shift_magnitudes(features).view(x1.shape[0], -1)
