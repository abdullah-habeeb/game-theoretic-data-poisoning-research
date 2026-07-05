"""
models/resnet.py
================
ResNet-18 for CIFAR-10 classification.

WHY RESNET AND NOT JUST A BIGGER CNN?
  A simple CNN saturates quickly on CIFAR-10 (~85-90% ceiling). ResNet-18
  uses residual connections (skip connections) that allow much deeper
  networks to be trained effectively. With this architecture:
  - Clean CIFAR-10 accuracy: ~93-94%
  - Poisoned accuracy (under attack) shows a meaningful, measurable drop
  - This gives your results credibility — the attack on a strong model
    is far more convincing than on a toy CNN.

RESIDUAL CONNECTION:
  Instead of learning y = F(x), the layer learns y = F(x) + x.
  This identity shortcut allows gradients to flow freely through the
  network, enabling training of deep networks without vanishing gradients.

ARCHITECTURE:
  Input: [B, 3, 32, 32]  (CIFAR-10 images)
  Stem:  Conv(3→64, 3×3) → BN → ReLU
  Layer1: 2 × BasicBlock(64→64)
  Layer2: 2 × BasicBlock(64→128, stride=2)
  Layer3: 2 × BasicBlock(128→256, stride=2)
  Layer4: 2 × BasicBlock(256→512, stride=2)
  AvgPool → FC(512→10)

CIFAR-10 vs ImageNet ResNet:
  The standard ResNet-18 uses a 7×7 stem conv + MaxPool, which downsamples
  too aggressively for 32×32 CIFAR images. We use a 3×3 stem and no initial
  MaxPool — this is the standard CIFAR-10 ResNet modification used in papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet basic block: two 3×3 conv layers with a skip connection."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Shortcut projection: needed when spatial dims or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # ← residual connection
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 input, 3×3 stem, no MaxPool).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # CIFAR-10 stem: 3×3 conv (no MaxPool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Residual layers
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

        # Weight initialization (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, n_blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer feature embeddings (used by defenses)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)   # [B, 512]


def get_model(device: torch.device, dataset: str = "cifar10",
              arch: str = "resnet18") -> nn.Module:
    """
    Return the appropriate model for the given dataset and architecture.

    Args:
        device:  torch.device to move model to.
        dataset: 'cifar10' | 'cifar100' | 'mnist' | 'gtsrb'
        arch:    'resnet18' (default) | 'resnet50' (for scalability study)
    """
    nc_map = {"cifar10": 10, "cifar100": 100, "mnist": 10, "gtsrb": 43}
    nc = nc_map.get(dataset, 10)

    if dataset == "mnist":
        from models.cnn import MnistCNN
        model = MnistCNN(num_classes=nc)
    elif arch == "resnet50":
        model = ResNet50CIFAR(num_classes=nc)
    else:
        model = ResNet18(num_classes=nc)
    return model.to(device)


# ── ResNet-50 (Bottleneck blocks) for scalability study ──────────────────────

class Bottleneck(nn.Module):
    """ResNet Bottleneck: 1x1→3x3→1x1 convolutions. Expansion=4."""
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet50CIFAR(nn.Module):
    """ResNet-50 for CIFAR-style 32x32 inputs (~23.5M params)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem    = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1  = self._make(64,   64,  3, 1)
        self.layer2  = self._make(256,  128, 4, 2)
        self.layer3  = self._make(512,  256, 6, 2)
        self.layer4  = self._make(1024, 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)

    def _make(self, in_p, p, n, s):
        return nn.Sequential(Bottleneck(in_p, p, s),
                             *[Bottleneck(p*4, p) for _ in range(1, n)])

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).view(x.size(0), -1))

    def get_features(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.avgpool(x).view(x.size(0), -1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(device, "cifar10")
    print(f"ResNet-18 for CIFAR-10")
    print(f"Parameters: {model.count_parameters():,}")
    dummy = torch.zeros(4, 3, 32, 32, device=device)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")   # Expected: (4, 10)
    assert out.shape == (4, 10)
    print("✓ Forward pass OK")
