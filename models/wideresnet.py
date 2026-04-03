"""
models/wideresnet.py
=====================
Wide Residual Network (WideResNet-28-10) for CIFAR-100.

PAPER: "Wide Residual Networks"
  Zagoruyko & Komodakis — BMVC 2016
  https://arxiv.org/abs/1605.07146

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHY WIDERESNET FOR CIFAR-100?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CIFAR-100 has 100 fine-grained classes that are significantly harder to
discriminate than CIFAR-10's 10 classes. A standard ResNet-18 achieves
~77–78% on CIFAR-100, which leaves very little headroom to observe
the effect of a poisoning attack (a 5% drop matters much more when
you start at 78% than at 94%).

WideResNet-28-10 (WRN-28-10) is the standard benchmark model for
CIFAR-100 adversarial robustness papers:
  - ResNet depth: 28 layers
  - Widening factor: 10 (channels × 10 vs standard ResNet)
  - Clean CIFAR-100 accuracy: ~80–81%
  - Used in: Madry et al. (2018), Hendrycks et al. (2019),
    Rice et al. (2020), and dozens of adversarial ML papers

ARCHITECTURE:
  Input: [B, 3, 32, 32]
  Stem:  Conv(3 → 16, 3×3)
  Group1: 4 × WideBlock(16 → 160, stride=1)
  Group2: 4 × WideBlock(160 → 320, stride=2)
  Group3: 4 → WideBlock(320 → 640, stride=2)
  BN → ReLU → AvgPool(8) → FC(640 → n_classes)

  Parameters: ~36.5M (depth=28, widen=10, classes=100)

WIDENING FACTOR:
  Standard ResNet-18 has 64 → 128 → 256 → 512 channels.
  WRN-28-10 has 160 → 320 → 640 channels.
  Wider channels increase representational capacity, which is
  critical for CIFAR-100's fine-grained discrimination task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBlock(nn.Module):
    """
    Wide residual block with dropout.

    Architecture:
      BN → ReLU → Conv(3×3) → BN → ReLU → Dropout → Conv(3×3) → + shortcut
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float = 0.3):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(p=dropout)

        # Shortcut: project if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    WideResNet-d-k: depth=d, widening factor=k.

    Standard configuration for CIFAR-100: WRN-28-10 with dropout=0.3.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout: float = 0.3,
        num_classes: int = 100,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must satisfy (depth-4) % 6 == 0."
        n_blocks = (depth - 4) // 6          # Blocks per group
        widths   = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.stem   = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)

        self.group1 = self._make_group(widths[0], widths[1], n_blocks, stride=1, dropout=dropout)
        self.group2 = self._make_group(widths[1], widths[2], n_blocks, stride=2, dropout=dropout)
        self.group3 = self._make_group(widths[2], widths[3], n_blocks, stride=2, dropout=dropout)

        self.bn_out = nn.BatchNorm2d(widths[3])
        self.fc     = nn.Linear(widths[3], num_classes)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_group(self, in_planes, out_planes, n_blocks, stride, dropout):
        blocks = [WideBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, n_blocks):
            blocks.append(WideBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*blocks)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings (used by Spectral/SEVER defenses)."""
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn_out(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.fc(feats)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── CIFAR-100 superclass mapping (for semantically meaningful attacks) ────────
CIFAR100_SUPERCLASSES = {
    "aquatic_mammals":     [4, 30, 55, 72, 95],
    "fish":                [1, 32, 67, 73, 91],
    "flowers":             [54, 62, 70, 82, 92],
    "food_containers":     [9, 10, 16, 28, 61],
    "fruit_and_veg":       [0, 51, 53, 57, 83],
    "household_elec":      [22, 39, 40, 86, 87],
    "household_furniture": [5, 20, 25, 84, 94],
    "insects":             [6, 7, 14, 18, 24],
    "large_carnivores":    [3, 42, 43, 88, 97],
    "large_man_made":      [12, 17, 37, 68, 76],
    "large_natural":       [23, 33, 49, 60, 71],
    "large_omnivores":     [15, 19, 21, 31, 38],
    "medium_mammals":      [34, 63, 64, 66, 75],
    "non_insect_inverts":  [26, 45, 77, 79, 99],
    "people":              [2, 11, 35, 46, 98],
    "reptiles":            [27, 29, 44, 78, 93],
    "small_mammals":       [36, 50, 65, 74, 80],
    "trees":               [47, 52, 56, 59, 96],
    "vehicles_1":          [8, 13, 48, 58, 90],
    "vehicles_2":          [41, 69, 81, 85, 89],
}

# Best attack pair: within same superclass (similar appearance → harder to detect)
# aquarium_fish (1) → flatfish (32): same superclass 'fish'
CIFAR100_DEFAULT_SRC = 1   # aquarium_fish
CIFAR100_DEFAULT_TGT = 32  # flatfish  (same 'fish' superclass → realistic attack)


def get_model(device: torch.device, dataset: str = "cifar10") -> nn.Module:
    """
    Return the appropriate model for the given dataset on the given device.

    Dataset → Architecture mapping:
      mnist    → MnistCNN
      cifar10  → ResNet-18 (num_classes=10)
      cifar100 → WideResNet-28-10 (num_classes=100)
    """
    if dataset == "cifar100":
        model = WideResNet(depth=28, widen_factor=10, dropout=0.3, num_classes=100)
    elif dataset == "cifar10":
        from models.resnet import ResNet18
        model = ResNet18(num_classes=10)
    elif dataset == "mnist":
        from models.cnn import MnistCNN
        model = MnistCNN(num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'.")
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WideResNet(depth=28, widen_factor=10, num_classes=100).to(device)
    print(f"WideResNet-28-10 for CIFAR-100")
    print(f"Parameters: {model.count_parameters():,}")
    dummy  = torch.zeros(4, 3, 32, 32, device=device)
    out    = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (4, 100)
    assert out.shape == (4, 100)
    print("✓ Forward pass OK")
