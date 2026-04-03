"""
models/cnn.py
=============
CNN architecture for MNIST classification.

ARCHITECTURE EXPLAINED (layer by layer):

  Input: [batch, 1, 28, 28]
    → 1 channel (grayscale), 28×28 pixels

  Conv2d(1→32, kernel=3): [batch, 32, 26, 26]
    → Learns 32 different feature detectors (edges, curves, etc.)
    → Each 3×3 kernel slides over the image
    → Output spatial size: (28 - 3 + 1) = 26

  ReLU: [batch, 32, 26, 26]
    → Non-linear activation. Sets negative values to 0.
    → Without this, stacking layers is equivalent to one linear layer.

  Conv2d(32→64, kernel=3): [batch, 64, 24, 24]
    → Learns 64 higher-level features (combinations of the 32 lower features)
    → Output spatial size: (26 - 3 + 1) = 24

  ReLU: [batch, 64, 24, 24]

  MaxPool2d(kernel=2): [batch, 64, 12, 12]
    → Downsamples by taking the max in each 2×2 block
    → Reduces spatial size: 24 // 2 = 12
    → Makes the representation more compact and translation-invariant

  Flatten: [batch, 64×12×12] = [batch, 9216]
    → Stretches the 3D tensor into a 1D vector for the FC layers
    → NOTE: The flattened size is 9,216 (not 1,600 as originally stated;
      1,600 would require a 5×5 input to the pool, which is a different setup)

  FC(9216→128): [batch, 128]
    → A fully connected layer that learns global combinations of features

  ReLU: [batch, 128]

  FC(128→10): [batch, 10]
    → Output layer with 10 neurons, one per digit class (0–9)
    → Raw logits (scores); apply softmax for probabilities

  Output: [batch, 10] logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.

    Based on the architecture from the research baseline:
        Conv(1→32) → Conv(32→64) → MaxPool(2) → FC → out
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Args:
            num_classes: Number of output classes. 10 for MNIST (digits 0–9).
        """
        super(MnistCNN, self).__init__()

        # ── Convolutional feature extractor ──────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # ── Classifier head ───────────────────────────────────────────────────
        # Flattened size: 64 channels × 12 × 12 spatial = 9,216
        # Derivation:
        #   28x28 → conv(3) → 26x26 → conv(3) → 24x24 → pool(2) → 12x12
        #   Channels: 64.  Total: 64 × 12 × 12 = 9,216
        self._flat_features = 64 * 12 * 12  # = 9216
        self.fc1 = nn.Linear(self._flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch, 1, 28, 28]

        Returns:
            Logits tensor of shape [batch, 10]
        """
        # Convolutional block
        x = F.relu(self.conv1(x))   # [B, 32, 26, 26]
        x = F.relu(self.conv2(x))   # [B, 64, 24, 24]
        x = self.pool(x)             # [B, 64, 12, 12]

        # Flatten: [B, 64, 12, 12] → [B, 9216]
        x = x.view(x.size(0), -1)

        # Fully connected classifier
        x = F.relu(self.fc1(x))     # [B, 128]
        x = self.fc2(x)             # [B, 10]

        return x

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(device: torch.device) -> MnistCNN:
    """
    Convenience function: instantiate and move the model to the target device.

    Args:
        device: torch.device('cuda') or torch.device('cpu')

    Returns:
        MnistCNN model on the specified device.
    """
    model = MnistCNN()
    model.to(device)
    return model


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    print(f"Model: MnistCNN")
    print(f"Trainable parameters: {model.count_parameters():,}")

    dummy = torch.zeros(8, 1, 28, 28).to(device)  # batch of 8 blank images
    out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # Expected: torch.Size([8, 10])
    assert out.shape == (8, 10), "Unexpected output shape!"
    print("✓ Forward pass OK")
