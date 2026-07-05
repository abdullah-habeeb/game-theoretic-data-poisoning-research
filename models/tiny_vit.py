"""
models/tiny_vit.py
==================
Tiny Vision Transformer (ViT) for CIFAR-style 32×32 inputs.
NO EXTERNAL DEPENDENCY (no timm, no huggingface) — pure PyTorch implementation.

WHY EVALUATE ON A TRANSFORMER?
  ResNet-18 and ResNet-50 use convolutional inductive biases:
    - Local receptive fields → each neuron attends to a patch
    - Translation equivariance → features are position-independent
  These properties make CNN-based models RELATIVELY resistant to
  small trigger patches (the trigger affects only one local region).

  ViT has NO such inductive bias — each patch attends to ALL other patches.
  This means:
    (a) ViT may be MORE vulnerable to trigger patches due to global attention
        amplifying the trigger signal across the entire feature map.
    (b) ViT may be MORE robust to gradient matching attacks since features
        are distributed globally, not localized to the trigger region.
  This is an OPEN RESEARCH QUESTION that our experiments address.

ARCHITECTURE (Tiny ViT — ~4.5M params):
  Input:    [B, C, 32, 32]  (C=3 for CIFAR/GTSRB, C=1 for MNIST)
  Patching: 4×4 patches → 8×8 = 64 tokens + 1 [CLS] token = 65 tokens
  Embed:    dim=192
  Layers:   6 transformer blocks
  Heads:    3 attention heads
  MLP dim:  192 × 4 = 768
  Head:     Linear(192 → n_classes)

COMPARISON TO RESNET-18:
  ResNet-18: 11.2M params, inductive bias (conv)
  Tiny ViT:   4.5M params, no inductive bias (attention)

NOTE ON TRAINING:
  ViT requires more training epochs and stronger augmentation than CNNs.
  With 32×32 images and short training schedules, ViT underperforms CNNs.
  For a fair comparison, use the same hyperparameters but interpret results
  in light of this training time disadvantage.

REFERENCE:
  Dosovitskiy et al. (2021). "An Image is Worth 16×16 Words." ICLR 2021.
  (We use 4×4 patches because 32×32 / 4 = 8, giving 64 tokens — comparable
   to the 196 tokens used in ViT-Base for 224×224 images.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.
    Input:  [B, C, H, W]
    Output: [B, N, D]  where N = (H/patch_size)^2
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.n_patches = (img_size // patch_size) ** 2
        # Convolutional projection: each patch_size×patch_size region → embed_dim vector
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, D, H/P, W/P] → [B, D, N] → [B, N, D]
        x = self.proj(x)
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1, 2)   # [B, N, D]


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional dropout."""

    def __init__(self, dim: int, n_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]     # [B, H, N, Dh] each

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, N, N]
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        return self.proj(x)


class MLPBlock(nn.Module):
    """Two-layer MLP with GELU activation (standard ViT MLP block)."""

    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    ViT Transformer Block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLPBlock(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # residual + attention
        x = x + self.mlp(self.norm2(x))    # residual + MLP
        return x


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer for CIFAR-style 32×32 images.
    ~4.5M parameters, 6 transformer layers, 3 attention heads.
    """

    def __init__(
        self,
        img_size:    int = 32,
        patch_size:  int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim:   int = 192,
        depth:       int = 6,
        n_heads:     int = 3,
        mlp_ratio:   int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.patch_embed  = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches         = self.patch_embed.n_patches

        # Learnable [CLS] token and positional embedding
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                      # [B, N, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)              # [B, N+1, D]
        x   = self.pos_drop(x + self.pos_embed)       # add positional embedding

        x = self.blocks(x)                            # [B, N+1, D]
        x = self.norm(x)
        return self.head(x[:, 0])                     # CLS token → logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token embedding (penultimate features). Used by defenses."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.blocks(x)
        return self.norm(x)[:, 0]                    # [B, embed_dim]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_tiny_vit(device: torch.device, dataset: str = "cifar10") -> TinyViT:
    """
    Return a TinyViT model ready for the specified dataset.

    MNIST:   1-channel, img_size=28, patch_size=4 → 7×7=49 patches
    CIFAR:   3-channel, img_size=32, patch_size=4 → 8×8=64 patches
    GTSRB:   3-channel, img_size=32, patch_size=4 → 64 patches
    """
    configs = {
        "mnist":   dict(img_size=28, in_channels=1, num_classes=10),
        "cifar10": dict(img_size=32, in_channels=3, num_classes=10),
        "cifar100":dict(img_size=32, in_channels=3, num_classes=100),
        "gtsrb":   dict(img_size=32, in_channels=3, num_classes=43),
    }
    cfg = configs.get(dataset, configs["cifar10"])
    model = TinyViT(
        img_size=cfg["img_size"],
        patch_size=4,
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        embed_dim=192,
        depth=6,
        n_heads=3,
    )
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10
    vit = get_tiny_vit(device, "cifar10")
    dummy = torch.zeros(4, 3, 32, 32, device=device)
    out   = vit(dummy)
    assert out.shape == (4, 10), f"Wrong output: {out.shape}"
    print(f"TinyViT CIFAR-10: {vit.count_parameters():,} params, output {out.shape}")

    # MNIST (1 channel, 28×28)
    # patch 4 → 7×7=49 patches
    vit_m = get_tiny_vit(device, "mnist")
    dummy_m = torch.zeros(4, 1, 28, 28, device=device)
    out_m   = vit_m(dummy_m)
    assert out_m.shape == (4, 10)
    print(f"TinyViT MNIST: {vit_m.count_parameters():,} params, output {out_m.shape}")

    # GTSRB (43 classes)
    vit_g = get_tiny_vit(device, "gtsrb")
    out_g = vit_g(dummy)
    assert out_g.shape == (4, 43)
    print(f"TinyViT GTSRB: {vit_g.count_parameters():,} params, output {out_g.shape}")

    print("\n✓ All TinyViT forward passes correct")
