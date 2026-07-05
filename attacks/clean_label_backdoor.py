"""
attacks/clean_label_backdoor.py
================================
Clean-Label Backdoor Attack (Turner et al., NeurIPS 2019 + BadNets variant).

PAPER: "Label-Consistent Backdoor Attacks"
  Turner, Tsipras, Madry — 2019
  https://arxiv.org/abs/1912.02771

THREAT MODEL:
  - Attacker CANNOT change training labels (labels are audited/verified)
  - Attacker CAN modify training images within a small perturbation budget
    OR attach a visible trigger patch
  - At test time, attacker can modify test images by adding the trigger

WHY THIS IS HARDER TO DETECT THAN LABEL-FLIP:
  - Labels look correct to a human auditor
  - The only anomaly is in the pixel space — a small patch or perturbation
  - Automated label verification (a common defense) is completely bypassed
  - The model behaves normally on clean test images; the backdoor only fires
    when the trigger is present at TEST time

TWO VARIANTS IMPLEMENTED:
  1. BadNets (Gu et al. 2019): Simple pixel trigger — a white square patch
     in a fixed corner. Visible but highly effective.
  2. Blended Trigger (Chen et al. 2017): Blend poison image with a trigger
     pattern (e.g., noise pattern) at blending ratio α. Nearly invisible.

ATTACK MECHANISM:
  Training: Add trigger to src_class images (keep src label = CLEAN)
  Test: Add same trigger to any image → model predicts tgt_class
  This creates a BACKDOORED model that behaves normally on clean inputs
  but misclassifies trigger inputs.

ASR MEASUREMENT:
  ASR = P(model predicts tgt | trigger added to src image)
  Use add_trigger() function on test-set source-class images.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import copy


# ─────────────────────────────────────────────────────────────────────────────
# Trigger Patterns
# ─────────────────────────────────────────────────────────────────────────────

def create_badnets_trigger(
    image_size: int = 32,
    channels: int = 3,
    patch_size: int = 5,
    corner: str = "bottom-right",
    value: float = 1.0,
) -> np.ndarray:
    """
    Create a BadNets-style trigger: a solid patch of constant value.

    Args:
        image_size:  H=W of the image.
        channels:    Number of channels (1 for MNIST grayscale, 3 for CIFAR).
        patch_size:  Size of the square trigger patch.
        corner:      Position of trigger: 'bottom-right', 'top-left', etc.
        value:       Pixel value for the trigger (normalized space).

    Returns:
        trigger_mask: [C, H, W] float array — 1.0 at trigger pixels, 0 elsewhere.
        trigger_pattern: [C, H, W] float array — trigger value at trigger pixels.
    """
    mask    = np.zeros((channels, image_size, image_size), dtype=np.float32)
    pattern = np.zeros_like(mask)

    if corner == "bottom-right":
        r_start = image_size - patch_size
        c_start = image_size - patch_size
    elif corner == "top-left":
        r_start, c_start = 0, 0
    elif corner == "bottom-left":
        r_start = image_size - patch_size
        c_start = 0
    else:
        r_start = image_size - patch_size
        c_start = image_size - patch_size

    for c in range(channels):
        mask[c, r_start:r_start+patch_size, c_start:c_start+patch_size]    = 1.0
        pattern[c, r_start:r_start+patch_size, c_start:c_start+patch_size] = value

    return mask, pattern


def create_blended_trigger(
    image_size: int = 32,
    channels: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Create a blended trigger: random noise pattern added at ratio α.
    The trigger is the SAME noise pattern for all poison samples.

    Returns:
        trigger_pattern: [C, H, W] noise pattern.
    """
    rng = np.random.default_rng(seed)
    pattern = rng.standard_normal((channels, image_size, image_size)).astype(np.float32)
    pattern = pattern / (np.abs(pattern).max() + 1e-8)  # normalize to [-1, 1]
    return pattern


def apply_trigger(
    image: np.ndarray,
    trigger_type: str,
    trigger_pattern: np.ndarray,
    trigger_mask: Optional[np.ndarray] = None,
    alpha: float = 0.2,
    channels: int = 3,
    image_size: int = 32,
) -> np.ndarray:
    """
    Apply a trigger to a single image (numpy array [C, H, W]).

    For BadNets: replace trigger region with pattern (mask-based).
    For Blended: x' = (1-α)·x + α·trigger_pattern.
    """
    img = image.copy()

    if trigger_type == "badnets":
        assert trigger_mask is not None
        img = img * (1 - trigger_mask) + trigger_pattern * trigger_mask

    elif trigger_type == "blended":
        img = (1.0 - alpha) * img + alpha * trigger_pattern

    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Poisoned Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CleanLabelBackdoorDataset(Dataset):
    """
    Dataset wrapper for clean-label backdoor poisoning.

    For poisoned samples:
      - Image: original image + trigger (pixel modification)
      - Label: ORIGINAL label (clean! — this is the key property)

    For clean samples:
      - Image: unchanged
      - Label: unchanged
    """

    def __init__(
        self,
        dataset: Dataset,
        poisoned_indices: np.ndarray,
        trigger_type: str,
        trigger_pattern: np.ndarray,
        trigger_mask: Optional[np.ndarray] = None,
        alpha: float = 0.15,
    ):
        self.dataset          = dataset
        self.poisoned_set     = set(int(i) for i in poisoned_indices)
        self.poisoned_indices = poisoned_indices
        self.trigger_type     = trigger_type
        self.trigger_pattern  = trigger_pattern
        self.trigger_mask     = trigger_mask
        self.alpha            = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple:
        img, label = self.dataset[idx]

        if idx in self.poisoned_set:
            img_np = img.numpy() if isinstance(img, torch.Tensor) else img
            img_np = apply_trigger(
                img_np,
                trigger_type    = self.trigger_type,
                trigger_pattern = self.trigger_pattern,
                trigger_mask    = self.trigger_mask,
                alpha           = self.alpha,
            )
            img = torch.tensor(img_np)

        return img, label   # label is UNCHANGED — clean label!


# ─────────────────────────────────────────────────────────────────────────────
# Backdoor ASR Evaluation Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TriggeredTestDataset(Dataset):
    """
    Test dataset with triggers added to all images (for ASR measurement).

    ASR = fraction of triggered images classified as tgt_class.
    """

    def __init__(
        self,
        test_dataset: Dataset,
        trigger_type: str,
        trigger_pattern: np.ndarray,
        trigger_mask: Optional[np.ndarray],
        tgt_class: int,
        src_class: Optional[int] = None,
        alpha: float = 0.15,
    ):
        self.dataset         = test_dataset
        self.trigger_type    = trigger_type
        self.trigger_pattern = trigger_pattern
        self.trigger_mask    = trigger_mask
        self.tgt_class       = tgt_class
        self.src_class       = src_class
        self.alpha           = alpha

        # If src_class given, only include src-class test samples
        if src_class is not None:
            if hasattr(test_dataset, "targets"):
                test_labels = np.array(test_dataset.targets)
            else:
                test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
            self.valid_indices = np.where(test_labels == src_class)[0]
        else:
            self.valid_indices = np.arange(len(test_dataset))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Tuple:
        global_idx = self.valid_indices[idx]
        img, _ = self.dataset[int(global_idx)]

        img_np = img.numpy() if isinstance(img, torch.Tensor) else img
        img_np = apply_trigger(
            img_np,
            trigger_type    = self.trigger_type,
            trigger_pattern = self.trigger_pattern,
            trigger_mask    = self.trigger_mask,
            alpha           = self.alpha,
        )
        img = torch.tensor(img_np)

        # Label is tgt_class for ASR computation (what SHOULD happen if attack works)
        return img, self.tgt_class


def compute_backdoor_asr(
    model: torch.nn.Module,
    test_dataset: Dataset,
    trigger_type: str,
    trigger_pattern: np.ndarray,
    trigger_mask: Optional[np.ndarray],
    tgt_class: int,
    src_class: int,
    device: torch.device,
    batch_size: int = 128,
    alpha: float = 0.15,
) -> float:
    """
    Compute Attack Success Rate for backdoor attack.
    ASR = P(model(x + trigger) = tgt_class | x ∈ test set, true label = src_class)

    Returns:
        asr: float ∈ [0, 1]
    """
    triggered_ds = TriggeredTestDataset(
        test_dataset, trigger_type, trigger_pattern, trigger_mask,
        tgt_class, src_class=src_class, alpha=alpha
    )
    if len(triggered_ds) == 0:
        return 0.0

    loader = torch.utils.data.DataLoader(
        triggered_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            preds  = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_label_backdoor_attack(
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    poison_fraction: float,
    trigger_type: str = "badnets",
    image_size: int = 32,
    channels: int = 3,
    patch_size: int = 5,
    alpha: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[CleanLabelBackdoorDataset, np.ndarray, dict]:
    """
    Create a clean-label backdoor poisoned dataset.

    Args:
        trigger_type: 'badnets' (visible patch) or 'blended' (subtle noise).
        patch_size:   Size of BadNets trigger patch.
        alpha:        Blending ratio for blended trigger.

    Returns:
        (poisoned_dataset, poisoned_indices, trigger_info)
        trigger_info: dict with trigger_type, trigger_pattern, trigger_mask
                      needed for ASR computation at test time.
    """
    # Get source class indices
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    src_indices = np.where(labels == src_class)[0]
    n_to_poison = max(1, int(len(src_indices) * poison_fraction))

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(src_indices, size=n_to_poison, replace=False)
    selected_indices = np.sort(selected_indices)

    # Create trigger
    if trigger_type == "badnets":
        trigger_mask, trigger_pattern = create_badnets_trigger(
            image_size=image_size, channels=channels, patch_size=patch_size
        )
    else:  # blended
        trigger_pattern = create_blended_trigger(
            image_size=image_size, channels=channels, seed=seed
        )
        trigger_mask = None

    trigger_info = {
        "trigger_type":    trigger_type,
        "trigger_pattern": trigger_pattern,
        "trigger_mask":    trigger_mask,
        "alpha":           alpha,
        "src_class":       src_class,
        "tgt_class":       tgt_class,
    }

    poisoned_ds = CleanLabelBackdoorDataset(
        dataset, selected_indices, trigger_type,
        trigger_pattern, trigger_mask, alpha
    )

    if verbose:
        print(f"  [CleanLabelBackdoor] trigger={trigger_type}  patch={patch_size}px  "
              f"poisoned={n_to_poison}/{len(src_indices)} src-class samples")
        print(f"  NOTE: Labels are UNCHANGED — attack is undetectable by label audit")

    return poisoned_ds, selected_indices, trigger_info
