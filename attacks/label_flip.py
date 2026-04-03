"""
attacks/label_flip.py
=====================
Targeted label-flip poisoning attack.

WHAT IS DATA POISONING?
  An attacker corrupts a fraction of the training data before the model sees it.
  Instead of attacking the model directly, the attacker attacks the training set.
  The corrupted data causes the model to learn wrong associations.

TARGETED LABEL FLIP (our attack):
  We pick samples belonging to a SOURCE class (e.g., digit '1')
  and relabel them as a TARGET class (e.g., digit '7').

  Effect: The model learns that '1' images should be classified as '7'.
          This is a targeted attack — it specifically degrades performance
          on class 1, potentially without affecting other classes much.

POISON FRACTION:
  The fraction of class-1 training samples that are corrupted.
  fraction=0.5 means 50% of all '1' images are relabeled as '7'.
  Higher fraction → stronger attack → lower model accuracy on class 1.

GAME THEORY CONNECTION:
  The attacker's "strategy" is the choice of (src_class, tgt_class, fraction).
  In the min-max game, the attacker tries to MAXIMIZE model error by
  choosing these parameters. The defender tries to MINIMIZE the error
  by updating model weights.
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


class PoisonedDataset(Dataset):
    """
    A dataset wrapper that applies label flipping to a subset of samples.

    This wraps an existing PyTorch dataset (e.g., MNIST train set) and
    returns poisoned labels for selected samples.

    Args:
        original_dataset: The clean dataset to poison.
        poisoned_indices: Set of sample indices whose labels should be flipped.
        target_label:     The label to assign to poisoned samples.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        poisoned_indices: np.ndarray,
        target_label: int,
    ) -> None:
        self.dataset = original_dataset
        self.poisoned_idx_set = set(poisoned_indices.tolist())
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        if idx in self.poisoned_idx_set:
            label = self.target_label   # Flip the label
        return image, label


def poison_dataset(
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    poison_fraction: float,
    seed: int = 42,
) -> Tuple[PoisonedDataset, np.ndarray]:
    """
    Create a poisoned version of a dataset via targeted label flipping.

    How it works:
      1. Find all samples belonging to src_class.
      2. Randomly select `poison_fraction` of them (using the fixed seed).
      3. In the returned dataset, their labels are flipped to tgt_class.

    Args:
        dataset:         Original clean dataset (e.g., MNIST train set).
        src_class:       Class to attack (source). E.g., 1 (digit '1').
        tgt_class:       Class to flip to (target). E.g., 7 (digit '7').
        poison_fraction: Fraction of src_class samples to poison (0.0 to 1.0).
        seed:            Random seed for reproducible sample selection.

    Returns:
        (poisoned_dataset, poisoned_indices)
        - poisoned_dataset: PoisonedDataset wrapping the original
        - poisoned_indices: numpy array of which indices were poisoned

    Example:
        >>> poisoned_train, p_idx = poison_dataset(
        ...     train_dataset, src_class=1, tgt_class=7, poison_fraction=0.5, seed=42
        ... )
        >>> print(f"Poisoned {len(p_idx)} samples")
    """
    # Get all labels as a numpy array for fast indexing
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)        # torchvision datasets have .targets
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Find indices of all samples in the source class
    src_indices = np.where(labels == src_class)[0]
    n_to_poison = int(len(src_indices) * poison_fraction)

    if n_to_poison == 0:
        raise ValueError(
            f"poison_fraction={poison_fraction} is too small — "
            f"0 samples would be poisoned from class {src_class}."
        )

    # Deterministically select which samples to poison
    rng = np.random.default_rng(seed)
    poisoned_indices = rng.choice(src_indices, size=n_to_poison, replace=False)
    poisoned_indices = np.sort(poisoned_indices)

    poisoned_ds = PoisonedDataset(dataset, poisoned_indices, tgt_class)

    print(
        f"[Attack] Poisoned {n_to_poison}/{len(src_indices)} samples of class {src_class} "
        f"→ class {tgt_class}  (fraction={poison_fraction:.0%}, seed={seed})"
    )

    return poisoned_ds, poisoned_indices


def get_poison_stats(
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    poison_fraction: float,
) -> dict:
    """
    Return statistics about what a poisoning attack would look like,
    without actually applying it.

    Useful for verifying the attack configuration before running it.
    """
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    src_count = int((labels == src_class).sum())
    n_to_poison = int(src_count * poison_fraction)

    return {
        "src_class": src_class,
        "tgt_class": tgt_class,
        "poison_fraction": poison_fraction,
        "total_src_samples": src_count,
        "samples_to_poison": n_to_poison,
        "samples_kept_clean": src_count - n_to_poison,
        "total_training_samples": len(dataset),
        "poison_rate_overall": n_to_poison / len(dataset),
    }
