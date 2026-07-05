"""
utils/validation.py
===================
Reproducible, stratified train / validation split.

WHY A VALIDATION SET?
  Without a validation set, there is no honest way to:
    1. Select hyperparameters (epochs, lr, early stopping threshold)
    2. Apply early stopping during defender training
    3. Avoid implicitly using the test set for model selection

  Standard ML practice: 80-90% train, 10-20% val, separate test.
  We use a 90/10 split to keep enough training data for the poisoning
  experiments to be meaningful.

STRATIFIED SPLIT:
  A stratified split ensures each class is proportionally represented
  in both the train and val subsets. This is critical for poisoning
  experiments — a random split might accidentally exclude or over-represent
  the attacked class in validation, which would corrupt the metrics.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Tuple


def train_val_split(
    dataset: Dataset,
    val_fraction: float = 0.1,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[Subset, Subset]:
    """
    Split a dataset into train and validation subsets.

    Args:
        dataset:      Any PyTorch Dataset. Must have numeric labels accessible
                      via `.targets` or by iterating (torchvision standard).
        val_fraction: Fraction of data to use for validation (default: 0.10).
        seed:         Random seed for reproducibility.
        stratified:   If True, maintain class proportions in both splits.
                      Recommended: always True for classification.

    Returns:
        (train_subset, val_subset) — two Subset objects over the original dataset.

    Example:
        >>> train_sub, val_sub = train_val_split(raw_train_ds, val_fraction=0.1, seed=42)
        >>> len(train_sub), len(val_sub)
        (54000, 6000)   # for MNIST with 60000 samples
    """
    n = len(dataset)

    # Retrieve labels efficiently
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        # fallback: iterate (slow but safe)
        labels = np.array([dataset[i][1] for i in range(n)])

    rng = np.random.default_rng(seed)

    if stratified:
        train_idx, val_idx = [], []
        classes = np.unique(labels)
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            n_val = max(1, int(len(cls_indices) * val_fraction))
            val_idx.extend(cls_indices[:n_val].tolist())
            train_idx.extend(cls_indices[n_val:].tolist())
    else:
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        n_val = max(1, int(n * val_fraction))
        val_idx   = all_idx[:n_val].tolist()
        train_idx = all_idx[n_val:].tolist()

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_val_loader(
    val_subset: Subset,
    batch_size: int = 128,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Convenience wrapper to create a DataLoader for the validation subset."""
    from torch.utils.data import DataLoader
    return DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────── Self-test ───────────────────────────────────────────
if __name__ == "__main__":
    from torchvision import datasets, transforms

    print("=== Validation Split Self-Test ===")
    tf = transforms.ToTensor()
    ds = datasets.MNIST(root="./data/raw", train=True, download=True, transform=tf)

    train_sub, val_sub = train_val_split(ds, val_fraction=0.1, seed=42, stratified=True)
    print(f"  Total  : {len(ds):,}")
    print(f"  Train  : {len(train_sub):,}  (expected ~54000)")
    print(f"  Val    : {len(val_sub):,}    (expected ~6000)")

    # Verify stratification: check class distribution in val set
    val_labels = [ds[i][1] for i in val_sub.indices]
    import collections
    dist = collections.Counter(val_labels)
    print(f"  Val class distribution: {dict(sorted(dist.items()))}")
    # Each class should have ~600 samples (MNIST has ~6000 per class)
    counts = list(dist.values())
    assert max(counts) - min(counts) < 100, "Val split is not stratified"
    print("  Stratification check passed ✓")
    print("All validation split tests passed ✓")
