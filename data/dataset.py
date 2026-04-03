"""
data/dataset.py
===============
Multi-dataset loader: MNIST, CIFAR-10, and CIFAR-100.

MNIST → quick ablations (28×28 grayscale, 10 classes)
CIFAR-10 → primary experiments (32×32 RGB, 10 classes)
CIFAR-100 → hardest benchmark (32×32 RGB, 100 fine-grained classes)

WHY THREE DATASETS?
  Using three datasets of increasing difficulty demonstrates that the
  attack and defense scale. This is a key requirement for conference papers.
  Results on multiple datasets eliminate the concern that findings are
  dataset-specific.

VALIDATION SPLIT:
  10% of training set is carved out as a validation set used by the
  gradient-based attacker as its oracle for computing poisoning gradients.
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ── Dataset statistics (pre-computed) ────────────────────────────────────────
STATS = {
    "mnist":    {"mean": (0.1307,),               "std": (0.3081,),               "channels": 1, "size": 28},
    "cifar10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010), "channels": 3, "size": 32},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761), "channels": 3, "size": 32},
}


def get_transforms(dataset: str, augment: bool = True):
    """Return train and test transforms for the specified dataset."""
    s = STATS[dataset]
    norm = transforms.Normalize(s["mean"], s["std"])

    if dataset == "mnist":
        train_tf = transforms.Compose([transforms.ToTensor(), norm])
        test_tf  = transforms.Compose([transforms.ToTensor(), norm])

    elif dataset in ("cifar10", "cifar100"):
        if augment:
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                norm,
            ])
        else:
            train_tf = transforms.Compose([transforms.ToTensor(), norm])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist', 'cifar10', or 'cifar100'.")

    return train_tf, test_tf


def get_dataloaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    data_root: str = "./data/raw",
    val_fraction: float = 0.1,
    num_workers: int = 2,
    augment: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download and return (train_loader, val_loader, test_loader).

    The validation set is carved from the training set. It is used by the
    gradient-based attacker as its oracle to compute poisoning gradients.

    Args:
        dataset:      'mnist', 'cifar10', or 'cifar100'.
        batch_size:   Mini-batch size.
        data_root:    Download directory.
        val_fraction: Fraction of training data used for validation.
        num_workers:  DataLoader workers (use 0 on Windows if errors occur).
        augment:      Apply data augmentation to training set.
        seed:         Seed for train/val split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    os.makedirs(data_root, exist_ok=True)
    train_tf, test_tf = get_transforms(dataset, augment=augment)

    if dataset == "mnist":
        full_train = datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
        test_ds    = datasets.MNIST(data_root, train=False, download=True, transform=test_tf)
    elif dataset == "cifar10":
        full_train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
        test_ds    = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    else:  # cifar100
        full_train = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        test_ds    = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)

    # Deterministic train / val split
    g = torch.Generator().manual_seed(seed)
    n_val = int(len(full_train) * val_fraction)
    n_trn = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_trn, n_val], generator=g)

    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader


def get_raw_train_dataset(
    dataset: str = "cifar10",
    data_root: str = "./data/raw",
    augment: bool = False,
):
    """
    Return just the raw training dataset object (no DataLoader).
    Required by the attacker and defenses to directly index samples.
    augment=False by default for deterministic access.
    """
    _, test_tf = get_transforms(dataset, augment=False)
    train_tf, _ = get_transforms(dataset, augment=augment)
    if dataset == "mnist":
        return datasets.MNIST("./data/raw", train=True, download=True, transform=train_tf)
    elif dataset == "cifar10":
        return datasets.CIFAR10("./data/raw", train=True, download=True, transform=train_tf)
    else:  # cifar100
        return datasets.CIFAR100("./data/raw", train=True, download=True, transform=train_tf)


def inspect_dataset(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Print a quick summary of the dataset split."""
    print("\n[Dataset Info]")
    print(f"  Training samples   : {len(train_loader.dataset)}")
    print(f"  Validation samples : {len(val_loader.dataset)}")
    print(f"  Test samples       : {len(test_loader.dataset)}")
    print(f"  Batch size         : {train_loader.batch_size}")
    images, labels = next(iter(train_loader))
    print(f"  Image tensor shape : {images.shape}")
    print(f"  Label classes      : {sorted(labels.unique().tolist())}")
