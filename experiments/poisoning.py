"""
experiments/poisoning.py
========================
Targeted label-flip attack experiment.

PURPOSE:
  We inject poisoned labels into the training set, then train the CNN on
  this corrupted data and measure how much the attack degrades accuracy.

SETUP:
  - Source class:    1  (digit '1')
  - Target class:    7  (digit '7')
  - Poison fraction: 0.5 (50% of all '1' samples are relabeled as '7')
  - 3 independent runs with fixed seeds

INTERPRETING THE RESULTS:
  If poisoned mean accuracy < baseline mean accuracy → attack is effective.
  If std is large → the attack is inconsistent (not good for a paper).
  We want consistent, repeatable drops in accuracy.

HOW TO RUN:
  python -m experiments.poisoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import get_model          # routes: mnist→MnistCNN, cifar10→ResNet18
from data.dataset import get_raw_train_dataset
from train.trainer import train_model
from train.evaluator import evaluate
from attacks.label_flip import poison_dataset
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary


def _get_test_loader(dataset: str, batch_size: int, data_root: str = "./data/raw"):
    """Return a test DataLoader for the given dataset."""
    from torch.utils.data import DataLoader
    from data.dataset import get_transforms
    _, test_tf = get_transforms(dataset, augment=False)
    if dataset == "mnist":
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
    elif dataset == "cifar10":
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    else:
        test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_poisoned(
    n_runs: int = 3,
    seeds: list = None,
    dataset: str = "mnist",
    src_class: int = 1,
    tgt_class: int = 7,
    poison_fraction: float = 0.5,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True,
) -> dict:
    """
    Train and evaluate the model on poisoned data multiple times.

    Args:
        n_runs:          Number of independent runs.
        seeds:           Per-run random seeds.
        dataset:         'mnist', 'cifar10', or 'cifar100'.
        src_class:       Class whose labels are flipped (source).
        tgt_class:       Class to flip the labels to (target).
        poison_fraction: Fraction of src_class samples to poison.
        epochs:          Training epochs per run.
        batch_size:      Mini-batch size.
        lr:              Learning rate.
        verbose:         Print progress.

    Returns:
        Summary dict with mean, std, min, max over all runs.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    assert len(seeds) == n_runs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*55}")
        print(f"  POISONING EXPERIMENT  ({n_runs} runs, {epochs} epochs)")
        print(f"  Attack: {src_class} → {tgt_class}, fraction={poison_fraction:.0%}")
        print(f"  Device: {device}")
        print(f"{'='*55}")

    accuracies = []
    test_loader = _get_test_loader(dataset, batch_size)

    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        set_seed(seed)

        if verbose:
            print(f"\n[Run {run_idx+1}/{n_runs}]  seed={seed}")

        # Load raw training dataset for the specified split
        train_ds = get_raw_train_dataset(dataset=dataset, augment=False)

        # Apply the poisoning attack
        poisoned_train_ds, _ = poison_dataset(
            train_ds,
            src_class=src_class,
            tgt_class=tgt_class,
            poison_fraction=poison_fraction,
            seed=seed,
        )

        train_loader = DataLoader(
            poisoned_train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Use the correct model architecture for this dataset
        model = get_model(device, dataset)
        train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=verbose)

        # Evaluate on the CLEAN test set (this is standard: test the model honestly)
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)

        if verbose:
            print(f"  → Test accuracy (poisoned model): {acc:.2f}%")

    summary = summarize_runs(accuracies)
    if verbose:
        print_summary(
            f"POISONING RESULTS  (src={src_class}→{tgt_class}, frac={poison_fraction:.0%})",
            summary,
        )

    return summary


if __name__ == "__main__":
    results = run_poisoned(
        n_runs=3,
        src_class=1,
        tgt_class=7,
        poison_fraction=0.5,
        epochs=5,
    )
