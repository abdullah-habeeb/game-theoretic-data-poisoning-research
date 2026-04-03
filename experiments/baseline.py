"""
experiments/baseline.py
========================
Clean baseline: train the CNN on unmodified MNIST and evaluate it.

PURPOSE:
  The baseline tells us how good the model is WITHOUT any attack.
  This is our reference point. Every adversarial result is compared against this.

WHY 3 RUNS?
  A single training run is not reliable — the result depends on the initial
  random weights and data shuffle. Running 3 times with different seeds and
  reporting mean ± std is standard practice in ML research papers.

HOW TO RUN:
  python -m experiments.baseline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from data.dataset import get_dataloaders
from models.cnn import get_model
from train.trainer import train_model
from train.evaluator import evaluate
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary


def run_baseline(
    n_runs: int = 3,
    seeds: list = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True,
) -> dict:
    """
    Train and evaluate the clean CNN baseline multiple times.

    Args:
        n_runs:     Number of independent training runs.
        seeds:      List of seeds, one per run. Default: [0, 1, 2].
        epochs:     Training epochs per run.
        batch_size: Mini-batch size.
        lr:         Learning rate.
        verbose:    Print progress.

    Returns:
        Summary dict: {'runs': [...], 'mean': ..., 'std': ..., 'min': ..., 'max': ...}
    """
    if seeds is None:
        seeds = list(range(n_runs))
    assert len(seeds) == n_runs, "Need one seed per run."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\n{'='*55}")
        print(f"  BASELINE EXPERIMENT  ({n_runs} runs, {epochs} epochs each)")
        print(f"  Device: {device}")
        print(f"{'='*55}")

    accuracies = []

    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        set_seed(seed)

        if verbose:
            print(f"\n[Run {run_idx+1}/{n_runs}]  seed={seed}")

        # Fresh dataloaders (shuffle is re-seeded via set_seed above)
        train_loader, _, test_loader = get_dataloaders(dataset="mnist", batch_size=batch_size)

        # Fresh model (weights re-initialized with the current seed)
        model = get_model(device)

        # Train on clean data
        train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=verbose)

        # Evaluate on test set
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)

        if verbose:
            print(f"  → Test accuracy: {acc:.2f}%")

    summary = summarize_runs(accuracies)
    if verbose:
        print_summary("BASELINE RESULTS", summary)

    return summary


if __name__ == "__main__":
    results = run_baseline(n_runs=3, epochs=5)
