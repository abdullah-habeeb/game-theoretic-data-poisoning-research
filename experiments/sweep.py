"""
experiments/sweep.py
====================
Poison fraction sweep experiment.

PURPOSE:
  We run the poisoning attack at multiple levels of intensity (fractions)
  and record how accuracy changes. This produces a curve:

      Accuracy vs Poison Fraction

  This is a standard ablation study in adversarial ML papers.
  It shows:
    - How sensitive the model is to poison intensity
    - At what threshold the attack becomes significant
    - The relationship between attacker effort and attack effectiveness

SWEEP FRACTIONS: [0.2, 0.4, 0.6, 0.8]
  Each fraction is tested in 3 independent runs for reliability.

HOW TO RUN:
  python -m experiments.sweep
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch

from experiments.poisoning import run_poisoned
from utils.plotting import plot_sweep

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")


def run_sweep(
    fractions: list = None,
    dataset: str = "mnist",
    n_runs: int = 3,
    seeds: list = None,
    src_class: int = 1,
    tgt_class: int = 7,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep over multiple poison fractions and record mean/std accuracy.

    Args:
        fractions:  List of poison fractions to sweep over.
        n_runs:     Number of runs per fraction.
        seeds:      Seeds (same seeds across all fractions for consistency).
        src_class:  Source class for label flip.
        tgt_class:  Target class for label flip.
        epochs:     Training epochs per run.
        batch_size: Mini-batch size.
        lr:         Learning rate.
        verbose:    Print progress.

    Returns:
        pandas DataFrame with columns: fraction, mean, std, min, max
        Also saves sweep.csv and sweep.png.
    """
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8]
    if seeds is None:
        seeds = list(range(n_runs))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  POISON FRACTION SWEEP")
        print(f"  Fractions: {fractions}")
        print(f"  Runs per fraction: {n_runs}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"{'='*60}")

    rows = []
    all_means = []
    all_stds = []

    for frac in fractions:
        if verbose:
            print(f"\n── Fraction = {frac:.0%} ──────────────────────────────────")

        summary = run_poisoned(
            n_runs=n_runs,
            seeds=seeds,
            dataset=dataset,
            src_class=src_class,
            tgt_class=tgt_class,
            poison_fraction=frac,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )
        rows.append({
            "fraction": frac,
            "mean": round(summary["mean"], 4),
            "std":  round(summary["std"],  4),
            "min":  round(summary["min"],  4),
            "max":  round(summary["max"],  4),
        })
        all_means.append(summary["mean"])
        all_stds.append(summary["std"])

    df = pd.DataFrame(rows)

    # Save table
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "sweep.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\n[Sweep Table saved] {csv_path}")
        print(df.to_string(index=False))

    # Save plot
    plot_sweep(
        fractions=fractions,
        means=all_means,
        stds=all_stds,
        save_name="sweep.png",
        title=f"Accuracy vs Poison Fraction (src={src_class}→{tgt_class})",
    )

    return df


if __name__ == "__main__":
    df = run_sweep(fractions=[0.2, 0.4, 0.6, 0.8], n_runs=3, epochs=5)
    print("\nFinal sweep results:")
    print(df)
