"""
experiments/defender.py
========================
Alternating Min–Max attacker–defender training.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE RESEARCH CONTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GAME-THEORETIC BACKGROUND:
  We model the attacker–defender interaction as a two-player zero-sum game:

    - ATTACKER (maximizer):
        Poisons a fraction of training data to MAXIMIZE the model's loss.
        Strategy: Choose which samples to corrupt and how to relabel them.

    - DEFENDER (minimizer):
        Trains the model parameters to MINIMIZE the loss on available data
        (which includes poisoned samples it cannot detect).

  The combined objective is a Min-Max problem:

        min_θ  max_a  L(θ, data_poisoned_by_a)

  Where:
    θ = model parameters (defender's decision variable)
    a = attacker's strategy (poison fraction, src/tgt class)
    L = cross-entropy loss on the training set

STACKELBERG GAME APPROXIMATION:
  In the first (academically valid) version, the attacker commits first
  (Stackelberg leader) with a fixed strategy, and the defender responds
  (Stackelberg follower) by updating model weights.

  This is implemented as an ALTERNATING LOOP:
    Round 1:  Attacker poisons data  →  Defender trains on poisoned data  →  Evaluate
    Round 2:  Attacker re-poisons    →  Defender re-trains with fresh init →  Evaluate
    ...

  In each round, the defender gets a fresh model (starting from scratch), which
  isolates the question: "How much can the defender recover if they simply
  retrain the model knowing it might be poisoned?"

INTERPRETATION:
  - Each round represents one iteration of the alternating game.
  - The defender accuracy across rounds shows whether robust training
    (under fixed attack intensity) converges or fluctuates.
  - Final comparison: poisoned_baseline vs defended shows the value of
    the defender's min-max strategy.

FUTURE EXTENSIONS (mentioned in paper as future work):
  - Adaptive attacker: Attacker also updates strategy using gradients on val set
  - Gradient-based attack: PGD on input features
  - Alternating gradient ascent/descent (full GAN-like min-max)

HOW TO RUN:
  python -m experiments.defender
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import get_model          # routes: mnist→MnistCNN, cifar10→ResNet18
from data.dataset import get_raw_train_dataset, get_transforms
from train.trainer import train_model
from train.evaluator import evaluate
from attacks.label_flip import poison_dataset
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from utils.plotting import plot_minmax_rounds, plot_comparison

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")


def _get_raw_datasets(dataset: str = "mnist", data_root: str = "./data/raw"):
    """Return (train_ds, test_ds) for the requested dataset."""
    from torchvision import datasets as tvds
    _, test_tf = get_transforms(dataset, augment=False)
    train_tf, _ = get_transforms(dataset, augment=False)
    if dataset == "mnist":
        train_ds = tvds.MNIST(root=data_root, train=True,  download=True, transform=train_tf)
        test_ds  = tvds.MNIST(root=data_root, train=False, download=True, transform=test_tf)
    elif dataset == "cifar10":
        train_ds = tvds.CIFAR10(root=data_root, train=True,  download=True, transform=train_tf)
        test_ds  = tvds.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    else:
        train_ds = tvds.CIFAR100(root=data_root, train=True,  download=True, transform=train_tf)
        test_ds  = tvds.CIFAR100(root=data_root, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


def run_one_minmax_round(
    device: torch.device,
    dataset: str,
    src_class: int,
    tgt_class: int,
    poison_fraction: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    round_num: int,
    verbose: bool,
) -> float:
    """
    Execute one round of the alternating min–max game:
      1. Attacker poisons the dataset.
      2. Defender trains a fresh model on poisoned data.
      3. Returns test accuracy.

    The "freshness" of the model in each round is intentional:
    It simulates a defender that can retrain from scratch each round.
    In a more advanced setting, the defender would inherit weights from
    the previous round (warm-starting).
    """
    set_seed(seed)

    # Load clean datasets for the selected dataset
    train_ds, test_ds = _get_raw_datasets(dataset=dataset)

    # ── ATTACKER STEP ────────────────────────────────────────────────────────
    # Attacker applies label-flip with fixed strategy
    poisoned_train_ds, poisoned_idx = poison_dataset(
        train_ds,
        src_class=src_class,
        tgt_class=tgt_class,
        poison_fraction=poison_fraction,
        seed=seed,
    )

    train_loader = DataLoader(
        poisoned_train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── DEFENDER STEP ────────────────────────────────────────────────────────
    # Defender trains a fresh model to minimize cross-entropy on the (poisoned) training data.
    # This is the "minimization" step in the min-max formulation.
    model = get_model(device, dataset)
    train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=verbose)

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    acc = evaluate(model, test_loader, device)

    if verbose:
        print(f"  [Round {round_num}] Defended accuracy: {acc:.2f}%")
        print(f"    └─ {len(poisoned_idx)} samples poisoned  "
              f"({poison_fraction:.0%} of class {src_class})")

    return acc


def run_minmax(
    n_rounds: int = 5,
    n_runs: int = 3,
    seeds: list = None,
    dataset: str = "mnist",
    src_class: int = 1,
    tgt_class: int = 7,
    poison_fraction: float = 0.5,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    baseline_mean: float = None,
    poisoned_mean: float = None,
    verbose: bool = True,
) -> dict:
    """
    Run the alternating min-max game for multiple rounds and runs.

    Each (round, run) pair uses a unique seed to ensure independent and
    reproducible results.

    Args:
        n_rounds:        Number of alternating attacker-defender rounds.
        n_runs:          Number of independent runs per round (for statistics).
        seeds:           Base seeds; each (round, run) uses seed = seeds[run] + round*100.
        src_class:       Attacker's source class.
        tgt_class:       Attacker's target class.
        poison_fraction: Attacker's poison intensity.
        epochs:          Defender's training budget per round.
        batch_size:      Mini-batch size.
        lr:              Defender's learning rate.
        baseline_mean:   Clean baseline accuracy (for comparison plot).
        poisoned_mean:   One-shot poisoned accuracy (for comparison plot).
        verbose:         Print progress.

    Returns:
        dict with:
          'round_means':  List of mean accuracies per round.
          'round_stds':   List of std deviations per round.
          'final_summary': Summarize_runs over the LAST round.
          'all_rounds_df': DataFrame with all round statistics.
    """
    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MIN–MAX DEFENDER EXPERIMENT")
        print(f"  Rounds: {n_rounds}  |  Runs/round: {n_runs}  |  Epochs: {epochs}")
        print(f"  Attack: {src_class}→{tgt_class}, fraction={poison_fraction:.0%}")
        print(f"  Device: {device}")
        print(f"{'='*60}")

    round_means = []
    round_stds  = []
    all_rows    = []

    for round_num in range(1, n_rounds + 1):
        if verbose:
            print(f"\n── Round {round_num}/{n_rounds} ──────────────────────────────────")

        round_accs = []
        for run_idx in range(n_runs):
            # Unique seed per (round, run) so each run is independent
            seed = seeds[run_idx] + round_num * 100
            acc = run_one_minmax_round(
                device=device,
                dataset=dataset,
                src_class=src_class,
                tgt_class=tgt_class,
                poison_fraction=poison_fraction,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                round_num=round_num,
                verbose=False,  # Show summary only, not per-epoch detail
            )
            round_accs.append(acc)

        summary = summarize_runs(round_accs)
        round_means.append(summary["mean"])
        round_stds.append(summary["std"])
        all_rows.append({
            "round": round_num,
            "mean": round(summary["mean"], 4),
            "std":  round(summary["std"],  4),
            "min":  round(summary["min"],  4),
            "max":  round(summary["max"],  4),
        })

        if verbose:
            print(f"  Round {round_num} summary:  "
                  f"mean={summary['mean']:.2f}%  std={summary['std']:.2f}%")

    # ── Final summary (last round = most "converged" defender) ───────────────
    final_summary = {
        "runs": round_accs,   # final round runs
        "mean": round_means[-1],
        "std":  round_stds[-1],
        "min":  min(round_accs),
        "max":  max(round_accs),
    }

    df = pd.DataFrame(all_rows)

    # Save table
    os.makedirs(TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(TABLES_DIR, "minmax_rounds.csv")
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"\n[Min–Max Table saved] {csv_path}")
        print(df.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_minmax_rounds(
        rounds=list(range(1, n_rounds + 1)),
        defended_accs=round_means,
        poisoned_baseline=poisoned_mean if poisoned_mean is not None else 92.0,
        clean_baseline=baseline_mean if baseline_mean is not None else 99.07,
        save_name="minmax_rounds.png",
    )

    # Final 3-way comparison (only useful if we have all three reference points)
    if baseline_mean is not None and poisoned_mean is not None:
        plot_comparison(
            labels=["Clean Baseline", "Poisoned Baseline", "Defended (Min–Max)"],
            means=[baseline_mean, poisoned_mean, round_means[-1]],
            stds=[0.0, 0.0, round_stds[-1]],
            save_name="comparison.png",
            title="Clean vs Poisoned vs Defended Model Accuracy",
        )

    if verbose:
        print_summary("DEFENDED MODEL (final round)", {
            "runs": round_means,
            "mean": round_means[-1],
            "std":  round_stds[-1],
            "min":  min(round_means),
            "max":  max(round_means),
        })

    return {
        "round_means":    round_means,
        "round_stds":     round_stds,
        "final_summary":  final_summary,
        "all_rounds_df":  df,
    }


if __name__ == "__main__":
    # Run the full min-max experiment.
    # Pass in previously computed baseline/poisoned means if available.
    results = run_minmax(
        n_rounds=5,
        n_runs=3,
        poison_fraction=0.5,
        epochs=5,
        baseline_mean=99.07,   # From baseline experiment
        poisoned_mean=92.23,   # From poisoning experiment
    )
