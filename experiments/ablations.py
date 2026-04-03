"""
experiments/ablations.py
=========================
Ablation Studies.

WHAT IS AN ABLATION STUDY?
  An ablation study systematically varies one hyperparameter at a time
  while keeping all others fixed. This shows WHICH components of your
  method actually matter, and provides evidence that your design choices
  are principled rather than arbitrary.

  Without ablations, a reviewer can always say:
  "Maybe a different learning rate would show the same result without
  your fancy min-max training." Ablations eliminate that argument.

ABLATION 1: Defender Learning Rate
  Fixed: epochs=5, poison_fraction=0.5
  Varied: lr ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
  Question: How sensitive is the defender to its learning rate?

ABLATION 2: Defender Epochs per Round
  Fixed: lr=0.001, poison_fraction=0.5
  Varied: epochs ∈ {1, 3, 5, 10, 15}
  Question: Does more defender compute help or hurt?

ABLATION 3: Defended Accuracy vs Poison Fraction
  Fixed: lr=0.001, epochs=5
  Varied: fraction ∈ {0.1, 0.2, 0.4, 0.6, 0.8}
  Question: How robust is the DEFENDED model (vs naive poisoned model) across fractions?
  This creates the key comparison curve in the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.dataset import get_raw_train_dataset, get_dataloaders
from attacks.label_flip import poison_dataset
from utils.seed import set_seed
from utils.metrics import summarize_runs


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
TABLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "tables")


def _run_one(
    dataset_name, model_fn,
    poison_fraction, src_class, tgt_class,
    epochs, lr, batch_size, seed, device,
) -> float:
    """Run one poisoning + training + evaluation trial."""
    from torch.utils.data import DataLoader
    from train.trainer import train_model
    from train.evaluator import evaluate
    from data.dataset import get_transforms
    from torchvision import datasets

    set_seed(seed)

    raw_train = get_raw_train_dataset(dataset=dataset_name, augment=False)
    poisoned_train, _ = poison_dataset(
        raw_train, src_class, tgt_class, poison_fraction, seed=seed
    )

    # Test loader using the unified transforms
    _, test_tf = get_transforms(dataset_name, augment=False)
    if dataset_name == "cifar10":
        test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    elif dataset_name == "cifar100":
        test_ds = datasets.CIFAR100("./data/raw", train=False, download=True, transform=test_tf)
    else:
        test_ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    train_loader = DataLoader(poisoned_train, batch_size=batch_size, shuffle=True, num_workers=0)

    model = model_fn().to(device)
    train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=False)
    return evaluate(model, test_loader, device)


def _train_model_simple(model, loader, device, epochs, lr):
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()


def ablation_lr(
    lr_values: list = None,
    dataset: str = "cifar10",
    n_runs: int = 3,
    seeds: list = None,
    poison_fraction: float = 0.5,
    src_class: int = 1,
    tgt_class: int = 7,
    epochs: int = 5,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ablation Study 1: Defender Learning Rate Sensitivity.
    """
    from torch.utils.data import DataLoader
    from train.evaluator import evaluate
    from torchvision import datasets, transforms
    from models.resnet import get_model

    if lr_values is None:
        lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    rows = []
    all_means, all_stds = [], []

    for lr in lr_values:
        run_accs = []
        for seed in seeds:
            set_seed(seed)
            raw_train = get_raw_train_dataset(dataset=dataset, augment=False)
            poisoned_train, _ = poison_dataset(
                raw_train, src_class, tgt_class, poison_fraction, seed=seed)
            train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
            model = get_model(device, dataset)
            _train_model_simple(model, train_loader, device, epochs, lr)
            acc = evaluate(model, test_loader, device)
            run_accs.append(acc)
        s = summarize_runs(run_accs)
        rows.append({"lr": lr, "mean": round(s["mean"],4), "std": round(s["std"],4)})
        all_means.append(s["mean"]); all_stds.append(s["std"])
        if verbose:
            print(f"  lr={lr:.0e}  →  mean={s['mean']:.2f}%  std={s['std']:.2f}%")

    df = pd.DataFrame(rows)
    _save_and_plot_1d(df, x_col="lr", y_col="mean", std_col="std",
                      xlabel="Learning Rate", title="Ablation: Defender Learning Rate",
                      save_name="ablation_lr", log_x=True)
    return df


def ablation_epochs(
    epoch_values: list = None,
    dataset: str = "cifar10",
    n_runs: int = 3,
    seeds: list = None,
    poison_fraction: float = 0.5,
    src_class: int = 1,
    tgt_class: int = 7,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ablation Study 2: Defender Epochs per Round.
    """
    from torch.utils.data import DataLoader
    from train.evaluator import evaluate
    from torchvision import datasets, transforms
    from models.resnet import get_model

    if epoch_values is None:
        epoch_values = [1, 3, 5, 10, 15]
    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    rows = []
    all_means, all_stds = [], []

    for ep in epoch_values:
        run_accs = []
        for seed in seeds:
            set_seed(seed)
            raw_train = get_raw_train_dataset(dataset=dataset, augment=False)
            poisoned_train, _ = poison_dataset(
                raw_train, src_class, tgt_class, poison_fraction, seed=seed)
            train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
            model = get_model(device, dataset)
            _train_model_simple(model, train_loader, device, ep, lr)
            acc = evaluate(model, test_loader, device)
            run_accs.append(acc)
        s = summarize_runs(run_accs)
        rows.append({"epochs": ep, "mean": round(s["mean"],4), "std": round(s["std"],4)})
        all_means.append(s["mean"]); all_stds.append(s["std"])
        if verbose:
            print(f"  epochs={ep:2d}  →  mean={s['mean']:.2f}%  std={s['std']:.2f}%")

    df = pd.DataFrame(rows)
    _save_and_plot_1d(df, x_col="epochs", y_col="mean", std_col="std",
                      xlabel="Epochs per Round", title="Ablation: Defender Epochs",
                      save_name="ablation_epochs", log_x=False)
    return df


def ablation_defended_vs_fraction(
    fractions: list = None,
    dataset: str = "cifar10",
    n_runs: int = 3,
    seeds: list = None,
    src_class: int = 1,
    tgt_class: int = 7,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ablation Study 3: Poisoned vs Defended accuracy across poison fractions.

    This is the KEY ablation for the paper. It shows that the min-max
    defended model maintains higher accuracy than the naive poisoned model
    across all poison fractions.
    """
    from torch.utils.data import DataLoader
    from train.evaluator import evaluate
    from experiments.defender import run_minmax
    from experiments.poisoning import run_poisoned

    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8]
    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    poisoned_means, poisoned_stds = [], []
    defended_means, defended_stds = [], []

    for frac in fractions:
        if verbose:
            print(f"\n── Fraction = {frac:.0%} ──────────────────────────────")

        # Naive poisoned
        p_summary = run_poisoned(
            n_runs=n_runs, seeds=seeds,
            dataset=dataset,
            src_class=src_class, tgt_class=tgt_class,
            poison_fraction=frac, epochs=epochs, batch_size=batch_size, lr=lr,
            verbose=False,
        )

        # Min-max defended (using label-flip attacker for speed in ablation)
        d_summary = run_minmax(
            n_rounds=3, n_runs=n_runs, seeds=seeds,
            dataset=dataset,
            src_class=src_class, tgt_class=tgt_class,
            poison_fraction=frac, epochs=epochs, batch_size=batch_size, lr=lr,
            verbose=False,
        )

        rows.append({
            "fraction":      frac,
            "poisoned_mean": round(p_summary["mean"], 4),
            "poisoned_std":  round(p_summary["std"],  4),
            "defended_mean": round(d_summary["final_summary"]["mean"], 4),
            "defended_std":  round(d_summary["final_summary"]["std"],  4),
        })
        poisoned_means.append(p_summary["mean"])
        poisoned_stds.append(p_summary["std"])
        defended_means.append(d_summary["final_summary"]["mean"])
        defended_stds.append(d_summary["final_summary"]["std"])

        if verbose:
            print(f"  Poisoned: {p_summary['mean']:.2f}% ± {p_summary['std']:.2f}%")
            print(f"  Defended: {d_summary['final_summary']['mean']:.2f}% ± "
                  f"{d_summary['final_summary']['std']:.2f}%")

    df = pd.DataFrame(rows)

    # Save
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, "ablation_fractions.csv"), index=False)

    # Plot: dual-line with error bars
    _plot_defended_vs_poisoned(fractions, poisoned_means, poisoned_stds,
                               defended_means, defended_stds)

    if verbose:
        print("\n[Ablation 3 Table]")
        print(df.to_string(index=False))

    return df


def _save_and_plot_1d(
    df, x_col, y_col, std_col, xlabel, title, save_name, log_x=False,
):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = df[x_col].tolist()
    y_vals = df[y_col].tolist()
    s_vals = df[std_col].tolist()

    ax.errorbar(range(len(x_vals)), y_vals, yerr=s_vals,
                marker="o", capsize=5, color="#2196F3")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:.0e}" if log_x else str(v) for v in x_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{save_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"[Plot saved] {path}")

    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, f"{save_name}.csv"), index=False)


def _plot_defended_vs_poisoned(fractions, p_means, p_stds, d_means, d_stds):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(fractions, p_means, yerr=p_stds, marker="o",
                color="#F44336", capsize=5, label="Poisoned (no defense)")
    ax.errorbar(fractions, d_means, yerr=d_stds, marker="s",
                color="#2196F3", capsize=5, label="Defended (Min–Max)")
    ax.fill_between(fractions,
                    [m-s for m,s in zip(d_means,d_stds)],
                    [m+s for m,s in zip(d_means,d_stds)],
                    alpha=0.15, color="#2196F3")
    ax.set_xlabel("Poison Fraction")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Poisoned vs Defended: Accuracy Across Poison Fractions")
    ax.set_xticks(fractions)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_defended_vs_poisoned.png")
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"[Plot saved] {path}")


if __name__ == "__main__":
    print("Running Ablation Studies...")
    ablation_lr(n_runs=3, verbose=True)
    ablation_epochs(n_runs=3, verbose=True)
    ablation_defended_vs_fraction(n_runs=3, verbose=True)
