"""
experiments/defense_comparison.py
===================================
Head-to-head comparison of all three defenses vs poisoned baseline.

DEFENSES COMPARED:
  1. Min–Max Alternating Training (our contribution)
  2. Spectral Signatures (Tran et al., NeurIPS 2018)
  3. SEVER (Diakonikolas et al., ICML 2019)
  4. FedMedian (Byzantine-robust aggregation, Yin et al., ICML 2018)

This generates Tables 1 and Figure 3 in the paper.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.dataset import get_raw_train_dataset
from attacks.label_flip import poison_dataset
from models.resnet import get_model
from train.trainer import train_model
from train.evaluator import evaluate
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from utils.statistics import full_significance_report
from utils.plotting import plot_comparison


TABLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def _get_test_loader(dataset: str = "cifar10", batch_size: int = 128):
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465) if dataset == "cifar10" else (0.1307,),
            (0.2023,0.1994,0.2010) if dataset == "cifar10" else (0.3081,),
        )
    ])
    if dataset == "cifar10":
        ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    else:
        ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_defense_comparison(
    n_runs: int = 3,
    seeds: list = None,
    dataset: str = "cifar10",
    src_class: int = 1,
    tgt_class: int = 7,
    poison_fraction: float = 0.5,
    epochs: int = 10,
    defense_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all defenses on the same poisoned data and compare.

    Returns a DataFrame summary table ready for the paper.
    """
    from defenses.spectral_signatures import apply_spectral_defense
    from defenses.sever import apply_sever_defense
    from experiments.defender import run_minmax

    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _get_test_loader(dataset, batch_size)

    all_results = {}

    # ── 1. Clean Baseline ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [1/5] CLEAN BASELINE")
    print("═"*60)
    baseline_accs = []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=True)
        train_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True, num_workers=0)
        model = get_model(device, dataset)
        train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=False)
        acc = evaluate(model, test_loader, device)
        baseline_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Clean Baseline"] = summarize_runs(baseline_accs)
    print_summary("Clean Baseline", all_results["Clean Baseline"])

    # ── 2. Poisoned No Defense ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [2/5] POISONED (No Defense)")
    print("═"*60)
    poisoned_accs = []
    poisoned_models = []  # Save for defenses
    poisoned_datasets = []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=False)
        poisoned_train, _ = poison_dataset(raw_train, src_class, tgt_class,
                                           poison_fraction, seed=seed)
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        model = get_model(device, dataset)
        train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=False)
        acc = evaluate(model, test_loader, device)
        poisoned_accs.append(acc)
        poisoned_models.append(model)
        poisoned_datasets.append((poisoned_train, raw_train))
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Poisoned (No Defense)"] = summarize_runs(poisoned_accs)
    print_summary("Poisoned (No Defense)", all_results["Poisoned (No Defense)"])

    # ── 3. Spectral Signatures Defense ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  [3/5] SPECTRAL SIGNATURES (Tran et al. 2018)")
    print("═"*60)
    spec_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        poisoned_train, raw_train = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        _, acc, _ = apply_spectral_defense(
            model=poisoned_models[i],
            model_fn=lambda: get_model(device, dataset),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            verbose=verbose and i == 0,
        )
        spec_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Spectral Signatures"] = summarize_runs(spec_accs)
    print_summary("Spectral Signatures", all_results["Spectral Signatures"])

    # ── 4. SEVER Defense ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [4/5] SEVER (Diakonikolas et al. 2019)")
    print("═"*60)
    sever_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        poisoned_train, _ = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        _, acc, _ = apply_sever_defense(
            model=poisoned_models[i],
            model_fn=lambda: get_model(device, dataset),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            verbose=verbose and i == 0,
        )
        sever_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["SEVER"] = summarize_runs(sever_accs)
    print_summary("SEVER", all_results["SEVER"])

    # ── 5. Min–Max (Our Method) ───────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [5/5] OURS: Min–Max Alternating Training")
    print("═"*60)
    minmax_results = run_minmax(
        n_rounds=5, n_runs=n_runs, seeds=seeds,
        dataset=dataset,
        src_class=src_class, tgt_class=tgt_class,
        poison_fraction=poison_fraction,
        epochs=epochs, batch_size=batch_size, lr=lr,
        verbose=verbose,
    )
    minmax_accs = minmax_results["round_means"]
    final_minmax = minmax_results["final_summary"]
    all_results["Ours (Min–Max)"] = summarize_runs(minmax_results["round_means"])

    # ── Statistical Significance ──────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STATISTICAL SIGNIFICANCE TESTING")
    print("═"*60)
    sig_report = full_significance_report(
        baseline_accs=baseline_accs,
        poisoned_accs=poisoned_accs,
        defended_accs=final_minmax.get("runs", minmax_results["round_means"]),
    )

    # ── Build Summary Table ───────────────────────────────────────────────────
    rows = []
    for method, summary in all_results.items():
        rows.append({
            "Method":         method,
            "Mean Acc (%)":   round(summary["mean"], 2),
            "Std (%)":        round(summary["std"],  2),
            "vs Baseline (%)":round(summary["mean"] - all_results["Clean Baseline"]["mean"], 2),
        })
    df = pd.DataFrame(rows)

    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLES_DIR, "defense_comparison.csv"), index=False)

    print("\n" + "═"*60)
    print("  FINAL COMPARISON TABLE")
    print("═"*60)
    print(df.to_string(index=False))

    # ── Comparison Plot ───────────────────────────────────────────────────────
    colors = ["#4CAF50", "#F44336", "#9C27B0", "#FF9800", "#2196F3"]
    plot_comparison(
        labels=df["Method"].tolist(),
        means=df["Mean Acc (%)"].tolist(),
        stds=df["Std (%)"].tolist(),
        colors=colors,
        save_name="defense_comparison.png",
        title="Defense Method Comparison (CIFAR-10, poison fraction=50%)",
    )

    return df


if __name__ == "__main__":
    df = run_defense_comparison(n_runs=3, dataset="cifar10", epochs=10)
    print(df)
