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

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.dataset import get_raw_train_dataset
from attacks.label_flip import poison_dataset
from models.wideresnet import get_model   # handles mnist, cifar10, cifar100 (WRN-28-10)
from train.trainer import train_model
from train.evaluator import evaluate
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from utils.statistics import full_significance_report
from utils.plotting import plot_comparison


TABLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints")

def _get_test_loader(dataset: str = "cifar10", batch_size: int = 128, num_workers: int = 2):
    from data.dataset import get_transforms
    _, test_tf = get_transforms(dataset, augment=False)
    pin = torch.cuda.is_available()
    if dataset == "cifar10":
        ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100("./data/raw", train=False, download=True, transform=test_tf)
    else:
        ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
        
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)


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
    num_workers: int = 2,
    verbose: bool = True,
    use_sgd: bool = False,
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
    test_loader = _get_test_loader(dataset, batch_size, num_workers)
    pin = torch.cuda.is_available()

    all_results = {}
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    state_file = os.path.join(CKPT_DIR, f"{dataset}_defense_comparison_state.json")
    
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
            print(f"[Loaded state from {state_file}]")
    else:
        state = {
            "baseline_accs": [],
            "poisoned_accs": [],
            "spec_accs": [],
            "sever_accs": []
        }
        
    def save_state():
        with open(state_file, "w") as f:
            json.dump(state, f)

    # ── 1. Clean Baseline ────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [1/5] CLEAN BASELINE")
    print("═"*60)
    baseline_accs = state["baseline_accs"]
    for i, seed in enumerate(seeds):
        if i < len(baseline_accs):
            print(f"  Run {seed+1}: {baseline_accs[i]:.2f}% (from checkpoint)")
            continue
            
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=True)
        train_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin)
        model = get_model(device, dataset)
        
        epoch_ckpt = os.path.join(CKPT_DIR, f"{dataset}_baseline_s{seed}_epoch.pt")
        train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=False, use_sgd=use_sgd,
                    checkpoint_path=epoch_ckpt, resume_from_checkpoint=epoch_ckpt)
        
        acc = evaluate(model, test_loader, device)
        
        baseline_accs.append(acc)
        save_state()
        if os.path.exists(epoch_ckpt): os.remove(epoch_ckpt)
        print(f"  Run {seed+1}: {acc:.2f}%")
        
    all_results["Clean Baseline"] = summarize_runs(baseline_accs)
    print_summary("Clean Baseline", all_results["Clean Baseline"])

    # ── 2. Poisoned No Defense ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [2/5] POISONED (No Defense)")
    print("═"*60)
    poisoned_accs = state["poisoned_accs"]
    poisoned_models = []  # Save for defenses
    poisoned_datasets = []
    
    for i, seed in enumerate(seeds):
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=False)
        poisoned_train, _ = poison_dataset(raw_train, src_class, tgt_class,
                                           poison_fraction, seed=seed)
        poisoned_datasets.append((poisoned_train, raw_train))
        
        model_ckpt_path = os.path.join(CKPT_DIR, f"{dataset}_poisoned_model_seed_{seed}.pt")
        model = get_model(device, dataset)
        
        if i < len(poisoned_accs):
            # Load weights
            model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
            poisoned_models.append(model)
            print(f"  Run {seed+1}: {poisoned_accs[i]:.2f}% (from checkpoint)")
        else:
            train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=pin)
            
            epoch_ckpt = os.path.join(CKPT_DIR, f"{dataset}_poisoned_s{seed}_epoch.pt")
            train_model(model, train_loader, device, epochs=epochs, lr=lr, verbose=False, use_sgd=use_sgd,
                        checkpoint_path=epoch_ckpt, resume_from_checkpoint=epoch_ckpt)
            
            acc = evaluate(model, test_loader, device)
            
            torch.save(model.state_dict(), model_ckpt_path)
            poisoned_accs.append(acc)
            save_state()
            if os.path.exists(epoch_ckpt): os.remove(epoch_ckpt)
            
            poisoned_models.append(model)
            print(f"  Run {seed+1}: {acc:.2f}%")
            
    all_results["Poisoned (No Defense)"] = summarize_runs(poisoned_accs)
    print_summary("Poisoned (No Defense)", all_results["Poisoned (No Defense)"])

    # ── 3. Spectral Signatures Defense ───────────────────────────────────────
    print("\n" + "═"*60)
    print("  [3/5] SPECTRAL SIGNATURES (Tran et al. 2018)")
    print("═"*60)
    spec_accs = state["spec_accs"]
    for i, seed in enumerate(seeds):
        if i < len(spec_accs):
            print(f"  Run {seed+1}: {spec_accs[i]:.2f}% (from checkpoint)")
            continue
            
        set_seed(seed)
        poisoned_train, raw_train = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin)
        ds_snap = dataset          # snapshot for closure
        dev_snap = device          # snapshot for closure
        epoch_ckpt = os.path.join(CKPT_DIR, f"{dataset}_spectral_s{seed}_epoch.pt")
        _, acc, _ = apply_spectral_defense(
            model=poisoned_models[i],
            model_fn=lambda ds=ds_snap, dv=dev_snap: get_model(dv, ds),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            dataset=dataset,        # FIX: pass dataset so n_classes is inferred correctly
            num_workers=num_workers,
            verbose=verbose and i == 0,
            use_sgd=use_sgd,
            checkpoint_path=epoch_ckpt,
            resume_from_checkpoint=epoch_ckpt,
        )
        spec_accs.append(acc)
        save_state()
        if os.path.exists(epoch_ckpt): os.remove(epoch_ckpt)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Spectral Signatures"] = summarize_runs(spec_accs)
    print_summary("Spectral Signatures", all_results["Spectral Signatures"])

    # ── 4. SEVER Defense ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  [4/5] SEVER (Diakonikolas et al. 2019)")
    print("═"*60)
    sever_accs = state["sever_accs"]
    for i, seed in enumerate(seeds):
        if i < len(sever_accs):
            print(f"  Run {seed+1}: {sever_accs[i]:.2f}% (from checkpoint)")
            continue
            
        set_seed(seed)
        poisoned_train, _ = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin)
        ds_snap = dataset
        dev_snap = device
        epoch_ckpt = os.path.join(CKPT_DIR, f"{dataset}_sever_s{seed}_epoch.pt")
        _, acc, _ = apply_sever_defense(
            model=poisoned_models[i],
            model_fn=lambda ds=ds_snap, dv=dev_snap: get_model(dv, ds),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            verbose=verbose and i == 0,
            use_sgd=use_sgd,
            checkpoint_path=epoch_ckpt,
            resume_from_checkpoint=epoch_ckpt,
        )
        sever_accs.append(acc)
        save_state()
        if os.path.exists(epoch_ckpt): os.remove(epoch_ckpt)
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
        baseline_mean=all_results["Clean Baseline"]["mean"],
        poisoned_mean=all_results["Poisoned (No Defense)"]["mean"],
        verbose=verbose,
        use_sgd=use_sgd,
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
        title=f"Defense Method Comparison ({dataset.upper()}, poison fraction={int(poison_fraction*100)}%)",
    )

    return df


if __name__ == "__main__":
    df = run_defense_comparison(n_runs=3, dataset="cifar10", epochs=10)
    print(df)
