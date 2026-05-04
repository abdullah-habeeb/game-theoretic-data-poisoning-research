"""
experiments/cifar100_experiment.py
====================================
Complete CIFAR-100 experimental pipeline.

CIFAR-100 SPECIFICS:
  - 60,000 images (50k train, 10k test), 32×32 RGB
  - 100 fine-grained classes organized in 20 superclasses (5 classes each)
  - State-of-the-art clean accuracy with WRN-28-10: ~80–81%
  - Attack pair used: aquarium_fish (1) → flatfish (32)
    Both are in the "fish" superclass — a semantically covert attack
    (hard to detect because both classes look similar)

WHY THIS ATTACK PAIR?
  Attacking within the same superclass simulates a realistic adversary
  who wants to be subtle. The labels look plausible to human reviewers.
  An aquarium_fish mislabeled as flatfish is hard to detect without
  careful per-sample inspection.

EXPERIMENT PLAN:
  1. Clean WRN-28-10 baseline (3 runs)
  2. Label-flip attack (src=1, tgt=32, fraction=0.5) (3 runs)
  3. Gradient-based attack (3 runs) ← stronger attacker
  4. Spectral Signatures defense (3 runs)
  5. SEVER defense (3 runs)
  6. Min–Max defender (5 rounds, 3 runs)
  7. Statistical significance for all comparisons

HOW TO RUN:
  python -m experiments.cifar100_experiment
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.dataset import get_raw_train_dataset, get_dataloaders
from models.wideresnet import get_model, CIFAR100_DEFAULT_SRC, CIFAR100_DEFAULT_TGT
from train.trainer import train_model
from train.evaluator import evaluate
from attacks.label_flip import poison_dataset
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from utils.statistics import full_significance_report
from utils.plotting import plot_comparison

TABLES_DIR  = "./results/tables"
FIGURES_DIR = "./results/figures"


def _test_loader_cifar100(batch_size: int = 128) -> DataLoader:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    ds = datasets.CIFAR100("./data/raw", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_cifar100_baseline(
    n_runs: int = 3,
    seeds: list = None,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Clean WRN-28-10 baseline on CIFAR-100.

    TRAINING NOTES:
      WideResNet on CIFAR-100 uses SGD with momentum + cosine LR schedule
      (the standard setup for CIFAR-100 in adversarial ML papers).
      We train for 100 epochs with:
        - SGD + momentum=0.9 + weight_decay=5e-4
        - Cosine LR schedule: lr_max=0.1 → 0 over 100 epochs
      This is the setup from Zagoruyko & Komodakis (2016).
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.nn as nn

    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _test_loader_cifar100(batch_size)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CIFAR-100 BASELINE  (WideResNet-28-10)")
        print(f"  {n_runs} runs × {epochs} epochs  |  Device: {device}")
        print(f"{'='*65}")

    accs = []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset("cifar100", augment=True)
        train_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True, num_workers=2)

        model = get_model(device, "cifar100")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        # ── Checkpoint: resume if a prior run was interrupted ────────────────────
        ckpt_path = f"./results/checkpoints/cifar100_baseline_s{seed}.pt"
        os.makedirs("./results/checkpoints", exist_ok=True)
        start_epoch = 1
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"  [Checkpoint] Seed={seed}: resumed from epoch {ckpt['epoch']}/{epochs}")

        if verbose:
            print(f"\n  [Run seed={seed}]  Training WRN-28-10 on CIFAR-100...")

        model.train()
        for epoch in range(start_epoch, epochs + 1):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()
            # Save checkpoint after every epoch
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, ckpt_path)
            if verbose and (epoch % 20 == 0 or epoch == epochs):
                acc = evaluate(model, test_loader, device)
                print(f"    Epoch {epoch:3d}/{epochs}  test_acc={acc:.2f}%  lr={scheduler.get_last_lr()[0]:.5f}")

        acc = evaluate(model, test_loader, device)
        accs.append(acc)
        # Clean up checkpoint on successful completion
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)
        if verbose:
            print(f"  → Final baseline accuracy: {acc:.2f}%")

    summary = summarize_runs(accs)
    if verbose:
        print_summary("CIFAR-100 BASELINE", summary)
    return summary


def run_cifar100_poisoned(
    n_runs: int = 3,
    seeds: list = None,
    src_class: int = CIFAR100_DEFAULT_SRC,
    tgt_class: int = CIFAR100_DEFAULT_TGT,
    poison_fraction: float = 0.5,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    WRN-28-10 trained on poisoned CIFAR-100 (label-flip attack).

    Default attack: aquarium_fish (1) → flatfish (32)
    Both in the same 'fish' superclass — a covert, semantically plausible attack.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.nn as nn

    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = _test_loader_cifar100(batch_size)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CIFAR-100 POISONING  (WideResNet-28-10)")
        print(f"  Attack: class {src_class} → class {tgt_class}, fraction={poison_fraction:.0%}")
        print(f"{'='*65}")

    accs = []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset("cifar100", augment=True)
        poisoned_train, _ = poison_dataset(
            raw_train, src_class, tgt_class, poison_fraction, seed=seed)

        train_loader = DataLoader(
            poisoned_train, batch_size=batch_size, shuffle=True, num_workers=2)

        model = get_model(device, "cifar100")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        # ── Checkpoint: resume if a prior run was interrupted ────────────────────
        ckpt_path = f"./results/checkpoints/cifar100_poisoned_s{seed}.pt"
        os.makedirs("./results/checkpoints", exist_ok=True)
        start_epoch = 1
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"  [Checkpoint] Seed={seed}: resumed from epoch {ckpt['epoch']}/{epochs}")

        if verbose:
            print(f"\n  [Run seed={seed}]  Training on poisoned CIFAR-100...")

        model.train()
        for epoch in range(start_epoch, epochs + 1):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()
            # Save checkpoint after every epoch
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, ckpt_path)
            if verbose and (epoch % 20 == 0 or epoch == epochs):
                acc = evaluate(model, test_loader, device)
                print(f"    Epoch {epoch:3d}/{epochs}  test_acc={acc:.2f}%")

        acc = evaluate(model, test_loader, device)
        accs.append(acc)
        # Clean up checkpoint on successful completion
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)
        if verbose:
            print(f"  → Poisoned accuracy: {acc:.2f}%")

    summary = summarize_runs(accs)
    if verbose:
        print_summary(
            f"CIFAR-100 POISONED (src={src_class}→tgt={tgt_class}, frac={poison_fraction:.0%})",
            summary)
    return summary


def run_cifar100_sweep(
    fractions: list = None,
    n_runs: int = 3,
    seeds: list = None,
    src_class: int = CIFAR100_DEFAULT_SRC,
    tgt_class: int = CIFAR100_DEFAULT_TGT,
    epochs: int = 100,
    batch_size: int = 128,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Poison fraction sweep on CIFAR-100.
    Same fractions as CIFAR-10 for direct comparison in the paper.
    """
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8]
    if seeds is None:
        seeds = list(range(n_runs))

    rows = []
    means, stds = [], []

    for frac in fractions:
        if verbose:
            print(f"\n── CIFAR-100 Sweep: fraction = {frac:.0%} ──────────────")
        s = run_cifar100_poisoned(
            n_runs=n_runs, seeds=seeds,
            src_class=src_class, tgt_class=tgt_class,
            poison_fraction=frac, epochs=epochs,
            batch_size=batch_size, verbose=False,
        )
        rows.append({"fraction": frac, "mean": round(s["mean"],4), "std": round(s["std"],4)})
        means.append(s["mean"]); stds.append(s["std"])
        if verbose:
            print(f"  mean={s['mean']:.2f}%  std={s['std']:.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    df.to_csv(f"{TABLES_DIR}/cifar100_sweep.csv", index=False)

    from utils.plotting import plot_sweep
    plot_sweep(fractions, means, stds,
               save_name="cifar100_sweep.png",
               title=f"CIFAR-100: Accuracy vs Poison Fraction (class {src_class}→{tgt_class})")

    if verbose:
        print(df.to_string(index=False))
    return df


def run_cifar100_full_pipeline(
    n_runs: int = 3,
    seeds: list = None,
    epochs: int = 100,
    batch_size: int = 128,
    verbose: bool = True,
) -> dict:
    """
    Run the complete CIFAR-100 research pipeline using the bulletproof
    defense_comparison.py engine, which handles epoch-level checkpointing
    and state persistence for all 5 methods:
      1. Clean baseline (WRN-28-10)
      2. Poisoned (label-flip, no defense)
      3. Spectral Signatures defense
      4. SEVER defense
      5. Min–Max alternating training (our method)

    Returns a dict with all results.
    """
    from experiments.defense_comparison import run_defense_comparison

    if seeds is None:
        seeds = list(range(n_runs))

    print("\n" + "🔬"*30)
    print("  CIFAR-100 FULL RESEARCH PIPELINE")
    print("🔬"*30)
    print(f"\n  Routing through defense_comparison engine")
    print(f"  Dataset : CIFAR-100  |  Model: WideResNet-28-10")
    print(f"  Attack  : aquarium_fish (1) → flatfish (32)  |  fraction=50%")
    print(f"  Epochs  : {epochs}  |  Runs: {n_runs}  |  Optimizer: SGD+CosineAnneal")

    df = run_defense_comparison(
        n_runs=n_runs,
        seeds=seeds,
        dataset="cifar100",
        src_class=CIFAR100_DEFAULT_SRC,   # 1 = aquarium_fish
        tgt_class=CIFAR100_DEFAULT_TGT,   # 32 = flatfish
        poison_fraction=0.5,
        epochs=epochs,
        defense_epochs=epochs,
        batch_size=batch_size,
        lr=0.1,            # WRN standard lr
        num_workers=2,
        use_sgd=True,      # SGD + CosineAnneal is the standard for WRN/CIFAR-100
        verbose=verbose,
    )

    print("\n\n✅ CIFAR-100 Full Pipeline Complete!")
    print(df.to_string(index=False))

    # Poison fraction sweep (after the main run)
    print("\n\n[STEP 2] Poison Fraction Sweep")
    sweep_df = run_cifar100_sweep(
        n_runs=n_runs, seeds=seeds, epochs=epochs,
        batch_size=batch_size, verbose=verbose,
    )

    return {
        "defense_comparison": df,
        "sweep": sweep_df,
    }


if __name__ == "__main__":
    results = run_cifar100_full_pipeline(n_runs=3, epochs=100, batch_size=128)

