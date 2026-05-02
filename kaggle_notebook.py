# ╔══════════════════════════════════════════════════════════════╗
# ║              KAGGLE MASTER NOTEBOOK                          ║
# ║  Self-contained. All bugs patched inline. No re-upload.     ║
# ╚══════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════
# CELL 1 — Setup & Copy
# ════════════════════════════════════════════════════════════════
import os, sys, json, shutil, torch, pickle, pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DST = "/kaggle/working/ml-research"
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree("/kaggle/input/ml-research1", DST)
os.chdir(DST)
sys.path.insert(0, DST)

SAVE_DIR = "/kaggle/working/saved_results"
for d in [SAVE_DIR, "results/figures", "results/tables", "results/checkpoints"]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"✅ GPU    : {torch.cuda.get_device_name(0)}")
print(f"✅ Root   : {DST}")


# ════════════════════════════════════════════════════════════════
# CELL 2 — Patch ALL bugs inline
# ════════════════════════════════════════════════════════════════

# ── PATCH 1: train_model — SGD + CosineAnnealingLR + new AMP API ────────────
def _patched_train_model(
    model, loader, device, epochs=5, lr=0.001,
    show_progress=False, verbose=True,
    checkpoint_path=None, resume_from_checkpoint=None,
    use_sgd=False,
):
    criterion = nn.CrossEntropyLoss()
    if use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None

    start_epoch = 1
    if resume_from_checkpoint and os.path.isfile(resume_from_checkpoint):
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        if verbose:
            print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']}")

    if verbose:
        opt = "SGD+CosineAnneal" if use_sgd else "Adam"
        print(f"  Training {epochs-(start_epoch-1)} epoch(s) "
              f"[{start_epoch}..{epochs}], lr={lr}, opt={opt}")

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if scheduler:
            scheduler.step()
        if verbose:
            cur_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"  Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}  lr={cur_lr:.6f}")

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            }
            if scheduler:
                ckpt_data["scheduler_state"] = scheduler.state_dict()
            torch.save(ckpt_data, checkpoint_path)
    return model

# Patch the function in ALL modules that imported it at load time
import train.trainer as _trainer_mod
import experiments.baseline as _base_mod
import experiments.poisoning as _pois_mod
import experiments.defender as _def_mod

_trainer_mod.train_model = _patched_train_model
_base_mod.train_model    = _patched_train_model   # baseline.py imported it at top
_pois_mod.train_model    = _patched_train_model   # poisoning.py imported it at top
_def_mod.train_model     = _patched_train_model   # defender.py imported it at top ← CRITICAL
print("✅ Patch 1: train_model (SGD+CosineAnnealingLR, patched in all modules)")


# ── PATCH 2: apply_spectral_defense — add use_sgd ───────────────────────────
import defenses.spectral_signatures as _spec_mod

def _patched_spectral(
    model, model_fn, train_dataset, train_loader, test_loader,
    device, defender_epochs=10, defender_lr=0.001,
    suspicious_quantile=0.95, n_classes=10, verbose=True, use_sgd=False,
):
    from train.evaluator import evaluate
    clean_idx, stats = _spec_mod.spectral_signatures_filter(
        model, train_dataset, device,
        suspicious_quantile=suspicious_quantile,
        n_classes=n_classes, verbose=verbose,
    )
    clean_loader = DataLoader(
        Subset(train_dataset, clean_idx),
        batch_size=train_loader.batch_size,
        shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    if verbose:
        print(f"\n[SpectralSignatures] Retraining on {len(clean_idx):,} clean samples...")
    clean_model = model_fn().to(device)
    _patched_train_model(clean_model, clean_loader, device,
                         epochs=defender_epochs, lr=defender_lr,
                         verbose=verbose, use_sgd=use_sgd)
    acc = evaluate(clean_model, test_loader, device)
    stats["final_accuracy"] = acc
    if verbose:
        print(f"[SpectralSignatures] Final accuracy: {acc:.2f}%")
    return clean_model, acc, stats

_spec_mod.apply_spectral_defense = _patched_spectral
print("✅ Patch 2: apply_spectral_defense (use_sgd)")


# ── PATCH 3: apply_sever_defense — add use_sgd ──────────────────────────────
import defenses.sever as _sever_mod

def _patched_sever(
    model, model_fn, train_dataset, train_loader, test_loader,
    device, defender_epochs=10, defender_lr=0.001,
    removal_fraction=0.05, verbose=True, use_sgd=False,
):
    from train.evaluator import evaluate
    clean_idx, stats = _sever_mod.sever_filter(
        model, train_dataset, device,
        removal_fraction=removal_fraction, verbose=verbose,
    )
    clean_loader = DataLoader(
        Subset(train_dataset, clean_idx),
        batch_size=train_loader.batch_size,
        shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    if verbose:
        print(f"\n[SEVER] Retraining on {len(clean_idx):,} samples...")
    clean_model = model_fn().to(device)
    _patched_train_model(clean_model, clean_loader, device,
                         epochs=defender_epochs, lr=defender_lr,
                         verbose=verbose, use_sgd=use_sgd)
    acc = evaluate(clean_model, test_loader, device)
    stats["final_accuracy"] = acc
    if verbose:
        print(f"[SEVER] Final accuracy: {acc:.2f}%")
    return clean_model, acc, stats

_sever_mod.apply_sever_defense = _patched_sever
print("✅ Patch 3: apply_sever_defense (use_sgd)")


# ── PATCH 4: run_defense_comparison — fully rewired ─────────────────────────
from data.dataset import get_raw_train_dataset
from attacks.label_flip import poison_dataset
from models.resnet import get_model
from train.evaluator import evaluate
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from utils.statistics import full_significance_report

def _patched_defense_comparison(
    n_runs=3, seeds=None, dataset="cifar10",
    src_class=1, tgt_class=7, poison_fraction=0.5,
    epochs=50, defense_epochs=50, batch_size=128,
    lr=0.1, verbose=False, use_sgd=True,
):
    if seeds is None:
        seeds = list(range(n_runs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test loader
    norm_stats = {
        "cifar10": ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        "mnist":   ((0.1307,),(0.3081,)),
    }
    mean, std = norm_stats.get(dataset, norm_stats["cifar10"])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == "cifar10":
        test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    else:
        test_ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_results = {}

    # 1. Clean Baseline
    print("\n" + "═"*60)
    print("  [1/5] CLEAN BASELINE")
    print("═"*60)
    baseline_accs = []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=True)
        train_loader = DataLoader(raw_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        model = get_model(device, dataset)
        _patched_train_model(model, train_loader, device, epochs=epochs,
                             lr=lr, verbose=False, use_sgd=use_sgd)
        acc = evaluate(model, test_loader, device)
        baseline_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Clean Baseline"] = summarize_runs(baseline_accs)
    print_summary("Clean Baseline", all_results["Clean Baseline"])

    # 2. Poisoned (No Defense)
    print("\n" + "═"*60)
    print("  [2/5] POISONED (No Defense)")
    print("═"*60)
    poisoned_accs, poisoned_models, poisoned_datasets = [], [], []
    for seed in seeds:
        set_seed(seed)
        raw_train = get_raw_train_dataset(dataset=dataset, augment=False)
        poisoned_train, _ = poison_dataset(raw_train, src_class, tgt_class,
                                           poison_fraction, seed=seed)
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        model = get_model(device, dataset)
        _patched_train_model(model, train_loader, device, epochs=epochs,
                             lr=lr, verbose=False, use_sgd=use_sgd)
        acc = evaluate(model, test_loader, device)
        poisoned_accs.append(acc)
        poisoned_models.append(model)
        poisoned_datasets.append(poisoned_train)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Poisoned (No Defense)"] = summarize_runs(poisoned_accs)
    print_summary("Poisoned (No Defense)", all_results["Poisoned (No Defense)"])

    # 3. Spectral Signatures
    print("\n" + "═"*60)
    print("  [3/5] SPECTRAL SIGNATURES")
    print("═"*60)
    spec_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        poisoned_train = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        dv, ds = device, dataset
        _, acc, _ = _patched_spectral(
            model=poisoned_models[i],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            verbose=False,
            use_sgd=use_sgd,
        )
        spec_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Spectral Signatures"] = summarize_runs(spec_accs)
    print_summary("Spectral Signatures", all_results["Spectral Signatures"])

    # 4. SEVER
    print("\n" + "═"*60)
    print("  [4/5] SEVER")
    print("═"*60)
    sever_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        poisoned_train = poisoned_datasets[i]
        train_loader = DataLoader(poisoned_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        dv, ds = device, dataset
        _, acc, _ = _patched_sever(
            model=poisoned_models[i],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_train,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            defender_epochs=defense_epochs,
            defender_lr=lr,
            verbose=False,
            use_sgd=use_sgd,
        )
        sever_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["SEVER"] = summarize_runs(sever_accs)
    print_summary("SEVER", all_results["SEVER"])

    # 5. Min-Max (Ours)
    print("\n" + "═"*60)
    print("  [5/5] OURS: Min-Max")
    print("═"*60)
    minmax_results = _def_mod.run_minmax(
        n_rounds=5, n_runs=n_runs, seeds=seeds,
        dataset=dataset, src_class=src_class, tgt_class=tgt_class,
        poison_fraction=poison_fraction,
        epochs=epochs, batch_size=batch_size, lr=lr,
        verbose=True, use_sgd=use_sgd,
    )
    all_results["Ours (Min-Max)"] = summarize_runs(minmax_results["round_means"])

    # Build & save table
    baseline_mean = all_results["Clean Baseline"]["mean"]
    rows = [
        {
            "Method":          method,
            "Mean Acc (%)":    round(s["mean"], 2),
            "Std (%)":         round(s["std"],  2),
            "vs Baseline (%)": round(s["mean"] - baseline_mean, 2),
        }
        for method, s in all_results.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv("./results/tables/defense_comparison.csv", index=False)
    print("\n" + "═"*60)
    print("  FINAL COMPARISON TABLE")
    print("═"*60)
    print(df.to_string(index=False))
    return df

import experiments.defense_comparison as _dc_mod
_dc_mod.run_defense_comparison = _patched_defense_comparison
print("✅ Patch 4: run_defense_comparison (fully rewired)")

print("\n" + "✅"*20)
print("ALL PATCHES APPLIED — safe to run experiments")
print("✅"*20)


# ════════════════════════════════════════════════════════════════
# CELL 3 — MNIST (~5 mins)
# ════════════════════════════════════════════════════════════════
from experiments.baseline import run_baseline
from experiments.poisoning import run_poisoned
from experiments.defender import run_minmax

print("\n--- MNIST BASELINE ---")
mnist_base = run_baseline(n_runs=3, epochs=5, batch_size=64)
with open(f"{SAVE_DIR}/mnist_base.json", "w") as f:
    json.dump(mnist_base, f)
print("✅ mnist_base saved")

print("\n--- MNIST POISONED ---")
mnist_pois = run_poisoned(n_runs=3, epochs=5, batch_size=64,
                          src_class=1, tgt_class=7, poison_fraction=0.5)
with open(f"{SAVE_DIR}/mnist_pois.json", "w") as f:
    json.dump(mnist_pois, f)
print("✅ mnist_pois saved")

print("\n--- MNIST MIN-MAX ---")
mnist_def = run_minmax(n_rounds=3, n_runs=3, epochs=5, batch_size=64,
                       src_class=1, tgt_class=7, poison_fraction=0.5)
mnist_def_save = {k: v for k, v in mnist_def.items() if k != "all_rounds_df"}
mnist_def["all_rounds_df"].to_csv(f"{SAVE_DIR}/mnist_def_rounds.csv", index=False)
with open(f"{SAVE_DIR}/mnist_def.json", "w") as f:
    json.dump(mnist_def_save, f)
print("✅ mnist_def saved")
print(f"\n✅ MNIST COMPLETE — files: {os.listdir(SAVE_DIR)}")


# ════════════════════════════════════════════════════════════════
# CELL 4 — CIFAR-10 (~5-6 hrs)
# ════════════════════════════════════════════════════════════════
from experiments.defense_comparison import run_defense_comparison

cifar10_df = run_defense_comparison(
    n_runs=3, dataset='cifar10',
    epochs=50, defense_epochs=50,
    lr=0.1, batch_size=128,
    use_sgd=True, verbose=False,
)
cifar10_df.to_csv(f"{SAVE_DIR}/cifar10_df.csv", index=False)
print("\n✅ CIFAR-10 done & saved")
print(cifar10_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════
# CELL 5 — CIFAR-100 (~3 hrs, checkpointed per epoch)
# NOTE: We call baseline + poisoned directly.
#       run_cifar100_full_pipeline also runs a poison-fraction SWEEP
#       (4 fractions × 3 runs × 100 epochs = 1,200 extra WRN epochs ≈ 12 hrs)
#       which would exceed Kaggle's session limit. We skip it.
# ════════════════════════════════════════════════════════════════
from experiments.cifar100_experiment import run_cifar100_baseline, run_cifar100_poisoned
from utils.statistics import full_significance_report
from utils.plotting import plot_comparison

print("\n[CIFAR-100 STEP 1] Clean Baseline (WideResNet-28-10, 100 epochs × 3 runs)")
c100_baseline = run_cifar100_baseline(n_runs=3, epochs=100, batch_size=128, verbose=True)

print("\n[CIFAR-100 STEP 2] Label-Flip Attack (100 epochs × 3 runs)")
c100_poisoned = run_cifar100_poisoned(n_runs=3, epochs=100, batch_size=128, verbose=True)

print("\n[CIFAR-100 STEP 3] Statistical Significance")
full_significance_report(
    baseline_accs=c100_baseline["runs"],
    poisoned_accs=c100_poisoned["runs"],
    defended_accs=c100_poisoned["runs"],  # placeholder; no CIFAR-100 defense run
    verbose=True,
)

plot_comparison(
    labels=["CIFAR-100\nBaseline", "CIFAR-100\nPoisoned"],
    means=[c100_baseline["mean"], c100_poisoned["mean"]],
    stds=[c100_baseline["std"],  c100_poisoned["std"]],
    save_name="cifar100_baseline_vs_poisoned.png",
    title="CIFAR-100: Baseline vs Poisoned (WideResNet-28-10)",
)

cifar100_results = {
    "baseline": c100_baseline,
    "poisoned": c100_poisoned,
}

with open(f"{SAVE_DIR}/cifar100_results.pkl", "wb") as f:
    pickle.dump(cifar100_results, f)
print("\n✅ CIFAR-100 done & saved")
print(f"   Baseline : {c100_baseline['mean']:.2f}% ± {c100_baseline['std']:.2f}%")
print(f"   Poisoned : {c100_poisoned['mean']:.2f}% ± {c100_poisoned['std']:.2f}%")


# ════════════════════════════════════════════════════════════════
# CELL 6 — Final Table (load from disk, no retraining)
# ════════════════════════════════════════════════════════════════
from experiments.multidataset_comparison import build_multidataset_table

with open(f"{SAVE_DIR}/mnist_base.json") as f: mnist_base = json.load(f)
with open(f"{SAVE_DIR}/mnist_pois.json") as f: mnist_pois = json.load(f)
with open(f"{SAVE_DIR}/mnist_def.json")  as f: mnist_def  = json.load(f)
cifar10_df = pd.read_csv(f"{SAVE_DIR}/cifar10_df.csv")
with open(f"{SAVE_DIR}/cifar100_results.pkl", "rb") as f:
    cifar100_results = pickle.load(f)

def _row(df, method):
    r = df[df["Method"] == method]
    if len(r) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(r["Mean Acc (%)"].values[0]),
            "std":  float(r["Std (%)"].values[0])}

all_res = {
    "mnist": {
        "baseline": mnist_base,
        "poisoned": mnist_pois,
        "defended": {
            "mean": mnist_def["final_summary"]["mean"],
            "std":  mnist_def["final_summary"]["std"],
        },
    },
    "cifar10": {
        "baseline": _row(cifar10_df, "Clean Baseline"),
        "poisoned": _row(cifar10_df, "Poisoned (No Defense)"),
        "defended": _row(cifar10_df, "Ours (Min-Max)"),
        "spectral": _row(cifar10_df, "Spectral Signatures"),
        "sever":    _row(cifar10_df, "SEVER"),
    },
    "cifar100": {
        "baseline": cifar100_results["baseline"],
        "poisoned": cifar100_results["poisoned"],
    },
}

build_multidataset_table(all_res)
print("\n✅ ALL EXPERIMENTS COMPLETE")
