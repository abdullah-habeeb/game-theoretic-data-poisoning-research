
# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — ALL PATCHES  (fixes every known bug inline)           ║
# ║  Run this immediately after Cell 1. Takes ~5 seconds.           ║
# ╚══════════════════════════════════════════════════════════════════╝

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ══════════════════════════════════════════════════════════════════════════════
# PATCH 1 — train_model
# Fixes: Adam+lr=0.1 divergence, missing CosineAnnealingLR, deprecated AMP API
# Also patches the frozen import binding in defender.py, baseline.py, poisoning.py
# ══════════════════════════════════════════════════════════════════════════════
def _patched_train_model(
    model, loader, device, epochs=5, lr=0.001,
    show_progress=False, verbose=True,
    checkpoint_path=None, resume_from_checkpoint=None,
    use_sgd=False,
):
    criterion = nn.CrossEntropyLoss()
    if use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
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
        print(f"  Training {epochs-(start_epoch-1)} epoch(s) [{start_epoch}..{epochs}], lr={lr}, opt={opt}")

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

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
            n_batches  += 1

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

import train.trainer       as _trainer_mod
import experiments.baseline  as _base_mod
import experiments.poisoning  as _pois_mod
import experiments.defender   as _def_mod

_trainer_mod.train_model = _patched_train_model
_base_mod.train_model    = _patched_train_model
_pois_mod.train_model    = _patched_train_model
_def_mod.train_model     = _patched_train_model  # ← critical: frozen import fix
print("✅ Patch 1: train_model (SGD+CosineAnnealingLR, patched in all 4 modules)")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 2 — apply_spectral_defense
# Fixes: missing use_sgd parameter, num_workers bypass via _patched_extract_features
# ══════════════════════════════════════════════════════════════════════════════
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
print("✅ Patch 2: apply_spectral_defense (use_sgd wired)")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 3 — apply_sever_defense
# Fixes: missing use_sgd parameter
# ══════════════════════════════════════════════════════════════════════════════
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
print("✅ Patch 3: apply_sever_defense (use_sgd wired)")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 4 — run_defense_comparison
# Fixes: lambda closure bug, use_sgd not propagated, num_workers
# ══════════════════════════════════════════════════════════════════════════════
from data.dataset import get_raw_train_dataset
from attacks.label_flip import poison_dataset
from models.resnet import get_model
from train.evaluator import evaluate
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary

def _patched_defense_comparison(
    n_runs=3, seeds=None, dataset="cifar10",
    src_class=1, tgt_class=7, poison_fraction=0.5,
    epochs=50, defense_epochs=50, batch_size=128,
    lr=0.1, verbose=False, use_sgd=True,
):
    if seeds is None:
        seeds = list(range(n_runs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = {
        "cifar10": ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        "mnist":   ((0.1307,),(0.3081,)),
    }
    mean, std = norm.get(dataset, norm["cifar10"])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == "cifar10":
        test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    else:
        test_ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_results = {}

    print("\n" + "═"*60 + "\n  [1/5] CLEAN BASELINE\n" + "═"*60)
    baseline_accs = []
    for seed in seeds:
        set_seed(seed)
        raw = get_raw_train_dataset(dataset=dataset, augment=True)
        ldr = DataLoader(raw, batch_size=batch_size, shuffle=True, num_workers=0)
        m   = get_model(device, dataset)
        _patched_train_model(m, ldr, device, epochs=epochs, lr=lr, verbose=False, use_sgd=use_sgd)
        acc = evaluate(m, test_loader, device)
        baseline_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Clean Baseline"] = summarize_runs(baseline_accs)

    print("\n" + "═"*60 + "\n  [2/5] POISONED (No Defense)\n" + "═"*60)
    poisoned_accs, poisoned_models, poisoned_datasets = [], [], []
    for seed in seeds:
        set_seed(seed)
        raw = get_raw_train_dataset(dataset=dataset, augment=False)
        pt, _ = poison_dataset(raw, src_class, tgt_class, poison_fraction, seed=seed)
        ldr = DataLoader(pt, batch_size=batch_size, shuffle=True, num_workers=0)
        m   = get_model(device, dataset)
        _patched_train_model(m, ldr, device, epochs=epochs, lr=lr, verbose=False, use_sgd=use_sgd)
        acc = evaluate(m, test_loader, device)
        poisoned_accs.append(acc); poisoned_models.append(m); poisoned_datasets.append(pt)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Poisoned (No Defense)"] = summarize_runs(poisoned_accs)

    print("\n" + "═"*60 + "\n  [3/5] SPECTRAL SIGNATURES\n" + "═"*60)
    spec_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        dv, ds = device, dataset
        _, acc, _ = _patched_spectral(
            model=poisoned_models[i],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_datasets[i],
            train_loader=DataLoader(poisoned_datasets[i], batch_size=batch_size, shuffle=True, num_workers=0),
            test_loader=test_loader, device=device,
            defender_epochs=defense_epochs, defender_lr=lr, verbose=False, use_sgd=use_sgd,
        )
        spec_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["Spectral Signatures"] = summarize_runs(spec_accs)

    print("\n" + "═"*60 + "\n  [4/5] SEVER\n" + "═"*60)
    sever_accs = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        dv, ds = device, dataset
        _, acc, _ = _patched_sever(
            model=poisoned_models[i],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_datasets[i],
            train_loader=DataLoader(poisoned_datasets[i], batch_size=batch_size, shuffle=True, num_workers=0),
            test_loader=test_loader, device=device,
            defender_epochs=defense_epochs, defender_lr=lr, verbose=False, use_sgd=use_sgd,
        )
        sever_accs.append(acc)
        print(f"  Run {seed+1}: {acc:.2f}%")
    all_results["SEVER"] = summarize_runs(sever_accs)

    print("\n" + "═"*60 + "\n  [5/5] OURS: Min-Max\n" + "═"*60)
    mm = _def_mod.run_minmax(
        n_rounds=5, n_runs=n_runs, seeds=seeds,
        dataset=dataset, src_class=src_class, tgt_class=tgt_class,
        poison_fraction=poison_fraction, epochs=epochs,
        batch_size=batch_size, lr=lr, verbose=True, use_sgd=use_sgd,
    )
    all_results["Ours (Min-Max)"] = summarize_runs(mm["round_means"])

    import pandas as pd
    baseline_mean = all_results["Clean Baseline"]["mean"]
    rows = [{"Method": k, "Mean Acc (%)": round(v["mean"],2),
             "Std (%)": round(v["std"],2),
             "vs Baseline (%)": round(v["mean"]-baseline_mean,2)}
            for k, v in all_results.items()]
    df = pd.DataFrame(rows)
    df.to_csv("./results/tables/defense_comparison.csv", index=False)
    print("\n" + "═"*60 + "\n  FINAL COMPARISON TABLE\n" + "═"*60)
    print(df.to_string(index=False))
    return df

import experiments.defense_comparison as _dc_mod
_dc_mod.run_defense_comparison = _patched_defense_comparison
print("✅ Patch 4: run_defense_comparison (fully rewired)")


# ══════════════════════════════════════════════════════════════════════════════
# PATCH 5 — extract_features  ← ROOT CAUSE of Spectral + SEVER giving 10%
# Fixes: hardcoded num_workers=2 inside spectral_signatures.py crashes Kaggle,
#        corrupting feature extraction → model stays at random init → 10%
# ══════════════════════════════════════════════════════════════════════════════
def _patched_extract_features(model, dataset, device, batch_size=128):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.get_features(x) if hasattr(model, "get_features") else model(x)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
    return np.concatenate(all_feats), np.concatenate(all_labels)

_spec_mod.extract_features = _patched_extract_features
print("✅ Patch 5: extract_features (num_workers=0 — fixes the 10% Spectral/SEVER bug)")


print("\n" + "✅ " * 20)
print("ALL 5 PATCHES APPLIED — safe to run Cell 4")
print("✅ " * 20)
