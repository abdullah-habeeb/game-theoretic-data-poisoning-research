"""
defenses/anti_backdoor_learning.py
====================================
Anti-Backdoor Learning (ABL) — Li et al., NeurIPS 2021.

PAPER: "Anti-Backdoor Learning: Training Clean Models on Poisoned Data"
  Li, Li, Wu, Li, He, Lyu — NeurIPS 2021
  https://arxiv.org/abs/2110.11571

WHY ABL BEATS SPECTRAL SIGNATURES:
  Spectral Signatures looks for outliers in feature SPACE (post-training).
  ABL exploits a key behavioral difference during TRAINING:

  KEY INSIGHT: Backdoored samples memorize their spurious trigger pattern
  FASTER than clean samples memorize their true semantic class.

  Visualized:
    Epoch 1:  clean loss ≈ high, poison loss ≈ also high (equal)
    Epoch 3:  clean loss ≈ medium, poison loss ≈ already low (!)
    Epoch 10: clean loss ≈ low,    poison loss ≈ near zero

  The SPEED of loss decrease is the signal:
    - Low loss after only a few epochs → likely poisoned (trigger memorized)
    - Normal loss trajectory → likely clean

ABL ALGORITHM:
  Phase 1 — Warmup: Train on ALL data for K epochs. Measure per-sample loss.
  Phase 2 — Splitting: Sort by loss. Flag bottom gamma% as "isolated" (suspect).
  Phase 3 — Unlearning: On the isolated set, MAXIMIZE loss (gradient ascent)
             to unlearn the trigger. This forces the model to forget the shortcut.
  Phase 4 — Fine-tuning: Train normally on the CLEAN set (non-isolated).

  The final model has "forgotten" the backdoor while retaining semantic accuracy.

HYPERPARAMETERS:
  K (warmup epochs):   5-10 (enough to separate loss trajectories)
  gamma (isolation %): 0.05-0.10 (top 5-10% lowest-loss samples)
  unlearn_epochs:      5-10 (gradient ascent on isolated set)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Tuple, Optional, Dict
from train.evaluator import evaluate


class PerSampleLossTracker:
    """
    Tracks per-sample loss across training epochs.
    Stores loss for each sample indexed by dataset position.
    """

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.losses    = np.ones(n_samples, dtype=np.float32) * 999.0

    def update(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 256,
        criterion: nn.Module = None,
    ):
        """Compute per-sample loss for the current model state."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction="none")

        model.eval()
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_losses = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                logits = model(imgs)
                losses = criterion(logits, lbls)
                all_losses.append(losses.cpu().numpy())

        self.losses = np.concatenate(all_losses)
        return self.losses


def anti_backdoor_learning(
    raw_poisoned_dataset: Dataset,
    model_fn,
    device: torch.device,
    n_classes: int = 10,
    warmup_epochs: int = 5,
    isolation_gamma: float = 0.05,
    unlearn_epochs: int = 5,
    finetune_epochs: int = 50,
    lr: float = 0.1,
    batch_size: int = 128,
    use_sgd: bool = True,
    patience: int = 8,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray, Dict]:
    """
    Full Anti-Backdoor Learning pipeline.

    Args:
        raw_poisoned_dataset: Full poisoned training dataset.
        model_fn:             Callable () → fresh untrained model (on CPU).
        device:               Compute device.
        isolation_gamma:      Fraction of lowest-loss samples to isolate (suspect).
        warmup_epochs:        Epochs to train before measuring loss.
        unlearn_epochs:       Gradient ascent epochs on isolated set.
        finetune_epochs:      Normal training epochs on clean set.
        val_loader:           Optional validation loader for early stopping.

    Returns:
        (defended_model, isolated_indices, stats_dict)
    """
    n = len(raw_poisoned_dataset)

    def _make_scaler():
        try:
            return torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def _autocast():
        try:
            return torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.float16)
        except Exception:
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16)

    def _make_opt(model):
        if use_sgd:
            return (torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        torch.optim.SGD(model.parameters(), lr=lr), T_max=finetune_epochs, eta_min=1e-6))
        else:
            return torch.optim.Adam(model.parameters(), lr=lr), None

    def _train_epoch(model, loader, optimizer, scaler, criterion):
        model.train()
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    if verbose:
        print(f"\n  [ABL] Starting Anti-Backdoor Learning")
        print(f"  [ABL] warmup={warmup_epochs} | gamma={isolation_gamma:.0%} | "
              f"unlearn={unlearn_epochs} | finetune={finetune_epochs}")

    # ── Phase 1: Warmup Training ──────────────────────────────────────────────
    if verbose:
        print(f"\n  [ABL] Phase 1: Warmup ({warmup_epochs} epochs)...")

    warmup_model = model_fn().to(device)
    criterion_mean = nn.CrossEntropyLoss()
    criterion_each = nn.CrossEntropyLoss(reduction="none")

    if use_sgd:
        opt = torch.optim.SGD(warmup_model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=warmup_epochs, eta_min=1e-6)
    else:
        opt   = torch.optim.Adam(warmup_model.parameters(), lr=lr)
        sched = None

    scaler = _make_scaler()
    full_loader = DataLoader(raw_poisoned_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    for ep in range(warmup_epochs):
        _train_epoch(warmup_model, full_loader, opt, scaler, criterion_mean)
        if sched: sched.step()
        if verbose:
            val_acc = evaluate(warmup_model, val_loader, device) if val_loader else float('nan')
            print(f"    Warmup epoch {ep+1}/{warmup_epochs}  val={val_acc:.2f}%")

    # ── Phase 2: Loss-based Splitting ─────────────────────────────────────────
    if verbose:
        print(f"\n  [ABL] Phase 2: Computing per-sample loss for isolation...")

    tracker = PerSampleLossTracker(n)
    tracker.update(warmup_model, raw_poisoned_dataset, device,
                   batch_size=batch_size, criterion=criterion_each)

    # LOW loss = fast memorization = suspect (likely poisoned)
    n_isolate = max(1, int(n * isolation_gamma))
    sorted_by_loss = np.argsort(tracker.losses)        # ascending: lowest loss first
    isolated_indices = sorted_by_loss[:n_isolate]       # bottom gamma% = most suspect
    clean_indices    = sorted_by_loss[n_isolate:]       # rest = clean

    if verbose:
        iso_mean_loss  = tracker.losses[isolated_indices].mean()
        clean_mean_loss = tracker.losses[clean_indices].mean()
        print(f"    Isolated {n_isolate:,} samples (bottom {isolation_gamma:.0%} by loss)")
        print(f"    Mean loss — isolated: {iso_mean_loss:.4f}  |  clean: {clean_mean_loss:.4f}")
        print(f"    Clean set size: {len(clean_indices):,}")

    # ── Phase 3: Unlearning (Gradient Ascent on Isolated Set) ─────────────────
    if verbose:
        print(f"\n  [ABL] Phase 3: Gradient ascent unlearning on {n_isolate} suspect samples "
              f"({unlearn_epochs} epochs)...")

    isolated_subset = Subset(raw_poisoned_dataset, isolated_indices.tolist())
    isolated_loader = DataLoader(isolated_subset, batch_size=min(64, n_isolate),
                                 shuffle=True, num_workers=0)

    unlearn_model = copy.deepcopy(warmup_model)
    if use_sgd:
        unlearn_opt = torch.optim.SGD(unlearn_model.parameters(), lr=lr * 0.1,
                                       momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        unlearn_opt = torch.optim.Adam(unlearn_model.parameters(), lr=lr * 0.1)

    unlearn_scaler = _make_scaler()

    for ep in range(unlearn_epochs):
        unlearn_model.train()
        for imgs, lbls in isolated_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            unlearn_opt.zero_grad(set_to_none=True)
            with _autocast():
                loss = criterion_mean(unlearn_model(imgs), lbls)
            # GRADIENT ASCENT: negate the loss to MAXIMIZE it → unlearn the trigger
            unlearn_scaler.scale(-loss).backward()
            unlearn_scaler.step(unlearn_opt)
            unlearn_scaler.update()

        if verbose:
            print(f"    Unlearn epoch {ep+1}/{unlearn_epochs}")

    # ── Phase 4: Fine-tune on Clean Set ──────────────────────────────────────
    if verbose:
        print(f"\n  [ABL] Phase 4: Fine-tuning on {len(clean_indices):,} clean samples "
              f"({finetune_epochs} epochs)...")

    clean_subset  = Subset(raw_poisoned_dataset, clean_indices.tolist())
    clean_loader  = DataLoader(clean_subset, batch_size=batch_size,
                               shuffle=True, num_workers=0)

    final_model = copy.deepcopy(unlearn_model)
    if use_sgd:
        ft_opt   = torch.optim.SGD(final_model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=5e-4, nesterov=True)
        ft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            ft_opt, T_max=finetune_epochs, eta_min=1e-6)
    else:
        ft_opt   = torch.optim.Adam(final_model.parameters(), lr=lr * 0.1)
        ft_sched = None

    ft_scaler  = _make_scaler()
    best_val   = 0.0
    best_state = copy.deepcopy(final_model.state_dict())
    patience_ctr = 0

    for ep in range(finetune_epochs):
        _train_epoch(final_model, clean_loader, ft_opt, ft_scaler, criterion_mean)
        if ft_sched: ft_sched.step()

        if val_loader is not None:
            val_acc = evaluate(final_model, val_loader, device)
            if val_acc > best_val:
                best_val     = val_acc
                best_state   = copy.deepcopy(final_model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"    [ABL] Early stop at epoch {ep+1}")
                break
            if verbose and ep % max(1, finetune_epochs // 5) == 0:
                print(f"    Finetune epoch {ep+1}/{finetune_epochs}  val={val_acc:.2f}%  "
                      f"best={best_val:.2f}%")

    if val_loader is not None:
        final_model.load_state_dict(best_state)

    stats = {
        "n_total":     n,
        "n_isolated":  n_isolate,
        "n_clean":     len(clean_indices),
        "gamma":       isolation_gamma,
        "warmup_eps":  warmup_epochs,
        "unlearn_eps": unlearn_epochs,
        "best_val":    best_val,
    }

    if verbose:
        print(f"\n  [ABL] Complete — best val acc: {best_val:.2f}%")
        print(f"  [ABL] Isolated {n_isolate} suspect samples out of {n} total")

    return final_model, isolated_indices, stats
