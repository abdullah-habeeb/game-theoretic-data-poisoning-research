"""
defenses/confusion_training.py
================================
Confusion Training Defense — inspired by Kurita et al. (2023).

CORE IDEA:
  Backdoor triggers survive training because the trigger→label mapping is
  strongly overfit. Confusion training intentionally injects CONTROLLED label
  noise during training — enough to disrupt spurious trigger correlations
  while preserving genuine semantic learning.

  Key insight: semantic features (true class signal) are REDUNDANT across
  many pixels/patches. A trigger patch (small, localized) is NOT redundant —
  it is a single fragile correlation. Label noise is uniquely harmful to the
  trigger correlation while being survivable by the semantic signal.

ALGORITHM:
  During each training batch:
    1. Sample a fraction `gamma` of the batch indices.
    2. Randomly shuffle their labels (uniform over all classes).
    3. Train on the mixed batch: some samples have true labels, some have noise.
  The model is forced to learn ROBUST features that survive label perturbation.

HOW IT DIFFERS FROM STANDARD LABEL SMOOTHING:
  Label smoothing distributes mass uniformly: y_soft = (1-ε)·y + ε/k.
  This is a scalar shift — the model still strongly prefers the true label.
  Confusion training uses RANDOM relabeling (not softening) at gamma% of samples,
  creating a much stronger disruption to spurious memorization.

HYPERPARAMETERS:
  gamma: Confusion fraction (0.05–0.20 recommended).
         Too low: insufficient disruption. Too high: too much clean data corrupted.
  Schedule: Can anneal gamma from high (early epochs) to low (fine-tune).

REFERENCE:
  Inspired by: Kurita, K. et al. (2023). "Revisiting Data Poisoning Defenses
  with Auxiliary Data." ACL 2023. (NLP setting; principle adapted to vision.)
  Also: Borgnia et al. (2021). "Strong Data Augmentation Sanitizes Poisoning."
  ICASSP 2021.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional, Dict, Tuple
from train.evaluator import evaluate


def _autocast_ctx(device_type: str, enabled: bool):
    try:
        return torch.amp.autocast(device_type, enabled=enabled, dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast(enabled=(enabled and device_type == "cuda"),
                                       dtype=torch.float16)


def confusion_training(
    poisoned_dataset: Dataset,
    model_fn,
    device: torch.device,
    n_classes: int = 10,
    gamma: float = 0.10,
    epochs: int = 50,
    lr: float = 0.1,
    batch_size: int = 128,
    use_sgd: bool = True,
    patience: int = 8,
    gamma_anneal: bool = True,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with confusion label noise to suppress backdoor memorization.

    Args:
        poisoned_dataset: Full (potentially poisoned) training dataset.
        model_fn:         Callable () → fresh untrained model.
        n_classes:        Number of output classes.
        gamma:            Fraction of each batch to relabel randomly.
        epochs:           Total training epochs.
        gamma_anneal:     If True, linearly anneal gamma → 0 in the final 20% of training.
        val_loader:       Validation loader for early stopping and model selection.

    Returns:
        (model, stats_dict)
    """
    if verbose:
        print(f"\n  [ConfusionTraining] γ={gamma:.0%}  anneal={gamma_anneal}  "
              f"epochs={epochs}  use_sgd={use_sgd}")

    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()

    if use_sgd:
        opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    else:
        opt, sched = torch.optim.Adam(model.parameters(), lr=lr), None

    use_amp = (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    full_loader = DataLoader(poisoned_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    best_val, best_state, patience_ctr = 0.0, copy.deepcopy(model.state_dict()), 0
    anneal_start = int(epochs * 0.80)   # start annealing at 80% of training

    for ep in range(1, epochs + 1):
        # Compute effective gamma (anneal if requested)
        if gamma_anneal and ep > anneal_start:
            ep_gamma = gamma * (1.0 - (ep - anneal_start) / (epochs - anneal_start))
        else:
            ep_gamma = gamma

        model.train()
        epoch_loss = 0.0

        for imgs, lbls in full_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            # ── Confusion injection ───────────────────────────────────────
            if ep_gamma > 0:
                B = imgs.size(0)
                n_confuse = max(1, int(B * ep_gamma))
                confuse_idx = torch.randperm(B, device=device)[:n_confuse]
                # Random labels drawn uniformly from all classes
                random_lbls = torch.randint(0, n_classes, (n_confuse,), device=device)
                lbls = lbls.clone()
                lbls[confuse_idx] = random_lbls

            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(device.type, use_amp):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        if sched: sched.step()

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val, best_state, patience_ctr = val_acc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"    [ConfusionTraining] Early stop at epoch {ep}")
                break
            if verbose and ep % max(1, epochs // 5) == 0:
                print(f"    Epoch {ep:3d}/{epochs}  γ={ep_gamma:.3f}  "
                      f"loss={epoch_loss/len(full_loader):.4f}  val={val_acc:.2f}%")

    if val_loader is not None:
        model.load_state_dict(best_state)

    stats = {
        "gamma":       gamma,
        "gamma_anneal":gamma_anneal,
        "epochs_run":  ep,
        "best_val":    best_val,
    }

    if verbose:
        print(f"  [ConfusionTraining] Complete — best val: {best_val:.2f}%")

    return model, stats
