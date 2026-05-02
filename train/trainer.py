"""
train/trainer.py
================
Reusable training loop for the CNN.

WHAT IS A TRAINING LOOP?
  During training, we:
    1. Feed a mini-batch of images through the model (forward pass).
    2. Compute how wrong the model was (the loss).
    3. Compute gradients of the loss w.r.t. each model parameter (backpropagation).
    4. Update the parameters in the direction that reduces the loss (optimizer step).
  We repeat this for every batch, and one full pass through all batches = one epoch.

CROSS-ENTROPY LOSS:
  The standard loss function for classification. It penalizes the model more
  heavily when it is confidently wrong than when it is uncertain.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = False,
) -> float:
    """
    Run one full pass over the training data (one epoch).

    Args:
        model:         The neural network.
        loader:        DataLoader for the training set.
        optimizer:     Optimizer (Adam, SGD, etc.) that updates model parameters.
        criterion:     Loss function (typically CrossEntropyLoss).
        device:        Device to run on ('cpu' or 'cuda').
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Average loss over the epoch (float).
    """
    model.train()   # Set model to training mode (enables dropout, batch norm, etc.)
    total_loss = 0.0
    total_batches = 0
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    iterator = tqdm(loader, desc="  Training", leave=False) if show_progress else loader

    for images, labels in iterator:
        # Move data to the same device as the model
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass (AMP FP16 on GPU) ────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp, dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)

        # ── Backward pass ──────────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 0.001,
    show_progress: bool = False,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    use_sgd: bool = False,
) -> nn.Module:
    """
    Full training loop: train the model for `epochs` epochs.

    Args:
        model:                   The neural network.
        loader:                  DataLoader for the training set.
        device:                  Device to run on.
        epochs:                  Number of complete passes over the training data.
        lr:                      Learning rate for the Adam optimizer.
        show_progress:           Show per-batch tqdm bar.
        verbose:                 Print per-epoch loss summary.
        checkpoint_path:         If set, save a checkpoint here after every epoch.
                                 Saves model weights, optimizer state, and last epoch.
        resume_from_checkpoint:  If set and file exists, resume training from this checkpoint.

    Returns:
        The trained model (same object, modified in-place).
    """
    import os
    criterion = nn.CrossEntropyLoss()

    if use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        # Cosine annealing: the standard LR schedule for ResNet on CIFAR-10/100
        # Decays LR smoothly from `lr` → 1e-6 over all epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None

    start_epoch = 1

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    if resume_from_checkpoint and os.path.isfile(resume_from_checkpoint):
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        if verbose:
            print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']} "
                  f"(loss={ckpt.get('loss', 'N/A'):.4f})")

    if verbose:
        remaining = epochs - (start_epoch - 1)
        sched_name = "SGD+CosineAnneal" if use_sgd else "Adam"
        print(f"  Training for {remaining} epoch(s) "
              f"[{start_epoch}..{epochs}], lr={lr}, opt={sched_name}")

    for epoch in range(start_epoch, epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, show_progress)
        if scheduler:
            scheduler.step()
        if verbose:
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            print(f"  Epoch [{epoch}/{epochs}]  Loss: {avg_loss:.4f}  lr={current_lr:.6f}")

        # ── Save checkpoint after each epoch ─────────────────────────────────
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            ckpt_data = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            avg_loss,
            }
            if scheduler:
                ckpt_data["scheduler_state"] = scheduler.state_dict()
            torch.save(ckpt_data, checkpoint_path)

    return model
