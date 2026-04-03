"""
defenses/sever.py
=================
SEVER Defense (Diakonikolas et al., ICML 2019).

PAPER: "Sever: A Robust Meta-Algorithm for Stochastic Optimization"
  Diakonikolas, Kamath, Kane, Li, Steinhardt, Stewart — ICML 2019
  https://arxiv.org/abs/1803.02815

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEVER is a robust meta-algorithm that detects and removes poisoned samples
based on the GRADIENT space rather than the feature space.

Key insight:
  Poisoned samples produce anomalous gradients — gradients that are
  far from the mean gradient computed on clean data. By projecting gradients
  onto the top singular vector of the gradient covariance matrix, we can
  identify outlier samples.

  This is more aggressive than Spectral Signatures because it operates
  directly on what the model "learns" from each sample (its gradient),
  not just the features.

ALGORITHM:
  For each training step:
    1. Compute per-sample gradients (gradient of loss w.r.t. model parameters
       for each individual sample).
    2. Stack gradients into a matrix G ∈ R^(N × P) where P = num parameters.
    3. Center and compute top singular vector via SVD.
    4. Compute outlier scores = |projection onto top singular vector|.
    5. Remove samples with scores exceeding threshold (top ε fraction).
    6. Train on remaining samples.

PRACTICAL APPROXIMATION:
  Computing per-sample gradients for all parameters of ResNet-18 (~11M params)
  is memory-prohibitive. We use:
  - Gradients only from the final FC layer (most discriminative).
  - Or gradient norms as scalar outlier scores (efficient approximation).

  This is the standard practical SEVER implementation used in papers that
  cite SEVER without using the exact full-parameter version.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List, Dict


def compute_per_sample_gradients(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    layer_name: str = "fc",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample gradient vectors from the last classification layer.

    For efficiency and memory, we only use the gradient w.r.t. the final
    linear layer's parameters. This captures the learning signal relevant
    to the poisoning objective.

    Args:
        model:      Trained model (used to compute gradients).
        dataset:    Training dataset.
        device:     Compute device.
        batch_size: Processing batch size (1 typically, but we use mini-batches).
        layer_name: The name of the final linear layer to extract gradients from.

    Returns:
        (grad_matrix, labels): [N, P] gradient matrix and [N] labels.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Get the last linear layer
    last_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module

    if last_layer is None:
        raise ValueError("No Linear layer found in model.")

    all_grads  = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        output = model(x)
        loss   = criterion(output, y).mean()
        loss.backward()

        # Collect gradient of last layer weights: [out_features × in_features]
        if last_layer.weight.grad is not None:
            grad = last_layer.weight.grad.detach().cpu().numpy().flatten()
            all_grads.append(grad)
            all_labels.append(y.item())

        if (i + 1) % 1000 == 0:
            print(f"  [SEVER] Gradient extraction: {i+1}/{len(dataset)}")

    return np.array(all_grads), np.array(all_labels)


def compute_gradient_norms(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faster approximation: compute per-sample gradient L2 norms.

    Samples with anomalously high gradient norms are likely poisoned,
    since they produce large updates pulling the model toward the wrong class.

    This is an O(N) operation (vs O(N × P) for full gradient matrix).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_norms  = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # Total gradient norm across all parameters
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        all_norms.append(total_norm)
        all_labels.append(y.item())

    return np.array(all_norms), np.array(all_labels)


def sever_filter(
    model: nn.Module,
    dataset,
    device: torch.device,
    removal_fraction: float = 0.05,
    use_full_gradients: bool = False,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """
    Apply SEVER: filter out samples with anomalous gradient signals.

    Args:
        model:              Trained (potentially poisoned) model.
        dataset:            Training dataset.
        device:             Compute device.
        removal_fraction:   Fraction of top-scored samples to remove (e.g. 0.05 = 5%).
        use_full_gradients: If True, use full per-sample gradient vectors + SVD.
                            If False, use gradient norm approximation (faster).
        verbose:            Print progress.

    Returns:
        (clean_indices, stats)
    """
    n_total = len(dataset)
    n_remove = int(n_total * removal_fraction)

    if verbose:
        print(f"\n[SEVER] Computing gradient scores for {n_total:,} samples...")
        print(f"        Removal fraction: {removal_fraction:.0%} ({n_remove:,} samples)")

    if use_full_gradients:
        # Full SVD-based SEVER
        grad_matrix, labels = compute_per_sample_gradients(model, dataset, device)
        # Center and SVD
        mean_grad = grad_matrix.mean(axis=0, keepdims=True)
        centered  = grad_matrix - mean_grad
        _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
        top_sv    = Vt[0]
        scores    = np.abs(centered @ top_sv)
    else:
        # Gradient norm approximation
        scores, labels = compute_gradient_norms(model, dataset, device)

    # Identify outliers
    if n_remove > 0:
        threshold = np.sort(scores)[-n_remove]
        flagged   = set(np.where(scores >= threshold)[0].tolist())
    else:
        flagged = set()

    clean_idx = sorted(set(range(n_total)) - flagged)

    stats = {
        "n_total":   n_total,
        "n_removed": len(flagged),
        "n_clean":   len(clean_idx),
        "threshold": float(np.sort(scores)[-n_remove]) if n_remove > 0 else float('inf'),
        "score_mean": float(scores.mean()),
        "score_std":  float(scores.std()),
    }

    if verbose:
        print(f"  Removed : {len(flagged):,} samples")
        print(f"  Kept    : {len(clean_idx):,} samples")
        print(f"  Grad norm: mean={scores.mean():.4f}, std={scores.std():.4f}")

    return clean_idx, stats


def apply_sever_defense(
    model: nn.Module,
    model_fn,
    train_dataset,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    defender_epochs: int = 10,
    defender_lr: float = 0.001,
    removal_fraction: float = 0.05,
    verbose: bool = True,
) -> Tuple[nn.Module, float, Dict]:
    """
    Full SEVER pipeline:
      1. Extract gradient scores from trained (poisoned) model.
      2. Remove top-scored (most anomalous) samples.
      3. Retrain on filtered dataset.
      4. Return model and accuracy.
    """
    from train.trainer import train_model
    from train.evaluator import evaluate

    # Filter
    clean_idx, stats = sever_filter(
        model, train_dataset, device,
        removal_fraction=removal_fraction,
        verbose=verbose,
    )

    # Retrain
    clean_subset = Subset(train_dataset, clean_idx)
    clean_loader = DataLoader(
        clean_subset, batch_size=train_loader.batch_size,
        shuffle=True, num_workers=0,
    )

    if verbose:
        print(f"\n[SEVER] Retraining on {len(clean_idx):,} samples...")

    clean_model = model_fn().to(device)
    train_model(clean_model, clean_loader, device,
                epochs=defender_epochs, lr=defender_lr, verbose=verbose)

    acc = evaluate(clean_model, test_loader, device)
    stats["final_accuracy"] = acc

    if verbose:
        print(f"[SEVER] Final accuracy after filtering: {acc:.2f}%")

    return clean_model, acc, stats
