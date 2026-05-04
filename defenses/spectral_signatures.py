"""
defenses/spectral_signatures.py
================================
Spectral Signatures Defense (Tran et al., NeurIPS 2018).

PAPER: "Spectral Signatures in Backdoor Attacks"
  Tran, Li, Madry — NeurIPS 2018
  https://arxiv.org/abs/1811.00636

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CORE IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Poisoned (backdoored) samples leave a detectable signature in the feature
representations of a neural network. Specifically:

  1. Train the model on the (potentially poisoned) dataset.
  2. Extract feature representations (penultimate layer activations)
     for all training samples in a given class.
  3. Compute the top singular vector of the feature covariance matrix via SVD.
  4. Project each sample's representation onto this singular vector.
     Poisoned samples tend to have anomalously large projections because
     their features cluster differently from clean samples.
  5. Remove samples whose projection score exceeds a threshold ε_spec.
  6. Retrain the model on the filtered dataset.

WHY IT WORKS:
  A poisoned model learns a "shortcut" feature (the trigger or adversarial
  correlation) specific to poisoned samples. This shortcut appears as an
  outlier in the SVD decomposition of the feature space — the top singular
  value captures it efficiently.

THRESHOLD:
  We use a fixed quantile threshold (e.g., top 5% of scores flagged as
  suspicious). This is the standard setting in the original paper.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List, Dict


def extract_features(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer feature representations for all samples.

    Args:
        model:      Trained neural network. Must have a `get_features` method
                    or be a ResNet-style model. Falls back to second-to-last
                    layer hook if `get_features` not available.
        dataset:    Dataset to extract features from.
        device:     Compute device.
        batch_size: Batch size for extraction.

    Returns:
        (features, labels): numpy arrays of shape [N, D] and [N].
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_feats  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if hasattr(model, "get_features"):
                feats = model.get_features(x)
            else:
                # Generic fallback: hook penultimate layer output
                feats = model(x)   # Use logits as proxy features
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.numpy() if isinstance(y, torch.Tensor)
                               else np.array(y))

    return np.concatenate(all_feats), np.concatenate(all_labels)


def compute_spectral_scores(
    features: np.ndarray,
) -> np.ndarray:
    """
    Compute spectral signature scores for a set of features.

    The score for sample i is its projection onto the top singular vector
    of the centered feature matrix.

    Args:
        features: [N, D] feature matrix.

    Returns:
        [N] array of spectral scores.
    """
    # Center features
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean  # [N, D]

    # SVD: U [N×N], S [min(N,D)], Vt [min(N,D)×D]
    # We only need the top singular vector (first column of Vt)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    top_sv = Vt[0]   # [D] — top singular vector

    # Projection score: distance along the top singular direction
    scores = np.abs(centered @ top_sv)   # [N]
    return scores


def spectral_signatures_filter(
    model: nn.Module,
    dataset,
    device: torch.device,
    suspicious_quantile: float = 0.95,
    n_classes: int = 10,
    batch_size: int = 128,
    verbose: bool = True,
) -> Tuple[List[int], Dict]:
    """
    Apply Spectral Signatures defense: identify and filter suspicious samples.

    For each class:
      1. Extract features for all class members.
      2. Compute spectral scores.
      3. Flag samples with score > suspicious_quantile as poisoned.

    Args:
        model:               Trained (possibly poisoned) model.
        dataset:             Training dataset (potentially poisoned).
        device:              Compute device.
        suspicious_quantile: Top quantile of scores to flag (e.g. 0.95 = top 5%).
        n_classes:           Number of classes.
        batch_size:          Batch size for feature extraction.
        verbose:             Print statistics.

    Returns:
        (clean_indices, stats):
          clean_indices: list of indices NOT flagged as suspicious.
          stats: dict with per-class counts.
    """
    if verbose:
        print(f"\n[SpectralSignatures] Extracting features...")

    features, labels = extract_features(model, dataset, device, batch_size)

    # Infer actual number of classes dynamically (prevents CIFAR-100 bug)
    if len(labels) > 0:
        n_classes = max(n_classes, int(np.max(labels)) + 1)

    suspicious_idx = set()
    stats = {"n_flagged": 0, "n_total": len(labels), "per_class": {}}

    for c in range(n_classes):
        class_mask = (labels == c)
        class_idx  = np.where(class_mask)[0]
        if len(class_idx) < 10:
            continue

        class_feats  = features[class_idx]
        scores       = compute_spectral_scores(class_feats)

        threshold    = np.quantile(scores, suspicious_quantile)
        flagged_local = (scores > threshold)
        flagged_global = class_idx[flagged_local]

        suspicious_idx.update(flagged_global.tolist())
        stats["per_class"][c] = {
            "total": len(class_idx),
            "flagged": int(flagged_local.sum()),
            "threshold": float(threshold),
        }
        if verbose:
            print(f"  Class {c:2d}: total={len(class_idx):5d}, "
                  f"flagged={flagged_local.sum():4d} "
                  f"({flagged_local.mean()*100:.1f}%)")

    all_idx    = set(range(len(labels)))
    clean_idx  = sorted(all_idx - suspicious_idx)
    stats["n_flagged"] = len(suspicious_idx)

    if verbose:
        print(f"\n  Total flagged : {len(suspicious_idx):,} / {len(labels):,} "
              f"({len(suspicious_idx)/len(labels)*100:.1f}%)")
        print(f"  Clean kept    : {len(clean_idx):,}")

    return clean_idx, stats


def apply_spectral_defense(
    model: nn.Module,
    model_fn,
    train_dataset,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    defender_epochs: int = 10,
    defender_lr: float = 0.001,
    suspicious_quantile: float = 0.95,
    n_classes: int = None,   # Auto-inferred from dataset if None
    dataset: str = "cifar10",
    num_workers: int = 2,
    verbose: bool = True,
    use_sgd: bool = False,
    checkpoint_path: str = None,
    resume_from_checkpoint: str = None,
) -> Tuple[nn.Module, float, Dict]:
    """
    Full Spectral Signatures pipeline:
      1. Train on poisoned data (already done — pass trained model).
      2. Filter suspicious samples.
      3. Retrain on filtered dataset.
      4. Return retrained model and accuracy.

    Args:
        model:        Already-trained (poisoned) model used for feature extraction.
        model_fn:     Callable () → fresh untrained model.
        train_dataset: Full training dataset (with poison).
        train_loader:  DataLoader for training (poisoned).
        test_loader:   Clean test DataLoader.
        device:        Compute device.
        defender_epochs: Epochs for retraining.
        defender_lr:   Retraining learning rate.
        suspicious_quantile: Flagging threshold.
        n_classes:     Number of classes.
        verbose:       Print progress.

    Returns:
        (retrained_model, test_accuracy, stats)
    """
    from train.trainer import train_model
    from train.evaluator import evaluate

    # Auto-infer n_classes from dataset name if not provided
    if n_classes is None:
        _n_classes_map = {"mnist": 10, "cifar10": 10, "cifar100": 100}
        n_classes = _n_classes_map.get(dataset, 10)

    # Step 1: Get clean indices
    clean_idx, stats = spectral_signatures_filter(
        model, train_dataset, device,
        suspicious_quantile=suspicious_quantile,
        n_classes=n_classes,
        verbose=verbose,
    )

    # Step 2: Build filtered DataLoader
    clean_subset = Subset(train_dataset, clean_idx)
    pin = torch.cuda.is_available()
    clean_loader = DataLoader(
        clean_subset, batch_size=train_loader.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin,
    )

    # Step 3: Retrain on clean subset
    if verbose:
        print(f"\n[SpectralSignatures] Retraining on {len(clean_idx):,} clean samples...")
    clean_model = model_fn().to(device)
    train_model(clean_model, clean_loader, device,
                epochs=defender_epochs, lr=defender_lr, verbose=verbose,
                use_sgd=use_sgd, checkpoint_path=checkpoint_path, resume_from_checkpoint=resume_from_checkpoint)

    # Step 4: Evaluate
    acc = evaluate(clean_model, test_loader, device)
    stats["final_accuracy"] = acc
    if verbose:
        print(f"[SpectralSignatures] Final accuracy after filtering: {acc:.2f}%")

    return clean_model, acc, stats
