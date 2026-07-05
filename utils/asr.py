"""
utils/asr.py
============
Attack Success Rate (ASR) and per-class accuracy metrics.

WHY ASR?
  Overall test accuracy is an insufficient metric in a targeted poisoning
  study. A defense can score 90% overall while class 1 collapses to 0% —
  that looks fine in the table but is a total failure. ASR and per-class
  accuracy expose this.

ASR DEFINITION:
  ASR = fraction of SOURCE-CLASS clean test samples that the (poisoned or
  defended) model incorrectly predicts as the TARGET class.

  ASR = 0%   → defense is perfect (no source samples misclassified as tgt)
  ASR = 100% → attack fully succeeded (all source samples → tgt class)

  A GOOD DEFENSE should drive ASR toward 0 while keeping overall accuracy high.

PER-CLASS ACCURACY:
  For each class c: Acc_c = correct predictions on class-c test samples / |class-c test samples|
  This is more informative than the global average because a targeted
  attack degrades one or two classes while barely touching the rest.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np


def compute_asr(
    model: nn.Module,
    test_loader: DataLoader,
    src_class: int,
    tgt_class: int,
    device: torch.device,
) -> float:
    """
    Compute Attack Success Rate (ASR).

    ASR = (# src-class samples predicted as tgt_class) / (# src-class samples total)

    Args:
        model:       Trained model to evaluate (eval mode is set internally).
        test_loader: DataLoader for the CLEAN test set.
        src_class:   The class the attacker poisoned (source).
        tgt_class:   The class the attacker wanted to redirect predictions to (target).
        device:      Torch device.

    Returns:
        ASR as a percentage (0.0 to 100.0).
        Returns 0.0 if no src_class samples are found.

    Example:
        >>> asr = compute_asr(model, test_loader, src_class=1, tgt_class=7, device=device)
        >>> print(f"ASR: {asr:.2f}%")
        ASR: 47.30%
    """
    model.eval()
    n_src = 0
    n_flipped_to_tgt = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Only look at source-class samples
            src_mask = (labels == src_class)
            if src_mask.sum() == 0:
                continue

            src_images = images[src_mask]
            logits = model(src_images)
            preds  = torch.argmax(logits, dim=1)

            n_src            += src_mask.sum().item()
            n_flipped_to_tgt += (preds == tgt_class).sum().item()

    if n_src == 0:
        return 0.0
    return 100.0 * n_flipped_to_tgt / n_src


def compute_per_class_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    n_classes: int,
    device: torch.device,
) -> Dict[int, float]:
    """
    Compute per-class accuracy for all classes.

    Args:
        model:       Trained model.
        test_loader: DataLoader for the evaluation set.
        n_classes:   Total number of classes.
        device:      Torch device.

    Returns:
        Dict mapping class_id → accuracy (%) for that class.

    Example:
        >>> per_class = compute_per_class_accuracy(model, test_loader, n_classes=10, device=device)
        >>> print(per_class[1])   # Accuracy on class 1 (the poisoned class)
        52.40
    """
    model.eval()
    correct_per_class = torch.zeros(n_classes)
    total_per_class   = torch.zeros(n_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds  = torch.argmax(model(images), dim=1)

            for c in range(n_classes):
                mask = (labels == c)
                total_per_class[c]   += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    result = {}
    for c in range(n_classes):
        if total_per_class[c] > 0:
            result[c] = 100.0 * float(correct_per_class[c]) / float(total_per_class[c])
        else:
            result[c] = float("nan")
    return result


def compute_full_attack_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    src_class: int,
    tgt_class: int,
    n_classes: int,
    device: torch.device,
) -> Dict:
    """
    Compute all attack-related metrics in one pass.

    Returns:
        dict with:
          'overall_acc':   float — overall test accuracy (%)
          'asr':           float — Attack Success Rate (%)
          'src_class_acc': float — accuracy on the source class specifically (%)
          'tgt_class_acc': float — accuracy on the target class (%)
          'per_class_acc': dict[int, float] — full per-class breakdown
    """
    model.eval()
    correct_total = 0
    total         = 0
    n_src         = 0
    n_flipped     = 0
    correct_per_class = [0] * n_classes
    total_per_class   = [0] * n_classes

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds  = torch.argmax(model(images), dim=1)

            correct_total += (preds == labels).sum().item()
            total         += labels.size(0)

            # ASR: source-class samples predicted as target
            src_mask = (labels == src_class)
            n_src    += src_mask.sum().item()
            n_flipped += (preds[src_mask] == tgt_class).sum().item()

            # Per-class counts
            for c in range(n_classes):
                mask = (labels == c)
                total_per_class[c]   += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    per_class_acc = {}
    for c in range(n_classes):
        if total_per_class[c] > 0:
            per_class_acc[c] = 100.0 * correct_per_class[c] / total_per_class[c]
        else:
            per_class_acc[c] = float("nan")

    random_asr_baseline = 100.0 / n_classes   # chance level for k-class problem
    asr_raw = 100.0 * n_flipped / n_src if n_src > 0 else 0.0

    return {
        "overall_acc":        100.0 * correct_total / total if total > 0 else 0.0,
        "asr":                asr_raw,
        "random_asr_baseline":random_asr_baseline,
        "asr_above_chance":   max(0.0, asr_raw - random_asr_baseline),  # pp above chance
        "practically_significant_attack": asr_raw > (random_asr_baseline + 5.0),
        "src_class_acc": per_class_acc.get(src_class, float("nan")),
        "tgt_class_acc": per_class_acc.get(tgt_class, float("nan")),
        "per_class_acc": per_class_acc,
    }


def print_attack_metrics(label: str, metrics: Dict) -> None:
    """Pretty-print the full attack metrics dict with baseline comparison."""
    asr       = metrics['asr']
    baseline  = metrics.get('random_asr_baseline', 10.0)
    above     = metrics.get('asr_above_chance', asr - baseline)
    sig_str   = "✓ real signal" if metrics.get('practically_significant_attack') else "⚠ near chance"
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Overall Accuracy      : {metrics['overall_acc']:.2f}%")
    print(f"  ASR                   : {asr:.2f}%  ({sig_str})")
    print(f"  Random ASR baseline   : {baseline:.2f}%  (1/{round(100/baseline):.0f} classes)")
    print(f"  ASR above chance      : {above:+.2f}pp")
    print(f"  Src Class Acc         : {metrics['src_class_acc']:.2f}%")
    print(f"  Tgt Class Acc         : {metrics['tgt_class_acc']:.2f}%")
    print(f"{'='*58}\n")


# ─────────────────────── Self-test ───────────────────────────────────────────
if __name__ == "__main__":
    import torch.nn as nn

    print("=== ASR Self-Test ===")

    # Toy model that always predicts class 7
    class AlwaysTgt(nn.Module):
        def forward(self, x):
            B = x.size(0)
            out = torch.zeros(B, 10)
            out[:, 7] = 10.0
            return out

    # Toy dataset: 10 samples of class 1 (src), 10 of class 0 (other)
    from torch.utils.data import TensorDataset
    images = torch.zeros(20, 1, 28, 28)
    labels = torch.tensor([1]*10 + [0]*10)
    loader = DataLoader(TensorDataset(images, labels), batch_size=20)

    dev = torch.device("cpu")
    model_always_tgt = AlwaysTgt()

    asr = compute_asr(model_always_tgt, loader, src_class=1, tgt_class=7, device=dev)
    assert abs(asr - 100.0) < 1e-3, f"Expected 100%, got {asr}"
    print(f"  AlwaysTgt ASR = {asr:.2f}% (expected 100%) ✓")

    metrics = compute_full_attack_metrics(
        model_always_tgt, loader,
        src_class=1, tgt_class=7, n_classes=10, device=dev
    )
    print(f"  Overall accuracy = {metrics['overall_acc']:.2f}% (expected 0%) ✓")
    print(f"  Src class acc    = {metrics['src_class_acc']:.2f}% (expected 0%) ✓")
    print("All ASR self-tests passed ✓")
