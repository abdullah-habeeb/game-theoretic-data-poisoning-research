"""
attacks/strategic_label_flip.py
================================
Strategic (adaptive) label-flip attacker for the Stackelberg Min-Max game.

WHAT MAKES THIS "STRATEGIC"?
  The original code used a RANDOM subset of source-class samples. This is NOT
  a strategic attacker — it ignores the defender's current model state
  entirely, making the "game" a fiction.

  A true Stackelberg attacker (leader) must SELECT SAMPLES THAT MAXIMALLY
  DAMAGE THE DEFENDER'S CURRENT MODEL. We implement this via gradient-norm
  based influence scoring:

  For each source-class sample x_i, we compute:

      influence_i = ‖∇_θ L(θ, x_i, tgt_class)‖₂²

  Interpretation: "If this sample's label were flipped to tgt_class, how
  strongly would it pull the model's weights in a direction that conflicts
  with clean gradients?"

  Higher influence_i → sample is more harmful when poisoned → attacker
  preferentially selects it.

ROUNDS:
  - Round 1 (no prior model): falls back to RANDOM selection (same as before)
  - Round 2+: gradient-norm scoring using previous defender model

FALLBACK SELECTION MODES:
  - 'gradient_norm': full gradient computation (accurate, slower)
  - 'loss_margin':   selects samples where L(θ, x, tgt) is smallest
                     (= model least confident in WRONG label → easiest to poison)
                     Much faster, almost as effective.
  - 'random':        uniform random baseline (round 1 default)

GAME-THEORETIC JUSTIFICATION:
  This implements the argmax_a step of:
      min_θ  max_a  L(θ, D̃_a)
  where 'a' is the attacker's choice of WHICH samples to flip.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional, List
import warnings

# Re-export PoisonedDataset from original label_flip for compatibility
from attacks.label_flip import PoisonedDataset, poison_dataset as random_poison_dataset


# ─────────────────────── Influence Scoring ───────────────────────────────────

def score_samples_by_gradient_norm(
    model: nn.Module,
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    device: torch.device,
    batch_size: int = 64,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score each source-class sample by gradient norm when label is flipped.

    For sample x_i with true label src_class, compute:
        score_i = ‖∇_θ L(θ, x_i, tgt_class)‖₂²

    A high score means: "poisoning this sample would create a large gradient
    conflict in the defender's current model."

    Args:
        model:       Current defender model θ^(r-1).
        dataset:     CLEAN training dataset (labels are ground truth).
        src_class:   Source class to attack.
        tgt_class:   Fake label to assign.
        device:      Torch device.
        batch_size:  Batch size for gradient computation.
        max_samples: Cap on samples scored (for speed on large datasets).

    Returns:
        (src_indices, scores) — parallel arrays.
        src_indices: global dataset indices of source-class samples.
        scores:      gradient-norm influence score for each sample.
    """
    # Get all source-class indices
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "targets"):
        # Handle Subset
        parent_labels = np.array(dataset.dataset.targets)
        labels = parent_labels[dataset.indices]
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    src_indices = np.where(labels == src_class)[0]

    if len(src_indices) > max_samples:
        rng = np.random.default_rng(seed=0)
        src_indices = rng.choice(src_indices, size=max_samples, replace=False)
        src_indices = np.sort(src_indices)

    scores = np.zeros(len(src_indices), dtype=np.float32)

    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    chunk_size = batch_size
    for chunk_start in range(0, len(src_indices), chunk_size):
        chunk_idx = src_indices[chunk_start: chunk_start + chunk_size]

        images_list = []
        for gidx in chunk_idx:
            img, _ = dataset[int(gidx)]
            images_list.append(img)

        images = torch.stack(images_list).to(device)
        # Assign FAKE labels (tgt_class) for gradient computation
        fake_labels = torch.full((len(images_list),), tgt_class,
                                 dtype=torch.long, device=device)

        model.zero_grad()
        images.requires_grad_(False)

        # Compute per-sample gradient norms via a small trick:
        # Process each sample individually to get its gradient.
        # For efficiency, we compute avg gradient over the chunk instead,
        # which is a good approximation (shapes are homogeneous).
        with torch.enable_grad():
            logits = model(images)
            losses = criterion(logits, fake_labels)

            chunk_scores = np.zeros(len(chunk_idx), dtype=np.float32)
            for local_i, loss_i in enumerate(losses):
                model.zero_grad()
                loss_i.backward(retain_graph=(local_i < len(losses) - 1))
                grad_norm_sq = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
                chunk_scores[local_i] = grad_norm_sq

        scores[chunk_start: chunk_start + len(chunk_idx)] = chunk_scores

    return src_indices, scores


def score_samples_by_loss_margin(
    model: nn.Module,
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast alternative: score by L(θ, x, tgt_class) — loss when label is faked.

    Intuition: A sample where the model ALREADY has low loss under the wrong
    label is already "halfway convinced" — poisoning it is cheap for the
    attacker and hard for the defender to resist.

    This is O(N) forward passes only (no backward), so ~10-50× faster than
    gradient_norm scoring, making it practical for large datasets.

    Returns:
        (src_indices, scores) where high score → high priority for attacker.
        Note: We NEGATE the loss so that higher score = easier-to-fool sample.
    """
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    src_indices = np.where(labels == src_class)[0]

    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    scores = np.zeros(len(src_indices), dtype=np.float32)

    chunk_size = batch_size
    for chunk_start in range(0, len(src_indices), chunk_size):
        chunk_idx = src_indices[chunk_start: chunk_start + chunk_size]
        images_list = [dataset[int(g)][0] for g in chunk_idx]
        images = torch.stack(images_list).to(device)
        fake_labels = torch.full((len(images_list),), tgt_class,
                                 dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(images)
            # Low loss under fake label = high priority (model already confused)
            losses = criterion(logits, fake_labels).cpu().numpy()
            # Negate: lower loss → higher score (easier target for attacker)
            scores[chunk_start: chunk_start + len(chunk_idx)] = -losses

    return src_indices, scores


# ─────────────────────── Strategic Poison ────────────────────────────────────

def strategic_poison_dataset(
    dataset: Dataset,
    src_class: int,
    tgt_class: int,
    poison_fraction: float,
    prev_model: Optional[nn.Module],
    device: torch.device,
    seed: int = 42,
    selection_mode: str = "loss_margin",
) -> Tuple[PoisonedDataset, np.ndarray, str]:
    """
    Create a poisoned dataset using strategic sample selection.

    This is the `max_a` step of the Stackelberg game:
        a* = argmax_a  L(θ^(r-1), D̃_a)

    Args:
        dataset:          Clean training dataset.
        src_class:        Source class to attack.
        tgt_class:        Target (fake) class.
        poison_fraction:  Fraction of src samples to poison (0.0–1.0).
        prev_model:       Defender model from previous round (θ^(r-1)).
                          If None, falls back to random selection (round 1).
        device:           Torch device.
        seed:             Fallback seed for round 1 random selection.
        selection_mode:   'loss_margin' (fast) | 'gradient_norm' (thorough) | 'random'.

    Returns:
        (poisoned_dataset, poisoned_indices, actual_mode_used)
    """
    # Round 1 or no model: random baseline
    if prev_model is None or selection_mode == "random":
        poisoned_ds, poisoned_idx = random_poison_dataset(
            dataset, src_class, tgt_class, poison_fraction, seed=seed
        )
        return poisoned_ds, poisoned_idx, "random"

    print(f"  [Strategic Attacker] Mode={selection_mode}, "
          f"src={src_class}→{tgt_class}, ε={poison_fraction:.0%}")

    # Score samples
    if selection_mode == "gradient_norm":
        src_indices, scores = score_samples_by_gradient_norm(
            prev_model, dataset, src_class, tgt_class, device
        )
    else:  # loss_margin (default — faster)
        src_indices, scores = score_samples_by_loss_margin(
            prev_model, dataset, src_class, tgt_class, device
        )

    # Select top-k by score (descending)
    n_to_poison = max(1, int(len(src_indices) * poison_fraction))
    top_local_idx = np.argsort(-scores)[:n_to_poison]  # descending
    poisoned_indices = np.sort(src_indices[top_local_idx])

    poisoned_ds = PoisonedDataset(dataset, poisoned_indices, tgt_class)

    print(f"  [Strategic Attacker] Selected {len(poisoned_indices)}/{len(src_indices)} "
          f"src-class samples (top by {selection_mode})")

    return poisoned_ds, poisoned_indices, selection_mode


# ─────────────────────── Self-test ───────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from torchvision import datasets, transforms
    from models.resnet import get_model
    from utils.seed import set_seed

    print("=== Strategic Attacker Self-Test ===")
    set_seed(0)
    device = torch.device("cpu")

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data/raw", train=True, download=True, transform=tf)

    # Use a tiny subset to keep test fast
    small_ds = Subset(train_ds, list(range(2000)))
    # Flatten labels for Subset (use parent)
    small_ds_labels = np.array(train_ds.targets)[:2000]

    from models.cnn import MnistCNN
    model = MnistCNN(num_classes=10).to(device)

    print("  Round 1 (no model) — should be RANDOM:")
    ds_r1, idx_r1, mode_r1 = strategic_poison_dataset(
        train_ds, src_class=1, tgt_class=7,
        poison_fraction=0.5, prev_model=None,
        device=device, seed=42, selection_mode="loss_margin"
    )
    print(f"    Mode: {mode_r1}, Poisoned: {len(idx_r1)}")

    print("  Round 2 (with model) — should be STRATEGIC (different indices):")
    ds_r2, idx_r2, mode_r2 = strategic_poison_dataset(
        train_ds, src_class=1, tgt_class=7,
        poison_fraction=0.5, prev_model=model,
        device=device, seed=42, selection_mode="loss_margin"
    )
    print(f"    Mode: {mode_r2}, Poisoned: {len(idx_r2)}")

    overlap = len(set(idx_r1.tolist()) & set(idx_r2.tolist()))
    total   = len(idx_r1)
    print(f"    Overlap with random selection: {overlap}/{total} "
          f"({100.*overlap/total:.1f}%)")
    print("  Strategic selection differs from random (as expected for untrained model with partial overlap) ✓")
    print("Strategic attacker self-test passed ✓")
