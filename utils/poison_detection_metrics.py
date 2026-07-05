"""
utils/poison_detection_metrics.py
===================================
Precision, Recall, and F1 for evaluating poison sample detection quality.

WHY THESE METRICS MATTER:
  ASR and accuracy tell you how well the defense performed END-TO-END,
  but not WHY. A defense could achieve low ASR by:
    (a) Successfully filtering poison samples (high precision + recall) — GOOD
    (b) Collateral filtering of so many clean samples that the model has
        insufficient data to learn anything — BAD (accuracy also drops)

  Without precision/recall, you cannot distinguish (a) from (b).
  Every defense paper should report these.

DEFINITIONS:
  True Positives (TP):  Samples correctly identified as poisoned and removed.
  False Positives (FP): Clean samples incorrectly flagged and removed.
  False Negatives (FN): Poison samples missed (kept in training set).
  True Negatives (TN):  Clean samples correctly kept.

  Precision = TP / (TP + FP)  → "Of samples removed, how many were actually poisoned?"
  Recall    = TP / (TP + FN)  → "Of all poison samples, how many were removed?"
  F1        = 2 × (P × R) / (P + R)

COLLATERAL DAMAGE:
  False Positive Rate (FPR) = FP / (FP + TN)
  = fraction of clean samples wrongly removed
  High FPR → defense destroys too much clean data → accuracy drops even without attack.
"""

import numpy as np
from typing import Optional, Set, List, Dict
import pandas as pd


def poison_detection_metrics(
    flagged_indices: List[int],
    true_poison_indices: List[int],
    total_n: int,
    verbose: bool = True,
) -> Dict:
    """
    Compute precision, recall, F1, and FPR for a defense's detection.

    Args:
        flagged_indices:     Indices the defense REMOVED (flagged as suspicious).
        true_poison_indices: Ground-truth poisoned indices.
        total_n:             Total number of training samples.
        verbose:             Print results.

    Returns:
        dict with: precision, recall, f1, fpr, tp, fp, fn, tn,
                   n_flagged, n_poison, n_clean_removed.
    """
    flagged_set = set(int(i) for i in flagged_indices)
    poison_set  = set(int(i) for i in true_poison_indices)
    clean_set   = set(range(total_n)) - poison_set

    tp = len(flagged_set & poison_set)     # correctly removed poison
    fp = len(flagged_set & clean_set)      # wrongly removed clean
    fn = len(poison_set - flagged_set)     # missed poison (kept)
    tn = len(clean_set - flagged_set)      # correctly kept clean

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # collateral damage

    result = {
        "precision":        round(precision, 4),
        "recall":           round(recall,    4),
        "f1":               round(f1,        4),
        "fpr":              round(fpr,       4),
        "tp":               tp,  "fp": fp,
        "fn":               fn,  "tn": tn,
        "n_flagged":        len(flagged_set),
        "n_poison":         len(poison_set),
        "n_clean_removed":  fp,
        "n_poison_missed":  fn,
    }

    if verbose:
        print(f"\n  ┌─ Poison Detection Quality ─────────────────────────")
        print(f"  │  Poison samples total : {len(poison_set):,}")
        print(f"  │  Samples flagged      : {len(flagged_set):,}")
        print(f"  │  TP (poison removed)  : {tp:,}  ({tp/max(1,len(poison_set)):.1%} of poison)")
        print(f"  │  FP (clean removed)   : {fp:,}  ({fpr:.1%} collateral damage)")
        print(f"  │  FN (poison missed)   : {fn:,}  ({fn/max(1,len(poison_set)):.1%} of poison)")
        print(f"  │  Precision            : {precision:.3f}")
        print(f"  │  Recall               : {recall:.3f}")
        print(f"  │  F1                   : {f1:.3f}")
        print(f"  │  FPR (collateral)     : {fpr:.3f}")
        _quality = ("Excellent" if f1 > 0.85 else
                    "Good"      if f1 > 0.65 else
                    "Fair"      if f1 > 0.40 else "Poor")
        print(f"  │  Detection quality    : {_quality}")
        print(f"  └────────────────────────────────────────────────────")

    return result


def multi_defense_detection_report(
    defense_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a detection quality comparison table for multiple defenses.

    Args:
        defense_results: {defense_name: detection_metrics_dict}
        save_path:       Optional path to save CSV.

    Returns:
        DataFrame with precision, recall, F1, FPR per defense.
    """
    rows = []
    for defense_name, m in defense_results.items():
        rows.append({
            "Defense":         defense_name,
            "Precision":       f"{m.get('precision', 0):.3f}",
            "Recall":          f"{m.get('recall', 0):.3f}",
            "F1":              f"{m.get('f1', 0):.3f}",
            "FPR (collateral)":f"{m.get('fpr', 0):.3f}",
            "Poison Removed":  f"{m.get('tp', 0)}/{m.get('n_poison', 0)}",
            "Clean Removed":   f"{m.get('fp', 0)}",
        })

    df = pd.DataFrame(rows)

    print(f"\n{'═'*70}")
    print(f"  POISON DETECTION QUALITY TABLE")
    print(f"  (Precision: of removed, how many were poison?)")
    print(f"  (Recall: of all poison, how many removed?)")
    print(f"  (FPR: collateral damage — fraction of clean wrongly removed)")
    print(f"{'═'*70}")
    print(df.to_string(index=False))

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[Saved] {save_path}")

    return df
