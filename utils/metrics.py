"""
utils/metrics.py
================
Helper functions for computing and summarizing accuracy across multiple runs.

WHY MULTIPLE RUNS?
  Neural network training is stochastic. A single run might get lucky or unlucky.
  Running 3+ times and reporting mean ± std is the standard in ML research
  because it shows both the average performance AND how consistent the method is.
"""

import numpy as np
from typing import List, Dict


def summarize_runs(accuracies: List[float]) -> Dict:
    """
    Compute summary statistics over multiple experimental runs.

    Args:
        accuracies: List of accuracy values (e.g., [99.13, 99.10, 98.98])

    Returns:
        dict with keys: 'runs', 'mean', 'std', 'min', 'max'

    Example:
        >>> summarize_runs([99.13, 99.10, 98.98])
        {'runs': [99.13, 99.1, 98.98], 'mean': 99.07, 'std': 0.06, ...}
    """
    arr = np.array(accuracies)
    return {
        "runs": accuracies,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_summary(label: str, summary: Dict) -> None:
    """Pretty-print a run summary."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for i, acc in enumerate(summary["runs"]):
        print(f"  Run {i+1}: {acc:.2f}%")
    print(f"  ─────────────────────────────")
    print(f"  Mean : {summary['mean']:.2f}%")
    print(f"  Std  : {summary['std']:.2f}%")
    print(f"  Min  : {summary['min']:.2f}%")
    print(f"  Max  : {summary['max']:.2f}%")
    print(f"{'='*50}\n")
