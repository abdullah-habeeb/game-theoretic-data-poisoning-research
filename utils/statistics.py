"""
utils/statistics.py
====================
Statistical significance testing for ML experiments.

WHY THIS MATTERS FOR A PAPER:
  Reporting mean ± std is necessary but not sufficient for a conference paper.
  Reviewers will ask: "Is the improvement statistically significant?"
  Without a significance test, your result could be due to random chance.

PAIRED t-TEST:
  We use a paired t-test (scipy.stats.ttest_rel) because the same seeds are
  used across conditions (baseline, poisoned, defended). When the same random
  seed produces run i in condition A and run i in condition B, the samples
  are PAIRED — this removes run-to-run noise from the comparison.

  Null hypothesis H0: The two conditions have equal means.
  We reject H0 at p < 0.05 (standard ML convention).

COHEN'S d (EFFECT SIZE):
  p-value only tells you IF the difference is significant.
  Cohen's d tells you HOW LARGE the difference is:
    d < 0.2  → negligible
    d = 0.2  → small
    d = 0.5  → medium
    d = 0.8  → large (this is what you want to see in a paper!)

REFERENCES:
  Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
  Welch, B.L. (1947). The generalization of Student's problem.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def paired_ttest(
    accs_a: List[float],
    accs_b: List[float],
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Paired two-sided t-test between two conditions.

    Args:
        accs_a: Accuracy values for condition A (same seeds as B).
        accs_b: Accuracy values for condition B.
        label_a: Name of condition A.
        label_b: Name of condition B.
        alpha:   Significance level (default 0.05).
        verbose: Print the test result.

    Returns:
        Dict with: t_stat, p_value, significant, cohens_d, ci_95.

    Example:
        >>> paired_ttest([99.1, 99.0, 99.2], [92.3, 90.1, 91.5],
        ...              label_a='Baseline', label_b='Poisoned')
    """
    assert len(accs_a) == len(accs_b), "Paired test requires equal-length lists."
    a = np.array(accs_a, dtype=float)
    b = np.array(accs_b, dtype=float)

    t_stat, p_value = stats.ttest_rel(a, b)
    significant     = p_value < alpha
    d               = cohens_d(a, b)
    ci              = confidence_interval_95(a - b)

    result = {
        "label_a":     label_a,
        "label_b":     label_b,
        "mean_a":      float(np.mean(a)),
        "mean_b":      float(np.mean(b)),
        "mean_diff":   float(np.mean(a - b)),
        "t_stat":      float(t_stat),
        "p_value":     float(p_value),
        "significant": bool(significant),
        "cohens_d":    float(d),
        "cohens_d_interp": interpret_cohens_d(d),
        "ci_95":       ci,
    }

    if verbose:
        sig_str = "✓ SIGNIFICANT" if significant else "✗ not significant"
        print(f"\n[paired t-test]  {label_a} vs {label_b}")
        print(f"  {label_a} mean : {np.mean(a):.3f}%")
        print(f"  {label_b} mean : {np.mean(b):.3f}%")
        print(f"  Mean diff     : {np.mean(a-b):+.3f}%")
        print(f"  t-statistic   : {t_stat:.4f}")
        print(f"  p-value       : {p_value:.6f}  → {sig_str} (α={alpha})")
        print(f"  Cohen's d     : {d:.3f}  ({interpret_cohens_d(d)} effect)")
        print(f"  95% CI [diff] : ({ci[0]:+.3f}%, {ci[1]:+.3f}%)")

    return result


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    d = mean(a - b) / std(a - b)
    """
    diff = a - b
    if diff.std() == 0:
        return float("inf") if diff.mean() != 0 else 0.0
    return float(diff.mean() / diff.std())


def interpret_cohens_d(d: float) -> str:
    """Return qualitative label for Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:         return "large"


def confidence_interval_95(
    values: np.ndarray,
) -> Tuple[float, float]:
    """
    95% confidence interval using t-distribution.

    When n is small (e.g. 3 runs), use t-distribution (not normal).
    """
    n    = len(values)
    mean = np.mean(values)
    se   = stats.sem(values)
    ci   = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


def full_significance_report(
    baseline_accs: List[float],
    poisoned_accs: List[float],
    defended_accs: List[float],
    verbose: bool = True,
) -> Dict:
    """
    Run all pairwise significance tests for the three main conditions.

    Reports:
      1. Baseline vs Poisoned  (attack effectiveness)
      2. Poisoned vs Defended  (defense effectiveness)
      3. Baseline vs Defended  (gap remaining)

    Args:
        baseline_accs: Clean model accuracies (per-run).
        poisoned_accs: Poisoned model accuracies (per-run).
        defended_accs: Defended model accuracies (per-run).
        verbose:       Print results.

    Returns:
        Dict with three test results.
    """
    if verbose:
        print("\n" + "="*60)
        print("  FULL STATISTICAL SIGNIFICANCE REPORT")
        print("="*60)

    r1 = paired_ttest(baseline_accs, poisoned_accs,
                      "Baseline", "Poisoned", verbose=verbose)
    r2 = paired_ttest(defended_accs, poisoned_accs,
                      "Defended", "Poisoned", verbose=verbose)
    r3 = paired_ttest(baseline_accs, defended_accs,
                      "Baseline", "Defended", verbose=verbose)

    return {
        "baseline_vs_poisoned": r1,
        "defended_vs_poisoned": r2,
        "baseline_vs_defended": r3,
    }


#  Phase-2: Wilcoxon + pairwise table 
import warnings, pandas as pd

def wilcoxon_test(a, b, alpha=0.05):
    import numpy as np; from scipy import stats
    a_, b_ = np.array(a, float), np.array(b, float)
    if len(a_) < 3: return {'stat': float('nan'), 'p_value': 1.0, 'significant': False, 'better': 'tie'}
    diff = a_ - b_
    if np.all(diff == 0): return {'stat': 0.0, 'p_value': 1.0, 'significant': False, 'better': 'tie'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, p = stats.wilcoxon(a_, b_, alternative='two-sided')
    return {'stat': float(stat), 'p_value': float(p), 'significant': bool(p < alpha),
            'better': 'a' if float(np.mean(a_)) > float(np.mean(b_)) else 'b',
            'mean_a': float(np.mean(a_)), 'mean_b': float(np.mean(b_)),
            'std_a': float(np.std(a_)), 'std_b': float(np.std(b_)), 'n': len(a_)}


def pairwise_significance_table(results, alpha=0.05, metric_name='Acc %'):
    import numpy as np, pandas as pd
    methods = list(results.keys())
    rows = []
    for i, m_a in enumerate(methods):
        for m_b in methods[i+1:]:
            res = wilcoxon_test(results[m_a], results[m_b], alpha=alpha)
            d   = cohens_d(np.array(results[m_a]), np.array(results[m_b]))
            rows.append({'Method A': m_a, 'Method B': m_b,
                         f'Mean A': round(res.get('mean_a', float('nan')), 2),
                         f'Mean B': round(res.get('mean_b', float('nan')), 2),
                         'Diff': round(res.get('mean_a', 0) - res.get('mean_b', 0), 2),
                         'p-value': round(res.get('p_value', 1.0), 4),
                         'Sig?': 'Yes' if res.get('significant') else 'No',
                         'Cohen d': round(d, 2), 'Effect': interpret_cohens_d(d)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()
