"""
utils/plotting.py
=================
Publication-ready graph generation.

All graphs are saved to results/figures/ and also displayed inline
when running inside a Jupyter notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional


# Use a clean, paper-friendly style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_sweep(
    fractions: List[float],
    means: List[float],
    stds: List[float],
    save_name: str = "sweep.png",
    title: str = "Accuracy vs Poison Fraction",
) -> str:
    """
    Line chart with error bars showing how accuracy degrades as poison fraction increases.

    Args:
        fractions: List of poison fractions, e.g. [0.2, 0.4, 0.6, 0.8]
        means:     Corresponding mean accuracies
        stds:      Corresponding std deviations
        save_name: Filename to save under results/figures/
        title:     Plot title

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_dir(FIGURES_DIR)
    save_path = os.path.join(FIGURES_DIR, save_name)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        fractions, means, yerr=stds,
        marker="o", capsize=5, capthick=2,
        color="#2196F3", ecolor="#1565C0", label="Poisoned Model"
    )

    ax.set_xlabel("Poison Fraction")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(fractions)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot saved] {save_path}")
    return save_path


def plot_comparison(
    labels: List[str],
    means: List[float],
    stds: List[float],
    colors: Optional[List[str]] = None,
    save_name: str = "comparison.png",
    title: str = "Model Comparison",
) -> str:
    """
    Grouped bar chart comparing baseline / poisoned / defended accuracy.

    Args:
        labels:    Bar labels, e.g. ['Baseline', 'Poisoned', 'Defended']
        means:     Mean accuracies for each bar
        stds:      Std deviations for each bar
        colors:    Optional list of hex colors
        save_name: Filename under results/figures/
        title:     Plot title

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_dir(FIGURES_DIR)
    save_path = os.path.join(FIGURES_DIR, save_name)

    if colors is None:
        colors = ["#4CAF50", "#F44336", "#2196F3"]  # green, red, blue

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors[:len(labels)],
                  edgecolor="black", linewidth=0.8, alpha=0.88)

    # Annotate bars with values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.3,
            f"{mean:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    # Set y-axis range slightly below min for readability
    y_min = max(0, min(means) - max(stds) - 5)
    ax.set_ylim(y_min, 102)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot saved] {save_path}")
    return save_path


def plot_minmax_rounds(
    rounds: List[int],
    defended_accs: List[float],
    poisoned_baseline: float,
    clean_baseline: float,
    save_name: str = "minmax_rounds.png",
    title: str = "Min–Max Defender: Accuracy Over Alternating Rounds",
) -> str:
    """
    Line chart showing how defended accuracy evolves over alternating attacker-defender rounds.

    Args:
        rounds:            Round numbers [1, 2, 3, ...]
        defended_accs:     Accuracy after each defender step
        poisoned_baseline: One-shot poisoned accuracy (horizontal reference line)
        clean_baseline:    Clean baseline accuracy (horizontal reference line)
        save_name:         Filename under results/figures/
        title:             Plot title
    """
    _ensure_dir(FIGURES_DIR)
    save_path = os.path.join(FIGURES_DIR, save_name)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(rounds, defended_accs, marker="s", color="#2196F3", label="Defended (Min–Max)")
    ax.axhline(poisoned_baseline, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Poisoned baseline ({poisoned_baseline:.1f}%)")
    ax.axhline(clean_baseline, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"Clean baseline ({clean_baseline:.1f}%)")

    ax.set_xlabel("Alternating Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(rounds)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot saved] {save_path}")
    return save_path
