"""
utils/threat_model.py
======================
Formal Threat Model Definition — Grey-Box Setting.

This module defines and enforces the threat model uniformly across ALL
experiments. Every experiment MUST import and use these constants to ensure
consistency. Mixing threat model assumptions across experiments is a
common flaw in adversarial ML papers.

═══════════════════════════════════════════════════════════════════════
  FORMAL THREAT MODEL: GREY-BOX DATA POISONING
═══════════════════════════════════════════════════════════════════════

SETTING:
  A third-party data supplier (attacker) contributes a fraction of the
  model owner's (defender's) training data. The model owner trains a
  classifier on the (potentially mixed) data.

ATTACKER CAPABILITIES (what the attacker CAN do):
  [A1] Modify training samples: add/change pixels within budget ε_pixel.
  [A2] Change training labels of contributed samples.
  [A3] Know the model ARCHITECTURE (ResNet-18/50, ViT) — grey-box.
  [A4] Know the training ALGORITHM (SGD + cosine annealing) — grey-box.
  [A5] Poison up to ε_data fraction of TOTAL training data.
  [A6] Choose which (class, target_class) pair to attack.

ATTACKER CANNOT (grey-box constraints):
  [NA1] Access model WEIGHTS or gradients during defender's training.
       (If attacker could, this would be white-box — unrealistic in supply chain.)
  [NA2] Modify test-time images (except for test-time trigger eval of backdoors).
  [NA3] Query the deployed model (no model stealing / evasion in addition).
  [NA4] Exceed the budget ε_data on total data fraction.

DEFENDER CAPABILITIES:
  [D1] Train on provided data with any optimizer and hyperparameters.
  [D2] Apply any data filtering or detection defense.
  [D3] Use a held-out validation set (assumed clean, 10% of data).
  [D4] Access model gradients and features for defense computation.

PRIMARY EVALUATION BUDGET:
  ε_data = 0.05  (5% of total training data)
  This is the REALISTIC threat scenario. An attacker controlling a
  data scraping pipeline can plausibly inject 5% bad samples.

STRESS TEST BUDGET:
  ε_data = [0.10, 0.20, 0.30, 0.50]
  Used for ablation studies to understand upper limits of attack/defense.

RANDOM ASR BASELINE:
  For a k-class problem, a model that predicts uniformly at random
  would achieve ASR = 1/k. Any reported ASR must be compared to this
  baseline to be interpretable.

REFERENCES:
  Goldblum et al. (2022). "Dataset Security for Machine Learning." IEEE TPAMI.
  Schwarzschild et al. (2021). "Just How Toxic is Data Poisoning?" ICML.
  Carlini & Wagner (2017). "Adversarial examples are not easily detected." AISec.

═══════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────── Budget Constants ────────────────────────────────────

# PRIMARY (realistic): attacker injects ≤5% of training data
EPSILON_PRIMARY: float = 0.05

# STRESS TEST fractions for ablation sweep
EPSILON_STRESS: List[float] = [0.05, 0.10, 0.20, 0.30, 0.50]

# Default pixel perturbation budget (L-inf, normalized to [0,1])
EPSILON_PIXEL: float = 8.0 / 255   # ≈ 0.031 — standard adversarial ML budget

# Default seeds (n=5 for adequate statistical power at α=0.05)
DEFAULT_SEEDS: List[int] = [0, 1, 2, 3, 4]
N_SEEDS: int = 5  # minimum for Wilcoxon significance test

# Significance level
ALPHA: float = 0.05

# Validation set fraction (assumed clean)
VAL_FRACTION: float = 0.10

# Dataset-specific class counts
N_CLASSES = {"mnist": 10, "cifar10": 10, "cifar100": 100, "gtsrb": 43}


# ─────────────────────── Random ASR Baseline ─────────────────────────────────

def random_asr_baseline(n_classes: int) -> float:
    """
    ASR for a random predictor: 1/n_classes.

    Any reported ASR must exceed this to demonstrate a real attack.
    An ASR of 12% on CIFAR-10 (10 classes) is barely above chance (10%).

    Args:
        n_classes: Number of output classes.

    Returns:
        float: Chance-level ASR ∈ (0, 1].
    """
    return 1.0 / max(1, n_classes)


# ─────────────────────── Threat Model Dataclass ──────────────────────────────

@dataclass
class ThreatModel:
    """
    Immutable specification of the threat model for one experiment.
    Pass this object to all experiment functions for consistency.
    """
    epsilon_data:       float           = EPSILON_PRIMARY
    epsilon_pixel:      float           = EPSILON_PIXEL
    attacker_knowledge: str             = "grey-box"  # "white-box", "grey-box", "black-box"
    src_class:          int             = 1
    tgt_class:          int             = 7
    seeds:              List[int]       = field(default_factory=lambda: list(DEFAULT_SEEDS))
    n_classes:          int             = 10
    val_fraction:       float           = VAL_FRACTION

    # Derived
    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def random_asr(self) -> float:
        return random_asr_baseline(self.n_classes)

    def describe(self) -> str:
        return (
            f"\n{'═'*60}\n"
            f"  THREAT MODEL\n"
            f"{'═'*60}\n"
            f"  Attacker knowledge : {self.attacker_knowledge}\n"
            f"  Poison budget (ε)  : {self.epsilon_data:.0%} of training data\n"
            f"  Pixel budget       : {self.epsilon_pixel:.4f} (L-inf)\n"
            f"  Attack pair        : class {self.src_class} → {self.tgt_class}\n"
            f"  Val set            : {self.val_fraction:.0%} (assumed clean)\n"
            f"  Seeds (n)          : {self.n_seeds}\n"
            f"  Random ASR baseline: {self.random_asr:.1%}  "
            f"(chance for {self.n_classes}-class problem)\n"
            f"  Significance level : α = {ALPHA}\n"
            f"{'═'*60}"
        )

    def validate_asr(self, asr: float, label: str = "") -> dict:
        """
        Check if an ASR value is above chance and practically significant.
        Returns interpretation.
        """
        above_chance = asr > self.random_asr
        margin       = asr - self.random_asr
        practical    = margin > 0.05   # >5pp above chance = practically significant

        result = {
            "asr":          asr,
            "random_baseline": self.random_asr,
            "above_chance": above_chance,
            "margin_pp":    round(margin * 100, 2),  # percentage points above chance
            "practically_significant": practical,
        }

        tag = label + " " if label else ""
        if not above_chance:
            verdict = f"⚠️  {tag}ASR={asr:.1%} is AT or BELOW chance ({self.random_asr:.1%}). Attack FAILED."
        elif not practical:
            verdict = f"⚠️  {tag}ASR={asr:.1%} is only {margin:.1%} above chance. Marginal attack."
        else:
            verdict = f"✓  {tag}ASR={asr:.1%} = {margin:.1%} above chance. Clear attack signal."

        result["verdict"] = verdict
        return result


# ─────────────────────── Standard Experiment Template ────────────────────────

def get_standard_threat_model(dataset: str = "cifar10",
                               src_class: int = 1,
                               tgt_class: int = 7) -> ThreatModel:
    """Return the standard grey-box threat model for a given dataset."""
    nc = N_CLASSES.get(dataset, 10)
    return ThreatModel(
        epsilon_data=EPSILON_PRIMARY,
        epsilon_pixel=EPSILON_PIXEL,
        attacker_knowledge="grey-box",
        src_class=src_class,
        tgt_class=tgt_class,
        seeds=list(DEFAULT_SEEDS),
        n_classes=nc,
    )


if __name__ == "__main__":
    tm = get_standard_threat_model("cifar10")
    print(tm.describe())
    print(tm.validate_asr(0.12, "Spectral Signatures")["verdict"])
    print(tm.validate_asr(0.47, "No Defense")["verdict"])
    print(tm.validate_asr(0.08, "ABL")["verdict"])
