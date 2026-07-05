"""
theory/convergence_analysis.py
================================
Theoretical conditions for Stackelberg game convergence.

This module provides:
  1. Mathematical conditions under which the Stackelberg Min-Max game
     converges to a unique Nash equilibrium.
  2. Empirical convergence diagnostics (variance of round-to-round
     defender accuracy and ASR) to detect convergence in practice.
  3. Lipschitz constant estimation for the loss function.

═══════════════════════════════════════════════════════════════════
  THEORETICAL FRAMEWORK
═══════════════════════════════════════════════════════════════════

GAME DEFINITION:
  Let θ ∈ Θ be the defender's model parameters (a convex compact set in R^P).
  Let a ∈ A be the attacker's strategy (a subset of indices to relabel, |A| = N·ε).
  Let L(θ, a) be the empirical loss of θ on poisoned dataset D̃_a.

  The STACKELBERG MIN-MAX game:
    (θ*, a*) = argmin_θ  max_a  L(θ, a)

  The attacker moves FIRST (leader) and the defender responds (follower).

EXISTENCE OF NASH EQUILIBRIUM (von Neumann 1928):
  Under the following conditions, a Nash equilibrium (θ*, a*) exists:
    C1. Θ is compact and convex.
    C2. A is finite (binary action space: relabel or not for each sample).
    C3. L(θ, a) is continuous in θ for all fixed a.
  Proof sketch: By Kakutani's fixed-point theorem applied to the
  best-response correspondences BR_θ(a) and BR_a(θ).

UNIQUENESS CONDITIONS:
  The Min-Max equilibrium is unique if additionally:
    C4. L(θ, a) is strictly convex in θ for all fixed a.
        (Satisfied for cross-entropy loss with L2 regularization and
         sufficiently overparameterized neural networks near convergence.)
    C5. L(θ, a) is strictly concave in a for all fixed θ.
        (Weaker condition: satisfied when samples are sufficiently
         independent, i.e., the poison samples do not fully span the
         feature space of the defender's network.)

CONVERGENCE OF THE ALTERNATING PROCEDURE:
  The alternating max_a → min_θ → max_a → ... procedure converges to the
  Stackelberg equilibrium if:

    Condition 1 (Sufficient Descent): Each min-θ step strictly decreases L.
      Formally: ∃ α > 0 s.t. L(θ^(r+1), a^(r)) ≤ L(θ^(r), a^(r)) - α·‖θ^(r+1) - θ^(r)‖²

    Condition 2 (Lipschitz Continuity): L(θ, ·) is L-Lipschitz in a.
      Formally: |L(θ, a1) - L(θ, a2)| ≤ L · |a1 - a2| / N
      This bounds how much the attacker can change the loss per round.

    Condition 3 (Bounded Action Space): |A| ≤ ε · N (budget constraint).
      The attacker cannot poison more than fraction ε of the training set.

  Under C1-C5 and Conditions 1-3, the sequence {(θ^(r), a^(r))} converges
  to the unique Stackelberg equilibrium (θ*, a*) as r → ∞.

  RATE OF CONVERGENCE: With warm-starting (which our implementation uses),
  the rate is O(exp(-αr)) (geometric), as opposed to O(1/r) for cold starts.
  Intuitively, warm-starting means the defender needs fewer gradient steps
  per round to reach the optimal response, accelerating overall convergence.

PRACTICAL CONVERGENCE CRITERION:
  We declare convergence when, for τ consecutive rounds:
    |overall_acc^(r) - overall_acc^(r-1)| < δ_acc
    |ASR^(r) - ASR^(r-1)| < δ_ASR
  Where δ_acc = 0.5% and δ_ASR = 1.0% are typical thresholds.

LIMITATIONS OF OUR IMPLEMENTATION:
  1. Neural networks are generally NOT convex in θ, so C4 may not hold
     globally. However, near local minima (which SGD finds), the loss
     is approximately convex in a small neighborhood.
  2. With finite n_rounds (typically 5), we do not claim full convergence
     but rather that we have computed several steps of the alternating
     procedure, providing a lower bound on the equilibrium defense quality.
  3. The discrete nature of A (we select which samples to poison) means
     A is actually a combinatorial space, not a continuous one. True
     convergence in this setting requires Condition 2 to hold approximately
     over the discrete-to-continuous relaxation.

REFERENCES:
  - von Neumann, J. (1928). "Zur Theorie der Gesellschaftsspiele." Math. Ann.
  - Stackelberg, H. v. (1934). "Marktform und Gleichgewicht." Springer.
  - Fiacco, A.V. & McCormick, G.P. (1968). Nonlinear Programming. SIAM.
  - Dempe, S. (2002). Foundations of Bilevel Programming. Springer.
  - Geiping et al. (2021). "Witches' Brew." ICLR.  (convergence in PGD attacks)

═══════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import List, Optional, Dict
import warnings


def check_roundwise_convergence(
    round_metrics: List[Dict],
    delta_acc:  float = 0.5,
    delta_asr:  float = 1.0,
    tau:        int   = 2,
    verbose:    bool  = True,
) -> Dict:
    """
    Test whether the Stackelberg game has converged based on round metrics.

    Convergence criterion: for the last tau rounds, both acc and ASR change
    by less than delta_acc and delta_asr respectively.

    Args:
        round_metrics: List of dicts with 'round', 'mean_acc', 'mean_asr' keys
                       (as returned by stackelberg_game.run_stackelberg_game).
        delta_acc:     Maximum allowed round-to-round change in accuracy (%).
        delta_asr:     Maximum allowed round-to-round change in ASR (%).
        tau:           Number of consecutive stable rounds required.

    Returns:
        dict with:
          'converged': bool
          'convergence_round': int or None
          'final_acc_variance': float
          'final_asr_variance': float
          'condition_satisfied': list of per-round booleans
    """
    if len(round_metrics) < 2:
        return {"converged": False, "convergence_round": None,
                "note": "Need at least 2 rounds to check convergence"}

    rounds = sorted(round_metrics, key=lambda x: x["round"])
    n = len(rounds)

    acc_changes  = [abs(rounds[i]["mean_acc"] - rounds[i-1]["mean_acc"]) for i in range(1, n)]
    asr_changes  = [abs(rounds[i]["mean_asr"] - rounds[i-1]["mean_asr"]) for i in range(1, n)]
    satisfied    = [a < delta_acc and b < delta_asr
                    for a, b in zip(acc_changes, asr_changes)]

    converged           = False
    convergence_round   = None

    for i in range(len(satisfied) - tau + 1):
        if all(satisfied[i:i+tau]):
            converged         = True
            convergence_round = rounds[i+1]["round"]
            break

    final_acc_var = float(np.var([r["mean_acc"] for r in rounds[-tau:]]))
    final_asr_var = float(np.var([r["mean_asr"] for r in rounds[-tau:]]))

    if verbose:
        print("\n" + "═"*55)
        print("  STACKELBERG GAME CONVERGENCE ANALYSIS")
        print("═"*55)
        print(f"  δ_acc={delta_acc}%  δ_ASR={delta_asr}%  τ={tau}")
        print(f"\n  {'Round':>5}  {'Δ Acc':>8}  {'Δ ASR':>8}  {'Stable?':>8}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
        for i, (r, ac, ar, s) in enumerate(
                zip(rounds[1:], acc_changes, asr_changes, satisfied)):
            print(f"  {r['round']:>5}  {ac:>+8.2f}  {ar:>+8.2f}  {'✓' if s else '✗':>8}")

        print()
        if converged:
            print(f"  CONVERGED at round {convergence_round}")
            print(f"  Final variance — Acc: {final_acc_var:.4f}  ASR: {final_asr_var:.4f}")
        else:
            print(f"  NOT CONVERGED after {n} rounds")
            print(f"  Consider increasing n_rounds (currently {n})")
        print("═"*55)

    return {
        "converged":          converged,
        "convergence_round":  convergence_round,
        "acc_changes":        acc_changes,
        "asr_changes":        asr_changes,
        "condition_satisfied":satisfied,
        "final_acc_variance": final_acc_var,
        "final_asr_variance": final_asr_var,
    }


def estimate_lipschitz_constant(
    model,
    dataset,
    device,
    n_pairs: int = 50,
    src_class: int = 1,
    tgt_class: int = 7,
    seed: int = 42,
) -> float:
    """
    Estimate the Lipschitz constant of L(θ, ·) in the attacker action space.

    L ≈ max_{i ≠ j} |L(θ, a_i) - L(θ, a_j)| / ‖a_i - a_j‖_1

    We estimate this by sampling random pairs of poison strategies.
    A small Lipschitz constant means the attacker has limited ability to
    change the loss across rounds → faster game convergence.

    Returns:
        L_hat: estimated Lipschitz constant (float)
    """
    import torch
    import torch.nn as nn
    from attacks.label_flip import poison_dataset

    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    src_idx = np.where(labels == src_class)[0]
    rng = np.random.default_rng(seed)
    n_src = len(src_idx)
    criterion = nn.CrossEntropyLoss()

    from torch.utils.data import DataLoader
    losses = []

    for trial in range(n_pairs):
        frac = rng.uniform(0.05, 0.60)
        pds, _ = poison_dataset(dataset, src_class, tgt_class, frac, seed=trial)
        loader = DataLoader(pds, batch_size=256, shuffle=False, num_workers=0)
        model.eval()
        total_loss = 0.0
        n_batches  = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                total_loss += criterion(model(imgs), lbls).item()
                n_batches  += 1
        losses.append((frac, total_loss / n_batches))

    losses.sort(key=lambda x: x[0])
    fracs = [l[0] for l in losses]
    vals  = [l[1] for l in losses]

    ratios = []
    for i in range(len(fracs)):
        for j in range(i+1, len(fracs)):
            df = abs(fracs[j] - fracs[i])
            dl = abs(vals[j]  - vals[i])
            if df > 1e-6:
                ratios.append(dl / df)

    L_hat = float(np.percentile(ratios, 95)) if ratios else float("nan")
    print(f"  Estimated Lipschitz constant (L̂) = {L_hat:.4f}")
    print(f"  (Lower is better for convergence speed)")
    return L_hat


def print_convergence_theorem() -> None:
    """Print the main convergence theorem as a formatted reference."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  THEOREM (Stackelberg Min-Max Convergence)                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Given:                                                      ║
║    1. Loss L(θ, a) is L-Lipschitz in a                      ║
║    2. L(θ, a) is α-strongly convex in θ (near local minimum)║
║    3. Defender uses SGD with warm-starting                   ║
║    4. Attacker budget |a| ≤ ε·N is fixed                    ║
║                                                              ║
║  Then: The alternating Stackelberg procedure                 ║
║    θ^(r+1) = argmin_θ L(θ, a^(r))    [defender step]       ║
║    a^(r+1) = argmax_a L(θ^(r+1), a)  [attacker step]       ║
║                                                              ║
║  converges geometrically:                                    ║
║    ‖(θ^(r), a^(r)) - (θ*, a*)‖ ≤ C · ρ^r                  ║
║    where C > 0 and ρ = (L²/α²) / (1 + L²/α²) < 1          ║
║                                                              ║
║  WARM-STARTING ACCELERATION:                                 ║
║    Without warm-start: convergence O(r · T_opt)             ║
║    With warm-start:    convergence O(log(r) · T_opt)        ║
║    where T_opt = epochs needed per round                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_convergence_theorem()

    # Example: check convergence on mock round metrics
    mock_rounds = [
        {"round": 1, "mean_acc": 89.2, "mean_asr": 45.3},
        {"round": 2, "mean_acc": 91.5, "mean_asr": 38.1},
        {"round": 3, "mean_acc": 92.8, "mean_asr": 33.2},
        {"round": 4, "mean_acc": 93.1, "mean_asr": 32.5},
        {"round": 5, "mean_acc": 93.2, "mean_asr": 32.1},
    ]

    result = check_roundwise_convergence(mock_rounds, delta_acc=0.5, delta_asr=1.0, tau=2)
    print(f"\nConverged: {result['converged']}  "
          f"at round: {result['convergence_round']}")
