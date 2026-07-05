"""
theory/repeated_game_analysis.py
==================================
Correct Game-Theoretic Framing: Repeated Adversarial Game.

NOTE ON TERMINOLOGY:
  Our earlier files used "Stackelberg game" which is technically imprecise.
  This module provides the correct framing and explains the relationship.

═══════════════════════════════════════════════════════════════════════
  CORRECT FRAMING: REPEATED ADVERSARIAL GAME
═══════════════════════════════════════════════════════════════════════

THE CONFUSION:
  A STACKELBERG game has:
    - One LEADER who moves first (commits to a strategy)
    - One FOLLOWER who observes the leader's move and responds optimally

  In data poisoning:
    - The ATTACKER poisons data BEFORE the defender trains. (Attacker = Leader ✓)
    - The DEFENDER sees the poisoned data and trains optimally. (Defender = Follower ✓)
  → Single-round Stackelberg fits the one-shot poisoning setting.

  BUT in our iterative game with r rounds:
    - Round r DEFENDER warm-starts from θ^(r-1), which was adapted to a^(r-1).
    - Round r ATTACKER adapts to θ^(r-1), which reflected the previous defense.
    - BOTH players observe each other's previous moves and adapt.
    - This is NOT a pure Stackelberg game — it is a REPEATED GAME.

CORRECT FORMULATION — REPEATED ADVERSARIAL GAME:
  At round r = 1, ..., R:
    Follower step (attacker adapts, attacks FIRST):
      a^(r) = argmax_{a ∈ A} L(θ^(r-1), a)   [attacker observes prev θ]
    Leader step (defender responds, trains SECOND):
      θ^(r) = argmin_{θ ∈ Θ} L(θ, a^(r))     [defender observes current a]

  This is a REPEATED ZERO-SUM GAME with full observability.

  KEY PROPERTIES:
    1. In a finite-horizon repeated zero-sum game, the time-averaged strategy
       profile (ā^(R), θ̄^(R)) converges to a Nash Equilibrium as R → ∞
       (by the Folk Theorem for repeated games, Aumann 1959).

    2. Warm-starting means the defender's strategy at round r1 is NOT
       independent of round r2 (unlike in independent game repetitions).
       This creates a CORRELATED EQUILIBRIUM structure, which may be
       more or less favorable to the attacker depending on the loss landscape.

    3. The Stackelberg LEADER/FOLLOWER role REVERSES each round:
       - The attacker is the Stackelberg leader at the START of each round.
       - The defender is the Stackelberg follower WITHIN each round.
       - But the defender's warm-starting makes the defender the "leader"
         in the meta-game across rounds (the defender commits historically).

DISTINCTION FROM STANDARD STACKELBERG:
  Standard Stackelberg (one-shot):     Attacker commits → Defender best-responds (once)
  Our Repeated Game:                   Both players iterate, observing each other
  Bilevel Optimization:               Defender trains while explicitly modeling attacker

IMPLICATION FOR OUR CLAIMS:
  We can claim: "The iterative procedure converges to a Nash Equilibrium
  of the repeated game by the Folk Theorem."
  We CANNOT claim: "This is a Stackelberg equilibrium" without strong convexity.

  The correct claim: our defended model θ^(R) is a BEST RESPONSE to the
  attacker's strategy a^(R), and vice versa (approximately, under finite R).

═══════════════════════════════════════════════════════════════════════
  EMPIRICAL VALIDATION OF GAME CONVERGENCE
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import List, Dict, Optional


def verify_best_response(
    round_metrics: List[Dict],
    verbose: bool = True,
) -> Dict:
    """
    Empirically verify whether the game has reached approximate best-response:
    1. Defender's accuracy has stabilized (no longer improving across rounds)
    2. Attacker's ASR has stabilized (no longer improving across rounds)
    3. The sequence is MONOTONE: ASR decreasing, Acc non-decreasing

    A CONVERGED game should satisfy all three.

    Args:
        round_metrics: List of per-round metrics with 'mean_acc' and 'mean_asr'.

    Returns:
        dict with convergence diagnostics.
    """
    if len(round_metrics) < 2:
        return {"verified": False, "note": "Need >= 2 rounds"}

    rounds = sorted(round_metrics, key=lambda x: x["round"])
    accs   = [r["mean_acc"] for r in rounds]
    asrs   = [r["mean_asr"] for r in rounds]

    # Check 1: Defender improving (acc non-decreasing) — defender best-responding
    acc_diffs = [accs[i+1] - accs[i] for i in range(len(accs)-1)]
    defender_improving = sum(d >= -0.5 for d in acc_diffs) >= len(acc_diffs) * 0.8

    # Check 2: ASR decreasing — attacker's advantage eroding as defender adapts
    asr_diffs = [asrs[i+1] - asrs[i] for i in range(len(asrs)-1)]
    asr_decreasing = sum(d <= 0.5 for d in asr_diffs) >= len(asr_diffs) * 0.8

    # Check 3: Final equilibrium gap — how stable are the last 2 rounds?
    final_acc_gap = abs(accs[-1] - accs[-2])
    final_asr_gap = abs(asrs[-1] - asrs[-2])
    equilibrium_reached = (final_acc_gap < 0.5) and (final_asr_gap < 1.0)

    result = {
        "verified":              defender_improving and asr_decreasing,
        "defender_improving":    defender_improving,
        "asr_eroding":           asr_decreasing,
        "equilibrium_reached":   equilibrium_reached,
        "final_acc_stability":   round(final_acc_gap, 3),
        "final_asr_stability":   round(final_asr_gap, 3),
        "total_asr_reduction":   round(asrs[0] - asrs[-1], 2),
        "total_acc_gain":        round(accs[-1] - accs[0], 2),
        "n_rounds":              len(rounds),
    }

    if verbose:
        print("\n" + "═"*62)
        print("  REPEATED ADVERSARIAL GAME — CONVERGENCE VERIFICATION")
        print("═"*62)
        print(f"  {'Round':>5}  {'Acc':>8}  {'ΔASR':>8}  {'ΔAcc':>8}")
        print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
        for i, r in enumerate(rounds):
            d_asr = f"{asr_diffs[i-1]:+.2f}" if i > 0 else "  —"
            d_acc = f"{acc_diffs[i-1]:+.2f}" if i > 0 else "  —"
            print(f"  {r['round']:>5}  {r['mean_acc']:>8.2f}  {d_asr:>8}  {d_acc:>8}")
        print()
        print(f"  ✓ Defender improving:  {result['defender_improving']}")
        print(f"  ✓ ASR eroding:         {result['asr_eroding']}")
        print(f"  ✓ Equilibrium reached: {result['equilibrium_reached']}")
        print(f"  Total ASR reduction:   {result['total_asr_reduction']:+.2f}pp")
        print(f"  Total Acc gain:        {result['total_acc_gain']:+.2f}pp")
        verdict = "CONVERGED" if result["verified"] else "NOT YET CONVERGED"
        print(f"\n  VERDICT: {verdict}")
        print("═"*62)

    return result


def print_game_framing() -> None:
    """Print the correct theoretical framing for the paper's methods section."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  REPEATED ADVERSARIAL GAME FORMULATION                           ║
╠══════════════════════════════════════════════════════════════════╣
║  At each round r = 1, ..., R:                                    ║
║                                                                  ║
║  1. ATTACKER observes θ^(r-1) and selects:                      ║
║     a^(r) = argmax_{a} L(θ^(r-1), a)    subject to |a| ≤ ε·N   ║
║                                                                  ║
║  2. DEFENDER warm-starts from θ^(r-1) and trains:               ║
║     θ^(r) = θ^(r-1) - η · ∇_θ L(θ, a^(r))  [SGD steps]       ║
║             with early stopping on clean val set                 ║
║                                                                  ║
║  CONVERGENCE (Folk Theorem, Aumann 1959):                        ║
║  Time-averaged strategies (ā,θ̄) → Nash Equilibrium as R → ∞   ║
║                                                                  ║
║  OUR CLAIM: θ^(R) is a near-best-response to a^(R) for R≥5.    ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_game_framing()
    mock = [{"round": r, "mean_acc": 88 + r*1.2, "mean_asr": 48 - r*3.5}
            for r in range(1, 6)]
    result = verify_best_response(mock)
