# Formal Game-Theoretic Framework
## Data Poisoning as a Two-Player Zero-Sum Stackelberg Game

---

## 1. Definitions and Notation

Let:
- **D** = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ — clean training dataset drawn from distribution **P**
- **f_θ** : X → ℝᶜ — neural network with parameters **θ** ∈ Θ
- **ℓ** : ℝᶜ × Y → ℝ₊ — loss function (cross-entropy)
- **L(θ, S)** = (1/|S|) Σᵢ∈S ℓ(f_θ(xᵢ), yᵢ) — empirical risk on set S
- **A** = {D_p : |D_p| ≤ εN, D_p ⊂ D} — attacker's feasible poison set (budget constraint)

---

## 2. The Bilevel Min–Max Objective

The interaction between attacker and defender is formalized as a **Stackelberg game**:

> **Attacker (Leader)** commits first: selects poisoned dataset D̃.
>
> **Defender (Follower)** responds: trains model to minimize loss on D ∪ D̃.

This yields the bilevel program:

```
OUTER (attacker, maximizer):
    max_{D̃ ∈ A}  E_{(x,y)~P} [ ℓ(f_{θ*(D̃)}(x), y) ]

INNER (defender, minimizer):
    θ*(D̃) = argmin_{θ ∈ Θ}  L(θ, D ∪ D̃)
```

Combined as a single min–max problem:

```
    min_θ  max_{D̃ ∈ A}  L(f_θ, D ∪ D̃)          ... (1)
```

Where the outer max is approximated by a gradient-based attacker and the
inner min is solved via SGD-based neural network training.

---

## 3. Attacker Strategy Space

**Definition (Label-Flip Attack):**
The attacker's strategy Φ_LF(src, tgt, ε) selects a subset D_p ⊆ {(x,y) ∈ D : y = src}
with |D_p| ≤ ε |{(x,y) ∈ D : y = src}| and assigns ỹ = tgt to all (x,ỹ) ∈ D_p.

**Definition (Gradient-Based Attack):**
The attacker optimizes poisoned feature perturbations δᵢ for each chosen sample:

```
    max_{δᵢ : ||δᵢ||_∞ ≤ ε}  L_val(f_{θ*}(xᵢ + δᵢ, yᵢ))
```

Implemented via projected gradient ascent:

```
    δᵢ^{t+1} ← Π_{||·||_∞ ≤ ε} [ δᵢ^t + α · sign(∇_{δᵢ} L_val(f_θ, xᵢ + δᵢ^t, ỹᵢ)) ]
```

This is the **PGD-based attacker** (analogous to PGD adversarial training,
Madry et al. 2018, but operating on training data rather than test inputs).

---

## 4. Defender Strategy

The defender solves the inner minimization via gradient descent:

```
    θ^{t+1} ← θ^t - β · ∇_θ L(f_{θ^t}, D ∪ D̃)
```

In the alternating setting, the defender does NOT know which samples are
poisoned — it trains on all available data. This is the central challenge:
the defender must become robust **without supervision** about which points
are corrupted.

---

## 5. Alternating Min–Max Algorithm

```
Algorithm 1: Alternating Min–Max Attacker-Defender Training

Input:  Clean dataset D, attacker budget ε, defender lr β,
        attacker lr α, rounds R, defender epochs K

Initialize: θ₀ ~ random, D̃₀ = Φ_LF(D, src, tgt, ε₀)

For r = 1, 2, ..., R:
  // ── DEFENDER STEP (minimization) ──────────────────────
  θ_r ← TRAIN(θ₀, D ∪ D̃_{r-1}, epochs=K, lr=β)
  //   θ_r = argmin_θ L(θ, D ∪ D̃_{r-1}) [K-step approximation]

  // ── ATTACKER STEP (maximization) ──────────────────────
  For i ∈ poison_indices:
    δᵢ^{t+1} ← Π_{||·||_∞ ≤ ε} [δᵢ^t + α · sign(∇_{δᵢ} L_val(f_{θ_r}, xᵢ+δᵢ^t))]
  D̃_r ← {(xᵢ + δᵢ, ỹᵢ) : i ∈ poison_indices}

  // ── EVALUATE ──────────────────────────────────────────
  Record acc_r = Acc(f_{θ_r}, D_test)

Output: Final model θ_R, per-round accuracies {acc_r}
```

**Interpretation:**
- Each round is one move in the game. The defender plays first (commits model weights), the attacker plays second (improves poison).
- This is the **Stackelberg follower-leader** reversal in alternating form.
- Convergence to a local Nash equilibrium is not guaranteed in general, but empirically this produces a robust model.

---

## 6. Nash Equilibrium and Convergence

**Definition (Nash Equilibrium):**
A pair (θ*, D̃*) constitutes a Nash equilibrium of the poisoning game if:

```
    L(θ*, D ∪ D̃*) ≤ L(θ', D ∪ D̃*)   ∀ θ' ∈ Θ     (defender cannot improve)
    L(θ*, D ∪ D̃*) ≥ L(θ*, D ∪ D̃')   ∀ D̃' ∈ A     (attacker cannot improve)
```

**Remark (Existence):**
Since L is continuous in θ and the attacker set A is compact (finite budget),
Nash equilibria exist by Glicksberg's theorem (Glicksberg, 1952) under mild
regularity conditions on the loss landscape.

**Remark (Computational Hardness):**
Finding the exact Nash equilibrium of the bilevel problem (1) is NP-hard
in general (Brückner & Scheffer, 2011). Our Algorithm 1 provides a practical
approximation via first-order methods, which is the standard approach in
adversarial training literature (Madry et al., 2018; Goodfellow et al., 2014).

**Remark (Convergence Claim):**
Under Lipschitz continuity of ∇_θ L and bounded gradient norms, the alternating
gradient descent-ascent procedure satisfies:

```
    min_{r ≤ R} || ∇_θ L + ∇_{D̃} L ||₂ ≤ O(1/√R)
```

This guarantees convergence to a **first-order stationary point** of the min-max
objective, which is the standard convergence notion for non-convex min-max
problems (Lin et al., 2020).

---

## 7. Federated Learning Extension

In the federated setting, N training samples are distributed across K clients,
each with local dataset D_k (|D_k| = Nₖ, ΣNₖ = N).

A fraction of clients (m < K) are **Byzantine (malicious)**: they apply the
poisoning attack locally and return a perturbed model update to the server.

The FedAvg global objective is:

```
    min_θ  Σ_k (Nₖ/N) · L(θ, D_k)
```

Under Byzantine poisoning, malicious client k' sends:

```
    Δ_{k'} = Δ_{honest} + γ · poison_direction
```

**FedAvg Vulnerability:**
Since FedAvg computes a weighted average, even one malicious client with
large γ can bias the global model significantly.

**FedMedian Robustness (Yin et al., 2018):**
FedMedian uses coordinate-wise median instead of mean:

```
    θ_global ← median_{k=1..K} {θ_k}
```

The median is provably robust when fewer than 50% of clients are malicious.
Specifically, FedMedian has **breakdown point 0.5** (maximum fraction of
corrupted clients beyond which the estimator fails).

**Our Contribution in FL:**
We show that under Byzantine poisoning, our min-max client-side defense
combined with FedMedian aggregation achieves the strongest robustness
among all evaluated methods (Table 2 in paper).

---

## 8. Threat Model Summary

| Property | This Work |
|---|---|
| Attacker type | White-box (knows model architecture) |
| Attacker knowledge | Knows training data distribution; no access to defender's weights during training |
| Attack target | Training phase (data poisoning); not test-time evasion |
| Attack capability | Can corrupt up to ε fraction of training samples |
| Attacker strategy | Label flip (static) + gradient-based feature perturbation |
| Defender knowledge | Knows data is potentially poisoned; does not know which samples |
| Defender strategy | Alternating min-max training + filtering (Spectral/SEVER) |
| Federated setting | 1 of 5 clients is Byzantine |

---

## References for Theory Section

- Madry et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR.
- Lin et al. (2020). Gradient Descent Ascent for Minimax Optimization Problems. ICML.
- Brückner & Scheffer (2011). Stackelberg Games for Adversarial Prediction. KDD.
- Glicksberg (1952). A further generalization of the Kakutani fixed point theorem. Proc. AMS.
- Yin et al. (2018). Byzantine-Robust Distributed Learning. ICML.
- Goodfellow et al. (2014). Generative Adversarial Nets. NeurIPS.
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks. AISTATS.
