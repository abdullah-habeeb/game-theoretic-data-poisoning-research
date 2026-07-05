"""Phase-3 comprehensive import and forward-pass verification."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 58)
print("  PHASE-3 VERIFICATION")
print("=" * 58)

# ── Threat Model ─────────────────────────────────────────────
from utils.threat_model import (
    get_standard_threat_model, EPSILON_PRIMARY, N_SEEDS,
    random_asr_baseline
)
tm = get_standard_threat_model("cifar10")
assert EPSILON_PRIMARY == 0.05, f"ε must be 0.05, got {EPSILON_PRIMARY}"
assert N_SEEDS == 5, f"n_seeds must be 5, got {N_SEEDS}"
assert abs(random_asr_baseline(10) - 0.10) < 1e-6
print("  [OK] utils.threat_model  ε=0.05  n_seeds=5  random_asr=10%")

# ── Poison Detection Metrics ─────────────────────────────────
from utils.poison_detection_metrics import poison_detection_metrics, multi_defense_detection_report
# 10 actual poison (0-9). Flagged: 0-4 (5 true), 10-11 (2 false)
m = poison_detection_metrics([0,1,2,3,4, 10,11], list(range(10)), 100, verbose=False)
assert m["tp"] == 5 and m["fp"] == 2 and m["fn"] == 5
print(f"  [OK] utils.poison_detection_metrics  F1={m['f1']:.3f}")

# ── Confusion Training ────────────────────────────────────────
from defenses.confusion_training import confusion_training
print("  [OK] defenses.confusion_training (imported)")

# ── Tiny ViT ─────────────────────────────────────────────────
import torch
from models.tiny_vit import get_tiny_vit
device = torch.device("cpu")
vit = get_tiny_vit(device, "cifar10")
dummy3 = torch.zeros(2, 3, 32, 32)
dummy1 = torch.zeros(2, 1, 28, 28)
assert vit(dummy3).shape == (2, 10)
feats = vit.get_features(dummy3)
assert feats.shape == (2, 192)
vit_m = get_tiny_vit(device, "mnist")
assert vit_m(dummy1).shape == (2, 10)
vit_g = get_tiny_vit(device, "gtsrb")
assert vit_g(dummy3).shape == (2, 43)
print(f"  [OK] models.tiny_vit  params={vit.count_parameters():,}  "
      f"get_features shape={feats.shape}")

# ── Repeated Game Analysis ────────────────────────────────────
from theory.repeated_game_analysis import verify_best_response, print_game_framing
mock = [{"round": r, "mean_acc": 88+r*1.5, "mean_asr": 45-r*4}
        for r in range(1, 6)]
res = verify_best_response(mock, verbose=False)
assert isinstance(res["verified"], bool)
print(f"  [OK] theory.repeated_game_analysis  converged={res['verified']}")

# ── ASR Baseline in Metrics ───────────────────────────────────
from utils.asr import compute_full_attack_metrics
from torch.utils.data import TensorDataset, DataLoader
class _Uniform(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 10)

imgs  = torch.zeros(20, 1, 28, 28)
lbls  = torch.tensor([1]*10 + [0]*10)
loader = DataLoader(TensorDataset(imgs, lbls), batch_size=20)
m2 = compute_full_attack_metrics(_Uniform(), loader, 1, 7, 10, device)
assert "random_asr_baseline" in m2, "Missing random_asr_baseline in metrics!"
assert abs(m2["random_asr_baseline"] - 10.0) < 1e-3
print(f"  [OK] utils.asr random_asr_baseline={m2['random_asr_baseline']:.1f}%  "
      f"above_chance={m2['asr_above_chance']:.1f}pp")

# ── Stackelberg defaults ─────────────────────────────────────
import inspect
from experiments.stackelberg_game import run_stackelberg_game
sig = inspect.signature(run_stackelberg_game)
assert sig.parameters["poison_fraction"].default == 0.05, "ε default not fixed!"
assert sig.parameters["n_runs"].default == 5, "n_runs default not fixed!"
assert "warm_start" in sig.parameters, "warm_start param missing!"
print(f"  [OK] stackelberg_game defaults: ε={sig.parameters['poison_fraction'].default}  "
      f"n_runs={sig.parameters['n_runs'].default}  warm_start={sig.parameters['warm_start'].default}")

print()
print("=" * 58)
print("  ALL PHASE-3 CHECKS PASSED ✓")
print("=" * 58)
