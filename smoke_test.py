"""Smoke test: verify the full Stackelberg game runs end-to-end."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.stackelberg_game import run_stackelberg_game

print("=" * 60)
print("  SMOKE TEST: True Stackelberg Game (2 rounds, 1 seed, 2 epochs)")
print("=" * 60)

results = run_stackelberg_game(
    n_rounds=2,
    n_runs=1,
    seeds=[42],
    dataset="mnist",
    src_class=1,
    tgt_class=7,
    poison_fraction=0.5,
    epochs=2,
    batch_size=128,
    lr=0.001,
    val_fraction=0.1,
    selection_mode="loss_margin",
    use_sgd=False,
    patience=2,
    verbose=True,
    smoke_test=False,  # manual control
)

print("\n" + "=" * 60)
print("  SMOKE TEST RESULTS")
print("=" * 60)
for rm in results["round_metrics"]:
    print(f"  Round {rm['round']}: "
          f"acc={rm['mean_acc']:.2f}%  "
          f"asr={rm['mean_asr']:.2f}%  "
          f"src_acc={rm['mean_src_acc']:.2f}%")

# Assertions
assert len(results["round_metrics"]) == 2, "Expected 2 rounds"
assert results["round_metrics"][0]["mean_acc"] > 0, "Accuracy should be > 0"
print("\n  All smoke test assertions passed ✓")
print("  Correct Python environment:", sys.executable)
