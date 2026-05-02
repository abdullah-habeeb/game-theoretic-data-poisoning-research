import os, json, torch, pandas as pd
from train.evaluator import evaluate
from data.dataset import get_raw_train_dataset
from attacks.label_flip import poison_dataset
from models.resnet import get_model
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = f"{SAVE_DIR}/cifar10_ckpt"
os.makedirs(CKPT, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_RUNS          = 3
SEEDS           = [0, 1, 2]
EPOCHS          = 50
DEFENSE_EPOCHS  = 50
LR              = 0.1
BATCH_SIZE      = 128
SRC_CLASS       = 1
TGT_CLASS       = 7
POISON_FRAC     = 0.5
USE_SGD         = True
DATASET         = "cifar10"
N_MM_ROUNDS     = 5   # min-max rounds

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def ck_load(name):
    p = f"{CKPT}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None

def ck_save(name, data):
    with open(f"{CKPT}/{name}.json", "w") as f:
        json.dump(data, f)

def save_model(tag, model):
    torch.save(model.state_dict(), f"{CKPT}/{tag}.pth")

def load_model(tag):
    p = f"{CKPT}/{tag}.pth"
    if not os.path.exists(p):
        return None
    m = get_model(DEVICE, DATASET)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    return m

# ── Test loader ───────────────────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_loader = DataLoader(
    datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
)

all_results = {}

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CLEAN BASELINE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60 + "\n  [1/5] CLEAN BASELINE\n" + "═"*60)
ck = ck_load("baseline")
baseline_accs = ck["accs"] if ck else []

if len(baseline_accs) == N_RUNS:
    print("  ✅ All runs loaded from checkpoint")
    for i, a in enumerate(baseline_accs): print(f"  Run {i+1}: {a:.2f}%")
else:
    start = len(baseline_accs)
    if start > 0: print(f"  ⏩ Resuming from run {start+1}/{N_RUNS}")
    for ri in range(start, N_RUNS):
        set_seed(SEEDS[ri])
        raw = get_raw_train_dataset(dataset=DATASET, augment=True)
        loader = DataLoader(raw, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        m = get_model(DEVICE, DATASET)
        _patched_train_model(m, loader, DEVICE, epochs=EPOCHS, lr=LR, verbose=False, use_sgd=USE_SGD)
        acc = evaluate(m, test_loader, DEVICE)
        baseline_accs.append(acc)
        ck_save("baseline", {"accs": baseline_accs})
        print(f"  Run {ri+1}: {acc:.2f}%  ✅ checkpointed")

all_results["Clean Baseline"] = summarize_runs(baseline_accs)
print_summary("Clean Baseline", all_results["Clean Baseline"])

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — POISONED (No Defense)  +  save model weights for stages 3 & 4
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60 + "\n  [2/5] POISONED (No Defense)\n" + "═"*60)
ck = ck_load("poisoned")
poisoned_accs = ck["accs"] if ck else []
poisoned_models   = []
poisoned_datasets = []

def _rebuild_poisoned_dataset(ri):
    set_seed(SEEDS[ri])
    raw = get_raw_train_dataset(dataset=DATASET, augment=False)
    ds, _ = poison_dataset(raw, SRC_CLASS, TGT_CLASS, POISON_FRAC, seed=SEEDS[ri])
    return ds

# Load already-completed runs (models + datasets)
for ri in range(len(poisoned_accs)):
    ds = _rebuild_poisoned_dataset(ri)
    poisoned_datasets.append(ds)
    m = load_model(f"poisoned_{ri}")
    if m is None:
        # Model file missing — retrain silently to get model object in memory
        ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        m = get_model(DEVICE, DATASET)
        _patched_train_model(m, ldr, DEVICE, epochs=EPOCHS, lr=LR, verbose=False, use_sgd=USE_SGD)
        save_model(f"poisoned_{ri}", m)
    poisoned_models.append(m)

if len(poisoned_accs) == N_RUNS:
    print("  ✅ All runs loaded from checkpoint")
    for i, a in enumerate(poisoned_accs): print(f"  Run {i+1}: {a:.2f}%")
else:
    start = len(poisoned_accs)
    if start > 0: print(f"  ⏩ Resuming from run {start+1}/{N_RUNS}")
    for ri in range(start, N_RUNS):
        ds = _rebuild_poisoned_dataset(ri)
        poisoned_datasets.append(ds)
        ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        m = get_model(DEVICE, DATASET)
        _patched_train_model(m, ldr, DEVICE, epochs=EPOCHS, lr=LR, verbose=False, use_sgd=USE_SGD)
        acc = evaluate(m, test_loader, DEVICE)
        poisoned_accs.append(acc)
        poisoned_models.append(m)
        save_model(f"poisoned_{ri}", m)
        ck_save("poisoned", {"accs": poisoned_accs})
        print(f"  Run {ri+1}: {acc:.2f}%  ✅ checkpointed (model saved)")

all_results["Poisoned (No Defense)"] = summarize_runs(poisoned_accs)
print_summary("Poisoned (No Defense)", all_results["Poisoned (No Defense)"])

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SPECTRAL SIGNATURES
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60 + "\n  [3/5] SPECTRAL SIGNATURES\n" + "═"*60)
ck = ck_load("spectral")
spec_accs = ck["accs"] if ck else []

if len(spec_accs) == N_RUNS:
    print("  ✅ All runs loaded from checkpoint")
    for i, a in enumerate(spec_accs): print(f"  Run {i+1}: {a:.2f}%")
else:
    start = len(spec_accs)
    if start > 0: print(f"  ⏩ Resuming from run {start+1}/{N_RUNS}")
    for ri in range(start, N_RUNS):
        set_seed(SEEDS[ri])
        dv, ds = DEVICE, DATASET
        _, acc, _ = _patched_spectral(
            model=poisoned_models[ri],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_datasets[ri],
            train_loader=DataLoader(poisoned_datasets[ri], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            test_loader=test_loader, device=DEVICE,
            defender_epochs=DEFENSE_EPOCHS, defender_lr=LR,
            verbose=False, use_sgd=USE_SGD,
        )
        spec_accs.append(acc)
        ck_save("spectral", {"accs": spec_accs})
        print(f"  Run {ri+1}: {acc:.2f}%  ✅ checkpointed")

all_results["Spectral Signatures"] = summarize_runs(spec_accs)
print_summary("Spectral Signatures", all_results["Spectral Signatures"])

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SEVER
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60 + "\n  [4/5] SEVER\n" + "═"*60)
ck = ck_load("sever")
sever_accs = ck["accs"] if ck else []

if len(sever_accs) == N_RUNS:
    print("  ✅ All runs loaded from checkpoint")
    for i, a in enumerate(sever_accs): print(f"  Run {i+1}: {a:.2f}%")
else:
    start = len(sever_accs)
    if start > 0: print(f"  ⏩ Resuming from run {start+1}/{N_RUNS}")
    for ri in range(start, N_RUNS):
        set_seed(SEEDS[ri])
        dv, ds = DEVICE, DATASET
        _, acc, _ = _patched_sever(
            model=poisoned_models[ri],
            model_fn=lambda dv=dv, ds=ds: get_model(dv, ds),
            train_dataset=poisoned_datasets[ri],
            train_loader=DataLoader(poisoned_datasets[ri], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            test_loader=test_loader, device=DEVICE,
            defender_epochs=DEFENSE_EPOCHS, defender_lr=LR,
            verbose=False, use_sgd=USE_SGD,
        )
        sever_accs.append(acc)
        ck_save("sever", {"accs": sever_accs})
        print(f"  Run {ri+1}: {acc:.2f}%  ✅ checkpointed")

all_results["SEVER"] = summarize_runs(sever_accs)
print_summary("SEVER", all_results["SEVER"])

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — MIN-MAX  (checkpointed per round)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60 + "\n  [5/5] OURS: Min-Max\n" + "═"*60)
ck = ck_load("minmax")
mm_round_means    = ck["round_means"] if ck else []
completed_rounds  = len(mm_round_means)

if completed_rounds == N_MM_ROUNDS:
    print("  ✅ All rounds loaded from checkpoint")
    for i, a in enumerate(mm_round_means): print(f"  Round {i+1}: mean={a:.2f}%")
else:
    if completed_rounds > 0:
        print(f"  ⏩ Resuming from round {completed_rounds+1}/{N_MM_ROUNDS}")

    for rnd in range(completed_rounds, N_MM_ROUNDS):
        # Run exactly ONE round — seeds mirror the internal pattern (r+1)*100 + run_idx
        rnd_seeds = [(rnd + 1) * 100 + i for i in range(N_RUNS)]
        rnd_result = _def_mod.run_minmax(
            n_rounds=1, n_runs=N_RUNS, seeds=rnd_seeds,
            dataset=DATASET, src_class=SRC_CLASS, tgt_class=TGT_CLASS,
            poison_fraction=POISON_FRAC, epochs=EPOCHS,
            batch_size=BATCH_SIZE, lr=LR,
            verbose=True, use_sgd=USE_SGD,
        )
        rnd_mean = rnd_result["round_means"][0]
        mm_round_means.append(float(rnd_mean))
        ck_save("minmax", {"round_means": mm_round_means})
        print(f"  Round {rnd+1}/{N_MM_ROUNDS}: mean={rnd_mean:.2f}%  ✅ checkpointed")

all_results["Ours (Min-Max)"] = summarize_runs(mm_round_means)
print_summary("Ours (Min-Max)", all_results["Ours (Min-Max)"])

# ═════════════════════════════════════════════════════════════════════════════
# FINAL TABLE
# ═════════════════════════════════════════════════════════════════════════════
baseline_mean = all_results["Clean Baseline"]["mean"]
rows = [
    {
        "Method":          method,
        "Mean Acc (%)":    round(s["mean"], 2),
        "Std (%)":         round(s["std"],  2),
        "vs Baseline (%)": round(s["mean"] - baseline_mean, 2),
    }
    for method, s in all_results.items()
]
df = pd.DataFrame(rows)
df.to_csv(f"{SAVE_DIR}/cifar10_df.csv", index=False)

print("\n" + "═"*60 + "\n  FINAL COMPARISON TABLE\n" + "═"*60)
print(df.to_string(index=False))
print(f"\n✅ CIFAR-10 COMPLETE!  Saved → {SAVE_DIR}/cifar10_df.csv")
print("   ⬇️  Download cifar10_df.csv from the Output panel on the right!")
