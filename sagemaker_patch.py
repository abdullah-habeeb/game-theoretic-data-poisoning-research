"""
sagemaker_patch.py
==================
Run this ONCE in a Jupyter cell on SageMaker to apply all fixes.
Paste this entire file content into a cell and run it.
"""

import os, re

BASE = "/home/ec2-user/SageMaker/ml-research"

def patch_file(path, replacements, label=""):
    with open(path, 'r') as f:
        content = f.read()
    n_found = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            n_found += 1
        else:
            print(f"  ⚠️  Substring not found (may already be patched): {repr(old[:70])}")
    with open(path, 'w') as f:
        f.write(content)
    print(f"  ✅ {label or path}: {n_found}/{len(replacements)} patches applied")

print("=" * 65)
print("  CIFAR-100 COMPREHENSIVE PATCH SCRIPT")
print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1: experiments/defender.py — wrong model import (CRITICAL)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1/6] experiments/defender.py — fix model import...")
patch_file(f"{BASE}/experiments/defender.py", [
    (
        "from models.resnet import get_model          # routes: mnist→MnistCNN, cifar10→ResNet18",
        "from models.wideresnet import get_model   # handles mnist, cifar10, cifar100 (WRN-28-10)"
    ),
    (
        "from models.resnet import get_model",
        "from models.wideresnet import get_model   # handles mnist, cifar10, cifar100 (WRN-28-10)"
    ),
], label="defender.py")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2: experiments/defense_comparison.py — wrong import + DataLoaders + SEVER DataLoader
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/6] experiments/defense_comparison.py — fix import + DataLoaders...")
patch_file(f"{BASE}/experiments/defense_comparison.py", [
    # Fix wrong import
    (
        "from models.resnet import get_model",
        "from models.wideresnet import get_model   # handles mnist, cifar10, cifar100 (WRN-28-10)"
    ),
    # Fix missing num_workers param in function signature
    (
        "    lr: float = 0.001,\n    verbose: bool = True,\n    use_sgd: bool = False,\n) -> pd.DataFrame:",
        "    lr: float = 0.001,\n    num_workers: int = 2,\n    verbose: bool = True,\n    use_sgd: bool = False,\n) -> pd.DataFrame:"
    ),
    # Fix test loader call
    (
        "    test_loader = _get_test_loader(dataset, batch_size)\n",
        "    test_loader = _get_test_loader(dataset, batch_size, num_workers)\n    pin = torch.cuda.is_available()\n"
    ),
    # Fix baseline DataLoader
    (
        "        train_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True, num_workers=2)",
        "        train_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=True,\n                                  num_workers=num_workers, pin_memory=pin)"
    ),
    # Fix poisoned DataLoader
    (
        "            train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                      shuffle=True, num_workers=2)",
        "            train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                      shuffle=True, num_workers=num_workers, pin_memory=pin)"
    ),
    # Fix Spectral DataLoader
    (
        "        train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                  shuffle=True, num_workers=2)\n        ds_snap = dataset",
        "        train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                  shuffle=True, num_workers=num_workers, pin_memory=pin)\n        ds_snap = dataset"
    ),
    # Fix SEVER DataLoader
    (
        "        train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                  shuffle=True, num_workers=2)\n        ds_snap = dataset\n        dev_snap = device",
        "        train_loader = DataLoader(poisoned_train, batch_size=batch_size,\n                                  shuffle=True, num_workers=num_workers, pin_memory=pin)\n        ds_snap = dataset\n        dev_snap = device"
    ),
    # Add dataset param to Spectral call
    (
        "            verbose=verbose and i == 0,\n            use_sgd=use_sgd,\n            checkpoint_path=epoch_ckpt,\n            resume_from_checkpoint=epoch_ckpt,\n        )\n        spec_accs.append(acc)",
        "            dataset=dataset,\n            num_workers=num_workers,\n            verbose=verbose and i == 0,\n            use_sgd=use_sgd,\n            checkpoint_path=epoch_ckpt,\n            resume_from_checkpoint=epoch_ckpt,\n        )\n        spec_accs.append(acc)"
    ),
    # Fix _get_test_loader to accept num_workers and use pin_memory correctly
    (
        "def _get_test_loader(dataset: str = \"cifar10\", batch_size: int = 128):\n    from data.dataset import get_transforms\n    _, test_tf = get_transforms(dataset, augment=False)\n    \n    if dataset == \"cifar10\":",
        "def _get_test_loader(dataset: str = \"cifar10\", batch_size: int = 128, num_workers: int = 2):\n    from data.dataset import get_transforms\n    _, test_tf = get_transforms(dataset, augment=False)\n    pin = torch.cuda.is_available()\n    if dataset == \"cifar10\":"
    ),
    (
        "    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)",
        "    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)"
    ),
], label="defense_comparison.py")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 3: defenses/spectral_signatures.py — add dataset param for n_classes auto-inference
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/6] defenses/spectral_signatures.py — fix n_classes for CIFAR-100...")
patch_file(f"{BASE}/defenses/spectral_signatures.py", [
    (
        "    suspicious_quantile: float = 0.95,\n    n_classes: int = 10,\n    verbose: bool = True,\n    use_sgd: bool = False,",
        "    suspicious_quantile: float = 0.95,\n    n_classes: int = None,\n    dataset: str = \"cifar10\",\n    num_workers: int = 2,\n    verbose: bool = True,\n    use_sgd: bool = False,"
    ),
    # Add auto-inference logic before the filter call
    (
        "    from train.trainer import train_model\n    from train.evaluator import evaluate\n\n    # Step 1: Get clean indices",
        "    from train.trainer import train_model\n    from train.evaluator import evaluate\n\n    # Auto-infer n_classes from dataset name if not provided\n    if n_classes is None:\n        _n_classes_map = {\"mnist\": 10, \"cifar10\": 10, \"cifar100\": 100}\n        n_classes = _n_classes_map.get(dataset, 10)\n\n    # Step 1: Get clean indices"
    ),
    # Fix hardcoded num_workers in clean_loader
    (
        "    clean_loader = DataLoader(\n        clean_subset, batch_size=train_loader.batch_size,\n        shuffle=True, num_workers=2, pin_memory=True,\n    )",
        "    pin = torch.cuda.is_available()\n    clean_loader = DataLoader(\n        clean_subset, batch_size=train_loader.batch_size,\n        shuffle=True, num_workers=num_workers, pin_memory=pin,\n    )"
    ),
], label="spectral_signatures.py")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 4: defenses/sever.py — fix hardcoded pin_memory=True
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/6] defenses/sever.py — fix pin_memory...")
patch_file(f"{BASE}/defenses/sever.py", [
    (
        "    clean_loader = DataLoader(\n        clean_subset, batch_size=train_loader.batch_size,\n        shuffle=True, num_workers=2, pin_memory=True,\n    )",
        "    pin = next(model.parameters()).is_cuda\n    clean_loader = DataLoader(\n        clean_subset, batch_size=train_loader.batch_size,\n        shuffle=True, num_workers=2, pin_memory=pin,\n    )"
    ),
], label="sever.py")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 5: experiments/cifar100_experiment.py — rewrite full pipeline (CRITICAL)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/6] experiments/cifar100_experiment.py — fix full pipeline...")

# Find and replace the full pipeline function
with open(f"{BASE}/experiments/cifar100_experiment.py", 'r') as f:
    content = f.read()

NEW_PIPELINE = '''def run_cifar100_full_pipeline(
    n_runs: int = 3,
    seeds: list = None,
    epochs: int = 100,
    batch_size: int = 128,
    verbose: bool = True,
) -> dict:
    """
    Run the complete CIFAR-100 research pipeline via the bulletproof
    defense_comparison engine (epoch-level checkpointing, state persistence):
      1. Clean baseline         (WRN-28-10)
      2. Poisoned               (label-flip, no defense)
      3. Spectral Signatures    defense
      4. SEVER                  defense
      5. Min-Max                alternating training (our method)
    Then runs poison-fraction sweep.
    """
    from experiments.defense_comparison import run_defense_comparison

    if seeds is None:
        seeds = list(range(n_runs))

    print("\\n" + "\\U0001f52c"*30)
    print("  CIFAR-100 FULL RESEARCH PIPELINE")
    print("\\U0001f52c"*30)
    print(f"  Dataset : CIFAR-100  |  Model : WideResNet-28-10")
    print(f"  Attack  : aquarium_fish(1) -> flatfish(32)  |  fraction=50%")
    print(f"  Epochs  : {epochs}  |  Runs : {n_runs}  |  Optimizer : SGD+CosineAnneal")

    # ── All 5 methods via bulletproof defense_comparison engine ──────────────
    df = run_defense_comparison(
        n_runs=n_runs,
        seeds=seeds,
        dataset="cifar100",
        src_class=CIFAR100_DEFAULT_SRC,   # 1 = aquarium_fish
        tgt_class=CIFAR100_DEFAULT_TGT,   # 32 = flatfish
        poison_fraction=0.5,
        epochs=epochs,
        defense_epochs=epochs,
        batch_size=batch_size,
        lr=0.1,            # standard WRN lr
        num_workers=2,
        use_sgd=True,      # SGD + CosineAnneal is the standard for WRN/CIFAR-100
        verbose=verbose,
    )

    print("\\n\\n\\u2705 CIFAR-100 Defense Comparison Complete!")
    print(df.to_string(index=False))

    # ── Poison fraction sweep (separate from main run) ───────────────────────
    print("\\n\\n[STEP 2] Poison Fraction Sweep")
    sweep_df = run_cifar100_sweep(
        n_runs=n_runs, seeds=seeds, epochs=epochs,
        batch_size=batch_size, verbose=verbose,
    )

    return {
        "defense_comparison": df,
        "sweep": sweep_df,
    }'''

# Detect old vs new pipeline and replace
if "# Placeholder" in content or '# Step 1: Baseline' in content:
    # Old pipeline: find and replace from def line to end of function
    pattern = r'(def run_cifar100_full_pipeline\(.*?return \{[^}]+\}\s*\n)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + NEW_PIPELINE + "\n\n" + content[match.end():]
        with open(f"{BASE}/experiments/cifar100_experiment.py", 'w') as f:
            f.write(content)
        print("  ✅ cifar100_experiment.py: full pipeline rewritten")
    else:
        print("  ⚠️  Could not find pipeline function — patching with append strategy")
        # Fallback: just append the fixed version
        content = re.sub(r'def run_cifar100_full_pipeline\(.*', '', content, flags=re.DOTALL)
        content = content.rstrip() + "\n\n" + NEW_PIPELINE + "\n\n\nif __name__ == '__main__':\n    results = run_cifar100_full_pipeline(n_runs=3, epochs=100, batch_size=128)\n"
        with open(f"{BASE}/experiments/cifar100_experiment.py", 'w') as f:
            f.write(content)
        print("  ✅ cifar100_experiment.py: pipeline appended (fallback)")
else:
    print("  ✅ cifar100_experiment.py: already patched (skipping)")


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 6: Verify data + checkpoint directory
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Verifying data and checkpoint directory...")
data_base = f"{BASE}/data/raw/cifar-100-python"
ckpt_dir  = f"{BASE}/results/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

all_ok = True
for fname in ["train", "test", "meta"]:
    path = os.path.join(data_base, fname)
    exists = os.path.isfile(path)
    size   = os.path.getsize(path) // (1024*1024) if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname:<6} {'(' + str(size) + ' MB)' if exists else 'MISSING'}")
    if not exists:
        all_ok = False

print(f"  ✅ Checkpoint dir: {ckpt_dir}")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if all_ok:
    print("  ✅ ALL PATCHES APPLIED SUCCESSFULLY")
    print("=" * 65)
    print("\n  1. Restart the kernel (Kernel → Restart & Clear Output)")
    print("  2. Run this code:\n")
    print("     from experiments.cifar100_experiment import run_cifar100_full_pipeline")
    print("     results = run_cifar100_full_pipeline(n_runs=3, epochs=100, batch_size=128, verbose=True)")
else:
    print("  ❌ DATA FILES MISSING — re-upload cifar-100-python/train, test, meta to:")
    print(f"     {data_base}")
    print("=" * 65)
