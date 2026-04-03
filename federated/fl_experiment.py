"""
federated/fl_experiment.py
===========================
FL poisoning experiment: poison, federate, defend, compare.

EXPERIMENT SETTINGS:
  - n_clients:       Total FL clients (e.g. 5).
  - n_malicious:     Number of malicious clients (e.g. 1 = 20% malicious).
  - fl_rounds:       Number of FL communication rounds.
  - local_epochs:    Local training epochs per client per round.
  - aggregation:     'fedavg' or 'fedmedian'.

WHAT WE MEASURE:
  1. Clean FL (no attack): all clients honest → baseline FL accuracy.
  2. Poisoned FL (FedAvg): malicious client(s) poison local shard
     → global model accuracy (attack effectiveness).
  3. Defended FL (FedMedian): same poison but robust aggregation
     → global model accuracy (defense effectiveness).
  4. Poisoned FL + Min-Max client: malicious client uses gradient-based
     poisoning instead of naive label flip → stronger attack.

This gives a 2×2 matrix in your paper:
  | Attack Strength \ Defense | FedAvg | FedMedian |
  |---------------------------|--------|-----------|
  | Label Flip                |  [A]   |    [B]    |
  | Gradient Poison           |  [C]   |    [D]    |
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from federated.client import FLClient
from federated.server import FLServer
from attacks.label_flip import poison_dataset
from data.dataset import get_raw_train_dataset
from torchvision import datasets, transforms
from utils.seed import set_seed
from utils.metrics import summarize_runs, print_summary


def partition_dataset(dataset, n_clients: int, seed: int = 42):
    """
    Partition a dataset into n_clients shards (IID split).

    IID (Independent and Identically Distributed) means each client
    receives roughly the same class distribution. This is the standard
    FL setup before studying non-IID effects.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    shard_size = n // n_clients
    sizes = [shard_size] * n_clients
    sizes[-1] += n - sum(sizes)   # Give remainder to last client
    subsets = random_split(dataset, sizes, generator=g)
    return subsets


def run_fl_experiment(
    n_clients: int = 5,
    n_malicious: int = 1,
    fl_rounds: int = 10,
    local_epochs: int = 3,
    aggregation: str = "fedavg",
    dataset: str = "cifar10",
    src_class: int = 1,
    tgt_class: int = 7,
    poison_fraction: float = 0.5,
    epochs_final_eval: int = 0,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run one complete FL poisoning experiment.

    Args:
        n_clients:     Total number of FL clients.
        n_malicious:   Number of clients that apply poisoning.
        fl_rounds:     FL communication rounds.
        local_epochs:  Local training epochs per client per round.
        aggregation:   FedAvg or FedMedian.
        dataset:       'cifar10' or 'mnist'.
        src_class:     Poisoning source class.
        tgt_class:     Poisoning target class.
        poison_fraction: Fraction of malicious client's src_class to poison.
        batch_size:    Local batch size.
        lr:            Local learning rate.
        seed:          Random seed.
        verbose:       Print progress.

    Returns:
        dict with 'round_accs', 'final_acc', 'n_malicious', 'aggregation'.
    """
    from models.resnet import get_model
    from train.evaluator import evaluate

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  FEDERATED LEARNING EXPERIMENT")
        print(f"  Clients: {n_clients} total, {n_malicious} malicious")
        print(f"  Aggregation: {aggregation}  |  FL Rounds: {fl_rounds}")
        print(f"  Attack: {src_class}→{tgt_class}, fraction={poison_fraction:.0%}")
        print(f"  Device: {device}")
        print(f"{'='*65}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    raw_train = get_raw_train_dataset(dataset=dataset, augment=False)

    # Test loader (clean, shared)
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465) if dataset=="cifar10" else (0.1307,),
            (0.2023,0.1994,0.2010) if dataset=="cifar10" else (0.3081,),
        )
    ])
    if dataset == "cifar10":
        test_ds = datasets.CIFAR10("./data/raw", train=False, download=True, transform=test_tf)
    else:
        test_ds = datasets.MNIST("./data/raw", train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # ── Partition data into client shards ─────────────────────────────────────
    shards = partition_dataset(raw_train, n_clients, seed=seed)

    # ── Build clients ─────────────────────────────────────────────────────────
    clients = []
    for cid in range(n_clients):
        is_malicious = (cid < n_malicious)

        if is_malicious:
            # Malicious client: poison its local shard
            poisoned_shard, _ = poison_dataset(
                shards[cid].dataset,   # Underlying dataset
                src_class=src_class,
                tgt_class=tgt_class,
                poison_fraction=poison_fraction,
                seed=seed + cid,
            )
            # We need to respect the subset indices
            # Build a poisoned version indexing the shard's subset
            local_ds = _poison_subset(shards[cid], src_class, tgt_class,
                                      poison_fraction, seed + cid)
            if verbose:
                print(f"  Client {cid}: MALICIOUS (poison fraction={poison_fraction:.0%})")
        else:
            local_ds = shards[cid]
            if verbose:
                print(f"  Client {cid}: honest ({len(shards[cid])} samples)")

        client = FLClient(
            client_id=cid,
            dataset=local_ds,
            device=device,
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
            is_malicious=is_malicious,
        )
        clients.append(client)

    # ── Initialize server ─────────────────────────────────────────────────────
    init_model = get_model(device, dataset)
    server = FLServer(init_model, device, aggregation=aggregation)

    # ── FL Training Loop ──────────────────────────────────────────────────────
    round_accs = []
    for fl_round in range(1, fl_rounds + 1):
        # Broadcast global model to all clients
        global_state = server.broadcast()

        # Each client trains locally
        client_states  = []
        client_weights = []
        for client in clients:
            client.set_model(server.get_global_model())
            local_state = client.local_train()
            client_states.append(local_state)
            client_weights.append(client.n_samples)

        # Server aggregates
        server.aggregate(client_states, client_weights)

        # Evaluate global model
        acc = server.evaluate(test_loader)
        round_accs.append(acc)

        if verbose:
            print(f"  [FL Round {fl_round:2d}/{fl_rounds}]  "
                  f"Global acc: {acc:.2f}%  (agg={aggregation})")

    final_acc = round_accs[-1]
    if verbose:
        print(f"\n  Final FL accuracy ({aggregation}, {n_malicious}/{n_clients} malicious): "
              f"{final_acc:.2f}%")

    return {
        "round_accs":  round_accs,
        "final_acc":   final_acc,
        "n_malicious": n_malicious,
        "n_clients":   n_clients,
        "aggregation": aggregation,
    }


def _poison_subset(subset, src_class, tgt_class, fraction, seed):
    """
    Apply label-flip to a Subset, respecting its index mapping.
    Returns a new dataset with poisoned labels embedded.
    """
    from attacks.label_flip import PoisonedDataset
    import numpy as np

    underlying = subset.dataset
    indices    = subset.indices

    # Get labels for this subset
    if hasattr(underlying, "targets"):
        import torch as _t
        labels_all = _t.tensor(underlying.targets)
    else:
        labels_all = _t.tensor([underlying[i][1] for i in range(len(underlying))])

    local_labels = labels_all[indices]
    src_local    = (local_labels == src_class).nonzero(as_tuple=True)[0].numpy()

    n_poison = int(len(src_local) * fraction)
    if n_poison == 0:
        return subset

    rng = np.random.default_rng(seed)
    chosen_local = rng.choice(src_local, size=n_poison, replace=False)
    global_chosen = np.array(indices)[chosen_local]

    poisoned_ds = PoisonedDataset(underlying, global_chosen, tgt_class)

    # Wrap back to the same subset indices
    return Subset(poisoned_ds, indices)


def run_fl_full_comparison(
    n_runs: int = 3,
    seeds: list = None,
    n_clients: int = 5,
    n_malicious: int = 1,
    fl_rounds: int = 10,
    local_epochs: int = 3,
    dataset: str = "cifar10",
    src_class: int = 1,
    tgt_class: int = 7,
    poison_fraction: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Run the full FL comparison matrix:
      - Clean FL (FedAvg)
      - Poisoned FL (FedAvg)
      - Poisoned FL (FedMedian)

    Each condition run n_runs times for mean/std.
    """
    import pandas as pd
    from utils.plotting import plot_comparison

    if seeds is None:
        seeds = list(range(n_runs))

    results = {}

    for condition, agg, n_mal in [
        ("Clean FL (FedAvg)",      "fedavg",   0),
        ("Poisoned FL (FedAvg)",   "fedavg",   n_malicious),
        ("Poisoned FL (FedMedian)","fedmedian", n_malicious),
    ]:
        print(f"\n{'═'*65}")
        print(f"  CONDITION: {condition}")
        print(f"{'═'*65}")
        accs = []
        for run_i, seed in enumerate(seeds):
            res = run_fl_experiment(
                n_clients=n_clients,
                n_malicious=n_mal,
                fl_rounds=fl_rounds,
                local_epochs=local_epochs,
                aggregation=agg,
                dataset=dataset,
                src_class=src_class,
                tgt_class=tgt_class,
                poison_fraction=poison_fraction,
                seed=seed,
                verbose=(run_i == 0 and verbose),  # Only verbose for first run
            )
            accs.append(res["final_acc"])
            print(f"  Run {run_i+1}/{n_runs}: {res['final_acc']:.2f}%")
        summary = summarize_runs(accs)
        results[condition] = summary
        print_summary(condition, summary)

    # Save table
    os.makedirs("results/tables", exist_ok=True)
    rows = [{"Condition": k, "Mean (%)": round(v["mean"],2),
             "Std (%)": round(v["std"],2)} for k,v in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/fl_comparison.csv", index=False)
    print(f"\n[FL Table saved] results/tables/fl_comparison.csv")
    print(df.to_string(index=False))

    # Plot
    plot_comparison(
        labels=list(results.keys()),
        means=[v["mean"] for v in results.values()],
        stds=[v["std"] for v in results.values()],
        save_name="fl_comparison.png",
        title="FL: Clean vs Poisoned (FedAvg) vs Robust (FedMedian)",
    )

    return results


if __name__ == "__main__":
    run_fl_full_comparison(n_runs=3, fl_rounds=10, local_epochs=3)
