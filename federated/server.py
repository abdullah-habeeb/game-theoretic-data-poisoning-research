"""
federated/server.py
====================
Federated Learning Server with FedAvg Aggregation.

PAPER: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
  McMahan et al., AISTATS 2017
  https://arxiv.org/abs/1602.05629

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FEDAVG ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FedAvg aggregates client updates by computing a WEIGHTED AVERAGE
of local model parameters, where weights are proportional to
each client's dataset size:

    w_global ← Σ_k (n_k / N) * w_k

Where:
  w_k   = local model weights of client k after local training
  n_k   = number of local training samples at client k
  N     = total number of samples across all participating clients

WHY WEIGHTED AVERAGE?
  Clients with more data should contribute more to the global model.
  Unweighted average would give a malicious client with few samples
  equal influence to a large honest client — which favors the attacker
  in naive setups.

BYZANTINE ATTACK IN FL:
  A malicious client sends a perturbed update:
    w_malicious = w_benign + α * poison_direction
  When α is large, this drags the global model toward the attacker's objective.
  FedAvg provides NO protection against Byzantine clients by default.
  This is what our min-max defender is designed to address.

AGGREGATION VARIANTS (implemented):
  1. FedAvg: weighted average (standard)
  2. FedMedian: coordinate-wise median (Byzantine-robust baseline)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class FLServer:
    """
    Central server that orchestrates federated training.

    Maintains the global model and aggregates client updates
    via FedAvg or FedMedian.

    Args:
        global_model: The initial global model.
        device:       Server compute device.
        aggregation:  'fedavg' or 'fedmedian'.
    """

    def __init__(
        self,
        global_model: nn.Module,
        device: torch.device,
        aggregation: str = "fedavg",
    ):
        self.global_model = copy.deepcopy(global_model)
        self.global_model.to(device)
        self.device      = device
        self.aggregation = aggregation
        self.round_num   = 0
        self.history: List[Dict] = []

    def broadcast(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the current global model's state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        client_weights: List[int],
    ) -> None:
        """
        Aggregate client model updates into the global model.

        Args:
            client_states:  List of state_dicts from participating clients.
            client_weights: List of sample counts (n_k for each client k).
        """
        self.round_num += 1
        total_weight = sum(client_weights)

        if self.aggregation == "fedavg":
            self._fedavg(client_states, client_weights, total_weight)
        elif self.aggregation == "fedmedian":
            self._fedmedian(client_states)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _fedavg(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        client_weights: List[int],
        total_weight: int,
    ) -> None:
        """
        Standard FedAvg: weighted mean of client parameters.

        w_global ← Σ_k (n_k / N) * w_k
        """
        global_state = self.global_model.state_dict()

        for key in global_state.keys():
            if not global_state[key].is_floating_point():
                # Integer buffers (e.g. running_mean count): use first client's value
                global_state[key] = client_states[0][key]
                continue

            # Weighted sum
            weighted_sum = torch.zeros_like(global_state[key], dtype=torch.float32)
            for state, weight in zip(client_states, client_weights):
                weighted_sum += (weight / total_weight) * state[key].float()
            global_state[key] = weighted_sum.to(global_state[key].dtype)

        self.global_model.load_state_dict(global_state)

    def _fedmedian(
        self,
        client_states: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        FedMedian: coordinate-wise median (Yin et al., 2018).

        More robust to Byzantine attackers than FedAvg because
        a single malicious update cannot move the median arbitrarily.
        However, it discards more information from honest clients.

        PAPER: "Byzantine-Robust Distributed Learning: Towards Optimal
                Statistical Rates" — Yin et al., ICML 2018
        """
        global_state = self.global_model.state_dict()

        for key in global_state.keys():
            if not global_state[key].is_floating_point():
                global_state[key] = client_states[0][key]
                continue

            # Stack all client tensors → [n_clients, *param_shape]
            stacked = torch.stack([s[key].float() for s in client_states], dim=0)
            # Coordinate-wise median
            global_state[key] = stacked.median(dim=0).values.to(global_state[key].dtype)

        self.global_model.load_state_dict(global_state)

    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> float:
        """Evaluate global model on the clean test set."""
        from train.evaluator import evaluate
        return evaluate(self.global_model, test_loader, self.device)

    def get_global_model(self) -> nn.Module:
        """Return a copy of the current global model."""
        return copy.deepcopy(self.global_model)
