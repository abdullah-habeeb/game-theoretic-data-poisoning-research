"""
federated/client.py
====================
Federated Learning Client.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHY FEDERATED LEARNING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In real-world machine learning deployments, data is often distributed:
  - Hospital A trains on its patients' data.
  - Hospital B trains on its own patients' data.
  - A central server aggregates model updates without seeing raw data.

This is the "collaborative neural network" setting from our paper title.
Data poisoning in this setting is MORE dangerous because:
  - The server cannot inspect raw data (privacy constraint).
  - A malicious client can inject poisoned updates undetected.
  - The game now has multiple attackers, each controlling one shard.

FEDERATED LEARNING PROTOCOL (FedAvg, McMahan et al. 2017):
  1. Server broadcasts global model weights to all clients.
  2. Each client trains locally on its data shard.
  3. Clients send model weight updates (deltas) back to server.
  4. Server aggregates updates via weighted average.
  5. Repeat for T rounds.

POISONING IN FL:
  A malicious client poisons its local data shard. Its local model update
  then propagates (partially) into the global model via FedAvg aggregation.
  This is called a "Byzantine poisoning attack."
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, Optional


class FLClient:
    """
    A single federated learning client.

    Each client holds:
      - A local data shard (potentially poisoned).
      - A local copy of the global model.
      - A local optimizer.

    The client receives the global model, trains locally, and returns
    its updated weights (a weight delta) to the server.

    Args:
        client_id:    Unique identifier (int).
        dataset:      Local training dataset shard.
        device:       Compute device.
        lr:           Local learning rate.
        local_epochs: Number of local training epochs per FL round.
        batch_size:   Local mini-batch size.
        is_malicious: Whether this client applies a poison attack.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        device: torch.device,
        lr: float = 0.001,
        local_epochs: int = 3,
        batch_size: int = 64,
        is_malicious: bool = False,
    ):
        self.client_id     = client_id
        self.dataset       = dataset
        self.device        = device
        self.lr            = lr
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.is_malicious  = is_malicious
        self._local_model: Optional[nn.Module] = None

    def set_model(self, global_model: nn.Module) -> None:
        """
        Download: receive the global model and make a local copy for training.
        """
        self._local_model = copy.deepcopy(global_model)
        self._local_model.to(self.device)

    def local_train(self) -> Dict[str, torch.Tensor]:
        """
        Train locally on the client's data shard.

        This is the "local SGD" step in FedAvg. The client runs
        `local_epochs` passes over its local data, then returns
        its updated model state dict.

        Returns:
            state_dict of locally trained model.
        """
        if self._local_model is None:
            raise RuntimeError("Call set_model() before local_train().")

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._local_model.parameters(), lr=self.lr)

        self._local_model.train()
        for epoch in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._local_model(x), y)
                loss.backward()
                optimizer.step()

        return copy.deepcopy(self._local_model.state_dict())

    def get_weight_delta(
        self, global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the weight update delta: local weights - global weights.

        Some aggregation algorithms work on deltas rather than full weights.

        Args:
            global_state: The global model's state_dict before local training.

        Returns:
            Dict mapping parameter name → (local - global) tensor.
        """
        local_state = self._local_model.state_dict()
        delta = {}
        for key in local_state:
            delta[key] = local_state[key].float() - global_state[key].float()
        return delta

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        role = "MALICIOUS" if self.is_malicious else "honest"
        return (f"FLClient(id={self.client_id}, role={role}, "
                f"n_samples={self.n_samples}, local_epochs={self.local_epochs})")
