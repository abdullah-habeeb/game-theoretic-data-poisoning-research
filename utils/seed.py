"""
utils/seed.py
=============
Seed control for reproducibility.

WHY THIS MATTERS:
  Neural network training involves many random operations:
  - Weight initialization
  - Data shuffling in DataLoaders
  - Dropout (if used)
  Setting all seeds to the same value before each run ensures that
  two runs with the same seed produce identical results. This is
  essential for scientific reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Use different seeds for independent runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Makes CUDA operations deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
