"""
train/evaluator.py
==================
Model evaluation on the test set.

WHY SEPARATE TRAIN AND EVAL?
  During evaluation:
  - We do NOT compute gradients (saves memory and time)
  - We set the model to eval() mode, which disables dropout and fixes batch norm
  This gives us an honest measure of how well the model generalizes to unseen data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on a DataLoader and return test accuracy.

    Args:
        model:  Trained neural network (in any mode; this function sets eval()).
        loader: DataLoader for the evaluation set (usually test_loader).
        device: Device the model lives on.

    Returns:
        Accuracy as a percentage (0.0 to 100.0).

    Example:
        >>> acc = evaluate(model, test_loader, device)
        >>> print(f"Test accuracy: {acc:.2f}%")
        Test accuracy: 99.07%
    """
    model.eval()   # Disable dropout and fix batch-norm statistics
    correct = 0
    total = 0

    with torch.no_grad():   # Disable gradient tracking during evaluation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)                      # Forward pass only
            predictions = torch.argmax(logits, dim=1)   # Take the class with highest score
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total
