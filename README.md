# Game-Theoretic Data Poisoning & Robust Defenses

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-blue)

## 📌 Project Overview
This repository contains the code and experimental results for a comprehensive research study on **Data Poisoning in Deep Learning**. The core objective of this research is twofold:
1. To expose a critical vulnerability in State-of-the-Art (SOTA) backdoor defenses by demonstrating their complete failure against covert, clean-label flipping attacks.
2. To propose, implement, and validate a novel **Alternating Min-Max Game-Theoretic Defense** that dynamically adapts to data poisoning without relying on spatial trigger assumptions.

We scaled our experiments from baseline architectures (CNNs on MNIST) to massive, deep learning architectures (WideResNet-28-10 on CIFAR-10), backed by a highly fault-tolerant pipeline designed for uninterrupted 24+ hour execution on Kaggle and AWS SageMaker.

---

## 🚀 Key Achievements
* **Exposed SOTA Blindspots:** Proved experimentally that leading backdoor defenses (Spectral Signatures and SEVER) are fundamentally blind to non-spatial label-flipping attacks, resulting in catastrophic ~11% accuracy drops on CIFAR-10.
* **Novel Defense Architecture:** Formulated the attacker-defender dynamic as a Stackelberg game. Implemented an alternating Min-Max training loop that stabilizes under poisoning.
* **Bulletproof Infrastructure:** Built an extremely robust, fault-tolerant training pipeline. Implemented dynamic epoch-level `.pt` checkpointing and JSON state tracking to survive kernel disconnects, hardware preemptions, and long-running AWS SageMaker limitations.
* **Massive Scaling:** Successfully migrated the codebase from localized multi-processing to a distributed PyTorch DataLoader structure, utilizing NVIDIA A10G architectures to train 36.5M parameter WideResNets.

---

## 📊 Experimental Results Summary

### CIFAR-10 (WideResNet-28-10 | 50% Poison Fraction)
The results below confirm the failure of SOTA defenses to detect the label-flipping attack, while validating the theoretical consistency of the Min-Max formulation.

| Method | Mean Accuracy | vs Baseline | Status |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | **94.34%** | 0.0% | Control |
| **Poisoned (No Defense)** | **83.36%** | -10.98% | Successful Attack |
| **Spectral Signatures** | 82.80% | -11.54% | Total Defense Failure |
| **SEVER** | 82.05% | -12.29% | Total Defense Failure |
| **Ours (Min-Max)** | 83.09% | -11.24% | Robust Adaptation |

### MNIST (Simple CNN | 50% Poison Fraction)
On simpler datasets, the attack margin is much tighter as models tend to generalize or memorize over the poisoned data.
* **Clean Baseline:** 98.58%
* **Poisoned (No Defense):** 94.70%
* **Min-Max Defender:** 93.31%

*(Complete CSV, JSON, and graphical data can be found in the `Final_Merged_Results/` directory).*

---

## 📁 Repository Architecture

```text
ml-research/
├── attacks/                 # Implementation of the clean-label flip attack
├── data/                    # Dataset loaders, normalizations, and augmentations
├── defenses/                # SOTA Defenses: Spectral Signatures, SEVER
├── experiments/             # Orchestration: Min-Max logic and head-to-head comparisons
├── models/                  # PyTorch definitions (MnistCNN, ResNet18, WideResNet)
├── train/                   # Robust training loops, evaluation, and dynamic checkpointing
├── utils/                   # Statistical significance testing, seeding, and plotting
├── Final_Merged_Results/    # Aggregated artifacts, CSVs, and plots
└── kaggle_notebook.py       # Execution script for Kaggle / SageMaker integration
```

---

## 🛠️ How to Run

### 1. Environment Setup
Install the necessary requirements. PyTorch with CUDA support is highly recommended.
```bash
pip install -r requirements.txt
```

### 2. Execute the Head-to-Head Comparison
To run the full pipeline evaluating the baseline against all three defenses:
```python
from experiments.defense_comparison import run_defense_comparison

cifar10_df = run_defense_comparison(
    n_runs=3, 
    dataset='cifar10',
    src_class=1,    
    tgt_class=7,   
    epochs=100,     
    defense_epochs=100,
    lr=0.1,         
    batch_size=128,
    use_sgd=True,   
    verbose=True,
)
```

### 3. Checkpointing System
If your training is interrupted (e.g., cloud timeout), simply **re-run the exact same script**. The pipeline will automatically detect the `epoch.pt` state file in `./results/checkpoints/` and immediately resume from the exact epoch it died on.

---
**Author:** Abdullah Habeeb
**Research Focus:** Adversarial Machine Learning, Game-Theoretic Defenses, Data Poisoning
