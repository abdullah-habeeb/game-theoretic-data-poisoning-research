# Game-Theoretic Modeling of Strategic Data Poisoning
**Min–Max Optimization in Collaborative Neural Networks**

## Research Question
> Can we improve the robustness of a neural network against poisoned training data by modeling the interaction between attacker and defender as a game and training the model using an alternating min–max process?

## Project Structure
```
ml-research/
├── requirements.txt
├── research_notebook.ipynb     ← Colab-ready unified notebook
├── utils/                      ← Seed control, metrics, plotting
├── data/                       ← MNIST dataset loader
├── models/                     ← CNN architecture
├── train/                      ← Training and evaluation loops
├── attacks/                    ← Label-flip poisoning attack
├── experiments/                ← Baseline, poisoning, sweep, defender
├── results/                    ← Saved figures and tables
└── paper/                      ← Paper writing template
```

## Quick Start (Local)
```bash
pip install -r requirements.txt
python -m experiments.baseline
python -m experiments.poisoning
python -m experiments.sweep
python -m experiments.defender
```

## Quick Start (Google Colab)
Upload the entire folder or open `research_notebook.ipynb` directly in Colab.

## Experiment Pipeline
| Step | Script | Description |
|------|--------|-------------|
| 1 | `experiments/baseline.py` | Train clean CNN × 3 runs |
| 2 | `experiments/poisoning.py` | Train on poisoned data × 3 runs |
| 3 | `experiments/sweep.py` | Sweep poison fractions 0.2→0.8 |
| 4 | `experiments/defender.py` | Alternating Min–Max defender |

## Key Design
- **Attack**: Targeted label-flip (class 1 → class 7)
- **Defender**: Alternating min–max game (Stackelberg approximation)
- **Reproducibility**: Fixed seeds per run, deterministic PyTorch
- **Dataset**: MNIST (extendable to CIFAR-10)
