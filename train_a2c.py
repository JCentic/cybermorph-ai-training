"""
train_a2c.py
────────────────────────────────────────────────────────────────
A2C TRAINING SCRIPT FOR CYBERMORPH RL COMPARISON

Trains an Advantage Actor-Critic (A2C) agent against the fake
CyberMorph player environment. The training loop, evaluation and
saving are shared by all three algorithms: see training_common.py.

HYPERPARAMETERS:
  learning_rate = 0.0007   ← Standard for A2C
  n_steps = 5              ← Rollout steps before update
  seed = [42] for Phase A, [42-46] for Phase B

OUTPUT:
  results/a2c_results_seed_<seed>.csv   (see training_common.py for the columns)
  models/a2c_seed_<seed>.zip

RUN:
  python train_a2c.py
────────────────────────────────────────────────────────────────
"""

from stable_baselines3 import A2C

from training_common import run_training

HYPERPARAMETERS = dict(
    learning_rate=0.0007,
    n_steps=5,
)


def train(seeds: list):
    """Train A2C for each seed. Phase A: [42]. Phase B: [42, 43, 44, 45, 46]."""
    run_training("A2C", A2C, HYPERPARAMETERS, seeds)


if __name__ == "__main__":
    # PHASE A: Single seed for validation
    train(seeds=[42])
