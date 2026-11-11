"""
train_dqn.py
────────────────────────────────────────────────────────────────
DQN TRAINING SCRIPT FOR CYBERMORPH RL COMPARISON

Trains a Deep Q-Network (DQN) agent against the fake CyberMorph
player environment. The training loop, evaluation and saving are
shared by all three algorithms: see training_common.py.

HYPERPARAMETERS:
  learning_rate = 0.0001   ← Standard for DQN
  buffer_size = 50000      ← Replay buffer size
  seed = [42] for Phase A, [42-46] for Phase B

OUTPUT:
  results/dqn_results_seed_<seed>.csv   (see training_common.py for the columns)
  models/dqn_seed_<seed>.zip

RUN:
  python train_dqn.py
────────────────────────────────────────────────────────────────
"""

from stable_baselines3 import DQN

from training_common import run_training

HYPERPARAMETERS = dict(
    learning_rate=0.0001,
    buffer_size=50000,
)


def train(seeds: list):
    """Train DQN for each seed. Phase A: [42]. Phase B: [42, 43, 44, 45, 46]."""
    run_training("DQN", DQN, HYPERPARAMETERS, seeds)


if __name__ == "__main__":
    # PHASE A: Single seed for validation
    train(seeds=[42])
