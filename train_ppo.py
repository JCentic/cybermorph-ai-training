"""
train_ppo.py
────────────────────────────────────────────────────────────────
PPO TRAINING SCRIPT FOR CYBERMORPH RL COMPARISON

Trains a Proximal Policy Optimization (PPO) agent against the fake
CyberMorph player environment. The training loop, evaluation and
saving are shared by all three algorithms: see training_common.py.

HYPERPARAMETERS:
  learning_rate = 0.0003   ← Standard for PPO
  n_steps = 2048           ← Batch size for policy updates
  seed = [42] for Phase A, [42-46] for Phase B

OUTPUT:
  results/ppo_results_seed_<seed>.csv   (see training_common.py for the columns)
  models/ppo_seed_<seed>.zip

RUN:
  python train_ppo.py
────────────────────────────────────────────────────────────────
"""

from stable_baselines3 import PPO

from training_common import run_training

HYPERPARAMETERS = dict(
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)


def train(seeds: list):
    """Train PPO for each seed. Phase A: [42]. Phase B: [42, 43, 44, 45, 46]."""
    run_training("PPO", PPO, HYPERPARAMETERS, seeds)


if __name__ == "__main__":
    # PHASE A: Single seed for validation
    train(seeds=[42])

    # PHASE B: train(seeds=[42, 43, 44, 45, 46])  (or: python run_all.py --phase B)

    print("\n" + "="*60)
    print("PPO TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train DQN: python train_dqn.py")
    print("  2. Train A2C: python train_a2c.py")
    print("  3. Or run all: python run_all.py")
