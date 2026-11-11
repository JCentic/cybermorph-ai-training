"""
check_model.py
────────────────────────────────────────────────────────────────
RELOAD TEST FOR A SAVED MODEL (claude-31 B)

A model that cannot be reloaded is not saved. This loads one file and
asks it for one deterministic action on a fresh observation.

    python check_model.py models/ppo_seed_42.zip

The algorithm is read from the file name (ppo_, dqn_ or a2c_).
────────────────────────────────────────────────────────────────
"""

import os
import sys

from stable_baselines3 import A2C, DQN, PPO

from environment.cybermorph_env import CyberMorphEnv


ALGORITHMS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


def main(path):
    name = os.path.basename(path).lower()
    algo = name.split("_")[0]
    if algo not in ALGORITHMS:
        sys.exit(f"Cannot tell the algorithm from '{name}' (expected ppo_, dqn_ or a2c_).")

    model = ALGORITHMS[algo].load(path)
    env = CyberMorphEnv()
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)

    assert 0 <= int(action) <= 15, "action out of range"
    print(f"[OK] {path} reloaded; deterministic action for a fresh observation: {int(action)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python check_model.py models/<algo>_seed_<seed>.zip")
    main(sys.argv[1])
