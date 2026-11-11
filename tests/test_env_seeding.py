"""
test_env_seeding.py
────────────────────────────────────────────────────────────────
Plain-assert checks for the claude-31 fixes in cybermorph_env.py.
Run from the repo root:

    python -m tests.test_env_seeding

Prints one line per check and "RESULT: OK" when all pass.
────────────────────────────────────────────────────────────────
"""

import numpy as np

from environment.cybermorph_env import CyberMorphEnv
from training_data import DERIVED_WEAKNESS


ACTIONS = [0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13] * 4


def run_episode(seed):
    """Reset with a seed, play a fixed action list, return everything seen."""
    env = CyberMorphEnv()
    obs, info = env.reset(seed=seed)
    seen = [obs.copy()]
    rewards = []
    profiles = [info["profile"]]
    for action in ACTIONS:
        obs, reward, terminated, truncated, _ = env.step(action)
        seen.append(obs.copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break
    # A second reset without a seed must still be repeatable given the first seed
    obs, info = env.reset()
    seen.append(obs.copy())
    profiles.append(info["profile"])
    return np.array(seen), rewards, profiles


def check_same_seed_same_episode():
    a = run_episode(7)
    b = run_episode(7)
    assert np.array_equal(a[0], b[0]), "observations differ for the same seed"
    assert a[1] == b[1], "rewards differ for the same seed"
    assert a[2] == b[2], "profiles differ for the same seed"
    print("ok  same seed gives the same episodes (including the next reset)")


def check_different_seeds_differ():
    # Five profiles plus noise: two different seeds should not match everywhere.
    first = run_episode(1)[0]
    for seed in (2, 3, 4):
        if not np.array_equal(first, run_episode(seed)[0]):
            print("ok  different seeds give different episodes")
            return
    raise AssertionError("seeds 1 to 4 all gave the same episode")


def check_noise_is_separate():
    # Use profile slots well away from 0 and 1 so clipping cannot hide the noise.
    env = CyberMorphEnv()
    for seed in range(20):
        env.reset(seed=seed)
        base = DERIVED_WEAKNESS[env.current_profile]
        w_delta = env.weakness_vector - np.array(base["weakness"], dtype=np.float32)
        f_delta = env.fp_vector - np.array(base["fp_rate"], dtype=np.float32)
        free = [
            i for i in range(8)
            if 0.2 < base["weakness"][i] < 0.8 and 0.2 < base["fp_rate"][i] < 0.8
        ]
        if len(free) >= 2:
            assert not np.allclose(w_delta[free], f_delta[free]), (
                "weakness and fp_rate noise are identical (shared array)"
            )
            print("ok  weakness and fp_rate noise are separate draws")
            return
    raise AssertionError("no profile had two unclipped slots to compare")


def check_observation_range():
    obs = run_episode(11)[0]
    assert obs.shape[1] == 20, "observation is not 20 numbers"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "observation left 0 to 1"
    print("ok  observations have 20 numbers, all in 0 to 1")


if __name__ == "__main__":
    check_same_seed_same_episode()
    check_different_seeds_differ()
    check_noise_is_separate()
    check_observation_range()
    print("RESULT: OK")
