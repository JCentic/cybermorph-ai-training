"""
test_env_stages.py
────────────────────────────────────────────────────────────────
Plain-assert checks for the claude-33 environment: map stages,
the judging start, hidden truth, estimates, invalid actions.
Run from the repo root:

    python -m tests.test_env_stages

Prints one line per check and "RESULT: OK" when all pass.
────────────────────────────────────────────────────────────────
"""

import numpy as np

from environment.cybermorph_env import CyberMorphEnv
from observation import action_mask


def make(stage, profile="dummy-player-001", seed=5):
    env = CyberMorphEnv()
    obs, info = env.reset(seed=seed, options={"stage": stage, "profile": profile})
    return env, obs, info


def check_judging_start():
    for stage in (1, 2, 3, 4):
        env, obs, info = make(stage)
        assert info["stage"] == stage and info["profile"] == "dummy-player-001"
        assert len(env.window) == 9, "the judging session is 9 cards"
        assert all(entry[0] < 2 * stage for entry in env.window), "judging cards must come from the stage's categories"
        for level in range(3):
            cards = env.window[level * 3:(level + 1) * 3]
            assert any(not entry[1] for entry in cards), "each level needs a safe card (when one can exist)"
        assert obs[19] == stage / 4.0, "the stage is observation slot 19"
    print("ok  every episode starts with 9 judging cards from the stage's categories, 1+ safe per level")


def check_truth_is_hidden():
    env, obs, _ = make(4)
    assert not np.allclose(obs[0:8], env.weakness_vector), "the agent must not see the true weaknesses"
    assert env.avg_response_ms is not None  # the truth exists inside the env, just not in obs
    print("ok  the observation holds estimates, not the true numbers")


def check_invalid_actions_change_nothing():
    env, obs, _ = make(1)
    window_before = list(env.window)
    credits_before = env.credits
    history_before = list(env.attack_history)
    invalid_actions = [a for a, ok in enumerate(action_mask(1)) if not ok]
    assert invalid_actions, "stage 1 must have invalid actions"
    for action in invalid_actions:
        new_obs, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0 and info["invalid"] is True, f"action {action} must pay 0 and be flagged"
        assert np.array_equal(new_obs, obs), f"action {action} changed the observation"
    assert env.window == window_before and env.credits == credits_before and env.attack_history == history_before
    assert env.episode_step == len(invalid_actions), "an invalid action still costs a step"
    print("ok  invalid actions pay 0, change nothing, but cost a step")


def check_valid_actions_update_estimates():
    env, obs, _ = make(2)
    size_before = len(env.window)
    valid = [a for a, ok in enumerate(action_mask(2)) if ok]
    new_obs, reward, _, _, info = env.step(valid[0])
    assert info["invalid"] is False and len(env.window) == size_before + 1
    assert 0.0 <= float(reward) <= 2.0
    print("ok  a valid action adds one answer to the window")


def check_stage_one_never_leaves_its_categories():
    env, _, _ = make(1)
    rng = np.random.default_rng(0)
    for _ in range(80):
        action = int(rng.integers(16))
        info = env.step(action)[4]
        if not info["invalid"]:
            assert info["attack_type"] in (0, 1), "stage 1 produced a category from a later map"
    assert all(entry[0] in (0, 1) for entry in env.window), "stage 1 window holds a later category"
    print("ok  on stage 1 only Phishing and Smishing ever happen")


def check_safe_card_means_blocking_is_failure():
    # A player that always blocks safe cards (fp 1.0) fails every safe card; one that never does passes them all.
    env, _, _ = make(1)
    env.fp_vector[:] = 1.0
    assert env._answer(0, False) is True
    env.fp_vector[:] = 0.0
    assert env._answer(0, False) is False
    print("ok  a safe card is failed only when the player wrongly blocks it")


def check_same_seed_same_episode():
    def play(seed):
        env = CyberMorphEnv()
        obs, info = env.reset(seed=seed)
        trail = [obs.copy()]
        for action in [0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13] * 3:
            obs, reward, terminated, truncated, _ = env.step(action)
            trail.append(obs.copy())
            if terminated or truncated:
                break
        return info["stage"], info["profile"], np.array(trail)

    a, b = play(11), play(11)
    assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2]), "same seed must give the same episode"
    stages = {play(seed)[0] for seed in range(40)}
    assert stages == {1, 2, 3, 4}, f"the stage draw should reach all four stages, got {stages}"
    print("ok  same seed, same episode; all four stages get drawn")


def check_observation_range_and_episode_end():
    for stage in (1, 2, 3, 4):
        env, obs, _ = make(stage)
        rng = np.random.default_rng(stage)
        steps = 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(int(rng.integers(16)))
            steps += 1
            assert obs.shape == (20,) and obs.dtype == np.float32
            assert obs.min() >= 0.0 and obs.max() <= 1.0, "observation left 0 to 1"
            if terminated or truncated:
                break
        assert steps <= 100, "an episode is at most 100 steps"
    print("ok  observations stay in 0 to 1 and episodes end within 100 steps")


if __name__ == "__main__":
    check_judging_start()
    check_truth_is_hidden()
    check_invalid_actions_change_nothing()
    check_valid_actions_update_estimates()
    check_stage_one_never_leaves_its_categories()
    check_safe_card_means_blocking_is_failure()
    check_same_seed_same_episode()
    check_observation_range_and_episode_end()
    print("RESULT: OK")
