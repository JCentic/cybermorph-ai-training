"""
test_learning_and_baselines.py
────────────────────────────────────────────────────────────────
Plain-assert checks for the "why RL?" comparison: the learning model of the
simulated player (environment/cybermorph_env.py), the non-RL policies
(baselines.py) and the evaluation (evaluate_policies.py).
Run from the repo root:

    python -m tests.test_learning_and_baselines
────────────────────────────────────────────────────────────────
"""

import numpy as np

import evaluate_policies as ev
from baselines import GreedyWeakest, GreedyWithVariety, RandomPolicy, StandardPattern, action_of
from environment.cybermorph_env import FP_FLOOR, WEAKNESS_FLOOR, CyberMorphEnv
from observation import action_mask


def make(learning, stage=4, profile="dummy-player-001", seed=3, **kwargs):
    env = CyberMorphEnv(learning=learning, **kwargs)
    obs, info = env.reset(seed=seed, options={"stage": stage, "profile": profile})
    return env, obs, info


# --- the learning model ---

def check_default_world_has_no_learning():
    env, _, _ = make(False)
    before = env.weakness_vector.copy()
    for _ in range(10):
        env.step(0)
    assert np.array_equal(env.weakness_vector, before), "training world: the player never changes"
    print("ok  by default the player never changes (the training world is untouched)")


def check_practice_lowers_the_practised_weakness_only_a_little():
    env, _, _ = make(True, forget_rate=0.0)
    before = env.weakness_vector.copy()
    env.step(action_of(2, True))  # a threat card of category 2
    after = env.weakness_vector
    assert after[2] < before[2], "practised: lower"
    assert np.allclose(np.delete(after, 2), np.delete(before, 2)), "others are untouched without forgetting"
    drop = before[2] - after[2]
    assert drop <= env.practise_rate * (before[2] - WEAKNESS_FLOOR) + 1e-6
    print("ok  one card lowers its own category's weakness by a small share of the gap")


def check_safe_cards_train_the_false_positive_rate():
    env, _, _ = make(True, forget_rate=0.0)
    weakness, fp = env.weakness_vector.copy(), env.fp_vector.copy()
    env.step(action_of(0, False))
    assert env.fp_vector[0] < fp[0] and env.weakness_vector[0] == weakness[0]
    print("ok  a safe card lowers the false-positive rate, not the weakness")


def check_floor_and_forgetting():
    env, _, _ = make(True, forget_rate=0.0)
    for _ in range(300):
        env.weakness_vector[1] = max(env.weakness_vector[1], 0.0)
        env._learn(1, True)
    assert env.weakness_vector[1] >= WEAKNESS_FLOOR - 1e-6, "a player never becomes perfect"
    env.fp_vector[0] = 0.9
    for _ in range(300):
        env._learn(0, False)
    assert env.fp_vector[0] >= FP_FLOOR - 1e-6
    forgetful, _, _ = make(True, practise_rate=0.0, forget_rate=0.5)
    forgetful.weakness_vector[3] = forgetful.start_weakness[3] - 0.2  # the player had improved
    forgetful._learn(5, True)
    assert forgetful.weakness_vector[3] > forgetful.start_weakness[3] - 0.2, "forgetting pulls it back toward the start"
    print("ok  weakness never goes below the floor, and unused skill fades back toward the start")


def check_the_environment_reports_the_mean_weakness():
    env, _, info = make(True, stage=2)
    assert abs(info["start_mean_weakness"] - float(np.mean(env.start_weakness[:4]))) < 1e-6, "only the categories of the map count"
    _, _, _, _, step_info = env.step(action_of(0, True))
    assert 0.0 <= step_info["mean_weakness"] <= 1.0
    print("ok  the mean weakness is measured over the categories of the map")


# --- the baselines ---

def check_baselines_only_choose_valid_actions():
    rng = np.random.default_rng(1)
    obs = np.random.default_rng(2).random(20).astype(np.float32)
    for stage in (1, 2, 3, 4):
        mask = action_mask(stage)
        for policy in (RandomPolicy(), StandardPattern(), GreedyWeakest(), GreedyWithVariety()):
            policy.reset(rng)
            for _ in range(60):
                assert mask[policy.act(obs, stage)], (policy.name, stage)
    print("ok  every baseline only chooses cards the map allows (and no safe card for the two types without one)")


def check_greedy_picks_the_weakest_and_variety_does_not_repeat_three_times():
    obs = np.zeros(20, dtype=np.float32)
    obs[3] = 0.9  # category 3 looks weakest
    greedy = GreedyWeakest()
    greedy.reset(np.random.default_rng(1))
    assert greedy.act(obs, 4) == action_of(3, True)
    variety = GreedyWithVariety()
    variety.reset(np.random.default_rng(1))
    picks = [variety.act(obs, 4) // 2 for _ in range(6)]
    assert picks[:2] == [3, 3] and picks[2] != 3, picks
    assert all(not (picks[i] == picks[i + 1] == picks[i + 2]) for i in range(len(picks) - 2)), picks
    print("ok  greedy picks the weakest category; the variety rule never repeats one three times in a row")


def check_the_standard_pattern_is_the_same_for_everyone():
    first = StandardPattern()
    second = StandardPattern()
    first.reset(np.random.default_rng(1))
    second.reset(np.random.default_rng(99))
    obs_a = np.random.default_rng(3).random(20).astype(np.float32)
    obs_b = np.random.default_rng(4).random(20).astype(np.float32)
    assert [first.act(obs_a, 4) for _ in range(20)] == [second.act(obs_b, 4) for _ in range(20)], "it ignores the player"
    first.reset(np.random.default_rng(1))
    categories = {first.act(obs_a, 2) // 2 for _ in range(8)}
    assert categories == {0, 1, 2, 3}, "all the categories of a 4-category map take turns"
    print("ok  the standard pattern ignores the player and rotates through the map's categories")


# --- the evaluation ---

def check_evaluation_is_repeatable_and_fair():
    profiles = {"p": {"username": "p", "description": "", "weakness": [0.5] * 8, "fp_rate": [0.2] * 8, "avg_response_time_ms": 3000}}
    a = ev.summarise(ev.evaluate(RandomPolicy(), profiles, 3, 0.06))
    b = ev.summarise(ev.evaluate(RandomPolicy(), profiles, 3, 0.06))
    assert a == b, "the same policy on the same episodes gives the same numbers"
    assert set(a) == set(ev.METRICS + ["invalid"])
    none = ev.summarise(ev.evaluate(StandardPattern(), profiles, 3, 0.0))
    assert abs(none["skill_gain"][0]) < 1e-9, "a player who cannot learn gains nothing"
    some = ev.summarise(ev.evaluate(StandardPattern(), profiles, 3, 0.06))
    assert some["skill_gain"][0] > 0.0, "with learning, practice helps"
    print("ok  the evaluation repeats exactly, and skill gain is zero without learning and positive with it")


def check_improvement_reward():
    # Model 2: the reward follows the fall of the player's weakness, not the player's failure.
    try:
        CyberMorphEnv(learning=False, reward_mode="improvement")
        raise SystemExit("improvement reward must need learning=True")
    except AssertionError:
        pass
    env, _, _ = make(True, reward_mode="improvement")
    rewards = [float(env.step((k * 2) % 16)[1]) for k in range(12)]
    assert all(0.0 <= r <= 2.0 for r in rewards), rewards
    assert sum(rewards) > 0.5, rewards
    # A player who cannot improve any more (everything at the floor) gives no improvement reward.
    env2, _, _ = make(True, reward_mode="improvement")
    env2.weakness_vector[:] = WEAKNESS_FLOOR
    env2.start_weakness[:] = WEAKNESS_FLOOR
    env2.step(0)
    assert float(env2.step(2)[1]) <= 0.2 + 1e-6
    print("ok  the improvement reward pays for a falling weakness only")


def check_observation_version_2():
    from observation import build_observation_v2
    env = CyberMorphEnv(learning=True, reward_mode="improvement", observation_version=2)
    obs, _ = env.reset(seed=5, options={"stage": 2})
    assert obs.shape == (29,) and env.observation_space.shape == (29,)
    assert abs(obs[28] - 1.0) < 1e-6, "the whole run is left at the start"
    first = obs[20:28].copy()
    assert first[0] > 0 or first[:4].sum() > 0, "the judging cards already count as practice"
    for _ in range(15):
        obs, _, _, truncated, _ = env.step(0)
    assert truncated and obs[28] == 0.0, "no cards left at the end of the run"
    assert obs[20] >= first[0], "practising category 0 raises its count"
    assert np.all((obs >= 0) & (obs <= 1))
    assert len(build_observation_v2([], 1, 0.5)) == 29
    print("ok  observation version 2 has 29 numbers: the 20, the practice counts and the share of the run left")


if __name__ == "__main__":
    check_observation_version_2()
    check_improvement_reward()
    check_default_world_has_no_learning()
    check_practice_lowers_the_practised_weakness_only_a_little()
    check_safe_cards_train_the_false_positive_rate()
    check_floor_and_forgetting()
    check_the_environment_reports_the_mean_weakness()
    check_baselines_only_choose_valid_actions()
    check_greedy_picks_the_weakest_and_variety_does_not_repeat_three_times()
    check_the_standard_pattern_is_the_same_for_everyone()
    check_evaluation_is_repeatable_and_fair()
    print("RESULT: OK")
