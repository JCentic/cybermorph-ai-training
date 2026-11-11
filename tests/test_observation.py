"""
test_observation.py
────────────────────────────────────────────────────────────────
Plain-assert checks for observation.py (claude-33): the estimates,
the 20 numbers and the action mask. Run from the repo root:

    python -m tests.test_observation

Prints one line per check and "RESULT: OK" when all pass.
────────────────────────────────────────────────────────────────
"""

import random

from observation import (
    N_ACTIONS,
    PRIOR_FP,
    PRIOR_RESPONSE_MS,
    PRIOR_WEAKNESS,
    PRIOR_WEIGHT,
    WINDOW_SIZE,
    action_mask,
    build_observation,
    estimate,
)
from training_data import DERIVED_WEAKNESS


def close(a, b, tolerance=1e-9):
    return abs(a - b) <= tolerance


def check_estimate_shrinks_to_the_prior():
    assert close(estimate(0, 0, 0.4), 0.4), "no cards must give the prior"
    # 3 cards, all failed, prior 0.4, weight 3: (3 + 1.2) / 6
    assert close(estimate(3, 3, 0.4), 4.2 / 6.0), "3 failed cards: wrong shrink"
    # many cards: close to the real rate
    assert abs(estimate(300, 600, 0.9) - 0.5) < 0.01, "many cards must follow the data"
    print("ok  estimate: prior with no cards, shrunk with few, data with many")


def check_empty_window():
    obs = build_observation([], 2)
    assert len(obs) == 20
    assert all(close(a, b, 1e-12) for a, b in zip(obs[0:8], PRIOR_WEAKNESS)), "empty window must show the weakness priors"
    assert all(close(a, b, 1e-12) for a, b in zip(obs[8:16], PRIOR_FP)), "empty window must show the fp priors"
    assert close(obs[16], PRIOR_RESPONSE_MS / 6000.0)
    assert obs[17] == 0.5 and obs[18] == 0.0 and close(obs[19], 0.5)
    print("ok  empty window gives the cold-start numbers")


def check_values_in_range():
    rng = random.Random(1)
    for _ in range(300):
        window = [
            (rng.randrange(8), rng.random() < 0.6, rng.random() < 0.5, rng.choice([0.0, 300.0, 2500.0, 6000.0, 99999.0]))
            for _ in range(rng.randrange(0, 120))
        ]
        obs = build_observation(window, rng.randint(1, 4))
        assert len(obs) == 20, "observation is not 20 numbers"
        assert all(0.0 <= v <= 1.0 for v in obs), f"value left 0 to 1: {obs}"
    print("ok  300 random windows: always 20 numbers in 0 to 1")


def check_only_the_last_50_count():
    old_failures = [(0, True, False, 2000.0)] * 10     # 10 failed Phishing cards, oldest
    newer = [(0, True, True, 2000.0)] * WINDOW_SIZE     # then 50 right ones
    assert close(build_observation(old_failures + newer, 1)[0], build_observation(newer, 1)[0]), (
        "answers older than the window must not count"
    )
    print("ok  only the last 50 answers count")


def check_estimates_follow_the_answers():
    # Three failed Smishing threat cards, nothing else
    window = [(1, True, False, 3000.0)] * 3
    obs = build_observation(window, 1)
    assert close(obs[1], (3 + PRIOR_WEAKNESS[1] * PRIOR_WEIGHT) / 6.0)
    assert close(obs[0], PRIOR_WEAKNESS[0], 1e-12), "a category with no answers keeps its prior"
    # A wrongly blocked safe Phishing card raises the fp estimate for Phishing only
    obs = build_observation([(0, False, False, 3000.0)], 1)
    assert obs[8] > PRIOR_FP[0] and close(obs[9], PRIOR_FP[1], 1e-12)
    print("ok  estimates move with the answers, other categories keep the prior")


def check_recent_accuracy_and_streak():
    window = [(0, True, True, 1000.0)] * 3 + [(0, True, False, 1000.0)] + [(0, True, True, 1000.0)] * 2
    obs = build_observation(window, 1)
    # last 5 answers: right, right, wrong, right, right -> 4 of 5; streak at the end: 2 -> 2 / 4
    assert close(obs[17], 0.8) and close(obs[18], 0.5)
    obs = build_observation([(0, True, True, 1000.0)] * 9, 1)
    assert obs[18] == 1.0, "streak is capped at 1"
    print("ok  recent accuracy and streak")


def check_action_mask():
    counts = {stage: sum(action_mask(stage)) for stage in (1, 2, 3, 4)}
    # stage 1: 2 categories x (threat, safe); stage 2: 4 x 2; stage 3: 6 x 2 minus 2 safe; stage 4: 8 x 2 minus 2 safe
    assert counts == {1: 4, 2: 8, 3: 10, 4: 14}, f"unexpected valid action counts {counts}"
    mask1 = action_mask(1)
    assert not any(mask1[4:]), "stage 1 must never offer categories 2 to 7"
    mask4 = action_mask(4)
    assert mask4[8] and not mask4[9], "Credential Theft: threat valid, safe invalid"
    assert mask4[10] and not mask4[11], "Public Wi-Fi: threat valid, safe invalid"
    assert len(mask4) == N_ACTIONS
    print("ok  action mask: stage sizes 4 / 8 / 10 / 14, no safe card for the two types without")


def check_priors_are_the_profile_means():
    profiles = list(DERIVED_WEAKNESS.values())
    for c in range(8):
        mean_w = sum(p["weakness"][c] for p in profiles) / len(profiles)
        mean_f = sum(p["fp_rate"][c] for p in profiles) / len(profiles)
        assert abs(mean_w - PRIOR_WEAKNESS[c]) < 0.006, f"weakness prior {c} is not the profile mean"
        assert abs(mean_f - PRIOR_FP[c]) < 0.006, f"fp prior {c} is not the profile mean"
    mean_ms = sum(p["avg_response_time_ms"] for p in profiles) / len(profiles)
    assert abs(mean_ms - PRIOR_RESPONSE_MS) < 50, "response-time prior is not the profile mean"
    print("ok  the cold-start priors are the mean of the five profiles")


if __name__ == "__main__":
    check_estimate_shrinks_to_the_prior()
    check_empty_window()
    check_values_in_range()
    check_only_the_last_50_count()
    check_estimates_follow_the_answers()
    check_recent_accuracy_and_streak()
    check_action_mask()
    check_priors_are_the_profile_means()
    print("RESULT: OK")
