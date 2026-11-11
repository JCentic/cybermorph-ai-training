"""
test_training_common.py
────────────────────────────────────────────────────────────────
Plain-assert checks for the evaluation part of training_common.py:
the players the evaluation holds out, and the reward per card.
Run from the repo root:

    python -m tests.test_training_common
────────────────────────────────────────────────────────────────
"""

import json
import os
import tempfile

import numpy as np

import training_common as tc
from environment.cybermorph_env import RUN_CARDS_BY_STAGE, CyberMorphEnv


def check_no_file_means_no_held_out_players():
    assert tc.load_held_out_profiles(os.path.join(tempfile.gettempdir(), "no-such-profiles-cm.json")) is None
    print("ok  without data/profiles_real.json the evaluation says it is not held out")


def check_real_profiles_are_converted_for_the_environment():
    data = {"attack_types": [], "profiles": {
        "beginner": {"weakness": [0.4] * 8, "fp_rate": [0.2] * 8, "avg_response_time_ms": 5000, "people": 3, "cards": 900},
        "expert": {"weakness": [0.2] * 8, "fp_rate": [0.1] * 8, "avg_response_time_ms": None, "people": 3, "cards": 500},
    }}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "profiles_real.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        profiles = tc.load_held_out_profiles(path)
    assert set(profiles) == {"real-beginner", "real-expert"}
    assert profiles["real-expert"]["avg_response_time_ms"] == tc.DEFAULT_RESPONSE_MS, "a missing time gets the default"
    env = CyberMorphEnv(profiles=profiles)
    _, info = env.reset(seed=1)
    assert info["profile"].startswith("real-")
    print("ok  real profiles become environment players, with a default time when there is none")


class _Model:
    """A stand-in model that always answers action 0."""
    num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array(0), None


def check_evaluation_writes_the_reward_per_card():
    with tempfile.TemporaryDirectory() as folder:
        csv_path = os.path.join(folder, "r.csv")
        env = CyberMorphEnv()
        callback = tc.EvaluationCallback("TEST", 1, env, csv_path, eval_freq=1, episodes_per_stage=2)
        callback.model = _Model()
        callback.model.num_timesteps = 10
        callback._on_step()
        lines = open(csv_path, encoding="utf-8").read().strip().splitlines()
    assert lines[0].split(",")[-1] == "reward_per_card", lines[0]
    assert len(lines) == 1 + 5, "one pooled row and four stage rows"
    rows = [line.split(",") for line in lines[1:]]
    for row in rows:
        assert 0.0 <= float(row[-1]) <= 2.0, "a reward per card stays inside the 0 to 2 range of one step"
    stage_one = [r for r in rows if r[2] == "1"][0]
    assert float(stage_one[3]) <= 2.0 * RUN_CARDS_BY_STAGE[1], "the plain reward of a short run is small"
    print("ok  the evaluation writes one row per stage and a reward per card that can be compared across stages")


if __name__ == "__main__":
    check_no_file_means_no_held_out_players()
    check_real_profiles_are_converted_for_the_environment()
    check_evaluation_writes_the_reward_per_card()
    print("RESULT: OK")
