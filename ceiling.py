"""
ceiling.py
────────────────────────────────────────────────────────────────
HOW MUCH ROOM IS THERE FOR ANY COACH TO BEAT THE SIMPLE RULES?

The oracle coaches here CHEAT: they read the simulated player's TRUE numbers
(env.weakness_vector), which no real coach and no RL agent can see. They are never
used in the game. They answer one question: if a coach knew everything, how much
better than the simple rules could it be? If the answer is "almost nothing", then
no RL agent can win by much in this simulator, and the problem is the simulator
(or the measure), not the training.

  OracleBestGain   each card, the threat card whose practice lowers the player's average
                   weakness the most right now (true numbers, no noise).
  OracleBestLong   the same idea, but it looks ahead: it tries each category, plays the
                   next few cards the greedy way, and keeps the first card of the best
                   line. (Copies the environment; slow but fine for a one-off check.)

Run from the repo root (venv active):

    python ceiling.py                          # 0.03 0.06 0.10 practise rates, 40 episodes
    python ceiling.py --episodes 100 --lookahead 4
────────────────────────────────────────────────────────────────
"""

import argparse
import copy
import sys

import numpy as np

from baselines import GreedyWithVariety, Policy, StandardPattern, action_of
from environment.cybermorph_env import WEAKNESS_FLOOR, CyberMorphEnv
from evaluate_policies import METRICS, evaluate, format_table, summarise
from training_common import load_held_out_profiles
from training_data import DERIVED_WEAKNESS


class OracleBestGain(Policy):
    name = "oracle (best gain now)"

    def bind(self, env):
        self.env = env

    def act(self, obs, stage):
        env = self.env
        offered = 2 * stage
        # The skill gain counts threat cards only (the average weakness), so a threat card of the category with the biggest gap to the floor is the best single card.
        gains = np.maximum(env.weakness_vector[:offered] - WEAKNESS_FLOOR, 0.0)
        best = np.flatnonzero(gains == gains.max())
        return action_of(int(best[int(self.rng.integers(len(best)))]), True)


class OracleBestLong(OracleBestGain):
    name = "oracle (looks ahead)"

    def __init__(self, lookahead=4):
        self.lookahead = lookahead

    def act(self, obs, stage):
        env = self.env
        offered = 2 * stage
        best_action, best_value = None, -1e9
        for category in range(offered):
            trial = copy.deepcopy(env)
            before = trial.mean_weakness(trial.weakness_vector)
            trial.step(action_of(category, True))
            for _ in range(self.lookahead - 1):
                gains = np.maximum(trial.weakness_vector[:offered] - WEAKNESS_FLOOR, 0.0)
                trial.step(action_of(int(np.argmax(gains)), True))
            value = before - trial.mean_weakness(trial.weakness_vector)
            if value > best_value + 1e-12:
                best_action, best_value = action_of(category, True), value
        return best_action


def main(argv):
    parser = argparse.ArgumentParser(description="How much better than the simple rules could a coach that knows everything be?")
    parser.add_argument("--episodes", type=int, default=40, help="episodes per player and stage")
    parser.add_argument("--learning-rates", nargs="*", type=float, default=[0.03, 0.06, 0.10])
    parser.add_argument("--lookahead", type=int, default=4)
    args = parser.parse_args(argv)

    profiles = load_held_out_profiles() or DERIVED_WEAKNESS
    policies = [StandardPattern(), GreedyWithVariety(), OracleBestGain(), OracleBestLong(args.lookahead)]
    for rate in args.learning_rates:
        rows = []
        for policy in policies:
            rows.append((policy.name, summarise(evaluate(policy, profiles, args.episodes, rate))))
        print(format_table(rows, rate))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
