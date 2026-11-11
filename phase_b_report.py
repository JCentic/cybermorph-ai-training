"""
phase_b_report.py
────────────────────────────────────────────────────────────────
THE PHASE B REPORT (five seeds per algorithm)

Part 1: the winner rule of RESULTS.md B3/B4, computed from results/<algo>_results_seed_<n>.csv
        (reward per card at the last five evaluations; the gate on invalid actions; the tie rule).
Part 2: the "why RL" test. Every saved model (3 algorithms x 5 seeds) plays the held-out
        real-data players in the learning world, with the invalid actions masked like in the game,
        and is compared with the simple rules. Skill gain is shown as the mean over the
        five seeds and the spread ACROSS seeds (this is what the single-seed numbers could not show).

Run from the repo root (venv active):

    python phase_b_report.py                  # 50 episodes per player and stage
    python phase_b_report.py --episodes 100 --write
────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import os
import sys

import numpy as np

from baselines import RLPolicy, all_baselines
from evaluate_policies import evaluate
from training_common import load_held_out_profiles
from training_data import DERIVED_WEAKNESS

ALGOS = ["ppo", "dqn", "a2c"]
SEEDS = [42, 43, 44, 45, 46]
RATES = [0.03, 0.06, 0.10]


def seed_score(algo, seed):
    """(reward per card, invalid actions per episode) averaged over the last five evaluations (steps 260,000 to 300,000), pooled stage 0."""
    path = os.path.join("results", "%s_results_seed_%d.csv" % (algo, seed))
    rows = [r for r in csv.DictReader(open(path)) if int(r["stage"]) == 0]
    rows.sort(key=lambda r: int(r["step"]))
    last = rows[-5:]
    return (float(np.mean([float(r["reward_per_card"]) for r in last])),
            float(np.mean([float(r["invalid_actions"]) for r in last])))


def winner_rule():
    print("PART 1 - the winner rule (RESULTS.md B3 and B4)")
    table = {}
    for algo in ALGOS:
        scores = [seed_score(algo, s) for s in SEEDS]
        rewards = np.array([r for r, _ in scores])
        invalid = float(np.mean([i for _, i in scores]))
        table[algo] = (rewards.mean(), rewards.std(ddof=1), invalid)
        print("  %-4s reward/card %.3f  std across seeds %.3f  invalid/episode (raw agent) %.2f  %s" % (
            algo.upper(), rewards.mean(), rewards.std(ddof=1), invalid, "passes the gate" if invalid < 0.5 else "FAILS the gate (<0.5)"))
    passing = {a: v for a, v in table.items() if v[2] < 0.5}
    if not passing:
        print("  No algorithm passes the invalid-action gate as a RAW agent (masking removes this in the game; see part 2).")
        return
    ranked = sorted(passing, key=lambda a: -passing[a][0])
    best = ranked[0]
    if len(ranked) > 1 and passing[best][0] - passing[ranked[1]][0] <= passing[best][1]:
        tied = [a for a in ranked if passing[best][0] - passing[a][0] <= passing[best][1]]
        best = min(tied, key=lambda a: passing[a][1])
        print("  TIE among %s -> smaller spread wins: %s" % (", ".join(t.upper() for t in tied), best.upper()))
    else:
        print("  Winner by the rule: %s" % best.upper())


def load(algo, seed):
    from stable_baselines3 import A2C, DQN, PPO
    cls = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[algo]
    return RLPolicy(cls.load(os.path.join("models", "%s_seed_%d.zip" % (algo, seed))), "RL %s s%d" % (algo.upper(), seed), masked=True)


def why_rl(episodes, write):
    profiles = load_held_out_profiles() or DERIVED_WEAKNESS
    print("\nPART 2 - skill gain on the held-out players, masked, mean over seeds (and the spread across the five seeds)")
    out = []
    for rate in RATES:
        print("\n  practise rate %.2f" % rate)
        for policy in all_baselines():
            mean = float(np.mean(evaluate(policy, profiles, episodes, rate)["skill_gain"]))
            print("    %-22s %.4f" % (policy.name, mean))
            out.append([rate, policy.name, mean, 0.0, 0])
        for algo in ALGOS:
            per_seed = [float(np.mean(evaluate(load(algo, s), profiles, episodes, rate)["skill_gain"])) for s in SEEDS]
            wins = ""
            print("    %-22s %.4f  +- %.4f across seeds   (best seed %.4f, worst %.4f)%s" % (
                "RL " + algo.upper(), np.mean(per_seed), np.std(per_seed, ddof=1), max(per_seed), min(per_seed), wins))
            out.append([rate, "RL " + algo.upper(), float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)), len(per_seed)])
    if write:
        with open(os.path.join("results", "phase_b_skill_gain.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["practise_rate", "policy", "skill_gain_mean", "std_across_seeds", "seeds"])
            w.writerows(out)
        print("\nwritten: results/phase_b_skill_gain.csv")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    winner_rule()
    why_rl(args.episodes, args.write)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
