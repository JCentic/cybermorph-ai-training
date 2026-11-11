"""
evaluate_policies.py
────────────────────────────────────────────────────────────────
THE COMPARISON THAT ANSWERS "WHY RL?"

Plays every policy (the RL models and the baselines of baselines.py) against the
SAME simulated players in the SAME episodes, in a world where the player LEARNS
from practice (environment/cybermorph_env.py, the learning model), and compares
four numbers:

  skill_gain      how much the player's average weakness fell during the run
                  (the educational result: higher is better)
  weak_spot_lift  how much more often than by chance the coach picked a card in the
                  player's weakest categories (the weakest quarter of the map's categories,
                  at least one). 1.0 is chance, 2.0 is twice as often (does the coach find the weak spots?)
  coverage        how many of the map's categories the player met (variety)
  reward_per_card the training reward (weak-spot targeting) per card

Players: the real-data groups of data/profiles_real.json that the RL agent never
trained on (python -m analysis.build_profiles --write); without that file the
five invented players are used and the report says so.

Run from the repo root (the venv must be active):

    python evaluate_policies.py                              # baselines only
    python evaluate_policies.py --models ppo dqn a2c         # also the saved seed-42 models
    python evaluate_policies.py --models ppo --episodes 200 --learning-rates 0.03 0.06 0.10
    python evaluate_policies.py --write                      # also writes results/policy_comparison.csv

The learning speed is an ASSUMPTION. The comparison is therefore repeated for
several speeds (--learning-rates): a conclusion that holds only for one guess is not a conclusion.
────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import os
import sys

import numpy as np

from baselines import RLPolicy, all_baselines
from environment.cybermorph_env import PRACTISE_RATE, CyberMorphEnv
from observation import MAX_STAGE
from training_common import load_held_out_profiles
from training_data import DERIVED_WEAKNESS

DEFAULT_EPISODES = 100          # per player, per map stage
BASE_SEED = 4242                # every policy plays the same episodes
METRICS = ["skill_gain", "weak_spot_lift", "coverage", "reward_per_card"]


def play_episode(env, policy, seed, stage, profile):
    """One episode. Returns the four metrics and the number of invalid actions."""
    obs, info = env.reset(seed=seed, options={"stage": stage, "profile": profile})
    policy.reset(np.random.default_rng(seed))
    if hasattr(policy, "bind"):
        policy.bind(env)  # an oracle (ceiling.py) may look at the player's TRUE numbers; no real policy can
    offered = 2 * stage
    weak_count = max(1, int(np.ceil(offered / 4.0)))  # the weakest quarter of the offered categories
    weakest = set(int(c) for c in np.argsort(-env.start_weakness[:offered])[:weak_count])
    chance = weak_count / offered  # the share of cards in those categories if the cards were spread evenly
    cards = valid = in_weak = 0
    categories = set()
    total_reward = 0.0
    invalid = 0
    last_weakness = info["start_mean_weakness"]
    done = False
    while not done:
        action = policy.act(obs, stage)
        obs, reward, terminated, truncated, step_info = env.step(action)
        cards += 1
        total_reward += float(reward)
        if step_info["invalid"]:
            invalid += 1
        else:
            valid += 1
            categories.add(step_info["attack_type"])
            in_weak += 1 if step_info["attack_type"] in weakest else 0
            last_weakness = step_info["mean_weakness"]
        done = terminated or truncated
    return {
        "skill_gain": info["start_mean_weakness"] - last_weakness,
        "weak_spot_lift": (in_weak / valid) / chance if valid else 0.0,
        "coverage": len(categories) / offered,
        "reward_per_card": total_reward / max(cards, 1),
        "invalid": invalid,
    }


def evaluate(policy, profiles, episodes, practise_rate):
    """All the episodes of one policy: every player x every stage x `episodes`. Returns {metric: [values...]} (one value per episode)."""
    env = CyberMorphEnv(profiles=profiles, learning=True, practise_rate=practise_rate, observation_version=getattr(policy, "obs_version", 1))
    values = {name: [] for name in METRICS + ["invalid"]}
    seed = BASE_SEED
    for profile in profiles:
        for stage in range(1, MAX_STAGE + 1):
            for _ in range(episodes):
                result = play_episode(env, policy, seed, stage, profile)
                for name in values:
                    values[name].append(result[name])
                seed += 1
    env.close()
    return values


def summarise(values):
    """Mean and the half-width of a 95% confidence interval for each metric."""
    out = {}
    for name, series in values.items():
        series = np.asarray(series, dtype=float)
        half = 1.96 * series.std(ddof=1) / np.sqrt(len(series)) if len(series) > 1 else 0.0
        out[name] = (float(series.mean()), float(half))
    return out


def load_models(names, folder="models", masked=True):
    """The saved seed-42 models of the algorithms asked for (ppo, dqn, a2c), as RL policies."""
    from stable_baselines3 import A2C, DQN, PPO
    classes = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    policies = []
    for name in names:
        path = os.path.join(folder, "%s_seed_42.zip" % name)
        if name not in classes or not os.path.exists(path):
            print("  skipped %s: no file %s" % (name, path))
            continue
        policies.append(RLPolicy(classes[name].load(path), "RL " + name.upper() + ("" if masked else " raw"), masked))
    return policies


def format_table(rows, rate):
    lines = ["", "learning speed (practise rate) %.2f" % rate,
             "%-22s %16s %16s %16s %16s %8s" % ("policy", "skill gain", "weak-spot lift", "coverage", "reward/card", "invalid")]
    for name, summary in rows:
        cells = ["%.3f ± %.3f" % summary[m] for m in METRICS]
        lines.append("%-22s %16s %16s %16s %16s %8.2f" % (name, cells[0], cells[1], cells[2], cells[3], summary["invalid"][0]))
    best = max(rows, key=lambda r: r[1]["skill_gain"][0])
    lines.append("best skill gain: %s" % best[0])
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(description="Compare RL models with non-RL baselines in a world where the player learns.")
    parser.add_argument("--models", nargs="*", default=[], help="saved models to include: ppo dqn a2c")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="episodes per player and stage")
    parser.add_argument("--learning-rates", nargs="*", type=float, default=[PRACTISE_RATE], help="practise rates to try (the assumption)")
    parser.add_argument("--models-dir", default="models", help="folder of the saved models (the older ones are in models/model1_... and models/model2_...)")
    parser.add_argument("--raw", action="store_true", help="do NOT mask the invalid actions (the raw agent); default is masked, like the game")
    parser.add_argument("--write", action="store_true", help="write results/policy_comparison.csv")
    args = parser.parse_args(argv)

    profiles = load_held_out_profiles()
    if profiles is None:
        profiles = DERIVED_WEAKNESS
        print("NOTE: data/profiles_real.json is missing: the five INVENTED players are used (not held out for the RL agent).")
    else:
        print("Players the RL agent never trained on: %s" % ", ".join(profiles))
    policies = all_baselines() + load_models(args.models, args.models_dir, not args.raw)
    csv_rows = []
    for rate in args.learning_rates:
        rows = []
        for policy in policies:
            summary = summarise(evaluate(policy, profiles, args.episodes, rate))
            rows.append((policy.name, summary))
            for metric, (mean, half) in summary.items():
                csv_rows.append([rate, policy.name, metric, mean, half])
        print(format_table(rows, rate))
    if args.write:
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", "policy_comparison.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["practise_rate", "policy", "metric", "mean", "ci95_half_width"])
            writer.writerows(csv_rows)
        print("\nwritten: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
