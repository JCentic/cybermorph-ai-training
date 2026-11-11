"""
training_common.py
────────────────────────────────────────────────────────────────
THE PART OF TRAINING THAT PPO, DQN AND A2C SHARE (claude-33)

One copy of the evaluation callback, the step logger and the training
loop, so the three algorithms are always trained and measured the
same way. The comparison is only fair if this code cannot drift apart.

Each train_<algo>.py holds only that algorithm's hyperparameters.

EVALUATION (every 10,000 steps, deterministic policy):
  EPISODES_PER_STAGE episodes on each of the 4 map stages, on an
  evaluation environment seeded from the training seed.
  One CSV row per stage (stage 1 to 4) plus one pooled row (stage 0):
      seed, step, stage, mean_reward, reward_std,
      unique_attacks_used, invalid_actions
  invalid_actions = mean invalid actions per episode; it should fall
  toward zero as the agent learns the stage's mask.
────────────────────────────────────────────────────────────────
"""

import csv
import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from environment.cybermorph_env import CyberMorphEnv
from observation import MAX_STAGE

TOTAL_TIMESTEPS = 300000
EVAL_FREQ = 10000
EPISODES_PER_STAGE = 10   # TUNABLE: more gives a smoother table, takes longer
EVAL_SEED_OFFSET = 1000   # the evaluation env is seeded with training seed + this

CSV_HEADER = ['seed', 'step', 'stage', 'mean_reward', 'reward_std', 'unique_attacks_used', 'invalid_actions', 'reward_per_card']

# The players the evaluation uses that the agent NEVER trains on: the three real groups made from the testers' data
# (python -m analysis.build_profiles --write). Without that file the evaluation falls back to the training players and says so.
HELD_OUT_FILE = os.path.join("data", "profiles_real.json")
DEFAULT_RESPONSE_MS = 3000.0
TRAIN_PRACTISE_RATE = 0.06


def load_held_out_profiles(path=HELD_OUT_FILE):
    """The real-data profiles in the shape the environment wants, or None when the file is not there.
    Keys are "real-beginner", "real-intermediate" and "real-expert"."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    profiles = {}
    for name, profile in data.get("profiles", {}).items():
        profiles["real-" + name] = {
            "username": "real " + name,
            "description": "built from %s tester(s), %s cards" % (profile.get("people", "?"), profile.get("cards", "?")),
            "weakness": list(profile["weakness"]),
            "fp_rate": list(profile["fp_rate"]),
            "avg_response_time_ms": profile.get("avg_response_time_ms") or DEFAULT_RESPONSE_MS,
        }
    return profiles or None


class EvaluationCallback(BaseCallback):
    """
    PERIODIC EVALUATION, PER MAP STAGE

    Every eval_freq steps: play EPISODES_PER_STAGE episodes on each stage,
    write the rows to this seed's CSV, print one progress line.
    """

    def __init__(self, algo_name, seed, eval_env, csv_file, eval_freq=EVAL_FREQ, episodes_per_stage=EPISODES_PER_STAGE):
        super().__init__()
        self.algo_name = algo_name
        self.seed = seed
        self.eval_env = eval_env
        self.csv_file = csv_file
        self.eval_freq = eval_freq
        self.episodes_per_stage = episodes_per_stage
        self.last_eval_step = 0

        # Seed the evaluation environment so evaluation episodes repeat.
        # (The training env is seeded by the model's seed=...)
        self.eval_env.reset(seed=seed + EVAL_SEED_OFFSET)

        # Start a fresh CSV for this seed (re-running a seed replaces its file)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

    def _run_episode(self, stage):
        """One evaluation episode on the given stage: reward, valid types used, invalid actions, and the number of cards (steps)."""
        obs, _ = self.eval_env.reset(options={"stage": stage})
        total_reward = 0.0
        attacks = set()
        invalid = 0
        cards = 0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += float(reward)
            cards += 1
            done = terminated or truncated
            if info['invalid']:
                invalid += 1
            else:
                attacks.add(info['attack_type'])
        return total_reward, len(attacks), invalid, cards

    def _on_step(self) -> bool:
        current_step = self.model.num_timesteps
        if current_step - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = current_step

        # An episode is as long as a real run of its map (8, 15, 22 or 30 cards), so the plain reward of a long stage is bigger than that of a short one. The winner rule therefore uses the
        # reward PER CARD (reward / cards), which can be compared across stages.
        rows = []        # (stage, rewards, diversities, invalids, rewards_per_card)
        pooled = ([], [], [], [])
        for stage in range(1, MAX_STAGE + 1):
            results = [self._run_episode(stage) for _ in range(self.episodes_per_stage)]
            rewards, diversities, invalids, cards = (list(column) for column in zip(*results))
            per_card = [r / max(c, 1) for r, c in zip(rewards, cards)]
            rows.append((stage, rewards, diversities, invalids, per_card))
            pooled[0].extend(rewards)
            pooled[1].extend(diversities)
            pooled[2].extend(invalids)
            pooled[3].extend(per_card)
        rows.insert(0, (0, *pooled))

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for stage, rewards, diversities, invalids, per_card in rows:
                writer.writerow([
                    self.seed, current_step, stage,
                    np.mean(rewards), np.std(rewards), np.mean(diversities), np.mean(invalids), np.mean(per_card),
                ])

        by_stage = " / ".join(f"{np.mean(c):.2f}" for stage, _, _, _, c in rows if stage > 0)
        print(f"[{self.algo_name}] Step {current_step:,} | "
              f"Reward per card: {np.mean(pooled[3]):.3f} | "
              f"Mean Reward: {np.mean(pooled[0]):.3f} ± {np.std(pooled[0]):.3f} | "
              f"Diversity: {np.mean(pooled[1]):.1f}/8 | "
              f"Invalid: {np.mean(pooled[2]):.1f}/episode | "
              f"Per card by stage: {by_stage}")
        return True


class StepLoggerCallback(BaseCallback):
    """
    HIGH-PERFORMANCE STEP-BY-STEP LOGGER
    Saves every single step's reward to RAM and writes to CSV at the end.
    """

    def __init__(self, csv_file):
        super().__init__()
        self.csv_file = csv_file
        self.data = []
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        self.data.append((self.step_count, self.locals['rewards'][0]))
        return True

    def _on_training_end(self):
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward'])
            writer.writerows(self.data)
        print(f"[OK] Saved high-resolution steps to: {self.csv_file}")


def run_training(algo_name, algo_class, model_kwargs, seeds, total_timesteps=TOTAL_TIMESTEPS):
    """
    TRAIN ONE ALGORITHM FOR EACH SEED

    Args:
        algo_name: "PPO", "DQN" or "A2C" (used in file names and messages)
        algo_class: the Stable-Baselines3 class
        model_kwargs: that algorithm's hyperparameters
        seeds: Phase A: [42]; Phase B: [42, 43, 44, 45, 46]

    Writes results/<algo>_results_seed_<seed>.csv and models/<algo>_seed_<seed>.zip.
    """
    name = algo_name.lower()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"{algo_name} TRAINING — Seed {seed}")
        print(f"{'='*60}")

        # Separate environments for training and evaluation
        # Model 2: the player learns from practice (practise rate 0.06) and the agent is paid for the player's improvement. The comparison afterwards
        # (evaluate_policies.py) also tests at 0.03 and 0.10, so the result does not depend on this one guess.
        env = CyberMorphEnv(learning=True, practise_rate=TRAIN_PRACTISE_RATE, reward_mode="improvement")
        held_out = load_held_out_profiles()
        if held_out is None:
            print("  NOTE: data/profiles_real.json is missing, so the evaluation uses the TRAINING players (not held out).")
            print("        Make it with:  python -m analysis.build_profiles --write")
        else:
            print(f"  Evaluation players the agent never trains on: {', '.join(held_out)}")
        eval_env = CyberMorphEnv(profiles=held_out, learning=True, practise_rate=TRAIN_PRACTISE_RATE, reward_mode="improvement")

        model = algo_class(policy="MlpPolicy", env=env, seed=seed, verbose=0, **model_kwargs)

        csv_file = f"results/{name}_results_seed_{seed}.csv"
        callback = EvaluationCallback(algo_name, seed, eval_env, csv_file)
        step_logger = StepLoggerCallback(f"results_all_steps/{name}_all_steps_seed_{seed}.csv")

        print(f"\nTraining {algo_name} for {total_timesteps:,} steps (seed={seed})...")
        print(f"Evaluation every {EVAL_FREQ:,} steps, {EPISODES_PER_STAGE} episodes on each of {MAX_STAGE} stages...\n")
        model.learn(total_timesteps=total_timesteps, callback=[callback, step_logger])

        # Save the trained model (Stable-Baselines3 adds the .zip itself)
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{name}_seed_{seed}"
        model.save(model_path)

        env.close()
        eval_env.close()

        print(f"\n[OK] {algo_name} training complete (seed={seed})")
        print(f"  Results saved to: {csv_file}")
        print(f"  Model saved to:   {model_path}.zip")
