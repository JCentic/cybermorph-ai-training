"""
train_ppo.py
────────────────────────────────────────────────────────────────
PPO TRAINING SCRIPT FOR CYBERMORPH RL COMPARISON

This script trains a Proximal Policy Optimization (PPO) agent
against the fake CyberMorph player environment.

WHAT THIS DOES:
  1. Creates the fake player environment
  2. Trains PPO for 300,000 steps
  3. Every 10,000 steps: evaluate on 10 test episodes
  4. Records: reward, std, unique attacks used
  5. Writes results to results/ppo_results.csv

HYPERPARAMETERS (tuned for this problem):
  learning_rate = 0.0003   ← Standard for PPO
  n_steps = 2048           ← Batch size for policy updates
  seed = [42] for Phase A, [42-46] for Phase B

TOTAL TRAINING TIME:
  Phase A (1 seed):  ~30-40 minutes
  Phase B (5 seeds): ~2.5-3 hours

────────────────────────────────────────────────────────────────
"""

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from environment.cybermorph_env import CyberMorphEnv


class EvaluationCallback(BaseCallback):
    """
    CUSTOM CALLBACK FOR PERIODIC EVALUATION
    
    Gymnasium allows us to hook into the training loop to run
    custom code at specific intervals. This callback:
      1. Runs every 10,000 steps
      2. Evaluates on 10 test episodes
      3. Records metrics to CSV
      4. Prints progress
    
    This is how we collect the data for comparison charts.
    """
    
    def __init__(self, eval_freq=10000, n_eval_episodes=10, csv_file="results/ppo_results.csv", eval_env=None):
        """
        Args:
            eval_freq: Run evaluation every N steps (10,000)
            n_eval_episodes: Run 10 episodes for evaluation
            csv_file: Where to write results
            eval_env: Reference to actual environment for evaluation
        """
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_file = csv_file
        self.eval_env = eval_env
        self.last_eval_step = 0  # Track last evaluation step
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'mean_reward', 'reward_std', 'unique_attacks_used'])
    
    def _on_step(self) -> bool:
        """
        Called after every environment step.
        
        We evaluate periodically (every eval_freq steps).
        """
        current_step = self.model.num_timesteps
        
        # Check if it's time to evaluate (every eval_freq timesteps)
        if current_step - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = current_step
            
            # Run evaluation episodes
            eval_rewards = []
            all_attacks_used = []
            
            for episode in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                episode_reward = 0.0
                episode_attacks = set()
                
                # Run one episode to completion
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    
                    # Track which attacks were used
                    if 'attack_type' in info:
                        episode_attacks.add(info['attack_type'])
                
                eval_rewards.append(episode_reward)
                all_attacks_used.append(len(episode_attacks))
            
            # Calculate statistics
            mean_reward = np.mean(eval_rewards)
            reward_std = np.std(eval_rewards)
            mean_diversity = np.mean(all_attacks_used)
            
            # Write to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_step, mean_reward, reward_std, mean_diversity])
            
            # Print progress
            print(f"[PPO] Step {current_step:,} | "
                  f"Mean Reward: {mean_reward:.3f} ± {reward_std:.3f} | "
                  f"Diversity: {mean_diversity:.1f}/8 attacks")
        
        return True


class StepLoggerCallback(BaseCallback):
    """
    HIGH-PERFORMANCE STEP-BY-STEP LOGGER
    Saves every single step's reward to RAM and writes to CSV at the end.
    """
    def __init__(self, csv_file="results_all_steps/ppo_all_steps.csv"):
        super().__init__()
        self.csv_file = csv_file
        self.data = []
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        # Extract immediate reward from the training environment
        reward = self.locals['rewards'][0]
        self.data.append((self.step_count, reward))
        return True

    def _on_training_end(self):
        # High-speed batch write at the very end of training
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'reward'])
            writer.writerows(self.data)
        print(f"✓ Saved high-resolution steps to: {self.csv_file}")


def train(seeds: list):
    """
    TRAIN PPO FOR SPECIFIED SEEDS
    
    Args:
        seeds: List of random seeds to use
               Phase A: [42]
               Phase B: [42, 43, 44, 45, 46]
    
    Example:
        train(seeds=[42])  # Single trial (Phase A)
        train(seeds=[42, 43, 44, 45, 46])  # 5 trials (Phase B)
    """
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"PPO TRAINING — Seed {seed}")
        print(f"{'='*60}")
        
        # ─────────────────────────────────────────────────────────
        # CREATE ENVIRONMENTS
        # ─────────────────────────────────────────────────────────
        # Two separate environments:
        #   1. env:      For training (PPO collects rollouts)
        #   2. eval_env: For evaluation (callback tests performance)
        # They MUST be separate to avoid conflicts.
        
        env = CyberMorphEnv()
        eval_env = CyberMorphEnv()
        
        # ─────────────────────────────────────────────────────────
        # CREATE PPO MODEL
        # ─────────────────────────────────────────────────────────
        # PPO (Proximal Policy Optimization) is a policy gradient
        # algorithm that's good at learning stable policies.
        #
        # Hyperparameters:
        #   policy = "MlpPolicy"      ← Multi-layer perceptron neural net
        #   learning_rate = 0.0003    ← How fast it learns (conservative)
        #   n_steps = 2048            ← Batch size for updates
        #   seed = seed               ← Reproducibility
        #   verbose = 0               ← Don't print training progress (we have callback)
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            seed=seed,
            verbose=0
        )
        
        # ─────────────────────────────────────────────────────────
        # CREATE EVALUATION CALLBACK
        # ─────────────────────────────────────────────────────────
        # This hooks into training to evaluate every 10,000 steps
        # and write results to CSV.
        
        callback = EvaluationCallback(
            eval_freq=10000,
            n_eval_episodes=10,
            csv_file="results/ppo_results.csv",
            eval_env=eval_env
        )
        
        step_logger_callback = StepLoggerCallback(
            csv_file=f"results_all_steps/ppo_all_steps_seed_{seed}.csv"
        )
        
        # ─────────────────────────────────────────────────────────
        # TRAIN THE MODEL
        # ─────────────────────────────────────────────────────────
        # This runs the main training loop:
        #   300,000 steps total
        #   Every 10,000 steps: evaluate and log results
        #
        # What happens internally:
        #   1. AI picks actions based on current policy
        #   2. Environment responds with rewards
        #   3. AI learns from rewards (policy gradient updates)
        #   4. Callback evaluates performance every 10k steps
        
        print(f"\nTraining PPO for 300,000 steps (seed={seed})...")
        print("Evaluation every 10,000 steps...\n")
        
        model.learn(total_timesteps=300000, callback=[callback, step_logger_callback])
        
        # ─────────────────────────────────────────────────────────
        # CLEANUP
        # ─────────────────────────────────────────────────────────
        env.close()
        eval_env.close()
        
        print(f"\n✓ PPO training complete (seed={seed})")
        print(f"  Results saved to: results/ppo_results.csv")


if __name__ == "__main__":
    """
    RUN THIS FILE TO TRAIN PPO
    
    Phase A (current): Single seed
        python train_ppo.py
        → Trains 1 PPO agent with seed=42
        → Takes ~30-40 minutes
        → Outputs: results/ppo_results.csv (30 rows, one per 10k steps)
    
    Phase B (later): Multiple seeds
        Edit line below to:
        train(seeds=[42, 43, 44, 45, 46])
        → Trains 5 PPO agents with different seeds
        → Takes ~2.5-3 hours
        → Outputs: results/ppo_results.csv (appends 30 more rows per seed)
    """
    
    # PHASE A: Single seed for validation
    train(seeds=[42])
    
    # PHASE B: (Uncomment later for final defense)
    # train(seeds=[42, 43, 44, 45, 46])
    
    print("\n" + "="*60)
    print("PPO TRAINING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train DQN: python train_dqn.py")
    print("  2. Train A2C: python train_a2c.py")
    print("  3. Or run all: python run_all.py")
