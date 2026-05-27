"""
train_dqn.py
────────────────────────────────────────────────────────────────
DQN TRAINING SCRIPT FOR CYBERMORPH RL COMPARISON

This script trains a Deep Q-Network (DQN) agent
against the fake CyberMorph player environment.

WHAT THIS DOES:
  1. Creates the fake player environment
  2. Trains DQN for 300,000 steps
  3. Every 10,000 steps: evaluate on 10 test episodes
  4. Records: reward, std, unique attacks used
  5. Writes results to results/dqn_results.csv

HYPERPARAMETERS (tuned for this problem):
  learning_rate = 0.0001   ← Standard for DQN
  buffer_size = 50000      ← Replay buffer size
  seed = [42] for Phase A, [42-46] for Phase B

────────────────────────────────────────────────────────────────
"""

import os
import csv
import numpy as np
from stable_baselines3 import DQN
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
    
    def __init__(self, eval_freq=10000, n_eval_episodes=10, csv_file="results/dqn_results.csv", eval_env=None):
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
        
        We evaluate periodically (every eval_freq timesteps).
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
            print(f"[DQN] Step {current_step:,} | "
                  f"Mean Reward: {mean_reward:.3f} ± {reward_std:.3f} | "
                  f"Diversity: {mean_diversity:.1f}/8 attacks")
        
        return True


def train(seeds: list):
    """
    TRAIN DQN FOR SPECIFIED SEEDS
    
    Args:
        seeds: List of random seeds to use
               Phase A: [42]
               Phase B: [42, 43, 44, 45, 46]
    """
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"DQN TRAINING — Seed {seed}")
        print(f"{'='*60}")
        
        # Create separate environments for training and evaluation
        env = CyberMorphEnv()
        eval_env = CyberMorphEnv()
        
        # Create DQN Model
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0001,
            buffer_size=50000,
            seed=seed,
            verbose=0
        )
        
        # Create evaluation callback
        callback = EvaluationCallback(
            eval_freq=10000,
            n_eval_episodes=10,
            csv_file="results/dqn_results.csv",
            eval_env=eval_env
        )
        
        print(f"\nTraining DQN for 300,000 steps (seed={seed})...")
        print("Evaluation every 10,000 steps...\n")
        
        model.learn(total_timesteps=300000, callback=callback)
        
        # Cleanup
        env.close()
        eval_env.close()
        
        print(f"\n✓ DQN training complete (seed={seed})")
        print(f"  Results saved to: results/dqn_results.csv")


if __name__ == "__main__":
    # PHASE A: Single seed for validation
    train(seeds=[42])
