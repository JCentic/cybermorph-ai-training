"""
run_all.py
────────────────────────────────────────────────────────────────
CYBERMORPH RL COMPARISON MASTER SEQUENCER

This script automates the sequential training of all three RL agents:
  1. Proximal Policy Optimization (PPO)
  2. Deep Q-Network (DQN)
  3. Advantage Actor-Critic (A2C)

WHAT THIS DOES:
  - Sequence: PPO -> DQN -> A2C
  - Shares the identical seed (42 for Phase A) for fair comparison
  - Logs durations and coordinates folder structure
  - Prepares the CSV outputs for the visualization script

RUNNING:
  python run_all.py [--phase A/B]
────────────────────────────────────────────────────────────────
"""

import sys
import time
import argparse
from train_ppo import train as train_ppo
from train_dqn import train as train_dqn
from train_a2c import train as train_a2c


def run_pipeline(phase: str):
    """
    Runs the comparison pipeline.
    
    Args:
        phase: 'A' for single-seed validation (seed=[42])
               'B' for full statistical comparison (seeds=[42, 43, 44, 45, 46])
    """
    
    # Define seeds based on phase
    if phase.upper() == 'A':
        seeds = [42]
        phase_name = "Phase A (Single Seed Validation)"
    elif phase.upper() == 'B':
        seeds = [42, 43, 44, 45, 46]
        phase_name = "Phase B (5-Seed Statistical Baseline)"
    else:
        print(f"Error: Unknown phase '{phase}'. Choose 'A' or 'B'.")
        return
        
    print("\n" + "="*80)
    print(f"          CYBERMORPH RL COMPARISON SUITE — {phase_name.upper()}")
    print("="*80)
    print(f"  Target Seeds    : {seeds}")
    print(f"  Algorithms      : PPO, DQN, A2C")
    print(f"  Total Steps/Run : 300,000 steps per agent")
    print("="*80)
    
    start_time = time.time()
    durations = {}
    
    # ─── 1. PPO TRAINING ───
    print("\n>>> Starting PPO Training Phase...")
    ppo_start = time.time()
    try:
        train_ppo(seeds=seeds)
        durations['PPO'] = time.time() - ppo_start
    except Exception as e:
        print(f"\n[FAILED] PPO Training failed: {e}")
        durations['PPO'] = "FAILED"
        
    # ─── 2. DQN TRAINING ───
    print("\n>>> Starting DQN Training Phase...")
    dqn_start = time.time()
    try:
        train_dqn(seeds=seeds)
        durations['DQN'] = time.time() - dqn_start
    except Exception as e:
        print(f"\n[FAILED] DQN Training failed: {e}")
        durations['DQN'] = "FAILED"
        
    # ─── 3. A2C TRAINING ───
    print("\n>>> Starting A2C Training Phase...")
    a2c_start = time.time()
    try:
        train_a2c(seeds=seeds)
        durations['A2C'] = time.time() - a2c_start
    except Exception as e:
        print(f"\n[FAILED] A2C Training failed: {e}")
        durations['A2C'] = "FAILED"
        
    # ─── 4. SUMMARY REPORT ───
    total_duration = time.time() - start_time
    
    print("\n" + "="*80)
    print("                      COMPARISON RUN COMPLETE SUMMARY")
    print("="*80)
    for algo, duration in durations.items():
        if isinstance(duration, float):
            print(f"  • {algo:<10} : Completed in {duration/60:.2f} minutes")
        else:
            print(f"  • {algo:<10} : {duration}")
    print(f"\n  • Total Pipeline Time: {total_duration/60:.2f} minutes")
    print("="*80)
    print("\nNext Step:")
    print("  Run plotting script to generate charts:")
    print("  python plot_metrics.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all CyberMorph RL comparison runs.")
    parser.add_argument(
        "--phase",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Comparison Phase. 'A' (1 seed, fast validation) or 'B' (5 seeds, final statistics)."
    )
    
    args = parser.parse_args()
    run_pipeline(args.phase)
