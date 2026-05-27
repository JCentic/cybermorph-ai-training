"""
plot_metrics.py
────────────────────────────────────────────────────────────────
CYBERMORPH RL COMPARISON VISUALIZATION & REPORTING ENGINE

This script reads the multi-seed evaluation logs (PPO, DQN, A2C)
from the results/ directory and generates:
  1. charts/reward_comparison.png (aggregated learning curves + shading)
  2. charts/diversity_comparison.png (attack diversity over time)
  3. charts/summary_table.md (step 300,000 final comparative report)

WHAT IT DOES:
  - Automatically aggregates multi-seed data (grouping by step)
  - Computes average performance and standard error bounds
  - Exports publication-quality visual plots into the charts/ folder
────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_comparison_plots():
    # Folder paths
    results_dir = "results"
    charts_dir = "charts"
    
    # Make sure the charts output directory exists
    os.makedirs(charts_dir, exist_ok=True)
    
    ppo_path = os.path.join(results_dir, "ppo_results.csv")
    dqn_path = os.path.join(results_dir, "dqn_results.csv")
    a2c_path = os.path.join(results_dir, "a2c_results.csv")
    
    # Load available datasets
    raw_data = {}
    if os.path.exists(ppo_path):
        raw_data['PPO'] = pd.read_csv(ppo_path)
    if os.path.exists(dqn_path):
        raw_data['DQN'] = pd.read_csv(dqn_path)
    if os.path.exists(a2c_path):
        raw_data['A2C'] = pd.read_csv(a2c_path)
        
    if not raw_data:
        print("Error: No training results found in results/ folder. Run your training scripts first!")
        return
        
    print(f"Found results for: {list(raw_data.keys())}. Aggregating multi-seed metrics...")
    
    # Aggregate data by Step (Averaging across the 5 seeds)
    aggregated_data = {}
    for algo, df in raw_data.items():
        # Group identical step intervals together (e.g. all 5 rows for step 10,000)
        # and calculate the mean for each column
        df_agg = df.groupby('step').mean().reset_index()
        aggregated_data[algo] = df_agg
        print(f"  • {algo:<4} : Aggregated {len(df)} raw rows down to {len(df_agg)} clean step steps.")
        
        # Export a clean, Excel-ready averaged CSV file to a dedicated results_averaged/ directory
        averaged_dir = "results_averaged"
        os.makedirs(averaged_dir, exist_ok=True)
        clean_csv_path = os.path.join(averaged_dir, f"{algo.lower()}_results_averaged.csv")
        df_agg.to_csv(clean_csv_path, index=False)
        print(f"✓ Saved clean averaged CSV: {clean_csv_path}")

    # Apply clean premium grid styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_color = '#F8F9FA'
    
    # ─── PLOT 1: MEAN REWARDS COMPARISON (WITH SD SHADING) ───
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=fig_color)
    ax.set_facecolor('#FFFFFF')
    
    colors = {'PPO': '#2E86C1', 'DQN': '#E74C3C', 'A2C': '#2ECC71'}
    
    for algo, df in aggregated_data.items():
        if 'step' in df.columns and 'mean_reward' in df.columns:
            # Draw mean line
            ax.plot(df['step'], df['mean_reward'], label=f"{algo} (Mean)", color=colors[algo], linewidth=2.5)
            
            # Fill standard deviation bounds (showing stability across seeds/profiles)
            if 'reward_std' in df.columns:
                lower = df['mean_reward'] - df['reward_std']
                upper = df['mean_reward'] + df['reward_std']
                ax.fill_between(df['step'], lower, upper, color=colors[algo], alpha=0.12)
                
    ax.set_title("CyberMorph RL Agent Learning Curves (5-Seed Average)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Training Steps", fontsize=12, labelpad=10)
    ax.set_ylabel("Evaluation Mean Reward (Shaded = ±1 Std Dev)", fontsize=12, labelpad=10)
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    reward_plot_path = os.path.join(charts_dir, "reward_comparison.png")
    plt.tight_layout()
    plt.savefig(reward_plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved reward comparison plot: {reward_plot_path}")
    
    # ─── PLOT 2: ATTACK DIVERSITY COMPARISON ───
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=fig_color)
    ax.set_facecolor('#FFFFFF')
    
    for algo, df in aggregated_data.items():
        if 'step' in df.columns and 'unique_attacks_used' in df.columns:
            ax.plot(df['step'], df['unique_attacks_used'], label=f"{algo} Attack Diversity", color=colors[algo], linewidth=2.5, linestyle='--')
            
    ax.set_title("CyberMorph RL Agent Attack Diversity Over Time (5-Seed Average)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Training Steps", fontsize=12, labelpad=10)
    ax.set_ylabel("Mean Unique Attack Types Used (Max: 8.0)", fontsize=12, labelpad=10)
    ax.set_ylim(0, 8.5)
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    div_plot_path = os.path.join(charts_dir, "diversity_comparison.png")
    plt.tight_layout()
    plt.savefig(div_plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved diversity comparison plot: {div_plot_path}")
    
    # ─── PLOT 3: STATISTICAL SUMMARY TABLE ───
    report_path = os.path.join(charts_dir, "summary_table.md")
    
    report_lines = [
        "# CyberMorph RL Algorithm Comparison - Performance Baseline",
        "\nThis table summarizes the final statistical benchmarks captured at step 300,000, averaged across all 5 random seeds:\n",
        "| Algorithm | Final Mean Reward (Score) | Final Consistency (Std Dev) | Final Attack Diversity (/8) |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    for algo, df in aggregated_data.items():
        if not df.empty:
            last_row = df.iloc[-1]
            mean_r = last_row.get('mean_reward', 0.0)
            std_r = last_row.get('reward_std', 0.0)
            div_r = last_row.get('unique_attacks_used', 0.0)
            report_lines.append(f"| **{algo}** | {mean_r:.2f} | ± {std_r:.2f} | {div_r:.1f} / 8 |")
            
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"✓ Generated Capstone Summary Report: {report_path}")


if __name__ == "__main__":
    generate_comparison_plots()
