#!/usr/bin/env python3
"""
Compare multiple trained Balatro models side-by-side
Usage: python compare_models.py model1.zip model2.zip model3.zip --episodes 20
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from balatro_env import BalatroEnv
from tabulate import tabulate  # pip install tabulate
import matplotlib.pyplot as plt


def evaluate_single_model(model_path, num_episodes=20, verbose=False):
    """Evaluate a single model"""
    if verbose:
        print(f"Evaluating {model_path}...")
    
    env = BalatroEnv()
    model = PPO.load(model_path)
    
    rewards = []
    antes = []
    wins = 0
    hands_used = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_hands = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            
            if info.get("action_type") == "play":
                ep_hands += 1
        
        rewards.append(ep_reward)
        antes.append(info.get("ante", 0))
        hands_used.append(ep_hands)
        if info.get("won", False):
            wins += 1
    
    env.close()
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_ante": np.mean(antes),
        "max_ante": np.max(antes),
        "win_rate": wins / num_episodes,
        "mean_hands": np.mean(hands_used),
        "all_rewards": rewards,
        "all_antes": antes
    }


def compare_models(model_paths, num_episodes=20, plot=False):
    """Compare multiple models"""
    
    print(f"\nEvaluating {len(model_paths)} models with {num_episodes} episodes each...\n")
    
    results = {}
    for i, path in enumerate(model_paths, 1):
        print(f"[{i}/{len(model_paths)}] {path}...")
        results[path] = evaluate_single_model(path, num_episodes, verbose=False)
    
    # Create comparison table
    table_data = []
    for model_name, stats in results.items():
        # Shorten model name for display
        short_name = model_name.split('/')[-1].replace('.zip', '')
        table_data.append([
            short_name,
            f"{stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}",
            f"{stats['mean_ante']:.2f}",
            f"{stats['max_ante']}",
            f"{stats['win_rate']*100:.1f}%",
            f"{stats['mean_hands']:.1f}"
        ])
    
    headers = ["Model", "Reward (μ±σ)", "Avg Ante", "Max Ante", "Win Rate", "Avg Hands"]
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")
    
    # Find best models
    best_reward_model = max(results.items(), key=lambda x: x[1]['mean_reward'])
    best_ante_model = max(results.items(), key=lambda x: x[1]['mean_ante'])
    best_winrate_model = max(results.items(), key=lambda x: x[1]['win_rate'])
    
    print("WINNERS:")
    print(f"  🏆 Best Reward: {best_reward_model[0]} ({best_reward_model[1]['mean_reward']:.1f})")
    print(f"  🏆 Best Ante: {best_ante_model[0]} ({best_ante_model[1]['mean_ante']:.2f})")
    print(f"  🏆 Best Win Rate: {best_winrate_model[0]} ({best_winrate_model[1]['win_rate']*100:.1f}%)")
    
    # Plotting
    if plot:
        plot_comparison(results)
    
    return results


def plot_comparison(results):
    """Create visualization comparing models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    short_names = [m.split('/')[-1].replace('.zip', '') for m in models]
    
    # 1. Mean Rewards
    mean_rewards = [results[m]['mean_reward'] for m in models]
    std_rewards = [results[m]['std_reward'] for m in models]
    axes[0, 0].bar(short_names, mean_rewards, yerr=std_rewards, capsize=5)
    axes[0, 0].set_title('Mean Episode Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Win Rate
    win_rates = [results[m]['win_rate'] * 100 for m in models]
    axes[0, 1].bar(short_names, win_rates, color='green', alpha=0.7)
    axes[0, 1].set_title('Win Rate')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Average Ante Reached
    mean_antes = [results[m]['mean_ante'] for m in models]
    axes[1, 0].bar(short_names, mean_antes, color='orange', alpha=0.7)
    axes[1, 0].set_title('Average Ante Reached')
    axes[1, 0].set_ylabel('Ante')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Reward Distribution
    for i, model in enumerate(models):
        axes[1, 1].hist(results[model]['all_rewards'], alpha=0.5, 
                       label=short_names[i], bins=15)
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison plot saved as 'model_comparison.png'\n")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple Balatro models")
    parser.add_argument("models", nargs="+", help="Paths to model files (.zip)")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes per model")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    if len(args.models) < 2:
        print("Error: Please provide at least 2 models to compare")
        return
    
    compare_models(args.models, args.episodes, args.plot)


if __name__ == "__main__":
    main()