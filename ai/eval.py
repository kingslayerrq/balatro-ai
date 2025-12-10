import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from balatro_env import BalatroEnv
import time


def evaluate_model(model_path, num_episodes=10, render=False, deterministic=True):
    """
    Evaluate a trained model over multiple episodes
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment
    env = BalatroEnv()
    
    # Load model
    model = PPO.load(model_path)
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    antes_reached = []
    wins = 0
    total_hands_used = []
    
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("="*60)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        hands_used = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track hands used
            if info.get("action_type") == "play":
                hands_used += 1
            
            if render:
                time.sleep(0.1)  # Slow down for viewing
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        ante = info.get("ante", 0)
        antes_reached.append(ante)
        
        won = info.get("won", False)
        if won:
            wins += 1
        
        total_hands_used.append(hands_used)
        
        # Print episode summary
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Ante Reached: {ante}")
        print(f"  Hands Used: {hands_used}")
        print(f"  Result: {'WIN' if won else 'LOSS'}")
    
    # Print overall statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Win Rate: {(wins/num_episodes)*100:.1f}%")
    print(f"  Avg Ante: {np.mean(antes_reached):.2f}")
    print(f"  Max Ante: {np.max(antes_reached)}")
    print(f"  Avg Hands Used: {np.mean(total_hands_used):.1f}")
    print(f"  Avg Episode Length: {np.mean(episode_lengths):.0f} steps")
    print("="*60)
    
    env.close()
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "win_rate": wins/num_episodes,
        "avg_ante": np.mean(antes_reached),
        "max_ante": np.max(antes_reached)
    }


def watch_agent(model_path, num_episodes=3):
    """
    Watch the agent play with detailed output
    """
    print(f"Loading model from: {model_path}")
    
    env = BalatroEnv()
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*60}\n")
        
        obs = env.reset()
        done = False
        step = 0
        
        while not done:
            # Get action
            action, _states = model.predict(obs, deterministic=True)
            
            # Display state before action
            state = env.get_state()
            if state:
                print(f"\nStep {step}:")
                print(f"  Chips: {state['chips']}/{state['blind_target']}")
                print(f"  Hands: {state['hands_left']} | Discards: {state['discards_left']}")
                print(f"  Ante: {state['ante']}")
                
                # Show hand
                if state.get('hand'):
                    hand_str = ", ".join([f"{c['rank']}{c['suit'][0]}" for c in state['hand']])
                    print(f"  Hand: {hand_str}")
                
                # Show action
                action_types = ["Toggle", "Play", "Discard"]
                if action[0] < len(action_types):
                    if action[0] == 0:
                        print(f"  Action: {action_types[action[0]]} card {action[1]}")
                    else:
                        print(f"  Action: {action_types[action[0]]}")
            
            # Take action
            obs, reward, done, info = env.step(action)
            
            if reward != 0:
                print(f"  Reward: {reward:+.2f}")
            
            step += 1
            time.sleep(0.3)  # Slow down for viewing
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} COMPLETE")
        print(f"  Final Ante: {info.get('ante', 0)}")
        print(f"  Result: {'WIN' if info.get('won', False) else 'LOSS'}")
        print(f"{'='*60}\n")
        
        input("Press Enter to continue to next episode...")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Balatro RL Agent")
    parser.add_argument("model_path", type=str,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--watch", action="store_true",
                       help="Watch agent play with detailed output")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (default: deterministic)")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_agent(args.model_path, num_episodes=min(args.episodes, 5))
    else:
        evaluate_model(
            args.model_path,
            num_episodes=args.episodes,
            deterministic=not args.stochastic
        )


if __name__ == "__main__":
    main()