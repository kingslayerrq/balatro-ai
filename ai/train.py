"""
Balatro RL Training Script
==========================
Training script for the Balatro environment using PPO with action masking.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# For action masking support with SB3
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    # [FIX] Import the ActionMasker wrapper
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_SB3_CONTRIB = True
except ImportError:
    HAS_SB3_CONTRIB = False
    print("Warning: sb3_contrib not installed. Action masking will use penalty method.")
    print("Install with: pip install sb3-contrib")

from balatro_env import BalatroEnv, BalatroEnvWithMasking
import configs


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class BalatroLoggingCallback(BaseCallback):
    """
    Custom callback for logging Balatro-specific metrics.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_stats = []
        self.best_ante = 0
        self.wins = 0
        self.total_episodes = 0
    
    def _on_step(self) -> bool:
        # Check for episode end
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                
                # Get info from the environment
                infos = self.locals.get("infos", [{}])
                info = infos[i] if i < len(infos) else {}
                
                ep_stats = info.get("episode_stats", {})
                max_ante = ep_stats.get("max_ante", 0)
                
                if max_ante > self.best_ante:
                    self.best_ante = max_ante
                    if self.verbose > 0:
                        print(f"New best ante: {max_ante}")
                
                # Check for win
                if max_ante >= 8:  # Assuming winning means beating ante 8
                    self.wins += 1
                
                # Log to tensorboard
                if self.logger:
                    self.logger.record("balatro/max_ante", max_ante)
                    self.logger.record("balatro/blinds_won", ep_stats.get("blinds_won", 0))
                    self.logger.record("balatro/hands_played", ep_stats.get("hands_played", 0))
                    self.logger.record("balatro/discards_used", ep_stats.get("discards_used", 0))
                    self.logger.record("balatro/jokers_bought", ep_stats.get("jokers_bought", 0))
                    self.logger.record("balatro/win_rate", self.wins / max(1, self.total_episodes))
                    self.logger.record("balatro/best_ante_ever", self.best_ante)
        
        return True
    
    def _on_training_end(self) -> None:
        print(f"\n{'='*50}")
        print("Training Complete!")
        print(f"Total episodes: {self.total_episodes}")
        print(f"Wins: {self.wins}")
        print(f"Win rate: {self.wins / max(1, self.total_episodes):.2%}")
        print(f"Best ante reached: {self.best_ante}")
        print(f"{'='*50}\n")


class ActionMaskingCallback(BaseCallback):
    """
    Callback that applies action masking penalty for invalid actions.
    Use this when sb3_contrib is not available.
    """
    
    def __init__(self, invalid_action_penalty: float = -10.0, verbose: int = 0):
        super().__init__(verbose)
        self.invalid_action_penalty = invalid_action_penalty
        self.invalid_actions = 0
        self.total_actions = 0
    
    def _on_step(self) -> bool:
        self.total_actions += 1
        
        # Check if action was invalid (look for invalid_action in info)
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info.get("invalid_action", False):
                self.invalid_actions += 1
        
        # Log invalid action rate
        if self.logger and self.total_actions % 1000 == 0:
            rate = self.invalid_actions / max(1, self.total_actions)
            self.logger.record("balatro/invalid_action_rate", rate)
        
        return True


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(rank: int, seed: int = 0, use_masking: bool = True):
    """
    Create a Balatro environment instance.
    
    Args:
        rank: Index of this environment (for port assignment)
        seed: Random seed
        use_masking: Whether to use the masking wrapper
    """
    def _init():
        port = 5000 + rank
        
        # [FIX] Use the Base Environment (Box Observation), NOT the WithMasking (Dict) one.
        env = BalatroEnv(port=port)
        
        if use_masking and HAS_SB3_CONTRIB:
            # [FIX] Wrap with ActionMasker so MaskablePPO can see the valid actions
            # This calls env.get_action_mask() automatically
            env = ActionMasker(env, lambda env: env.get_action_mask())
        
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    checkpoint_freq: int = 10000,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    resume_from: Optional[str] = None,
    use_masking: bool = True,
    device: str = "auto",
    verbose: int = 1,
):
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"balatro_ppo_{timestamp}"
    run_log_dir = os.path.join(log_dir, run_name)
    run_model_dir = os.path.join(model_dir, run_name)
    
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)
    
    print(f"Logging to: {run_log_dir}")
    print(f"Models to: {run_model_dir}")
    
    # Create environment
    env = DummyVecEnv([make_env(0, seed, use_masking)])
    
    # Select algorithm
    if use_masking and HAS_SB3_CONTRIB:
        print("Using MaskablePPO with action masking")
        algo_class = MaskablePPO
    else:
        print("Using standard PPO")
        algo_class = PPO
    
    # Create or load model
    if resume_from:
        print(f"Resuming from: {resume_from}")
        model = algo_class.load(resume_from, env=env, device=device)
        model.learning_rate = learning_rate  # Allow LR adjustment on resume
    else:
        # Policy kwargs for network architecture
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Policy network
                vf=[256, 256, 128],  # Value network
            ),
            activation_fn=torch.nn.ReLU,
        )
        
        model = algo_class(
            "MlpPolicy",  # [FIX] Back to MlpPolicy because we removed the Dict wrapper
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=run_log_dir,
            verbose=verbose,
            seed=seed,
            device=device,
        )
    
    # Create callbacks
    callbacks = [
        BalatroLoggingCallback(verbose=verbose),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=run_model_dir,
            name_prefix="balatro_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]
    
    if not (use_masking and HAS_SB3_CONTRIB):
        callbacks.append(ActionMaskingCallback(verbose=verbose))
    
    callback = CallbackList(callbacks)
    
    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Use Ctrl+C to stop training (model will be saved)")
    print("-" * 50)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = os.path.join(run_model_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")
    
    return model


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = True,
    verbose: int = 1,
):
    """Evaluate a trained model."""
    
    # Create environment
    env = BalatroEnv()
    
    # Load model
    if HAS_SB3_CONTRIB:
        try:
            model = MaskablePPO.load(model_path)
        except:
            model = PPO.load(model_path)
    else:
        model = PPO.load(model_path)
    
    print(f"Evaluating model: {model_path}")
    print(f"Running {n_episodes} episodes...")
    print("-" * 50)
    
    results = {
        "episode_rewards": [],
        "max_antes": [],
        "blinds_won": [],
        "wins": 0,
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get action
            action_mask = env.get_action_mask()
            
            if HAS_SB3_CONTRIB and isinstance(model, MaskablePPO):
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_mask)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if render:
                env.render()
            
            done = done or truncated
        
        # Record results
        ep_stats = info.get("episode_stats", {})
        max_ante = ep_stats.get("max_ante", 0)
        blinds = ep_stats.get("blinds_won", 0)
        
        results["episode_rewards"].append(episode_reward)
        results["max_antes"].append(max_ante)
        results["blinds_won"].append(blinds)
        
        if max_ante >= 8:
            results["wins"] += 1
        
        if verbose > 0:
            print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, "
                  f"Ante={max_ante}, Blinds={blinds}, Steps={step}")
    
    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {n_episodes}")
    print(f"Win rate: {results['wins'] / n_episodes:.1%}")
    print(f"Avg reward: {np.mean(results['episode_rewards']):.1f} ± {np.std(results['episode_rewards']):.1f}")
    print(f"Avg max ante: {np.mean(results['max_antes']):.1f}")
    print(f"Avg blinds won: {np.mean(results['blinds_won']):.1f}")
    print(f"Best ante: {max(results['max_antes'])}")
    print("=" * 50)
    
    env.close()
    return results


# =============================================================================
# HYPERPARAMETER SUGGESTIONS
# =============================================================================

def get_hyperparameters(difficulty: str = "default") -> Dict[str, Any]:
    presets = {
        "easy": {
            "learning_rate": 1e-4,
            "n_steps": 1024,
            "batch_size": 32,
            "n_epochs": 5,
            "gamma": 0.95,
            "ent_coef": 0.05,
            "clip_range": 0.3,
        },
        "default": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "clip_range": 0.2,
        },
        "hard": {
            "learning_rate": 1e-4,
            "n_steps": 4096,
            "batch_size": 128,
            "n_epochs": 15,
            "gamma": 0.995,
            "ent_coef": 0.005,
            "clip_range": 0.15,
        },
        "expert": {
            "learning_rate": 5e-5,
            "n_steps": 8192,
            "batch_size": 256,
            "n_epochs": 20,
            "gamma": 0.999,
            "ent_coef": 0.001,
            "clip_range": 0.1,
        },
    }
    
    return presets.get(difficulty, presets["default"])


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a Balatro RL agent")
    
    # Mode
    parser.add_argument("mode", choices=["train", "evaluate", "test_env"],
                       help="Mode: train, evaluate, or test_env")
    
    # Training args
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--preset", choices=["easy", "default", "hard", "expert"],
                       default="default", help="Hyperparameter preset")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model to resume training")
    parser.add_argument("--no-masking", action="store_true",
                       help="Disable action masking")
    
    # Evaluation args
    parser.add_argument("--model", type=str, help="Model path for evaluation")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        hyperparams = get_hyperparameters(args.preset)
        train(
            total_timesteps=args.timesteps,
            **hyperparams,
            seed=args.seed,
            resume_from=args.resume,
            use_masking=not args.no_masking,
            device=args.device,
            verbose=args.verbose,
        )
    
    elif args.mode == "evaluate":
        if not args.model:
            print("Error: --model required for evaluation")
            sys.exit(1)
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            verbose=args.verbose,
        )
    
    elif args.mode == "test_env":
        print("Testing environment...")
        env = BalatroEnv()
        if not args.no_masking and HAS_SB3_CONTRIB:
             env = ActionMasker(env, lambda env: env.get_action_mask())
             
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print("\nWaiting for game connection...")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Check for mask validity
        mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else info.get('action_mask')
        if mask is not None:
             print(f"Action mask shape: {mask.shape}")
             print(f"Valid actions: {np.sum(mask)}")
        
        env.render()
        env.close()


if __name__ == "__main__":
    main()