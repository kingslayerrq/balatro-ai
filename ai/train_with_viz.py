"""
Enhanced training script with better visualization
"""
import os
import configs
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from balatro_env import BalatroEnv

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better visualization: pip install rich")


class BalatroVisualizationCallback(BaseCallback):
    """
    Enhanced callback with rich visualization
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_antes = []
        self.wins = 0
        self.total_episodes = 0
        self.last_10_rewards = []
        self.last_10_antes = []

        if RICH_AVAILABLE:
            self.console = Console()

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.total_episodes += 1
                info = self.locals["infos"][idx]

                # Track metrics
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.episode_rewards.append(reward)
                    self.last_10_rewards.append(reward)
                    if len(self.last_10_rewards) > 10:
                        self.last_10_rewards.pop(0)

                if "ante" in info:
                    ante = info["ante"]
                    self.episode_antes.append(ante)
                    self.last_10_antes.append(ante)
                    if len(self.last_10_antes) > 10:
                        self.last_10_antes.pop(0)

                if info.get("won", False):
                    self.wins += 1

                # Display every 5 episodes
                if self.total_episodes % 5 == 0:
                    self._display_stats()

        return True

    def _display_stats(self):
        if not RICH_AVAILABLE:
            # Fallback to simple print
            avg_reward = sum(self.last_10_rewards) / len(self.last_10_rewards) if self.last_10_rewards else 0
            avg_ante = sum(self.last_10_antes) / len(self.last_10_antes) if self.last_10_antes else 0
            win_rate = (self.wins / self.total_episodes * 100) if self.total_episodes > 0 else 0

            print(f"\n{'='*60}")
            print(f"Episodes: {self.total_episodes}")
            print(f"Avg Reward (last 10): {avg_reward:.2f}")
            print(f"Avg Ante (last 10): {avg_ante:.2f}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"{'='*60}\n")
            return

        # Rich display
        table = Table(title=f"Training Progress - Episode {self.total_episodes}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        avg_reward = sum(self.last_10_rewards) / len(self.last_10_rewards) if self.last_10_rewards else 0
        avg_ante = sum(self.last_10_antes) / len(self.last_10_antes) if self.last_10_antes else 0
        win_rate = (self.wins / self.total_episodes * 100) if self.total_episodes > 0 else 0
        max_ante = max(self.episode_antes) if self.episode_antes else 0

        table.add_row("Total Episodes", str(self.total_episodes))
        table.add_row("Avg Reward (last 10)", f"{avg_reward:.2f}")
        table.add_row("Avg Ante (last 10)", f"{avg_ante:.2f}")
        table.add_row("Max Ante Reached", str(max_ante))
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Total Wins", str(self.wins))

        self.console.print(table)
        self.console.print()

        # Log to tensorboard
        if len(self.last_10_rewards) > 0:
            self.logger.record("balatro/avg_reward_10", avg_reward)
        if len(self.last_10_antes) > 0:
            self.logger.record("balatro/avg_ante_10", avg_ante)
        self.logger.record("balatro/win_rate", win_rate)
        self.logger.record("balatro/max_ante", max_ante)


def make_env(env_id=0):
    def _init():
        env = BalatroEnv()
        env = Monitor(env, filename=f"./logs/monitor_{env_id}.log")
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Balatro RL Agent with Visualization")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--config", type=str, default="default", choices=list(configs.CONFIGS.keys()))

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./models/{timestamp}"
    log_dir = f"./logs/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit(
            f"[bold cyan]Balatro RL Training[/bold cyan]\n\n"
            f"Session: {timestamp}\n"
            f"Models: {save_dir}\n"
            f"Logs: {log_dir}\n\n"
            f"[yellow]TensorBoard:[/yellow] tensorboard --logdir {log_dir}",
            border_style="green"
        ))
    else:
        print(f"{'=' * 60}")
        print(f"Training Session: {timestamp}")
        print(f"Models: {save_dir}")
        print(f"Logs: {log_dir}")
        print(f"TensorBoard: tensorboard --logdir {log_dir}")
        print(f"{'=' * 60}\n")

    # Create environment
    print("--- Initializing Environment ---")
    env = DummyVecEnv([make_env(0)])

    if args.normalize:
        print("Using observation normalization")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    print("--- Skipping Environment Check (socket issues) ---\n")

    # Create or load model
    if args.resume:
        print(f"--- Resuming from {args.resume} ---")
        model = PPO.load(args.resume, env=env, device=args.device)
        if args.lr != 3e-4:
            model.learning_rate = args.lr
    else:
        print("--- Creating New Model ---")
        config = configs.get_config(args.config)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            ent_coef=config["ent_coef"],
            clip_range=config["clip_range"],
            policy_kwargs=dict(net_arch=config["net_arch"], activation_fn=torch.nn.ReLU),
            tensorboard_log=log_dir,
            device=args.device,
        )

    print(f"\nModel Configuration:")
    print(f"  Device: {model.device}")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Batch Size: {model.batch_size}")
    print()

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="balatro_bot",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    viz_callback = BalatroVisualizationCallback()
    callback = CallbackList([checkpoint_callback, viz_callback])

    # Training
    print("--- Starting Training ---")
    print("Press Ctrl+C to stop\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        final_path = os.path.join(save_dir, "balatro_ppo_final")
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}.zip")

        if args.normalize:
            env.save(os.path.join(save_dir, "vec_normalize.pkl"))

        env.close()

        print("\n" + "=" * 60)
        print("Session Summary:")
        print(f"  Timesteps: {model.num_timesteps}")
        print(f"  Models: {save_dir}")
        print(f"  Logs: {log_dir}")
        print(f"\nResume: python train_with_viz.py --resume {final_path}.zip")
        print(f"TensorBoard: tensorboard --logdir {log_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
