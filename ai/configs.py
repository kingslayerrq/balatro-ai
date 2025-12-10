# configs.py - Different hyperparameter configurations to experiment with

CONFIGS = {
    "default": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "description": "Balanced configuration, good starting point"
    },
    
    "fast_learner": {
        "learning_rate": 5e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "n_epochs": 15,
        "gamma": 0.99,
        "ent_coef": 0.015,
        "clip_range": 0.2,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "description": "Faster learning, more exploration, smaller network"
    },
    
    "deep_thinker": {
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "ent_coef": 0.005,
        "clip_range": 0.2,
        "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
        "description": "Slower, deeper learning with larger network"
    },
    
    "explorer": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.05,  # High entropy = more exploration
        "clip_range": 0.2,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "description": "Emphasizes exploration, good for discovering strategies"
    },
    
    "stable": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.005,
        "clip_range": 0.1,  # Smaller clip range = more stable
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "description": "Stable, conservative learning"
    },
    
    "long_term": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.999,  # Values future rewards more
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "description": "Optimized for long-term strategy (high antes)"
    }
}


def get_config(name="default"):
    """Get configuration by name"""
    if name not in CONFIGS:
        print(f"Warning: Config '{name}' not found. Using 'default'")
        name = "default"
    
    config = CONFIGS[name].copy()
    description = config.pop("description")
    
    print(f"\nUsing configuration: {name}")
    print(f"Description: {description}")
    print(f"Parameters: {config}\n")
    
    return config


# Modified train.py to use configs:
"""
Add to train.py:

import configs

# In argument parser, add:
parser.add_argument("--config", type=str, default="default",
                   choices=list(configs.CONFIGS.keys()),
                   help="Training configuration preset")

# Then in model creation:
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
    policy_kwargs=dict(
        net_arch=config["net_arch"],
        activation_fn=torch.nn.ReLU
    ),
    tensorboard_log=log_dir,
    device=args.device
)

# Usage:
# python train.py --config fast_learner --timesteps 1000000
# python train.py --config deep_thinker --timesteps 2000000
"""


# Quick reference guide
HYPERPARAMETER_GUIDE = """
=============================================================================
HYPERPARAMETER TUNING GUIDE FOR BALATRO
=============================================================================

LEARNING RATE (learning_rate)
  - Too high (>1e-3): Unstable, agent forgets past learning
  - Too low (<1e-5): Very slow learning
  - Recommended: 3e-4 (default) or 1e-4 (stable)

N_STEPS (n_steps)
  - Higher = More stable, but slower updates
  - Lower = Faster updates, but higher variance
  - Recommended: 2048 (default), 4096 (for stability)

BATCH SIZE (batch_size)
  - Must divide n_steps evenly
  - Higher = More stable gradients, slower training
  - Lower = Faster training, higher variance
  - Recommended: 64 (default), 128 (for stability)

GAMMA (gamma)
  - Discount factor for future rewards
  - Higher (0.999) = Values long-term strategy
  - Lower (0.95) = Values immediate rewards
  - Recommended: 0.99 (balanced), 0.995 (long-term planning)

ENTROPY COEFFICIENT (ent_coef)
  - Controls exploration vs exploitation
  - Higher (0.05) = More random exploration
  - Lower (0.001) = More deterministic, exploits known strategies
  - Recommended: 0.01 (default), 0.05 (if stuck in local optimum)

NETWORK ARCHITECTURE (net_arch)
  - Deeper/wider = More capacity, slower training, needs more data
  - Shallower/narrower = Faster training, may underfit
  - Recommended: [256, 256] (default), [512, 512, 256] (complex)

=============================================================================
TROUBLESHOOTING
=============================================================================

IF AGENT ISN'T LEARNING:
  ✓ Increase learning_rate to 5e-4
  ✓ Increase ent_coef to 0.05 (more exploration)
  ✓ Decrease n_steps to 1024 (faster updates)
  ✓ Check reward function - are rewards too sparse?

IF LEARNING IS UNSTABLE:
  ✓ Decrease learning_rate to 1e-4
  ✓ Increase batch_size to 128
  ✓ Decrease clip_range to 0.1
  ✓ Increase n_steps to 4096

IF AGENT GETS STUCK IN LOCAL OPTIMUM:
  ✓ Increase ent_coef to 0.05
  ✓ Try "explorer" config
  ✓ Add curriculum learning (start with easier blinds)

IF TRAINING IS TOO SLOW:
  ✓ Try "fast_learner" config
  ✓ Reduce network size to [128, 128]
  ✓ Reduce n_epochs to 5

=============================================================================
"""

if __name__ == "__main__":
    print(HYPERPARAMETER_GUIDE)
    print("\nAvailable Configurations:")
    print("="*60)
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  {config['description']}")