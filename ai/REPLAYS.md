# Balatro RL Replay System

The replay recording system automatically saves successful runs to help analyze bot behavior and winning strategies.

## How It Works

- **Automatic Recording**: Every episode is recorded from start to finish
- **Saves Only Successful Runs**: Only episodes where the bot wins at least 1 blind are saved
- **One File Per Model**: Each training run gets its own replay files (based on model name)

## What Gets Recorded

Each replay file contains:

- **Game Seed**: The exact seed used for the run (for reproducibility)
- **All Actions**: Every action the bot took with rewards
- **State Snapshots**: Game state after each action (ante, money, chips, etc.)
- **Context Info**: What was bought, which cards were selected, etc.
- **Final Statistics**: Blinds won, max ante reached, total hands played, etc.

## Replay Files

Replays are saved in the `replays/` directory with the format:

```
replays/
  balatro_ppo_20251212_010203_20251212_010530.json
  balatro_ppo_20251212_010203_20251212_011015.json
  ...
```

Filename format: `{model_name}_{timestamp}.json`

## Viewing Replays

Use the provided viewer script:

```bash
# View full replay
python view_replay.py replays/balatro_ppo_20251212_010203_20251212_010530.json

# View summary only
python view_replay.py replays/balatro_ppo_20251212_010203_20251212_010530.json --summary-only

# View first 20 actions
python view_replay.py replays/balatro_ppo_20251212_010203_20251212_010530.json --actions 20
```

## Example Output

```
============================================================
REPLAY SUMMARY
============================================================
Model: balatro_ppo_20251212_010203
Timestamp: 2025-12-12T01:05:30.123456
Seed: 1234567890

Initial State:
  Ante: 1
  Money: $4

Final Stats:
  Blinds Won: 3
  Max Ante: 2
  Hands Played: 8
  Discards Used: 2
  Money Earned: $25
  Jokers Bought: 2

Total Actions: 45
============================================================

ACTIONS:
------------------------------------------------------------

1. SELECT_BLIND (index=0) | Reward: +1.0
   State: Ante 1, Round 1, $4, 0/300 chips, 0 hands

2. PLAY (index=0) | Reward: +5.2
   State: Ante 1, Round 1, $4, 120/300 chips, 3 hands
   Cards: [0, 1, 4]

3. PLAY (index=0) | Reward: +8.5
   State: Ante 1, Round 1, $4, 350/300 chips, 2 hands
   Cards: [0, 2, 3, 4]

4. CASH_OUT (index=0) | Reward: +12.0
   State: Ante 1, Round 1, $8, 350/0 chips, 2 hands

5. BUY (index=1) | Reward: +5.0
   State: Ante 1, Round 1, $4, 0/0 chips, 4 hands
   Bought: Joker Stencil (joker) for $4

... and 40 more actions
```

## Replaying a Run

To replay a specific seed:

1. Find the seed in the replay file
2. Start Balatro with that seed
3. Follow the actions in order

This is useful for:
- Understanding why certain strategies worked
- Debugging unexpected behavior
- Creating training demonstrations
- Analyzing decision patterns

## Integration with Training

The replay system is automatically integrated with the training pipeline:

- Each training run creates a unique model name (e.g., `balatro_ppo_20251212_010203`)
- All successful episodes from that run are saved with that model name
- Files are automatically overwritten if you reuse a model name (useful for testing)
