"""
Replay Viewer
=============
View and analyze recorded Balatro RL replays.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_replay(filepath: str) -> Dict[str, Any]:
    """Load a replay file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_replay_summary(replay: Dict[str, Any]):
    """Print a summary of the replay."""
    stats = replay.get("stats", {})

    print("\n" + "="*60)
    print("REPLAY SUMMARY")
    print("="*60)
    print(f"Model: {replay.get('model', 'unknown')}")
    print(f"Timestamp: {replay.get('timestamp', 'unknown')}")
    print(f"Seed: {replay.get('seed', 'unknown')}")
    print(f"\nInitial State:")
    print(f"  Ante: {replay.get('initial_ante', 1)}")
    print(f"  Money: ${replay.get('initial_money', 0)}")
    print(f"\nFinal Stats:")
    print(f"  Blinds Won: {stats.get('blinds_won', 0)}")
    print(f"  Max Ante: {stats.get('max_ante', 0)}")
    print(f"  Hands Played: {stats.get('hands_played', 0)}")
    print(f"  Discards Used: {stats.get('discards_used', 0)}")
    print(f"  Money Earned: ${stats.get('money_earned', 0)}")
    print(f"  Jokers Bought: {stats.get('jokers_bought', 0)}")
    print(f"\nTotal Actions: {replay.get('total_actions', 0)}")
    print("="*60 + "\n")


def print_actions(replay: Dict[str, Any], limit: int = None):
    """Print all actions in the replay."""
    actions = replay.get("actions", [])

    if limit:
        actions = actions[:limit]

    print("\nACTIONS:")
    print("-" * 60)

    for i, action in enumerate(actions, 1):
        action_type = action.get("action_type", "unknown")
        action_idx = action.get("action_index", 0)
        reward = action.get("reward", 0)
        state = action.get("state_snapshot", {})

        print(f"\n{i}. {action_type.upper()} (index={action_idx}) | Reward: {reward:+.1f}")
        print(f"   State: Ante {state.get('ante', '?')}, Round {state.get('round', '?')}, "
              f"${state.get('money', 0)}, {state.get('chips', 0)}/{state.get('blind_target', 0)} chips, "
              f"{state.get('hands_left', 0)} hands")

        # Show context-specific info
        if "item_bought" in action:
            item = action["item_bought"]
            print(f"   Bought: {item.get('name', '?')} ({item.get('type', '?')}) for ${item.get('cost', 0)}")

        if "cards_selected" in action:
            cards = action["cards_selected"]
            print(f"   Cards: {cards}")

        if "blind_type" in action:
            print(f"   Blind: {action['blind_type']}")

    if limit and len(replay.get("actions", [])) > limit:
        remaining = len(replay.get("actions", [])) - limit
        print(f"\n... and {remaining} more actions")


def main():
    parser = argparse.ArgumentParser(description="View Balatro RL replays")
    parser.add_argument("replay_file", type=str, help="Path to replay JSON file")
    parser.add_argument("--actions", type=int, default=None,
                       help="Number of actions to show (default: all)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only show summary, not actions")

    args = parser.parse_args()

    # Load replay
    replay_path = Path(args.replay_file)
    if not replay_path.exists():
        print(f"Error: File not found: {args.replay_file}")
        return

    replay = load_replay(args.replay_file)

    # Print summary
    print_replay_summary(replay)

    # Print actions if requested
    if not args.summary_only:
        print_actions(replay, limit=args.actions)


if __name__ == "__main__":
    main()
