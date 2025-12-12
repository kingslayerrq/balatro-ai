"""
Replay Recording System for Balatro RL
========================================
Records successful runs with all actions and game state for replay analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class ReplayRecorder:
    """Records game actions and state for successful runs."""

    def __init__(self, model_name: str = "default", replay_dir: str = "replays"):
        """
        Initialize the replay recorder.

        Args:
            model_name: Name of the model being used (for filename)
            replay_dir: Directory to save replay files
        """
        self.model_name = model_name
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(exist_ok=True)

        # Current recording
        self.recording = False
        self.current_replay: Dict[str, Any] = {}
        self.actions: List[Dict[str, Any]] = []

    def start_recording(self, initial_state: Dict[str, Any]):
        """Start recording a new episode."""
        self.recording = True
        self.actions = []

        self.current_replay = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "seed": initial_state.get("seed", "unknown"),
            "initial_ante": initial_state.get("ante", 1),
            "initial_money": initial_state.get("money", 0),
            "actions": [],
            "stats": {
                "blinds_won": 0,
                "hands_played": 0,
                "discards_used": 0,
                "money_earned": 0,
                "jokers_bought": 0,
                "max_ante": 0,
            }
        }

    def record_action(self, action_type: str, action_index: int,
                      state_before: Dict[str, Any],
                      state_after: Dict[str, Any],
                      reward: float):
        """
        Record a single action with before/after state.

        Args:
            action_type: Type of action (e.g., "play", "buy", "select_blind")
            action_index: Index of the action
            state_before: Game state before action
            state_after: Game state after action
            reward: Reward received for this action
        """
        if not self.recording:
            return

        action_record = {
            "action_type": action_type,
            "action_index": action_index,
            "reward": round(reward, 2),
            "state_snapshot": {
                "ante": state_after.get("ante", 1),
                "round": state_after.get("round", 1),
                "money": state_after.get("money", 0),
                "chips": state_after.get("chips", 0),
                "blind_target": state_after.get("blind_target", 0),
                "hands_left": state_after.get("hands_left", 0),
                "phase": state_after.get("phase", "unknown"),
            }
        }

        # Add context-specific info based on action type
        if action_type == "play" or action_type == "discard":
            # Record which cards were selected
            hand = state_before.get("hand", [])
            selected = [i for i, card in enumerate(hand) if card.get("selected", False)]
            action_record["cards_selected"] = selected

        elif action_type == "buy":
            # Record what was bought
            shop = state_before.get("shop", [])
            if action_index < len(shop):
                item = shop[action_index]
                action_record["item_bought"] = {
                    "type": item.get("type", "unknown"),
                    "name": item.get("name", "unknown"),
                    "cost": item.get("cost", 0)
                }

        elif action_type == "select_blind" or action_type == "skip_blind":
            # Record blind choice
            action_record["blind_type"] = state_after.get("blind_select", {})

        self.actions.append(action_record)

    def update_stats(self, stats: Dict[str, Any]):
        """Update episode statistics."""
        if not self.recording:
            return
        self.current_replay["stats"] = stats.copy()

    def finish_recording(self, success: bool = True, final_stats: Optional[Dict[str, Any]] = None):
        """
        Finish recording and save if successful.

        Args:
            success: Whether the run was successful (won at least one blind)
            final_stats: Final episode statistics
        """
        if not self.recording:
            return

        self.recording = False

        if final_stats:
            self.current_replay["stats"] = final_stats.copy()

        self.current_replay["actions"] = self.actions
        self.current_replay["total_actions"] = len(self.actions)

        # Only save if we won at least one blind
        blinds_won = self.current_replay["stats"].get("blinds_won", 0)
        if success and blinds_won > 0:
            self._save_replay()
        else:
            # Don't save unsuccessful runs
            pass

    def _save_replay(self):
        """Save the current replay to a file."""
        # Generate filename: modelname_YYYYMMDD_HHMMSS.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_{timestamp}.json"
        filepath = self.replay_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_replay, f, indent=2)

            stats = self.current_replay["stats"]
            print(f"\n{'='*60}")
            print(f"REPLAY SAVED: {filename}")
            print(f"{'='*60}")
            print(f"Seed: {self.current_replay['seed']}")
            print(f"Blinds Won: {stats.get('blinds_won', 0)}")
            print(f"Max Ante: {stats.get('max_ante', 0)}")
            print(f"Total Actions: {self.current_replay['total_actions']}")
            print(f"Hands Played: {stats.get('hands_played', 0)}")
            print(f"Money Earned: {stats.get('money_earned', 0)}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"ERROR: Failed to save replay: {e}")

    def get_replay_summary(self) -> str:
        """Get a summary of the current replay."""
        if not self.current_replay:
            return "No replay recorded"

        stats = self.current_replay.get("stats", {})
        return (f"Seed: {self.current_replay.get('seed', 'unknown')} | "
                f"Blinds: {stats.get('blinds_won', 0)} | "
                f"Ante: {stats.get('max_ante', 0)} | "
                f"Actions: {len(self.actions)}")
