"""
Balatro RL Environment v2
=========================
A comprehensive Gymnasium environment for training RL agents on Balatro.

Key improvements over v1:
- Phase-aware action space (Blind, Shop, Blind Selection)
- Expanded observations (jokers, consumables, hand levels, shop)
- Action masking for valid actions only
- Normalized rewards that scale with ante
- Proper shop phase handling
"""

import socket
import time
import json
import logging
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class GamePhase(IntEnum):
    BLIND = 0
    BLIND_SELECT = 1
    SHOP = 2
    PACK_OPENING = 3
    BLIND_COMPLETE = 4  # Screen after beating/failing a blind (cash out button)
    GAME_OVER = 5


class BlindAction(IntEnum):
    TOGGLE = 0
    PLAY = 1
    DISCARD = 2


class ShopAction(IntEnum):
    BUY_ITEM = 0       # Buy from shop slot
    SELL_JOKER = 1     # Sell a joker
    USE_CONSUMABLE = 2 # Use tarot/planet/spectral
    REROLL = 3         # Reroll shop
    NEXT_ROUND = 4     # Leave shop


class BlindSelectAction(IntEnum):
    SELECT_BLIND = 0   # Play this blind
    SKIP_BLIND = 1     # Skip (if tag available)


class PackAction(IntEnum):
    SELECT_CARD = 0    # Choose a card from pack
    SKIP = 1           # Skip remaining picks


# Joker encoding - common jokers mapped to indices
# You'll want to expand this based on which jokers you want to track
JOKER_TYPES = [
    "Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker",
    "Jolly Joker", "Zany Joker", "Mad Joker", "Crazy Joker", "Droll Joker",
    "Sly Joker", "Wily Joker", "Clever Joker", "Devious Joker", "Crafty Joker",
    "Half Joker", "Joker Stencil", "Four Fingers", "Mime", "Credit Card",
    "Ceremonial Dagger", "Banner", "Mystic Summit", "Marble Joker", "Loyalty Card",
    "8 Ball", "Misprint", "Dusk", "Raised Fist", "Chaos the Clown",
    "Fibonacci", "Steel Joker", "Scary Face", "Abstract Joker", "Delayed Gratification",
    "Hack", "Pareidolia", "Gros Michel", "Even Steven", "Odd Todd",
    "Scholar", "Business Card", "Supernova", "Ride the Bus", "Space Joker",
    "Egg", "Burglar", "Blackboard", "Runner", "Ice Cream",
    "DNA", "Splash", "Blue Joker", "Sixth Sense", "Constellation",
    "Hiker", "Faceless Joker", "Green Joker", "Superposition", "To Do List",
    "Cavendish", "Card Sharp", "Red Card", "Madness", "Square Joker",
    "Seance", "Riff-raff", "Vampire", "Shortcut", "Hologram",
    "Vagabond", "Baron", "Cloud 9", "Rocket", "Obelisk",
    "Midas Mask", "Luchador", "Photograph", "Gift Card", "Turtle Bean",
    "Erosion", "Reserved Parking", "Mail-In Rebate", "To the Moon", "Hallucination",
    "Fortune Teller", "Juggler", "Drunkard", "Stone Joker", "Golden Joker",
    "Lucky Cat", "Baseball Card", "Bull", "Diet Cola", "Trading Card",
    "Flash Card", "Popcorn", "Spare Trousers", "Ancient Joker", "Ramen",
    "Walkie Talkie", "Seltzer", "Castle", "Smiley Face", "Campfire",
    # Legendary jokers
    "Canio", "Triboulet", "Yorick", "Chicot", "Perkeo",
    # Add more as needed...
    "Unknown"  # Fallback
]
JOKER_MAP = {name: i for i, name in enumerate(JOKER_TYPES)}

HAND_TYPES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush",
    "Five of a Kind", "Flush House", "Flush Five"
]
HAND_MAP = {h: i for i, h in enumerate(HAND_TYPES)}

RANK_MAP = {str(i): i - 2 for i in range(2, 11)}
RANK_MAP.update({"Jack": 9, "Queen": 10, "King": 11, "Ace": 12})

SUIT_MAP = {"Spades": 0, "Hearts": 1, "Clubs": 2, "Diamonds": 3}

EDITION_MAP = {"None": 0, "Foil": 1, "Holographic": 2, "Polychrome": 3, "Negative": 4}
ENHANCEMENT_MAP = {
    "None": 0, "Bonus": 1, "Mult": 2, "Wild": 3, "Glass": 4,
    "Steel": 5, "Stone": 6, "Gold": 7, "Lucky": 8
}
SEAL_MAP = {"None": 0, "Gold": 1, "Red": 2, "Blue": 3, "Purple": 4}


# =============================================================================
# OBSERVATION SPACE CONFIGURATION
# =============================================================================

class ObsConfig:
    """Configuration for observation space dimensions."""
    
    # Global state
    GLOBAL_SIZE = 15
    
    # Deck composition
    DECK_SIZE = 6  # 4 suits + stone + total
    
    # Hand cards (up to 12 cards, each with multiple features)
    MAX_HAND_SIZE = 12
    CARD_FEATURES = 8  # rank, suit, selected, enhanced, edition, seal, scoring, debuffed
    HAND_SIZE = MAX_HAND_SIZE * CARD_FEATURES
    
    # Selected cards one-hot (for quick reference)
    SELECTED_SIZE = MAX_HAND_SIZE
    
    # Poker hand info
    POKER_HAND_SIZE = 12  # One-hot current hand type
    HAND_LEVELS_SIZE = 12  # Level of each hand type (normalized)
    
    # Jokers (5 slots, each with type + features)
    MAX_JOKERS = 5
    JOKER_FEATURES = 8  # type_idx (normalized), edition, sell_value, etc.
    JOKERS_SIZE = MAX_JOKERS * JOKER_FEATURES
    
    # Consumables (2 slots)
    MAX_CONSUMABLES = 2
    CONSUMABLE_FEATURES = 4
    CONSUMABLES_SIZE = MAX_CONSUMABLES * CONSUMABLE_FEATURES
    
    # Shop (during shop phase)
    MAX_SHOP_SLOTS = 5
    SHOP_ITEM_FEATURES = 6
    SHOP_SIZE = MAX_SHOP_SLOTS * SHOP_ITEM_FEATURES
    
    # Phase one-hot
    PHASE_SIZE = 5
    
    # Blind selection info
    BLIND_SELECT_SIZE = 6  # small/big/boss targets, tags, skip available
    
    @classmethod
    def total_size(cls) -> int:
        return (
            cls.GLOBAL_SIZE +
            cls.DECK_SIZE +
            cls.HAND_SIZE +
            cls.SELECTED_SIZE +
            cls.POKER_HAND_SIZE +
            cls.HAND_LEVELS_SIZE +
            cls.JOKERS_SIZE +
            cls.CONSUMABLES_SIZE +
            cls.SHOP_SIZE +
            cls.PHASE_SIZE +
            cls.BLIND_SELECT_SIZE
        )


# =============================================================================
# ACTION SPACE CONFIGURATION
# =============================================================================

class ActionConfig:
    """
    Unified action space that covers all phases.
    
    We use a flat Discrete space with action masking for simplicity.
    Actions are organized by phase:
    
    BLIND PHASE (0-38):
        0-11: Toggle card 0-11
        12: Play hand
        13: Discard hand
        14-25: (reserved for future card-specific actions)
        26-38: (reserved)
    
    SHOP PHASE (39-58):
        39-43: Buy shop slot 0-4
        44-48: Sell joker 0-4
        49-50: Use consumable 0-1
        51: Reroll shop
        52: Next round (leave shop)
        53-58: (reserved)
    
    BLIND SELECT PHASE (59-64):
        59: Select small blind
        60: Select big blind  
        61: Select boss blind
        62: Skip blind (if available)
        63-64: (reserved)
    
    PACK OPENING PHASE (65-74):
        65-72: Select card 0-7 from pack
        73: Skip remaining
        74: (reserved)

    BLIND COMPLETE PHASE (75-76):
        75: Cash out / Continue to shop
        76: (reserved)
    """

    # Blind phase actions
    TOGGLE_START = 0
    TOGGLE_END = 11
    PLAY_HAND = 12
    DISCARD_HAND = 13

    # Shop phase actions
    SHOP_START = 39
    BUY_SLOT_START = 39
    BUY_SLOT_END = 43
    SELL_JOKER_START = 44
    SELL_JOKER_END = 48
    USE_CONSUMABLE_START = 49
    USE_CONSUMABLE_END = 50
    REROLL_SHOP = 51
    NEXT_ROUND = 52

    # Blind select actions
    BLIND_SELECT_START = 59
    SELECT_SMALL = 59
    SELECT_BIG = 60
    SELECT_BOSS = 61
    SKIP_BLIND = 62

    # Pack opening actions
    PACK_START = 65
    PACK_CARD_START = 65
    PACK_CARD_END = 72
    PACK_SKIP = 73

    # Blind complete actions
    CASH_OUT = 75

    TOTAL_ACTIONS = 77
    
    @classmethod
    def get_action_type(cls, action: int) -> Tuple[str, int]:
        """Decode action into (type, index)."""
        if cls.TOGGLE_START <= action <= cls.TOGGLE_END:
            return ("toggle", action - cls.TOGGLE_START)
        elif action == cls.PLAY_HAND:
            return ("play", 0)
        elif action == cls.DISCARD_HAND:
            return ("discard", 0)
        elif cls.BUY_SLOT_START <= action <= cls.BUY_SLOT_END:
            return ("buy", action - cls.BUY_SLOT_START)
        elif cls.SELL_JOKER_START <= action <= cls.SELL_JOKER_END:
            return ("sell_joker", action - cls.SELL_JOKER_START)
        elif cls.USE_CONSUMABLE_START <= action <= cls.USE_CONSUMABLE_END:
            return ("use_consumable", action - cls.USE_CONSUMABLE_START)
        elif action == cls.REROLL_SHOP:
            return ("reroll", 0)
        elif action == cls.NEXT_ROUND:
            return ("next_round", 0)
        elif action == cls.SELECT_SMALL:
            return ("select_blind", 0)
        elif action == cls.SELECT_BIG:
            return ("select_blind", 1)
        elif action == cls.SELECT_BOSS:
            return ("select_blind", 2)
        elif action == cls.SKIP_BLIND:
            return ("skip_blind", 0)
        elif cls.PACK_CARD_START <= action <= cls.PACK_CARD_END:
            return ("pack_select", action - cls.PACK_CARD_START)
        elif action == cls.PACK_SKIP:
            return ("pack_skip", 0)
        elif action == cls.CASH_OUT:
            return ("cash_out", 0)
        else:
            return ("unknown", action)


# =============================================================================
# MAIN ENVIRONMENT
# =============================================================================

class BalatroEnv(gym.Env):
    """
    Balatro RL Environment v2
    
    A comprehensive environment for training agents on the full Balatro game loop.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        super().__init__()
        
        # Network config
        self.host = host
        self.port = port
        
        # Action space: flat discrete with masking
        self.action_space = spaces.Discrete(ActionConfig.TOTAL_ACTIONS)
        
        # Observation space
        obs_size = ObsConfig.total_size()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # State tracking
        self.last_state: Dict[str, Any] = {}
        self.prev_chips: int = 0
        self.prev_money: int = 0
        self.current_phase: GamePhase = GamePhase.BLIND
        self.episode_stats: Dict[str, Any] = {}
        
        # Networking
        self.server: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self.buffer: bytes = b""
        
        # Initialize server
        self._setup_server()
        
        logging.info(f"BalatroEnv v2 initialized. Observation size: {obs_size}")
    
    def _setup_server(self):
        """Set up the TCP server for game communication."""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        logging.info(f"Server listening on {self.host}:{self.port}")
    
    # =========================================================================
    # GYMNASIUM INTERFACE
    # =========================================================================
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.episode_stats = {
            "hands_played": 0,
            "discards_used": 0,
            "blinds_won": 0,
            "money_earned": 0,
            "jokers_bought": 0,
            "max_ante": 0,
        }
        
        while True:
            # Ensure connection
            if self.conn is None:
                self._wait_for_connection()
                self.buffer = b""
                self.prev_chips = 0
                self.prev_money = 0
                self.last_state = {}
            
            # Request state from game
            logging.info("RESET: Requesting state from game...")
            self._send_command({"type": "ping"})
            
            # Wait for response
            state = self._read_packet(block_forever=True)
            
            if state is not None:
                self.last_state = state
                self.prev_chips = state.get("chips", 0)
                self.prev_money = state.get("money", 0)
                self.current_phase = self._detect_phase(state)
                
                logging.info(f"RESET: Success. Phase={self.current_phase.name}, "
                           f"Money=${state.get('money', 0)}, Ante={state.get('ante', 1)}")
                
                obs = self._build_observation(state)
                info = {"action_mask": self.get_action_mask()}
                return obs, info
            
            logging.warning("RESET: Connection failed. Retrying...")
            self._close_connection()
    
    def step(self, action: int):
        """Execute an action and return the result."""
        action = int(action)
        
        # Decode action
        action_type, action_index = ActionConfig.get_action_type(action)
        
        # Get action mask and check validity
        mask = self.get_action_mask()
        if not mask[action]:
            # Invalid action - apply penalty and return current state
            logging.warning(f"Invalid action {action} ({action_type}, {action_index}) "
                          f"in phase {self.current_phase.name}")
            reward = -5.0
            obs = self._build_observation(self.last_state)
            return obs, reward, False, False, {"action_mask": mask, "invalid_action": True}
        
        # Build command for game
        command = self._build_command(action_type, action_index)
        
        # Log action
        self._log_action(action_type, action_index, command)
        
        # Send command
        if not self._send_command(command):
            return self._crash_exit()
        
        time.sleep(0.15)  # Give game time to process
        
        # Receive new state
        new_state = self._read_packet(block_forever=True)
        if not new_state:
            return self._crash_exit()
        
        # Calculate reward
        reward = self._calculate_reward(self.last_state, action_type, action_index, new_state)
        
        # Check for state change (penalize no-ops)
        if self._states_equal(self.last_state, new_state):
            reward -= 1.0  # Small penalty for wasted action
        
        # Update tracking
        old_phase = self.current_phase
        self.current_phase = self._detect_phase(new_state)
        self._update_episode_stats(action_type, self.last_state, new_state)
        
        self.last_state = new_state
        self.prev_chips = new_state.get("chips", 0)
        self.prev_money = new_state.get("money", 0)
        
        # Check termination
        done = new_state.get("game_over", False) or new_state.get("game_won", False)
        
        # Build observation
        obs = self._build_observation(new_state)
        
        info = {
            "action_mask": self.get_action_mask(),
            "phase": self.current_phase.name,
            "phase_changed": old_phase != self.current_phase,
            "episode_stats": self.episode_stats.copy(),
        }
        
        return obs, reward, done, False, info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask of valid actions for the current state.
        
        This is crucial for training - it prevents the agent from learning
        invalid actions and speeds up convergence.
        """
        mask = np.zeros(ActionConfig.TOTAL_ACTIONS, dtype=bool)
        state = self.last_state
        phase = self.current_phase
        
        if phase == GamePhase.BLIND:
            mask = self._get_blind_action_mask(state, mask)
        elif phase == GamePhase.SHOP:
            mask = self._get_shop_action_mask(state, mask)
        elif phase == GamePhase.BLIND_SELECT:
            mask = self._get_blind_select_action_mask(state, mask)
        elif phase == GamePhase.PACK_OPENING:
            mask = self._get_pack_action_mask(state, mask)
        elif phase == GamePhase.BLIND_COMPLETE:
            mask = self._get_blind_complete_action_mask(state, mask)
        else:
            # Game over - no valid actions
            pass
        
        # Ensure at least one action is valid to prevent crashes
        if not mask.any():
            # Default to a safe action based on phase
            if phase == GamePhase.BLIND:
                mask[ActionConfig.TOGGLE_START] = True
            elif phase == GamePhase.SHOP:
                mask[ActionConfig.NEXT_ROUND] = True
            elif phase == GamePhase.BLIND_SELECT:
                mask[ActionConfig.SELECT_SMALL] = True
            elif phase == GamePhase.BLIND_COMPLETE:
                mask[ActionConfig.CASH_OUT] = True
        
        return mask
    
    def _get_blind_action_mask(self, state: Dict, mask: np.ndarray) -> np.ndarray:
        """Mask for blind phase actions."""
        hand = state.get("hand", [])
        hand_size = len(hand)
        
        # Count selected cards
        selected_count = sum(1 for c in hand if c.get("selected", False))
        
        # Toggle is valid for existing cards only
        for i in range(min(hand_size, 12)):
            mask[ActionConfig.TOGGLE_START + i] = True
        
        # Play is valid if cards are selected
        if selected_count > 0:
            mask[ActionConfig.PLAY_HAND] = True
        
        # Discard is valid if cards selected AND discards remaining
        discards_left = state.get("discards_left", 0)
        if selected_count > 0 and discards_left > 0:
            mask[ActionConfig.DISCARD_HAND] = True
        
        return mask
    
    def _get_shop_action_mask(self, state: Dict, mask: np.ndarray) -> np.ndarray:
        """Mask for shop phase actions."""
        money = state.get("money", 0)
        shop_items = state.get("shop", [])
        jokers = state.get("jokers", [])
        consumables = state.get("consumables", [])
        
        # Buy items (if affordable and slot exists)
        for i, item in enumerate(shop_items[:5]):
            if item and item.get("cost", 999) <= money:
                mask[ActionConfig.BUY_SLOT_START + i] = True
        
        # Sell jokers (if jokers exist)
        for i, joker in enumerate(jokers[:5]):
            if joker:
                mask[ActionConfig.SELL_JOKER_START + i] = True
        
        # Use consumables (if they exist and are usable)
        for i, cons in enumerate(consumables[:2]):
            if cons and cons.get("usable", True):
                mask[ActionConfig.USE_CONSUMABLE_START + i] = True
        
        # Reroll (costs $5 by default, but can vary)
        reroll_cost = state.get("reroll_cost", 5)
        if money >= reroll_cost:
            mask[ActionConfig.REROLL_SHOP] = True
        
        # Next round is always valid in shop
        mask[ActionConfig.NEXT_ROUND] = True
        
        return mask
    
    def _get_blind_select_action_mask(self, state: Dict, mask: np.ndarray) -> np.ndarray:
        """Mask for blind selection phase."""
        blinds = state.get("blinds_available", ["small", "big", "boss"])
        can_skip = state.get("can_skip", False)
        
        # Select available blinds
        if "small" in blinds:
            mask[ActionConfig.SELECT_SMALL] = True
        if "big" in blinds:
            mask[ActionConfig.SELECT_BIG] = True
        if "boss" in blinds:
            mask[ActionConfig.SELECT_BOSS] = True
        
        # Skip if tag allows
        if can_skip:
            mask[ActionConfig.SKIP_BLIND] = True
        
        return mask
    
    def _get_pack_action_mask(self, state: Dict, mask: np.ndarray) -> np.ndarray:
        """Mask for pack opening phase."""
        pack_cards = state.get("pack_cards", [])
        picks_remaining = state.get("pack_picks_remaining", 0)
        
        if picks_remaining > 0:
            for i, card in enumerate(pack_cards[:8]):
                if card and not card.get("picked", False):
                    mask[ActionConfig.PACK_CARD_START + i] = True
        
        # Can always skip
        mask[ActionConfig.PACK_SKIP] = True

        return mask

    def _get_blind_complete_action_mask(self, state: Dict, mask: np.ndarray) -> np.ndarray:
        """Mask for blind complete phase (cash out screen)."""
        # Only action is to cash out and continue
        mask[ActionConfig.CASH_OUT] = True
        return mask

    # =========================================================================
    # OBSERVATION BUILDING
    # =========================================================================
    
    def _build_observation(self, state: Dict) -> np.ndarray:
        """Build the full observation vector from game state."""
        obs = np.zeros(ObsConfig.total_size(), dtype=np.float32)
        cursor = 0
        
        # 1. Global state
        cursor = self._encode_global(obs, cursor, state)
        
        # 2. Deck composition
        cursor = self._encode_deck(obs, cursor, state)
        
        # 3. Hand cards
        cursor = self._encode_hand(obs, cursor, state)
        
        # 4. Selected cards quick reference
        cursor = self._encode_selected(obs, cursor, state)
        
        # 5. Current poker hand (one-hot)
        cursor = self._encode_poker_hand(obs, cursor, state)
        
        # 6. Hand levels
        cursor = self._encode_hand_levels(obs, cursor, state)
        
        # 7. Jokers
        cursor = self._encode_jokers(obs, cursor, state)
        
        # 8. Consumables
        cursor = self._encode_consumables(obs, cursor, state)
        
        # 9. Shop
        cursor = self._encode_shop(obs, cursor, state)
        
        # 10. Phase (one-hot)
        cursor = self._encode_phase(obs, cursor)
        
        # 11. Blind selection info
        cursor = self._encode_blind_select(obs, cursor, state)
        
        return obs
    
    def _encode_global(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode global game state."""
        # Money (normalized to ~100 for early game visibility)
        obs[cursor + 0] = min(state.get("money", 0) / 100.0, 1.0)
        
        # Hands left (normalized to max ~5)
        obs[cursor + 1] = state.get("hands_left", 0) / 5.0
        
        # Discards left (normalized to max ~5)
        obs[cursor + 2] = state.get("discards_left", 0) / 5.0
        
        # Ante (normalized to 8)
        obs[cursor + 3] = state.get("ante", 1) / 8.0
        
        # Round within ante (normalized to 3)
        obs[cursor + 4] = state.get("round", 1) / 3.0
        
        # Chip progress toward blind (capped at 2x for overflow)
        blind_target = max(state.get("blind_target", 1), 1)
        chips = state.get("chips", 0)
        obs[cursor + 5] = min(chips / blind_target, 2.0)
        
        # Blind target (log-normalized for scaling)
        obs[cursor + 6] = min(np.log1p(blind_target) / 15.0, 1.0)
        
        # Is boss blind
        obs[cursor + 7] = 1.0 if state.get("is_boss", False) else 0.0
        
        # Joker slots used (normalized to 5)
        jokers = state.get("jokers", [])
        obs[cursor + 8] = len([j for j in jokers if j]) / 5.0
        
        # Consumable slots used
        consumables = state.get("consumables", [])
        obs[cursor + 9] = len([c for c in consumables if c]) / 2.0
        
        # Hand size (normalized to 12)
        obs[cursor + 10] = len(state.get("hand", [])) / 12.0
        
        # Deck size remaining (normalized to ~52)
        obs[cursor + 11] = state.get("deck_remaining", 52) / 52.0
        
        # Interest threshold indicator (at $5+ intervals)
        money = state.get("money", 0)
        obs[cursor + 12] = min(money // 5, 5) / 5.0  # Max interest at $25
        
        # Reroll cost (normalized)
        obs[cursor + 13] = min(state.get("reroll_cost", 5) / 10.0, 1.0)
        
        # Reserved
        obs[cursor + 14] = 0.0
        
        return cursor + ObsConfig.GLOBAL_SIZE
    
    def _encode_deck(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode deck composition."""
        deck = state.get("deck_stats", {})
        
        for i, suit in enumerate(["Spades", "Hearts", "Clubs", "Diamonds"]):
            obs[cursor + i] = deck.get(suit, 0) / 13.0
        
        obs[cursor + 4] = deck.get("Stone", 0) / 10.0
        obs[cursor + 5] = deck.get("total", 52) / 60.0  # Can exceed 52 with added cards
        
        return cursor + ObsConfig.DECK_SIZE
    
    def _encode_hand(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode cards in hand with full features."""
        hand = state.get("hand", [])
        
        for i, card in enumerate(hand[:ObsConfig.MAX_HAND_SIZE]):
            base = cursor + i * ObsConfig.CARD_FEATURES
            
            # Rank (normalized 0-12)
            rank = str(card.get("rank", "2"))
            obs[base + 0] = RANK_MAP.get(rank, 0) / 12.0
            
            # Suit (one-hot would be better, but normalized for space)
            suit = card.get("suit", "Spades")
            obs[base + 1] = SUIT_MAP.get(suit, 0) / 3.0
            
            # Selected
            obs[base + 2] = 1.0 if card.get("selected", False) else 0.0
            
            # Enhancement
            enhancement = card.get("enhancement", "None")
            obs[base + 3] = ENHANCEMENT_MAP.get(enhancement, 0) / 8.0
            
            # Edition
            edition = card.get("edition", "None")
            obs[base + 4] = EDITION_MAP.get(edition, 0) / 4.0
            
            # Seal
            seal = card.get("seal", "None")
            obs[base + 5] = SEAL_MAP.get(seal, 0) / 4.0
            
            # Is scoring (will this card score in current selection?)
            obs[base + 6] = 1.0 if card.get("scoring", False) else 0.0
            
            # Is debuffed
            obs[base + 7] = 1.0 if card.get("debuffed", False) else 0.0
        
        return cursor + ObsConfig.HAND_SIZE
    
    def _encode_selected(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Quick reference for which cards are selected."""
        hand = state.get("hand", [])
        
        for i, card in enumerate(hand[:ObsConfig.MAX_HAND_SIZE]):
            obs[cursor + i] = 1.0 if card.get("selected", False) else 0.0
        
        return cursor + ObsConfig.SELECTED_SIZE
    
    def _encode_poker_hand(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """One-hot encode current poker hand type."""
        hand_type = state.get("poker_hand", "High Card")
        idx = HAND_MAP.get(hand_type, 0)
        
        if 0 <= idx < len(HAND_TYPES):
            obs[cursor + idx] = 1.0
        
        return cursor + ObsConfig.POKER_HAND_SIZE
    
    def _encode_hand_levels(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode the level of each poker hand type."""
        hand_levels = state.get("hand_levels", {})
        
        for i, hand_type in enumerate(HAND_TYPES):
            level = hand_levels.get(hand_type, 1)
            obs[cursor + i] = min(level / 10.0, 1.0)  # Normalize to ~level 10
        
        return cursor + ObsConfig.HAND_LEVELS_SIZE
    
    def _encode_jokers(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode jokers in joker slots."""
        jokers = state.get("jokers", [])
        
        for i, joker in enumerate(jokers[:ObsConfig.MAX_JOKERS]):
            if not joker:
                continue
                
            base = cursor + i * ObsConfig.JOKER_FEATURES
            
            # Joker type (normalized index)
            joker_name = joker.get("name", "Unknown")
            type_idx = JOKER_MAP.get(joker_name, len(JOKER_TYPES) - 1)
            obs[base + 0] = type_idx / len(JOKER_TYPES)
            
            # Edition
            edition = joker.get("edition", "None")
            obs[base + 1] = EDITION_MAP.get(edition, 0) / 4.0
            
            # Sell value (normalized)
            obs[base + 2] = min(joker.get("sell_value", 0) / 10.0, 1.0)
            
            # Rarity (common=0, uncommon=1, rare=2, legendary=3)
            rarity = joker.get("rarity", "Common")
            rarity_map = {"Common": 0, "Uncommon": 1, "Rare": 2, "Legendary": 3}
            obs[base + 3] = rarity_map.get(rarity, 0) / 3.0
            
            # Has counter/accumulator (for scaling jokers)
            obs[base + 4] = 1.0 if joker.get("has_counter", False) else 0.0
            
            # Counter value (normalized)
            obs[base + 5] = min(joker.get("counter", 0) / 100.0, 1.0)
            
            # Is active/triggered this hand
            obs[base + 6] = 1.0 if joker.get("active", False) else 0.0
            
            # Reserved
            obs[base + 7] = 0.0
        
        return cursor + ObsConfig.JOKERS_SIZE
    
    def _encode_consumables(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode consumable items."""
        consumables = state.get("consumables", [])
        
        for i, cons in enumerate(consumables[:ObsConfig.MAX_CONSUMABLES]):
            if not cons:
                continue
                
            base = cursor + i * ObsConfig.CONSUMABLE_FEATURES
            
            # Type (tarot=0, planet=1, spectral=2)
            cons_type = cons.get("type", "tarot")
            type_map = {"tarot": 0, "planet": 1, "spectral": 2}
            obs[base + 0] = type_map.get(cons_type, 0) / 2.0
            
            # Sell value
            obs[base + 1] = min(cons.get("sell_value", 0) / 5.0, 1.0)
            
            # Is usable now
            obs[base + 2] = 1.0 if cons.get("usable", True) else 0.0
            
            # Reserved
            obs[base + 3] = 0.0
        
        return cursor + ObsConfig.CONSUMABLES_SIZE
    
    def _encode_shop(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode shop items (only relevant during shop phase)."""
        shop = state.get("shop", [])
        money = state.get("money", 0)
        
        for i, item in enumerate(shop[:ObsConfig.MAX_SHOP_SLOTS]):
            if not item:
                continue
                
            base = cursor + i * ObsConfig.SHOP_ITEM_FEATURES
            
            # Item type (joker=0, tarot=1, planet=2, voucher=3, pack=4)
            item_type = item.get("type", "joker")
            type_map = {"joker": 0, "tarot": 1, "planet": 2, "voucher": 3, "pack": 4}
            obs[base + 0] = type_map.get(item_type, 0) / 4.0
            
            # Cost (normalized)
            cost = item.get("cost", 0)
            obs[base + 1] = min(cost / 20.0, 1.0)
            
            # Can afford
            obs[base + 2] = 1.0 if cost <= money else 0.0
            
            # Rarity (for jokers)
            rarity = item.get("rarity", "Common")
            rarity_map = {"Common": 0, "Uncommon": 1, "Rare": 2, "Legendary": 3}
            obs[base + 3] = rarity_map.get(rarity, 0) / 3.0
            
            # Is sold out
            obs[base + 4] = 1.0 if item.get("sold", False) else 0.0
            
            # Reserved
            obs[base + 5] = 0.0
        
        return cursor + ObsConfig.SHOP_SIZE
    
    def _encode_phase(self, obs: np.ndarray, cursor: int) -> int:
        """One-hot encode current game phase."""
        obs[cursor + int(self.current_phase)] = 1.0
        return cursor + ObsConfig.PHASE_SIZE
    
    def _encode_blind_select(self, obs: np.ndarray, cursor: int, state: Dict) -> int:
        """Encode blind selection information."""
        blind_info = state.get("blind_select", {})
        
        # Small blind target (log normalized)
        small_target = blind_info.get("small_target", 300)
        obs[cursor + 0] = min(np.log1p(small_target) / 15.0, 1.0)
        
        # Big blind target
        big_target = blind_info.get("big_target", 450)
        obs[cursor + 1] = min(np.log1p(big_target) / 15.0, 1.0)
        
        # Boss blind target
        boss_target = blind_info.get("boss_target", 600)
        obs[cursor + 2] = min(np.log1p(boss_target) / 15.0, 1.0)
        
        # Has skip tag
        obs[cursor + 3] = 1.0 if state.get("can_skip", False) else 0.0
        
        # Current blind (0=small, 1=big, 2=boss)
        current = blind_info.get("current", 0)
        obs[cursor + 4] = current / 2.0
        
        # Reserved
        obs[cursor + 5] = 0.0
        
        return cursor + ObsConfig.BLIND_SELECT_SIZE
    
    # =========================================================================
    # REWARD CALCULATION
    # =========================================================================
    
    def _calculate_reward(self, old_state: Dict, action_type: str, 
                         action_index: int, new_state: Dict) -> float:
        """
        Calculate reward based on game state transition.
        
        Design principles:
        1. Rewards scale with blind target (ante-agnostic)
        2. Sparse rewards for major events (wins, losses)
        3. Shaping rewards for progress and good decisions
        4. Small penalties for inefficiency
        """
        reward = 0.0
        phase = self.current_phase
        
        # ==================== TERMINAL REWARDS ====================
        
        # Game won (beat ante 8)
        if new_state.get("game_won", False):
            return 200.0  # Big reward for winning
        
        # Game over (lost)
        if new_state.get("game_over", False) and not new_state.get("game_won", False):
            # Scale penalty by progress
            ante = new_state.get("ante", 1)
            return -50.0 + ante * 5.0  # Less penalty for dying at high ante
        
        # ==================== PHASE-SPECIFIC REWARDS ====================
        
        if phase == GamePhase.BLIND:
            reward += self._reward_blind_phase(old_state, action_type, action_index, new_state)
        elif phase == GamePhase.SHOP:
            reward += self._reward_shop_phase(old_state, action_type, action_index, new_state)
        elif phase == GamePhase.BLIND_SELECT:
            reward += self._reward_blind_select_phase(old_state, action_type, new_state)
        elif phase == GamePhase.PACK_OPENING:
            reward += self._reward_pack_phase(old_state, action_type, action_index, new_state)
        elif phase == GamePhase.BLIND_COMPLETE:
            # Simple transition phase - small positive for progressing
            if action_type == "cash_out":
                reward += 1.0

        return reward
    
    def _reward_blind_phase(self, old_state: Dict, action_type: str, 
                           action_index: int, new_state: Dict) -> float:
        """Rewards for blind (playing) phase."""
        reward = 0.0
        
        # Extract state
        blind_target = max(new_state.get("blind_target", 1), 1)
        old_chips = old_state.get("chips", 0)
        new_chips = new_state.get("chips", 0)
        
        old_hands = old_state.get("hands_left", 4)
        new_hands = new_state.get("hands_left", 4)
        
        old_discards = old_state.get("discards_left", 3)
        new_discards = new_state.get("discards_left", 3)
        
        # Normalized progress
        old_progress = old_chips / blind_target
        new_progress = new_chips / blind_target
        progress_delta = new_progress - old_progress
        
        # ========== BLIND VICTORY ==========
        if old_progress < 1.0 and new_progress >= 1.0:
            base_reward = 30.0
            efficiency_bonus = new_hands * 8.0  # Reward unused hands
            return base_reward + efficiency_bonus
        
        # ========== CHIP PROGRESS ==========
        if progress_delta > 0:
            # Base progress reward (scaled to blind)
            reward += progress_delta * 15.0
            
            # Bonus for big single plays (encourages building strong hands)
            if progress_delta > 0.5:
                reward += 8.0
            elif progress_delta > 0.25:
                reward += 3.0
        
        # ========== HAND PLAYED ==========
        if action_type == "play" and old_hands > new_hands:
            # Played a hand
            self.episode_stats["hands_played"] += 1
            
            # Penalty for weak plays when discards available
            if progress_delta < 0.1 and new_discards > 0:
                reward -= 2.0  # Should have discarded first
            
            # Penalty for no progress (invalid/blocked hand)
            if progress_delta == 0:
                reward -= 5.0
        
        # ========== DISCARD USED ==========
        if action_type == "discard" and old_discards > new_discards:
            self.episode_stats["discards_used"] += 1
            
            # Small positive - discards are strategic
            reward += 0.5
            
            # Bonus for discarding multiple cards (clearing bad hand)
            selected = sum(1 for c in old_state.get("hand", []) if c.get("selected"))
            if selected >= 3:
                reward += 1.0
        
        # ========== URGENCY PENALTIES ==========
        # Running low on hands without good progress
        if new_hands <= 1 and new_progress < 0.5:
            reward -= 5.0  # Danger zone
        elif new_hands <= 2 and new_progress < 0.3:
            reward -= 2.0  # Getting risky
        
        # ========== FAILURE ==========
        if new_hands == 0 and new_progress < 1.0:
            # Failed the blind
            reward -= 25.0
        
        return reward
    
    def _reward_shop_phase(self, old_state: Dict, action_type: str,
                          action_index: int, new_state: Dict) -> float:
        """Rewards for shop phase."""
        reward = 0.0
        
        old_money = old_state.get("money", 0)
        new_money = new_state.get("money", 0)
        
        old_jokers = len([j for j in old_state.get("jokers", []) if j])
        new_jokers = len([j for j in new_state.get("jokers", []) if j])
        
        # ========== BOUGHT JOKER ==========
        if new_jokers > old_jokers:
            self.episode_stats["jokers_bought"] += 1
            reward += 5.0  # Jokers are key to scaling
            
            # Extra bonus early game (need to build collection)
            ante = new_state.get("ante", 1)
            if ante <= 3 and new_jokers <= 3:
                reward += 3.0  # Priority early joker acquisition
        
        # ========== SOLD JOKER ==========
        if new_jokers < old_jokers and action_type == "sell_joker":
            # Selling is situational - small penalty unless replacing
            reward -= 1.0
        
        # ========== MONEY MANAGEMENT ==========
        # Interest threshold rewards
        if new_money >= 25:
            reward += 0.5  # Max interest
        elif new_money >= 5:
            reward += 0.2  # Some interest
        
        # Penalty for going broke
        if new_money == 0 and old_money > 5:
            reward -= 2.0
        
        # ========== USED CONSUMABLE ==========
        if action_type == "use_consumable":
            reward += 1.0  # Generally good to use consumables
        
        # ========== REROLL ==========
        if action_type == "reroll":
            # Neutral - rerolling is fine
            reward += 0.0
        
        # ========== LEFT SHOP ==========
        if action_type == "next_round":
            reward += 0.5  # Progress the game
        
        return reward
    
    def _reward_blind_select_phase(self, old_state: Dict, action_type: str,
                                   new_state: Dict) -> float:
        """Rewards for blind selection phase."""
        reward = 0.0
        
        if action_type == "select_blind":
            # Selected a blind - good, progress the game
            reward += 1.0
        elif action_type == "skip_blind":
            # Skipped - can be strategic, neutral reward
            # Could add logic to reward skipping bad boss blinds
            reward += 0.5
        
        return reward
    
    def _reward_pack_phase(self, old_state: Dict, action_type: str,
                          action_index: int, new_state: Dict) -> float:
        """Rewards for pack opening phase."""
        reward = 0.0
        
        if action_type == "pack_select":
            # Selected a card from pack
            reward += 1.0
        elif action_type == "pack_skip":
            # Skipped - sometimes correct
            reward += 0.2
        
        return reward
    
    # =========================================================================
    # COMMAND BUILDING
    # =========================================================================
    
    def _build_command(self, action_type: str, action_index: int) -> Dict:
        """Build the command to send to the game."""
        
        if action_type == "toggle":
            return {"type": "toggle", "index": action_index}
        elif action_type == "play":
            return {"type": "play"}
        elif action_type == "discard":
            return {"type": "discard"}
        elif action_type == "buy":
            return {"type": "buy", "slot": action_index}
        elif action_type == "sell_joker":
            return {"type": "sell_joker", "index": action_index}
        elif action_type == "use_consumable":
            return {"type": "use_consumable", "index": action_index}
        elif action_type == "reroll":
            return {"type": "reroll"}
        elif action_type == "next_round":
            return {"type": "next_round"}
        elif action_type == "select_blind":
            blind_names = ["small", "big", "boss"]
            return {"type": "select_blind", "blind": blind_names[action_index]}
        elif action_type == "skip_blind":
            return {"type": "skip_blind"}
        elif action_type == "pack_select":
            return {"type": "pack_select", "index": action_index}
        elif action_type == "pack_skip":
            return {"type": "pack_skip"}
        elif action_type == "cash_out":
            return {"type": "cash_out"}
        else:
            logging.warning(f"Unknown action type: {action_type}")
            return {"type": "ping"}
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _detect_phase(self, state: Dict) -> GamePhase:
        """Detect the current game phase from state."""
        phase_str = state.get("phase", "blind").lower()
        
        if state.get("game_over", False) or state.get("game_won", False):
            return GamePhase.GAME_OVER
        elif phase_str == "shop":
            return GamePhase.SHOP
        elif phase_str == "blind_select":
            return GamePhase.BLIND_SELECT
        elif phase_str == "pack" or phase_str == "pack_opening":
            return GamePhase.PACK_OPENING
        elif phase_str == "blind_complete":
            return GamePhase.BLIND_COMPLETE
        else:
            return GamePhase.BLIND
    
    def _states_equal(self, state1: Dict, state2: Dict) -> bool:
        """Check if two states are effectively equal (action did nothing)."""
        keys = ["chips", "hands_left", "discards_left", "hand", "money", "phase"]
        for key in keys:
            if state1.get(key) != state2.get(key):
                return False
        return True
    
    def _update_episode_stats(self, action_type: str, old_state: Dict, new_state: Dict):
        """Track episode statistics."""
        # Money earned
        money_diff = new_state.get("money", 0) - old_state.get("money", 0)
        if money_diff > 0:
            self.episode_stats["money_earned"] += money_diff
        
        # Blinds won (phase change from blind to shop with chips >= target)
        old_phase = self._detect_phase(old_state)
        new_phase = self._detect_phase(new_state)
        if old_phase == GamePhase.BLIND and new_phase == GamePhase.SHOP:
            self.episode_stats["blinds_won"] += 1
        
        # Max ante reached
        ante = new_state.get("ante", 1)
        if ante > self.episode_stats["max_ante"]:
            self.episode_stats["max_ante"] = ante
    
    def _log_action(self, action_type: str, action_index: int, command: Dict):
        """Log the action being taken."""
        hand = self.last_state.get("hand", [])
        selected = sum(1 for c in hand if c.get("selected", False))
        
        print(f"\n┌── 🤖 STEP ──────────────────────────────────")
        print(f"│ Phase: {self.current_phase.name}")
        print(f"│ State: Hands={self.last_state.get('hands_left', '?')} | "
              f"Chips={self.last_state.get('chips', 0)} | Selected={selected}")
        print(f"│ Action: {action_type.upper()} (index={action_index})")
        print(f"│ Command: {command}")
        print(f"└─────────────────────────────────────────────")
    
    # =========================================================================
    # NETWORKING
    # =========================================================================
    
    def _wait_for_connection(self):
        """Wait for the game to connect."""
        logging.info("Waiting for Balatro to connect...")
        self.conn, addr = self.server.accept()
        self.conn.settimeout(None)
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logging.info(f"Connected: {addr}")
    
    def _send_command(self, cmd: Dict) -> bool:
        """Send a command to the game."""
        if not self.conn:
            return False
        try:
            msg = json.dumps(cmd) + "\n"
            self.conn.sendall(msg.encode("utf-8"))
            return True
        except BrokenPipeError:
            logging.error("Connection lost (Broken Pipe)")
            self._close_connection()
            return False
        except Exception as e:
            logging.error(f"Send error: {e}")
            self._close_connection()
            return False
    
    def _read_packet(self, block_forever: bool = False) -> Optional[Dict]:
        """Read a JSON packet from the game."""
        if not self.conn:
            return None
        
        while True:
            try:
                if b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    if not line.strip():
                        continue
                    return json.loads(line.decode("utf-8"))
                
                chunk = self.conn.recv(4096)
                if not chunk:
                    logging.warning("Connection closed by game")
                    self._close_connection()
                    return None
                self.buffer += chunk
                
            except BlockingIOError:
                if not block_forever:
                    return None
                continue
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                self.buffer = b""
            except Exception as e:
                logging.error(f"Read error: {e}")
                self._close_connection()
                return None
    
    def _close_connection(self):
        """Close the current connection."""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        self.conn = None
        logging.info("Connection closed")
    
    def _crash_exit(self):
        """Return a crashed state."""
        obs = np.zeros(ObsConfig.total_size(), dtype=np.float32)
        return obs, -10.0, True, False, {"crashed": True}
    
    def close(self):
        """Clean up resources."""
        self._close_connection()
        if self.server:
            try:
                self.server.close()
            except:
                pass
        logging.info("Environment closed")
    
    def render(self, mode: str = "human"):
        """Render the current state."""
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Generate ASCII representation of game state."""
        state = self.last_state
        lines = [
            "=" * 50,
            f"Phase: {self.current_phase.name}",
            f"Ante: {state.get('ante', '?')} | Round: {state.get('round', '?')}",
            f"Money: ${state.get('money', '?')}",
            f"Chips: {state.get('chips', 0)} / {state.get('blind_target', '?')}",
            f"Hands: {state.get('hands_left', '?')} | Discards: {state.get('discards_left', '?')}",
            "-" * 50,
            "Hand: " + self._render_hand(state.get("hand", [])),
            "Jokers: " + ", ".join(j.get("name", "?") for j in state.get("jokers", []) if j),
            "=" * 50,
        ]
        return "\n".join(lines)
    
    def _render_hand(self, hand: List[Dict]) -> str:
        """Render hand cards as string."""
        def card_str(c):
            rank = str(c.get("rank", "?"))
            suit = c.get("suit", "?")[0]  # First letter
            sel = "*" if c.get("selected") else ""
            return f"{rank}{suit}{sel}"
        return " ".join(card_str(c) for c in hand)


# =============================================================================
# WRAPPER FOR ACTION MASKING (for libraries like SB3 that support it)
# =============================================================================

class BalatroEnvWithMasking(BalatroEnv):
    """
    Wrapper that includes action mask in observation for compatibility
    with algorithms that support action masking natively.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extend observation space to include action mask
        base_size = ObsConfig.total_size()
        mask_size = ActionConfig.TOTAL_ACTIONS
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0, high=1, shape=(base_size,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(mask_size,), dtype=np.float32),
        })
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        mask = self.get_action_mask().astype(np.float32)
        return {"observation": obs, "action_mask": mask}, info
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        mask = self.get_action_mask().astype(np.float32)
        return {"observation": obs, "action_mask": mask}, reward, done, truncated, info


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Balatro RL Environment v2")
    print(f"Observation size: {ObsConfig.total_size()}")
    print(f"Action space size: {ActionConfig.TOTAL_ACTIONS}")
    print("\nObservation breakdown:")
    for name, size in [
        ("Global", ObsConfig.GLOBAL_SIZE),
        ("Deck", ObsConfig.DECK_SIZE),
        ("Hand", ObsConfig.HAND_SIZE),
        ("Selected", ObsConfig.SELECTED_SIZE),
        ("Poker Hand", ObsConfig.POKER_HAND_SIZE),
        ("Hand Levels", ObsConfig.HAND_LEVELS_SIZE),
        ("Jokers", ObsConfig.JOKERS_SIZE),
        ("Consumables", ObsConfig.CONSUMABLES_SIZE),
        ("Shop", ObsConfig.SHOP_SIZE),
        ("Phase", ObsConfig.PHASE_SIZE),
        ("Blind Select", ObsConfig.BLIND_SELECT_SIZE),
    ]:
        print(f"  {name}: {size}")
    
    print("\nAction space breakdown:")
    print(f"  Blind phase: Toggle(0-11), Play(12), Discard(13)")
    print(f"  Shop phase: Buy(39-43), Sell(44-48), Consumable(49-50), Reroll(51), Next(52)")
    print(f"  Blind select: Small(59), Big(60), Boss(61), Skip(62)")
    print(f"  Pack opening: Select(65-72), Skip(73)")