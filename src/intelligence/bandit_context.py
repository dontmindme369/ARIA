#!/usr/bin/env python3
"""
bandit_context.py - LinUCB-Based Multi-Armed Bandit (NEW IMPLEMENTATION)

Replaces Thompson Sampling with LinUCB for better context-aware learning.

Key improvements:
1. Uses query features directly in learning (not just filtering)
2. Generalizes across similar query types
3. Faster convergence and better adaptation
4. Interpretable feature weights

Maintains backward-compatible API with Thompson implementation.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.presets import Preset
from intelligence.contextual_bandit import (
    ContextualBandit,
    DEFAULT_ARMS,
    ARM_TO_PRESET_ARGS
)


# Default presets for retrieval strategies (same as Thompson version)
DEFAULT_PRESETS = [
    {"name": "fast", "args": {"top_k": 40, "sem_limit": 64, "rotations": 1, "max_per_file": 8}},
    {"name": "balanced", "args": {"top_k": 64, "sem_limit": 128, "rotations": 2, "max_per_file": 6}},
    {"name": "deep", "args": {"top_k": 96, "sem_limit": 256, "rotations": 3, "max_per_file": 5}},
    {"name": "diverse", "args": {"top_k": 80, "sem_limit": 128, "rotations": 2, "max_per_file": 4}},
]

# EPSILON-GREEDY: 10% random exploration
EPSILON = 0.10


# ============================================================================
# LinUCB Wrapper - Maintains Thompson API Compatibility
# ============================================================================

class BanditState:
    """
    Wrapper around ContextualBandit to maintain Thompson API compatibility.

    This class exists ONLY for backward compatibility. It delegates all
    operations to the ContextualBandit (LinUCB) implementation.
    """

    def __init__(self, presets: Optional[List[Dict[str, Any]]] = None):
        """Initialize LinUCB bandit with presets"""
        self.presets = [Preset(p['name'], p['args']) for p in (presets or DEFAULT_PRESETS)]

        # Create LinUCB bandit with preset names as arms
        arm_names = [p.name for p in self.presets]
        self.bandit = ContextualBandit(
            arms=arm_names,
            feature_dim=10,
            alpha=1.0,  # Moderate exploration
            state_path=None  # Will be set in select_preset
        )

        # Track for compatibility
        self.selections = {p.name: 0 for p in self.presets}
        self.total_pulls = 0
        self.phase = "exploration"  # or "exploitation"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state (for compatibility - not used with LinUCB)

        LinUCB saves its own state automatically, but we provide this
        for API compatibility.
        """
        return {
            'selections': self.selections,
            'total_pulls': self.total_pulls,
            'phase': self.phase,
            'algorithm': 'LinUCB',
            'note': 'State managed by ContextualBandit, saved to separate file'
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], presets: Optional[List[Dict[str, Any]]] = None) -> 'BanditState':
        """Deserialize state (for compatibility)"""
        state = cls(presets)
        state.selections = data.get('selections', state.selections)
        state.total_pulls = data.get('total_pulls', 0)
        state.phase = data.get('phase', 'exploration')
        return state


def select_preset(
    features: Dict[str, Any],
    state_path: str = "~/.aria_contextual_bandit.json",
    epsilon: float = EPSILON
) -> Tuple[Preset, str, Dict[str, Any]]:
    """
    Select preset using LinUCB with epsilon-greedy exploration

    API-compatible replacement for Thompson Sampling version.

    Args:
        features: Query features from QueryFeatureExtractor
        state_path: Path to persistent state file
        epsilon: Probability of random exploration (default 0.10 = 10%)

    Returns:
        (selected_preset, selection_reason, metadata)
    """
    state_path_expanded = Path(os.path.expanduser(state_path))

    # Initialize or load LinUCB bandit
    arms = [p["name"] for p in DEFAULT_PRESETS]
    bandit = ContextualBandit(
        arms=arms,
        feature_dim=10,
        alpha=1.0,
        state_path=state_path_expanded
    )

    # Determine phase based on number of observations
    if bandit.n_observations < 20:
        phase = "exploration"
    else:
        phase = "exploitation"

    # Select arm using epsilon-greedy LinUCB
    arm_name, ucb_score, feature_vector, selection_method = bandit.select_arm_epsilon_greedy(
        query_context=features,
        epsilon=epsilon
    )

    # Convert arm name to Preset object
    preset_dict = next(p for p in DEFAULT_PRESETS if p["name"] == arm_name)
    selected_preset = Preset(preset_dict["name"], preset_dict["args"])

    # Create reason string (for logging/debugging)
    if selection_method == "epsilon_random":
        reason = f"epsilon-greedy random (ε={epsilon}) → {arm_name}"
    else:
        reason = f"LinUCB UCB (score={ucb_score:.3f}) → {arm_name}"

    # Metadata (compatible with Thompson version)
    meta = {
        'phase': phase,
        'total_pulls': bandit.n_observations,
        'selection_method': selection_method,
        'epsilon': epsilon,
        'ucb_score': ucb_score,
        'feature_vector': feature_vector.tolist(),  # For debugging
        'algorithm': 'LinUCB'
    }

    return selected_preset, reason, meta


def give_reward(
    preset_name: str,
    reward: float,
    state_path: str = "~/.aria_contextual_bandit.json",
    features: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update bandit state with reward

    LinUCB version - requires features for proper updating.
    Features are optional for backward compatibility but recommended.

    Args:
        preset_name: Name of preset that was used
        reward: Reward value (0.0-1.0)
        state_path: Path to persistent state file
        features: Query features (REQUIRED for LinUCB, optional for compatibility)
    """
    state_path_expanded = Path(os.path.expanduser(state_path))

    if not state_path_expanded.exists():
        # No state to update
        return

    try:
        # Load LinUCB bandit
        arms = [p["name"] for p in DEFAULT_PRESETS]
        bandit = ContextualBandit(
            arms=arms,
            feature_dim=10,
            alpha=1.0,
            state_path=state_path_expanded
        )

        if features is None:
            # Fallback: use default/zero features (not ideal but maintains compatibility)
            print(f"[WARNING] give_reward called without features - using defaults", flush=True)
            features = {
                "complexity": "moderate",
                "domain": "default",
                "length": 50
            }

        # Extract feature vector
        feature_vector = bandit.extract_features(features)

        # Update LinUCB matrices
        bandit.update(
            arm_name=preset_name,
            features=feature_vector,
            reward=reward,
            query_context=features
        )

        # State is automatically saved by ContextualBandit._save_state()

    except Exception as e:
        # Silently fail to maintain compatibility
        import sys
        print(f"[WARNING] LinUCB update failed: {e}", file=sys.stderr, flush=True)


# ============================================================================
# Utility Functions (for compatibility)
# ============================================================================

def map_features_to_candidates(features: Dict[str, Any]) -> List[str]:
    """
    Legacy function from Thompson implementation.

    LinUCB doesn't need rule-based candidate filtering since it learns
    patterns automatically. This is kept for API compatibility only.

    Returns all presets as candidates.
    """
    # LinUCB learns which presets work for which features automatically,
    # so we don't need manual filtering rules
    return [p["name"] for p in DEFAULT_PRESETS]


__all__ = ['select_preset', 'give_reward', 'BanditState', 'map_features_to_candidates']
