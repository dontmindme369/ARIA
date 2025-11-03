#!/usr/bin/env python3
"""
bandit_context.py - Multi-armed bandit for preset selection
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Implements Thompson Sampling with context features for adaptive preset selection
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from presets import Preset


# Default presets for retrieval strategies
DEFAULT_PRESETS = [
    {"name": "fast", "args": {"top_k": 40, "sem_limit": 64, "rotations": 1}},
    {"name": "balanced", "args": {"top_k": 64, "sem_limit": 128, "rotations": 2}},
    {"name": "deep", "args": {"top_k": 96, "sem_limit": 256, "rotations": 3}},
    {"name": "diverse", "args": {"top_k": 80, "sem_limit": 128, "rotations": 2, "max_per_file": 6}},
]


class BanditState:
    """Thompson Sampling bandit state"""
    
    def __init__(self, presets: Optional[List[Dict[str, Any]]] = None):
        self.presets = [Preset(p['name'], p['args']) for p in (presets or DEFAULT_PRESETS)]
        
        # Thompson Sampling: track alpha (successes) and beta (failures)
        self.alpha = {p.name: 1.0 for p in self.presets}  # Prior: 1 success
        self.beta = {p.name: 1.0 for p in self.presets}   # Prior: 1 failure
        
        # Track selections and rewards
        self.selections = {p.name: 0 for p in self.presets}
        self.total_reward = {p.name: 0.0 for p in self.presets}
        
        # Phase tracking (exploration vs exploitation)
        self.total_pulls = 0
        self.phase = "exploration"  # or "exploitation"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'selections': self.selections,
            'total_reward': self.total_reward,
            'total_pulls': self.total_pulls,
            'phase': self.phase
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], presets: Optional[List[Dict[str, Any]]] = None) -> 'BanditState':
        """Deserialize state"""
        state = cls(presets)
        state.alpha = data.get('alpha', state.alpha)
        state.beta = data.get('beta', state.beta)
        state.selections = data.get('selections', state.selections)
        state.total_reward = data.get('total_reward', state.total_reward)
        state.total_pulls = data.get('total_pulls', 0)
        state.phase = data.get('phase', 'exploration')
        return state


def select_preset(
    features: Dict[str, Any],
    state_path: str = "~/.rag_bandit_state.json"
) -> Tuple[Preset, str, Dict[str, Any]]:
    """
    Select preset using Thompson Sampling
    
    Args:
        features: Query features (not used in basic implementation)
        state_path: Path to persistent state file
    
    Returns:
        (selected_preset, selection_reason, metadata)
    """
    state_path_expanded = Path(os.path.expanduser(state_path))
    
    # Load or initialize state
    if state_path_expanded.exists():
        try:
            with open(state_path_expanded, 'r') as f:
                data = json.load(f)
            state = BanditState.from_dict(data)
        except:
            state = BanditState()
    else:
        state = BanditState()
    
    # Determine phase
    state.total_pulls += 1
    if state.total_pulls < 20:
        state.phase = "exploration"
    else:
        state.phase = "exploitation"
    
    # Thompson Sampling: sample from Beta distribution
    samples = {}
    for preset in state.presets:
        # Sample from Beta(alpha, beta)
        alpha = state.alpha[preset.name]
        beta = state.beta[preset.name]
        sample = random.betavariate(alpha, beta)
        samples[preset.name] = sample
    
    # Select preset with highest sample
    selected_name = max(samples.keys(), key=lambda k: samples[k])
    selected_preset = next(p for p in state.presets if p.name == selected_name)
    
    # Track selection
    state.selections[selected_name] += 1
    
    # Save state
    state_path_expanded.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path_expanded, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
    
    reason = f"Thompson sample: {samples[selected_name]:.3f}"
    meta = {
        'phase': state.phase,
        'total_pulls': state.total_pulls,
        'samples': samples
    }
    
    return selected_preset, reason, meta


def give_reward(
    preset_name: str,
    reward: float,
    state_path: str = "~/.rag_bandit_state.json"
) -> None:
    """
    Update bandit state with reward
    
    Args:
        preset_name: Name of preset that was used
        reward: Reward value (0.0-1.0)
        state_path: Path to persistent state file
    """
    state_path_expanded = Path(os.path.expanduser(state_path))
    
    if not state_path_expanded.exists():
        return
    
    try:
        with open(state_path_expanded, 'r') as f:
            data = json.load(f)
        state = BanditState.from_dict(data)
        
        # Update Thompson Sampling parameters
        if reward > 0.5:
            # Success: increment alpha
            state.alpha[preset_name] += reward
        else:
            # Failure: increment beta
            state.beta[preset_name] += (1.0 - reward)
        
        # Track total reward
        state.total_reward[preset_name] += reward
        
        # Save updated state
        with open(state_path_expanded, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
    except Exception as e:
        pass  # Silently fail


__all__ = ['select_preset', 'give_reward', 'BanditState']
