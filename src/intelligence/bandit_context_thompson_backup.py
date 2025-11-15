#!/usr/bin/env python3
"""
bandit_context.py - Query-Aware Multi-Armed Bandit

Implements Thompson Sampling with QUERY-AWARE preset selection:
- Maps query features to preset categories
- Thompson samples within appropriate category
- Adds epsilon-greedy for forced exploration
- Prevents "lock-in" on single preset

FIXES:
1. Different queries get different presets (not always "fast")
2. All presets get regular trials (no starvation)
3. Still learns optimal within each query type
4. 10% epsilon-greedy prevents permanent lock-in
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.presets import Preset


# Default presets for retrieval strategies
DEFAULT_PRESETS = [
    {"name": "fast", "args": {"top_k": 40, "sem_limit": 64, "rotations": 1, "max_per_file": 8}},
    {"name": "balanced", "args": {"top_k": 64, "sem_limit": 128, "rotations": 2, "max_per_file": 6}},
    {"name": "deep", "args": {"top_k": 96, "sem_limit": 256, "rotations": 3, "max_per_file": 5}},
    {"name": "diverse", "args": {"top_k": 80, "sem_limit": 128, "rotations": 2, "max_per_file": 4}},
]

# EPSILON-GREEDY: 10% random exploration even in exploitation phase
EPSILON = 0.10


def map_features_to_candidates(features: Dict[str, Any]) -> List[str]:
    """
    Map query features to candidate presets
    
    This is the KEY FIX that makes selection query-aware:
    - Simple queries → [fast, balanced]
    - Explanations → [balanced, deep]
    - Research → [deep, diverse]
    - Creative → [diverse, balanced]
    
    Returns:
        List of preset names that are appropriate for this query
    """
    # Extract features
    complexity = features.get('complexity', 'moderate')
    domain = features.get('domain', 'default')
    length = features.get('length', 0)
    word_count = features.get('word_count', 0)
    tech_density = features.get('technical_density', 0.0)
    has_multiple_questions = features.get('has_multiple_questions', False)
    
    # Default: all presets available
    candidates = ["fast", "balanced", "deep", "diverse"]
    
    # RULE 1: Simple + Short → Fast or Balanced
    if complexity == 'simple' and length < 100:
        candidates = ["fast", "balanced"]
    
    # RULE 2: Complex queries → Deep or Diverse
    elif complexity == 'complex':
        candidates = ["deep", "diverse"]
    
    # RULE 3: Technical queries → Balanced or Deep
    elif tech_density > 0.3:
        candidates = ["balanced", "deep"]
    
    # RULE 4: Multiple questions → Deep (needs thoroughness)
    elif has_multiple_questions:
        candidates = ["deep", "diverse"]
    
    # RULE 5: Conceptual explanations → Balanced or Deep
    elif domain == 'concept':
        candidates = ["balanced", "deep"]
    
    # RULE 6: Code/factual → Fast or Balanced (precise answers)
    elif domain in ['code', 'factual']:
        candidates = ["fast", "balanced"]
    
    # RULE 7: Research/paper → Deep or Diverse
    elif domain in ['paper', 'research']:
        candidates = ["deep", "diverse"]
    
    # RULE 8: Long queries (200+ chars) → Deep
    elif length > 200:
        candidates = ["deep", "diverse"]
    
    # RULE 9: Very short queries (< 20 chars) → Fast
    elif length < 20:
        candidates = ["fast", "balanced"]
    
    return candidates


class BanditState:
    """Thompson Sampling bandit state with query-aware selection"""
    
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
        
        # NEW: Track query-type based selections
        self.category_selections = {}  # Track which category led to each selection
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'selections': self.selections,
            'total_reward': self.total_reward,
            'total_pulls': self.total_pulls,
            'phase': self.phase,
            'category_selections': self.category_selections
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
        state.category_selections = data.get('category_selections', {})
        return state


def select_preset(
    features: Dict[str, Any],
    state_path: str = "~/.rag_bandit_state.json",
    epsilon: float = EPSILON
) -> Tuple[Preset, str, Dict[str, Any]]:
    """
    Select preset using QUERY-AWARE Thompson Sampling
    
    NEW BEHAVIOR:
    1. Map query features to candidate presets
    2. With probability (1-epsilon): Thompson sample from candidates
    3. With probability epsilon: Random selection (forced exploration)
    
    Args:
        features: Query features from QueryFeatureExtractor
        state_path: Path to persistent state file
        epsilon: Probability of random exploration (default 0.10 = 10%)
    
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
    
    # NEW: Map query features to candidate presets
    candidates = map_features_to_candidates(features)
    
    # Epsilon-greedy: Random exploration
    if random.random() < epsilon:
        # FORCED RANDOM EXPLORATION
        selected_name = random.choice([p.name for p in state.presets])
        selected_preset = next(p for p in state.presets if p.name == selected_name)
        reason = f"epsilon-greedy random (ε={epsilon})"
        selection_method = "epsilon_random"
    else:
        # QUERY-AWARE THOMPSON SAMPLING
        # Only sample from candidate presets appropriate for this query
        samples = {}
        for preset in state.presets:
            if preset.name in candidates:
                # Sample from Beta(alpha, beta)
                alpha = state.alpha[preset.name]
                beta = state.beta[preset.name]
                sample = random.betavariate(alpha, beta)
                samples[preset.name] = sample
        
        # Select preset with highest sample from candidates
        if samples:
            selected_name = max(samples.keys(), key=lambda k: samples[k])
        else:
            # Fallback if no candidates (shouldn't happen)
            selected_name = "balanced"
        
        selected_preset = next(p for p in state.presets if p.name == selected_name)
        reason = f"query-aware Thompson ({candidates}) → {selected_name} ({samples.get(selected_name, 0.0):.3f})"
        selection_method = "query_aware_thompson"
    
    # Track selection
    state.selections[selected_name] += 1
    
    # Track which candidates led to this selection
    candidates_key = ",".join(sorted(candidates))
    if candidates_key not in state.category_selections:
        state.category_selections[candidates_key] = {}
    if selected_name not in state.category_selections[candidates_key]:
        state.category_selections[candidates_key][selected_name] = 0
    state.category_selections[candidates_key][selected_name] += 1
    
    # Save state
    state_path_expanded.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path_expanded, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)
    
    meta = {
        'phase': state.phase,
        'total_pulls': state.total_pulls,
        'candidates': candidates,
        'selection_method': selection_method,
        'epsilon': epsilon
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


__all__ = ['select_preset', 'give_reward', 'BanditState', 'map_features_to_candidates']
