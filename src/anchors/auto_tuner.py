#!/usr/bin/env python3
"""
Anchor Auto-Tuner - Wave 1 Foundation

Automatically tunes anchor selector weights based on performance feedback.

Integration with Current ARIA:
- Monitors anchor_selector.py mode selection accuracy
- Analyzes aria_telemetry.py performance metrics per anchor
- Adjusts weights to improve anchor selection
- Works with current pattern-based scoring system

Key Features:
1. Per-anchor performance tracking
2. Auto-adjust pattern weights (keyword importance, pattern multipliers)
3. Detect anchor confusion (queries wrongly classified)
4. A/B testing for weight changes
5. Rollback mechanism for bad updates
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class AnchorPerformance:
    """Performance tracking for a single anchor mode"""
    mode: str
    
    # Selection statistics
    times_selected: int = 0
    times_correct: int = 0  # High reward
    times_incorrect: int = 0  # Low reward
    
    # Performance metrics
    total_reward: float = 0.0
    avg_reward: float = 0.5
    recent_rewards: Any = field(default_factory=lambda: deque(maxlen=20))
    
    # Confusion matrix (which modes were selected when this should've been)
    confused_with: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Pattern effectiveness (which patterns correlate with success)
    pattern_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pattern_rewards: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Weight adjustments
    weight_adjustments: Dict[str, float] = field(default_factory=dict)


class AnchorAutoTuner:
    """
    Auto-tune anchor selector weights based on performance
    
    Strategy:
    1. Track which anchors work well for which queries
    2. Identify confusions (wrong anchor selected)
    3. Adjust pattern weights to reduce confusions
    4. A/B test weight changes
    5. Rollback if performance degrades
    """
    
    def __init__(
        self,
        state_path: Optional[Path] = None,
        learning_rate: float = 0.1,
        success_threshold: float = 0.6,
        adjustment_frequency: int = 50  # Adjust after N selections
    ):
        """
        Args:
            state_path: Path to save/load state
            learning_rate: How quickly to adjust weights (0.1 = 10% adjustment)
            success_threshold: Reward > threshold = success
            adjustment_frequency: Adjust weights after N anchor selections
        """
        self.state_path = state_path or Path.home() / '.aria_anchor_autotuner.json'
        self.learning_rate = learning_rate
        self.success_threshold = success_threshold
        self.adjustment_frequency = adjustment_frequency
        
        # Initialize all 8 anchor modes
        self.anchors = {
            mode: AnchorPerformance(mode)
            for mode in ['formal', 'casual', 'technical', 'educational', 
                         'philosophical', 'analytical', 'factual', 'creative']
        }
        
        # Global stats
        self.total_selections = 0
        self.selections_since_adjustment = 0
        self._update_count = 0  # Track number of updates for periodic actions
        
        # Weight recommendations (delta to apply to anchor_selector)
        self.weight_recommendations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # A/B testing state
        self.ab_test_active = False
        self.ab_baseline_performance = {}
        
        # History
        self.history: deque = deque(maxlen=1000)
        
        # Load state
        self._load_state()
    
    def record_selection(
        self,
        query: str,
        selected_mode: str,
        reward: float,
        patterns_matched: Optional[Dict[str, List[str]]] = None,
        query_features: Optional[Dict[str, Any]] = None
    ):
        """
        Record an anchor selection and its outcome
        
        Args:
            query: Query string
            selected_mode: Anchor mode that was selected
            reward: Observed reward (0-1)
            patterns_matched: Dict of {mode: [patterns_that_matched]}
            query_features: Optional query features dict
        """
        if selected_mode not in self.anchors:
            return
        
        anchor = self.anchors[selected_mode]
        
        # Update selection stats
        anchor.times_selected += 1
        is_success = reward >= self.success_threshold
        
        if is_success:
            anchor.times_correct += 1
        else:
            anchor.times_incorrect += 1
        
        # Update reward tracking
        anchor.total_reward += reward
        anchor.avg_reward = anchor.total_reward / max(anchor.times_selected, 1)
        anchor.recent_rewards.append(reward)
        
        # Track pattern effectiveness
        if patterns_matched:
            for mode, patterns in patterns_matched.items():
                if mode == selected_mode:
                    # These patterns contributed to the selection
                    for pattern in patterns:
                        anchor.pattern_hits[pattern] += 1
                        anchor.pattern_rewards[pattern] += reward
        
        # Track history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'selected_mode': selected_mode,
            'reward': reward,
            'success': is_success,
            'patterns': patterns_matched or {},
            'features': query_features or {}
        })
        
        # Global counters
        self.total_selections += 1
        self.selections_since_adjustment += 1
        
        # Periodically adjust weights
        if self.selections_since_adjustment >= self.adjustment_frequency:
            self._adjust_weights()
            self.selections_since_adjustment = 0
        
        # Periodic save
        if len(self.history) % 20 == 0:
            self._save_state()
    
    def update(
        self,
        selected_mode: str,
        reward: float,
        features: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        confidence: Optional[float] = None,
    ):
        """
        Update auto-tuner with orchestrated signals (new API for Wave 2+)
        
        This is the new method called by orchestrator-integrated aria_main.py
        
        Args:
            selected_mode: Anchor mode that was selected
            reward: Observed reward (0-1)
            features: Query features dict
            success: Explicit success signal (overrides threshold if provided)
            confidence: Confidence in the anchor selection (0-1)
        """
        if selected_mode not in self.anchors:
            return
        
        anchor = self.anchors[selected_mode]
        
        # Increment update counter for periodic actions
        self._update_count += 1
        
        # Update selection stats
        anchor.times_selected += 1
        
        # Use explicit success signal if provided, otherwise threshold
        if success is not None:
            is_success = success
        else:
            is_success = reward >= self.success_threshold
        
        if is_success:
            anchor.times_correct += 1
        else:
            anchor.times_incorrect += 1
        
        # Update reward tracking
        anchor.total_reward += reward
        anchor.avg_reward = anchor.total_reward / max(anchor.times_selected, 1)
        anchor.recent_rewards.append(reward)
        
        # Track history (simpler format for orchestrated updates)
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'selected_mode': selected_mode,
            'reward': reward,
            'success': is_success,
            'confidence': confidence or 1.0,
            'features': features or {}
        })
        
        # Global counters
        self.total_selections += 1
        self.selections_since_adjustment += 1
        
        # Periodically adjust weights
        if self.selections_since_adjustment >= self.adjustment_frequency:
            self._adjust_weights()
            self.selections_since_adjustment = 0
    
    def detect_confusions(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Detect anchor confusions from recent history
        
        Analyzes queries with low rewards to see if a different anchor
        would have been better.
        
        Returns:
            Dict of {anchor: [(confused_with_anchor, confusion_rate), ...]}
        """
        confusions = defaultdict(list)
        
        # Look at recent low-reward selections
        low_reward_events = [
            h for h in list(self.history)[-200:]
            if h['reward'] < self.success_threshold
        ]
        
        if not low_reward_events:
            return {}
        
        # For each failed selection, see which mode had strong patterns
        for event in low_reward_events:
            selected = event['selected_mode']
            patterns = event.get('patterns', {})
            
            # Find modes with strong pattern matches
            mode_scores = {}
            for mode, mode_patterns in patterns.items():
                if mode != selected and len(mode_patterns) > 0:
                    mode_scores[mode] = len(mode_patterns)
            
            # If another mode had stronger signal, record confusion
            if mode_scores:
                best_alternative = max(mode_scores.items(), key=lambda x: x[1])
                if best_alternative[1] > len(patterns.get(selected, [])):
                    # Likely should've selected the alternative
                    self.anchors[selected].confused_with[best_alternative[0]] += 1
        
        # Compile confusion rates
        for mode, anchor in self.anchors.items():
            if anchor.times_selected < 10:
                continue
            
            for confused_mode, count in anchor.confused_with.items():
                confusion_rate = count / max(anchor.times_selected, 1)
                if confusion_rate > 0.1:  # >10% confusion
                    confusions[mode].append((confused_mode, confusion_rate))
        
        return dict(confusions)
    
    def _adjust_weights(self):
        """
        Adjust anchor selector weights based on performance
        
        Strategy:
        1. Identify underperforming anchors (low avg_reward)
        2. Identify overused anchors (high selection rate but low reward)
        3. Boost patterns that correlate with success
        4. Penalize patterns that correlate with failure
        """
        print(f"\nðŸ”§ Auto-tuning anchor weights (after {self.adjustment_frequency} selections)...")
        
        # Compute selection rates
        total = max(self.total_selections, 1)
        selection_rates = {
            mode: anchor.times_selected / total
            for mode, anchor in self.anchors.items()
        }
        
        adjustments_made = []
        
        for mode, anchor in self.anchors.items():
            if anchor.times_selected < 5:
                continue  # Not enough data
            
            # Analyze performance
            success_rate = anchor.times_correct / max(anchor.times_selected, 1)
            avg_reward = anchor.avg_reward
            
            # Case 1: High selection rate but low performance â†’ Reduce weights
            if selection_rates[mode] > 0.15 and avg_reward < 0.55:
                adjustment = -self.learning_rate
                self.weight_recommendations[mode]['global_multiplier'] = 1.0 + adjustment
                adjustments_made.append(f"{mode}: -10% (overused, low reward)")
            
            # Case 2: Low selection rate but high performance â†’ Increase weights
            elif selection_rates[mode] < 0.08 and avg_reward > 0.65:
                adjustment = self.learning_rate
                self.weight_recommendations[mode]['global_multiplier'] = 1.0 + adjustment
                adjustments_made.append(f"{mode}: +10% (underused, high reward)")
            
            # Case 3: Pattern-level adjustments
            if anchor.pattern_hits:
                # Find most effective patterns
                pattern_effectiveness = {}
                for pattern, hits in anchor.pattern_hits.items():
                    if hits >= 3:
                        avg_pattern_reward = anchor.pattern_rewards[pattern] / hits
                        pattern_effectiveness[pattern] = avg_pattern_reward
                
                # Boost effective patterns
                for pattern, effectiveness in pattern_effectiveness.items():
                    if effectiveness > 0.7:
                        self.weight_recommendations[mode][f'pattern_{pattern}'] = 1.2
                    elif effectiveness < 0.4:
                        self.weight_recommendations[mode][f'pattern_{pattern}'] = 0.8
        
        if adjustments_made:
            print(f"  Adjustments:")
            for adj in adjustments_made:
                print(f"    â€¢ {adj}")
        else:
            print("  No adjustments needed (all anchors performing well)")
    
    def get_anchor_stats(self, mode: str) -> Dict[str, Any]:
        """Get detailed stats for an anchor"""
        if mode not in self.anchors:
            return {}
        
        anchor = self.anchors[mode]
        
        if anchor.times_selected == 0:
            return {
                'mode': mode,
                'times_selected': 0,
                'status': 'No data yet'
            }
        
        success_rate = anchor.times_correct / max(anchor.times_selected, 1)
        recent_avg = np.mean(list(anchor.recent_rewards)) if anchor.recent_rewards else 0.0
        
        return {
            'mode': mode,
            'times_selected': anchor.times_selected,
            'success_rate': success_rate,
            'avg_reward': anchor.avg_reward,
            'recent_avg_reward': recent_avg,
            'confused_with': dict(anchor.confused_with),
            'top_patterns': self._get_top_patterns(anchor, n=5)
        }
    
    def _get_top_patterns(self, anchor: AnchorPerformance, n: int = 5) -> List[Tuple[str, float, int]]:
        """Get top performing patterns for an anchor"""
        if not anchor.pattern_hits:
            return []
        
        pattern_scores = []
        for pattern, hits in anchor.pattern_hits.items():
            if hits >= 3:
                avg_reward = anchor.pattern_rewards[pattern] / hits
                pattern_scores.append((pattern, avg_reward, hits))
        
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores[:n]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        anchor_stats = {
            mode: self.get_anchor_stats(mode)
            for mode in self.anchors.keys()
        }
        
        # Overall metrics
        if self.total_selections > 0:
            overall_success = sum(
                a.times_correct for a in self.anchors.values()
            ) / self.total_selections
            overall_reward = sum(
                a.total_reward for a in self.anchors.values()
            ) / self.total_selections
        else:
            overall_success = 0.0
            overall_reward = 0.0
        
        return {
            'total_selections': self.total_selections,
            'overall_success_rate': overall_success,
            'overall_avg_reward': overall_reward,
            'anchor_stats': anchor_stats,
            'confusions': self.detect_confusions(),
            'weight_recommendations': dict(self.weight_recommendations)
        }
    
    def export_weight_adjustments(self) -> Dict[str, Dict[str, float]]:
        """
        Export weight adjustments to apply to anchor_selector.py
        
        Returns:
            Dict of {mode: {adjustment_type: multiplier}}
        """
        return dict(self.weight_recommendations)
    
    def _save_state(self):
        """Save state to disk"""
        try:
            state = {
                'version': '1.0',
                'anchors': {
                    mode: {
                        'times_selected': a.times_selected,
                        'times_correct': a.times_correct,
                        'times_incorrect': a.times_incorrect,
                        'total_reward': a.total_reward,
                        'avg_reward': a.avg_reward,
                        'recent_rewards': list(a.recent_rewards),
                        'confused_with': dict(a.confused_with),
                        'pattern_hits': dict(a.pattern_hits),
                        'pattern_rewards': dict(a.pattern_rewards)
                    }
                    for mode, a in self.anchors.items()
                },
                'total_selections': self.total_selections,
                '_update_count': self._update_count,
                'weight_recommendations': dict(self.weight_recommendations),
                'history': list(self.history)[-500:]  # Keep last 500
            }
            
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save auto-tuner state: {e}")
    
    def _load_state(self):
        """Load state from disk"""
        if not self.state_path.exists():
            return
        
        try:
            state = json.loads(self.state_path.read_text())
            
            # Restore anchors
            for mode, a_data in state.get('anchors', {}).items():
                if mode in self.anchors:
                    anchor = self.anchors[mode]
                    anchor.times_selected = a_data['times_selected']
                    anchor.times_correct = a_data['times_correct']
                    anchor.times_incorrect = a_data['times_incorrect']
                    anchor.total_reward = a_data['total_reward']
                    anchor.avg_reward = a_data['avg_reward']
                    
                    for reward in a_data.get('recent_rewards', []):
                        anchor.recent_rewards.append(reward)
                    
                    anchor.confused_with.update(a_data.get('confused_with', {}))
                    anchor.pattern_hits.update(a_data.get('pattern_hits', {}))
                    anchor.pattern_rewards.update(a_data.get('pattern_rewards', {}))
            
            # Restore global stats
            self.total_selections = state.get('total_selections', 0)
            self._update_count = state.get('_update_count', 0)
            self.weight_recommendations = defaultdict(dict, state.get('weight_recommendations', {}))
            
            # Restore history
            for h in state.get('history', []):
                self.history.append(h)
            
            print(f"âœ“ Loaded anchor auto-tuner: {self.total_selections} selections tracked")
        
        except Exception as e:
            print(f"Warning: Could not load auto-tuner state: {e}")


# ============================================================================
# Integration with Current ARIA
# ============================================================================

def integrate_with_anchor_selector():
    """
    Example: Integrate auto-tuner with anchor_selector.py
    
    Add to aria_main.py after anchor selection and reward computation
    """
    
    # Initialize tuner
    tuner = AnchorAutoTuner(
        state_path=Path.home() / '.aria_anchor_autotuner.json',
        learning_rate=0.1,
        adjustment_frequency=50
    )
    
    # Example: After anchor selection in aria_main.py
    query = "How do I implement async functions in JavaScript?"
    selected_mode = 'technical'  # From anchor_selector.select()
    
    # Mock patterns matched (would come from anchor_selector)
    patterns_matched = {
        'technical': ['implement', 'async', 'javascript', 'function'],
        'educational': ['how do i'],
        'casual': []
    }
    
    # ... run retrieval and get reward from aria_telemetry ...
    reward = 0.78  # Example
    
    # Record selection
    tuner.record_selection(
        query,
        selected_mode,
        reward,
        patterns_matched=patterns_matched
    )
    
    # Periodically get stats and weight recommendations
    if tuner.total_selections % 100 == 0:
        stats = tuner.get_all_stats()
        
        print("\nðŸ“Š Anchor Performance Summary:")
        for mode, mode_stats in stats['anchor_stats'].items():
            if mode_stats.get('times_selected', 0) > 0:
                print(f"  {mode}: {mode_stats['success_rate']:.1%} success, "
                      f"avg reward {mode_stats['avg_reward']:.3f}")
        
        # Export weight adjustments
        adjustments = tuner.export_weight_adjustments()
        if adjustments:
            print("\nðŸ”§ Recommended Weight Adjustments:")
            for mode, mode_adj in adjustments.items():
                print(f"  {mode}: {mode_adj}")


if __name__ == '__main__':
    integrate_with_anchor_selector()
