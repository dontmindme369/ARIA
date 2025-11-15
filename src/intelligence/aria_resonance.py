#!/usr/bin/env python3
"""
ARIA Resonance Detector - Phase 6
Detects when system achieves stable resonance and extracts successful patterns
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque
import statistics


class ResonanceDetector:
    """
    Detect when ARIA reaches stable resonance state
    
    Resonance = high coverage + low gaps + high exemplar fit + system confidence
    Tracks history to identify stable patterns vs transient spikes
    """
    
    def __init__(
        self, 
        history_size: int = 10,
        stable_threshold: int = 5,
        resonance_floor: float = 0.85
    ):
        self.resonance_history = deque(maxlen=history_size)
        self.stable_threshold = stable_threshold
        self.resonance_floor = resonance_floor
        
        # Track what configurations led to resonance
        self.resonance_patterns: List[Dict[str, Any]] = []
        self.last_stable_config: Optional[Dict[str, Any]] = None
    
    def measure_resonance(
        self,
        coverage_score: float,
        gap_score: float,
        exemplar_fit: float,
        total_chunks: int,
        unique_sources: int,
        query_features: Dict[str, Any],
        config_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate resonance score from multiple signals
        
        Returns:
            {
                'resonance': float,        # 0.0 - 1.0
                'is_stable': bool,         # Sustained high resonance
                'pattern_detected': bool,  # New resonance pattern found
                'confidence': float        # Statistical confidence
            }
        """
        # Core resonance calculation
        # High coverage + low gaps + high fit = resonance
        resonance = (
            coverage_score * 0.35 +           # Context completeness
            (1 - gap_score) * 0.35 +          # Low gaps = high resonance
            exemplar_fit * 0.20 +             # Pattern alignment
            min(total_chunks / 30, 1.0) * 0.05 +  # Sufficient context
            min(unique_sources / 5, 1.0) * 0.05   # Source diversity
        )
        
        # Add to history
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'resonance': resonance,
            'coverage': coverage_score,
            'gap_score': gap_score,
            'exemplar_fit': exemplar_fit,
            'total_chunks': total_chunks,
            'unique_sources': unique_sources,
            'query_features': query_features,
            'config': config_snapshot
        }
        self.resonance_history.append(measurement)
        
        # Check for stable resonance
        is_stable = self._check_stability()
        pattern_detected = False
        
        if is_stable and not self.last_stable_config:
            # New stable resonance pattern detected
            pattern_detected = True
            pattern = self._extract_resonance_pattern()
            self.resonance_patterns.append(pattern)
            self.last_stable_config = config_snapshot.copy()
        
        # Calculate confidence (how stable is the measurement)
        confidence = self._calculate_confidence()
        
        return {
            'resonance': resonance,
            'is_stable': is_stable,
            'pattern_detected': pattern_detected,
            'confidence': confidence,
            'measurements': {
                'coverage': coverage_score,
                'gap_score': gap_score,
                'exemplar_fit': exemplar_fit,
                'total_chunks': total_chunks,
                'unique_sources': unique_sources
            }
        }
    
    def _check_stability(self) -> bool:
        """Check if recent measurements show stable high resonance"""
        if len(self.resonance_history) < self.stable_threshold:
            return False
        
        recent = list(self.resonance_history)[-self.stable_threshold:]
        recent_scores = [m['resonance'] for m in recent]
        
        # All recent measurements must be above floor
        all_high = all(r >= self.resonance_floor for r in recent_scores)
        
        # Low variance (stable, not spiking)
        variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0
        low_variance = variance < 0.01
        
        return all_high and low_variance
    
    def _calculate_confidence(self) -> float:
        """Statistical confidence in resonance measurement"""
        if len(self.resonance_history) < 3:
            return 0.5  # Not enough data
        
        recent = list(self.resonance_history)[-5:]
        scores = [m['resonance'] for m in recent]
        
        # Higher confidence = lower variance + more samples
        variance = statistics.variance(scores) if len(scores) > 1 else 1.0
        sample_factor = min(len(self.resonance_history) / 10, 1.0)
        
        confidence = (1 - min(variance, 1.0)) * sample_factor
        return max(0.1, min(confidence, 1.0))
    
    def _extract_resonance_pattern(self) -> Dict[str, Any]:
        """Extract the configuration pattern that led to stable resonance"""
        if not self.resonance_history:
            return {}
        
        # Get recent high-resonance measurements
        recent = [m for m in list(self.resonance_history)[-self.stable_threshold:] 
                  if m['resonance'] >= self.resonance_floor]
        
        if not recent:
            return {}
        
        # Find common configuration elements
        configs = [m['config'] for m in recent]
        query_features = [m['query_features'] for m in recent]
        
        # Extract pattern
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'avg_resonance': statistics.mean([m['resonance'] for m in recent]),
            'sample_size': len(recent),
            'common_config': self._find_common_config(configs),
            'query_characteristics': self._aggregate_query_features(query_features),
            'measurements': {
                'avg_coverage': statistics.mean([m['coverage'] for m in recent]),
                'avg_gap_score': statistics.mean([m['gap_score'] for m in recent]),
                'avg_exemplar_fit': statistics.mean([m['exemplar_fit'] for m in recent]),
                'avg_chunks': statistics.mean([m['total_chunks'] for m in recent]),
                'avg_sources': statistics.mean([m['unique_sources'] for m in recent])
            }
        }
        
        return pattern
    
    def _find_common_config(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find configuration parameters common to all high-resonance states"""
        if not configs:
            return {}
        
        common = {}
        
        # Check preset
        presets = [c.get('preset') for c in configs if c.get('preset')]
        if presets and len(set(presets)) == 1:
            common['preset'] = presets[0]
        
        # Check anchor mode
        modes = [c.get('reasoning_mode') for c in configs if c.get('reasoning_mode')]
        if modes and len(set(modes)) == 1:
            common['reasoning_mode'] = modes[0]
        
        # Check flags
        flags = [c.get('flags', {}) for c in configs]
        if flags:
            # Average numeric values
            for key in ['top_k', 'sem_limit', 'rotations', 'rotation_subspace']:
                values = [f.get(key) for f in flags if f.get(key) is not None]
                if values:
                    common[key] = statistics.mean(values)
        
        return common
    
    def _aggregate_query_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate query feature patterns"""
        if not features_list:
            return {}
        
        # Find common query characteristics
        agg = {
            'avg_length': statistics.mean([f.get('length', 0) for f in features_list]),
            'avg_word_count': statistics.mean([f.get('word_count', 0) for f in features_list])
        }
        
        # Check for common question types
        is_question = [f.get('is_question', False) for f in features_list]
        if is_question:
            agg['question_ratio'] = sum(is_question) / len(is_question)
        
        return agg
    
    def get_resonance_patterns(self, min_resonance: float = 0.85) -> List[Dict[str, Any]]:
        """Get all detected resonance patterns above threshold"""
        return [p for p in self.resonance_patterns 
                if p.get('avg_resonance', 0) >= min_resonance]
    
    def save_patterns(self, path: Path):
        """Save detected resonance patterns to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                'patterns': self.resonance_patterns,
                'last_stable_config': self.last_stable_config,
                'history_size': len(self.resonance_history)
            }, indent=2),
            encoding='utf-8'
        )
    
    def load_patterns(self, path: Path):
        """Load previously detected patterns"""
        if not path.exists():
            return
        
        data = json.loads(path.read_text(encoding='utf-8'))
        self.resonance_patterns = data.get('patterns', [])
        self.last_stable_config = data.get('last_stable_config')


class SelfTuningEngine:
    """
    Automatically adjust ARIA parameters based on resonance patterns
    
    When stable resonance detected, extract and apply successful configuration
    """
    
    def __init__(self, resonance_detector: ResonanceDetector):
        self.detector = resonance_detector
        self.tuning_history: List[Dict[str, Any]] = []
    
    def should_tune(self, current_resonance: float, avg_resonance: float) -> bool:
        """Decide if system should self-tune based on resonance difference"""
        # If current resonance significantly below average, tune
        return current_resonance < (avg_resonance - 0.15)
    
    def extract_tuning_recommendations(
        self,
        current_config: Dict[str, Any],
        query_features: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze resonance patterns and recommend config changes
        
        Returns recommended config changes or None if current config is optimal
        """
        patterns = self.detector.get_resonance_patterns(min_resonance=0.85)
        
        if not patterns:
            return None  # No successful patterns learned yet
        
        # Find patterns with similar query characteristics
        similar_patterns = self._find_similar_patterns(query_features, patterns)
        
        if not similar_patterns:
            return None
        
        # Get most successful pattern
        best_pattern = max(similar_patterns, key=lambda p: p['avg_resonance'])
        best_config = best_pattern.get('common_config', {})
        
        # Generate recommendations
        recommendations = {}
        
        # Check if preset should change
        if 'preset' in best_config and best_config['preset'] != current_config.get('preset'):
            recommendations['preset'] = best_config['preset']
        
        # Check if anchor mode should change
        if 'reasoning_mode' in best_config:
            recommendations['reasoning_mode'] = best_config['reasoning_mode']
        
        # Check retrieval parameters
        for param in ['top_k', 'sem_limit', 'rotations', 'rotation_subspace']:
            if param in best_config:
                current_val = current_config.get('flags', {}).get(param)
                recommended_val = best_config[param]
                
                # Only recommend if significantly different
                if current_val and abs(current_val - recommended_val) > (current_val * 0.15):
                    if 'flags' not in recommendations:
                        recommendations['flags'] = {}
                    recommendations['flags'][param] = int(recommended_val)
        
        return recommendations if recommendations else None
    
    def _find_similar_patterns(
        self, 
        query_features: Dict[str, Any], 
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find resonance patterns with similar query characteristics"""
        similar = []
        
        query_length = query_features.get('length', 0)
        is_question = query_features.get('is_question', False)
        
        for pattern in patterns:
            characteristics = pattern.get('query_characteristics', {})
            
            # Check length similarity (within 50%)
            pattern_length = characteristics.get('avg_length', 0)
            if pattern_length > 0:
                length_ratio = query_length / pattern_length
                if 0.5 <= length_ratio <= 2.0:
                    # Check question type similarity
                    pattern_q_ratio = characteristics.get('question_ratio', 0.5)
                    if (is_question and pattern_q_ratio > 0.5) or (not is_question and pattern_q_ratio <= 0.5):
                        similar.append(pattern)
        
        return similar
    
    def apply_tuning(
        self,
        current_config: Dict[str, Any],
        recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tuning recommendations to config"""
        tuned_config = current_config.copy()
        
        # Log tuning event
        tuning_event = {
            'timestamp': datetime.now().isoformat(),
            'original_config': current_config.copy(),
            'recommendations': recommendations.copy()
        }
        
        # Apply changes
        if 'preset' in recommendations:
            tuned_config['preset'] = recommendations['preset']
        
        if 'reasoning_mode' in recommendations:
            tuned_config['reasoning_mode'] = recommendations['reasoning_mode']
        
        if 'flags' in recommendations:
            if 'flags' not in tuned_config:
                tuned_config['flags'] = {}
            tuned_config['flags'].update(recommendations['flags'])
        
        tuning_event['tuned_config'] = tuned_config.copy()
        self.tuning_history.append(tuning_event)
        
        return tuned_config


class MetaPatternExtractor:
    """
    Extract meta-patterns from query sequences that lead to resonance
    Discovers higher-order patterns in system behavior
    """
    
    def __init__(self):
        self.query_sequences: List[List[Dict[str, Any]]] = []
        self.meta_patterns: List[Dict[str, Any]] = []
    
    def add_query_sequence(self, queries: List[Dict[str, Any]]):
        """Add a sequence of queries with their resonance scores"""
        if len(queries) >= 2:
            self.query_sequences.append(queries)
    
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze sequences to find meta-patterns
        
        Examples:
        - "Philosophical query followed by technical query achieves higher resonance"
        - "Iterative refinement (3+ follow-ups) leads to stable resonance"
        - "Exploratory questions before synthesis query improves outcomes"
        """
        patterns = []
        
        # Pattern: Mode transitions
        mode_transitions = self._detect_mode_transitions()
        if mode_transitions:
            patterns.append({
                'type': 'mode_transition',
                'pattern': mode_transitions,
                'description': 'Reasoning mode transitions that lead to resonance'
            })
        
        # Pattern: Iterative refinement
        refinement = self._detect_iterative_refinement()
        if refinement:
            patterns.append({
                'type': 'iterative_refinement',
                'pattern': refinement,
                'description': 'Multi-turn refinement leading to high resonance'
            })
        
        # Pattern: Follow-up effectiveness
        followup = self._detect_followup_effectiveness()
        if followup:
            patterns.append({
                'type': 'followup_effectiveness',
                'pattern': followup,
                'description': 'Follow-up queries improving resonance over time'
            })
        
        self.meta_patterns.extend(patterns)
        return patterns
    
    def _detect_mode_transitions(self) -> Optional[Dict[str, Any]]:
        """Detect if certain anchor mode transitions lead to higher resonance"""
        transitions = []
        
        for seq in self.query_sequences:
            if len(seq) < 2:
                continue
            
            for i in range(len(seq) - 1):
                mode1 = seq[i].get('reasoning_mode')
                mode2 = seq[i + 1].get('reasoning_mode')
                res1 = seq[i].get('resonance', 0)
                res2 = seq[i + 1].get('resonance', 0)
                
                if mode1 and mode2 and mode1 != mode2 and res2 > res1:
                    transitions.append({
                        'from': mode1,
                        'to': mode2,
                        'resonance_gain': res2 - res1
                    })
        
        if not transitions:
            return None
        
        # Find most effective transition
        best = max(transitions, key=lambda t: t['resonance_gain'])
        return {
            'most_effective': f"{best['from']} → {best['to']}",
            'avg_gain': statistics.mean([t['resonance_gain'] for t in transitions]),
            'sample_size': len(transitions)
        }
    
    def _detect_iterative_refinement(self) -> Optional[Dict[str, Any]]:
        """Detect if iterative follow-ups lead to resonance"""
        refinement_sequences = []
        
        for seq in self.query_sequences:
            if len(seq) < 3:
                continue
            
            # Check if it's a refinement sequence (follow-ups)
            is_refinement = all(q.get('is_followup', False) for q in seq[1:])
            
            if is_refinement:
                resonances = [q.get('resonance', 0) for q in seq]
                # Check if resonance improved
                if resonances[-1] > resonances[0]:
                    refinement_sequences.append({
                        'length': len(seq),
                        'resonance_gain': resonances[-1] - resonances[0],
                        'final_resonance': resonances[-1]
                    })
        
        if not refinement_sequences:
            return None
        
        return {
            'avg_length': statistics.mean([s['length'] for s in refinement_sequences]),
            'avg_gain': statistics.mean([s['resonance_gain'] for s in refinement_sequences]),
            'success_rate': len([s for s in refinement_sequences if s['final_resonance'] > 0.85]) / len(refinement_sequences),
            'sample_size': len(refinement_sequences)
        }
    
    def _detect_followup_effectiveness(self) -> Optional[Dict[str, Any]]:
        """Analyze follow-up query effectiveness"""
        followups = []
        
        for seq in self.query_sequences:
            for i in range(len(seq)):
                if seq[i].get('is_followup', False):
                    res_before = seq[i-1].get('resonance', 0) if i > 0 else 0
                    res_after = seq[i].get('resonance', 0)
                    
                    followups.append({
                        'resonance_before': res_before,
                        'resonance_after': res_after,
                        'improved': res_after > res_before
                    })
        
        if not followups:
            return None
        
        return {
            'total_followups': len(followups),
            'improvement_rate': sum(1 for f in followups if f['improved']) / len(followups),
            'avg_resonance_change': statistics.mean([f['resonance_after'] - f['resonance_before'] for f in followups])
        }
