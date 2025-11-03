#!/usr/bin/env python3
"""
Uncertainty Quantifier - Wave 2: Advanced Reasoning
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Quantifies uncertainty in retrieval results and generated answers.

Integration with Current ARIA:
- Analyzes aria_postfilter.py output confidence
- Uses contradiction_detector.py findings
- Feeds into aria_telemetry.py metrics
- Informs multi_hop_reasoner.py confidence scores

Key Features:
1. Epistemic uncertainty (knowledge gaps - "we don't know")
2. Aleatoric uncertainty (data noise - "sources disagree")
3. Confidence decomposition (where does uncertainty come from?)
4. Calibration (are confidence scores accurate?)
5. Uncertainty visualization (how to present to user)

Uncertainty Sources:
- Coverage: gaps in knowledge base
- Consistency: contradictions between sources
- Source reliability: trustworthiness of sources
- Retrieval quality: BM25/semantic scores
- Temporal: information freshness
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty"""
    EPISTEMIC = "epistemic"  # Knowledge gaps
    ALEATORIC = "aleatoric"  # Data noise/disagreement
    COVERAGE = "coverage"  # Incomplete retrieval
    RELIABILITY = "reliability"  # Source trustworthiness
    TEMPORAL = "temporal"  # Information freshness


@dataclass
class UncertaintyBreakdown:
    """Detailed uncertainty breakdown"""
    total_uncertainty: float  # Overall [0, 1]
    confidence: float  # 1 - uncertainty
    
    # Component uncertainties
    epistemic: float = 0.0  # Knowledge gaps
    aleatoric: float = 0.0  # Source disagreement
    coverage: float = 0.0  # Retrieval completeness
    reliability: float = 0.0  # Source trust
    temporal: float = 0.0  # Freshness
    
    # Contributing factors
    factors: Dict[str, float] = field(default_factory=dict)
    
    # Explanation
    explanation: str = ""
    
    # Calibration info
    is_calibrated: bool = False
    calibration_confidence: float = 0.0


class UncertaintyQuantifier:
    """
    Quantify and decompose uncertainty in retrieval/answers
    
    Process:
    1. Collect uncertainty signals from multiple sources
    2. Decompose into epistemic vs aleatoric
    3. Compute overall uncertainty score
    4. Generate human-readable explanation
    5. Track calibration over time
    """
    
    def __init__(
        self,
        state_path: Optional[Path] = None,
        calibration_history_size: int = 100
    ):
        """
        Args:
            state_path: Path to save/load calibration state
            calibration_history_size: Number of past predictions to track
        """
        self.state_path = state_path or Path.home() / '.aria_uncertainty.json'
        
        # Calibration tracking
        self.calibration_history = []  # List of (predicted_conf, actual_reward)
        self.calibration_history_size = calibration_history_size
        
        # Load state
        self._load_state()
    
    def quantify(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        contradictions: Optional[List[Any]] = None,
        retrieval_metrics: Optional[Dict[str, float]] = None
    ) -> UncertaintyBreakdown:
        """
        Quantify uncertainty for a retrieval result
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            contradictions: Optional list from contradiction_detector
            retrieval_metrics: Optional metrics dict with coverage, etc.
        
        Returns:
            UncertaintyBreakdown with detailed uncertainty analysis
        """
        breakdown = UncertaintyBreakdown(total_uncertainty=0.0, confidence=1.0)
        
        # 1. Coverage uncertainty (epistemic)
        coverage_unc = self._compute_coverage_uncertainty(chunks, retrieval_metrics)
        breakdown.coverage = coverage_unc
        breakdown.epistemic += coverage_unc * 0.5
        
        # 2. Source reliability uncertainty
        reliability_unc = self._compute_reliability_uncertainty(chunks)
        breakdown.reliability = reliability_unc
        breakdown.epistemic += reliability_unc * 0.3
        
        # 3. Contradiction uncertainty (aleatoric)
        if contradictions:
            contradiction_unc = self._compute_contradiction_uncertainty(contradictions)
            breakdown.aleatoric += contradiction_unc
            breakdown.factors['contradictions'] = contradiction_unc
        
        # 4. Score variance uncertainty (aleatoric)
        score_unc = self._compute_score_uncertainty(chunks)
        breakdown.aleatoric += score_unc * 0.5
        breakdown.factors['score_variance'] = score_unc
        
        # 5. Temporal uncertainty (epistemic)
        temporal_unc = self._compute_temporal_uncertainty(chunks)
        breakdown.temporal = temporal_unc
        breakdown.epistemic += temporal_unc * 0.2
        
        # Normalize epistemic and aleatoric
        breakdown.epistemic = min(1.0, breakdown.epistemic)
        breakdown.aleatoric = min(1.0, breakdown.aleatoric)
        
        # Compute total uncertainty (combination of epistemic + aleatoric)
        # Using quadrature sum: sqrt(epistemic^2 + aleatoric^2)
        breakdown.total_uncertainty = np.sqrt(
            breakdown.epistemic**2 + breakdown.aleatoric**2
        )
        breakdown.total_uncertainty = min(1.0, breakdown.total_uncertainty)
        
        # Confidence = 1 - uncertainty
        breakdown.confidence = 1.0 - breakdown.total_uncertainty
        
        # Generate explanation
        breakdown.explanation = self._generate_explanation(breakdown)
        
        # Apply calibration if available
        breakdown = self._apply_calibration(breakdown)
        
        return breakdown
    
    def _compute_coverage_uncertainty(
        self,
        chunks: List[Dict[str, Any]],
        metrics: Optional[Dict[str, float]]
    ) -> float:
        """
        Compute coverage uncertainty (knowledge gaps)
        
        Low coverage = high epistemic uncertainty
        """
        if not chunks:
            return 1.0  # Maximum uncertainty
        
        # Use coverage metric if available
        if metrics and 'coverage_score' in metrics:
            coverage = metrics['coverage_score']
            return 1.0 - coverage
        
        # Fallback: estimate from chunk count and scores
        num_chunks = len(chunks)
        avg_score = float(np.mean([c.get('score', 0.5) for c in chunks]))
        
        # Heuristic: good coverage needs 3+ chunks with avg score > 0.7
        if num_chunks >= 3 and avg_score > 0.7:
            return 0.2  # Low uncertainty
        elif num_chunks >= 2 and avg_score > 0.6:
            return 0.4  # Medium uncertainty
        else:
            return 0.7  # High uncertainty
    
    def _compute_reliability_uncertainty(
        self,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute source reliability uncertainty
        
        Unreliable sources = higher epistemic uncertainty
        """
        if not chunks:
            return 1.0
        
        # Default reliability scores
        reliability_scores = {
            'arxiv': 0.95,
            'pubmed': 0.95,
            'wikipedia': 0.75,
            'github': 0.65,
            'stackoverflow': 0.65,
            'reddit': 0.40,
            'unknown': 0.50
        }
        
        # Extract source reliability
        chunk_reliability = []
        for chunk in chunks:
            source = self._extract_source(chunk)
            reliability = reliability_scores.get(source, 0.5)
            chunk_reliability.append(reliability)
        
        # Uncertainty = 1 - avg reliability
        avg_reliability = float(np.mean(chunk_reliability))
        return 1.0 - avg_reliability
    
    def _compute_contradiction_uncertainty(
        self,
        contradictions: List[Any]
    ) -> float:
        """
        Compute uncertainty from contradictions
        
        More contradictions = higher aleatoric uncertainty
        """
        if not contradictions:
            return 0.0
        
        # Weight by contradiction confidence
        total_conf = sum(
            getattr(c, 'confidence', 0.5) for c in contradictions
        )
        
        # Normalize by number of contradictions
        uncertainty = min(1.0, total_conf / 2.0)  # Divide by 2 for scaling
        
        return uncertainty
    
    def _compute_score_uncertainty(
        self,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute uncertainty from score variance
        
        High variance = sources disagree on relevance = aleatoric uncertainty
        """
        if len(chunks) < 2:
            return 0.0
        
        scores = [c.get('score', 0.5) for c in chunks]
        score_std = float(np.std(scores))
        
        # Normalize std to [0, 1]
        # Max theoretical std for scores in [0, 1] is 0.5
        uncertainty = min(1.0, score_std / 0.5)
        
        return uncertainty
    
    def _compute_temporal_uncertainty(
        self,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Compute temporal uncertainty (information freshness)
        
        Old information = higher epistemic uncertainty for time-sensitive queries
        """
        # For now, return low temporal uncertainty
        # In production, would check chunk timestamps vs current date
        return 0.1
    
    def _extract_source(self, chunk: Dict[str, Any]) -> str:
        """Extract source name from chunk"""
        file_path = chunk.get('file', chunk.get('path', 'unknown'))
        
        if isinstance(file_path, Path):
            file_path = str(file_path)
        
        file_lower = file_path.lower()
        
        # Detect source type
        sources = ['arxiv', 'pubmed', 'wikipedia', 'github', 'stackoverflow', 'reddit']
        for source in sources:
            if source in file_lower:
                return source
        
        return 'unknown'
    
    def _generate_explanation(self, breakdown: UncertaintyBreakdown) -> str:
        """Generate human-readable uncertainty explanation"""
        conf = breakdown.confidence
        
        if conf > 0.9:
            base = "Very high confidence"
        elif conf > 0.75:
            base = "High confidence"
        elif conf > 0.6:
            base = "Moderate confidence"
        elif conf > 0.4:
            base = "Low confidence"
        else:
            base = "Very low confidence"
        
        # Add main uncertainty source
        main_source = ""
        if breakdown.epistemic > breakdown.aleatoric:
            if breakdown.coverage > 0.5:
                main_source = " (limited information available)"
            elif breakdown.reliability > 0.5:
                main_source = " (low source reliability)"
            elif breakdown.temporal > 0.3:
                main_source = " (information may be outdated)"
        else:
            if breakdown.aleatoric > 0.5:
                main_source = " (sources disagree)"
        
        return base + main_source
    
    def _apply_calibration(self, breakdown: UncertaintyBreakdown) -> UncertaintyBreakdown:
        """
        Apply calibration correction based on historical accuracy
        
        If model is overconfident (predicts 0.9 but actual is 0.7),
        adjust future predictions downward.
        """
        if len(self.calibration_history) < 20:
            breakdown.is_calibrated = False
            return breakdown
        
        # Compute calibration error
        predicted_confs = [p[0] for p in self.calibration_history[-50:]]
        actual_rewards = [p[1] for p in self.calibration_history[-50:]]
        
        # Average overconfidence
        overconfidence = float(np.mean([p - a for p, a in zip(predicted_confs, actual_rewards)]))
        
        # Apply correction
        if abs(overconfidence) > 0.05:  # >5% miscalibration
            breakdown.confidence -= overconfidence
            breakdown.confidence = np.clip(breakdown.confidence, 0.0, 1.0)
            breakdown.total_uncertainty = 1.0 - breakdown.confidence
            breakdown.is_calibrated = True
            breakdown.calibration_confidence = 1.0 - abs(overconfidence)
        
        return breakdown
    
    def update_calibration(self, predicted_confidence: float, actual_reward: float):
        """
        Update calibration with observed outcome
        
        Args:
            predicted_confidence: Confidence we predicted
            actual_reward: Actual reward received (0-1)
        """
        self.calibration_history.append((predicted_confidence, actual_reward))
        
        # Keep only recent history
        if len(self.calibration_history) > self.calibration_history_size:
            self.calibration_history = self.calibration_history[-self.calibration_history_size:]
        
        # Periodic save
        if len(self.calibration_history) % 10 == 0:
            self._save_state()
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics"""
        if len(self.calibration_history) < 10:
            return {'status': 'insufficient_data'}
        
        predicted = np.array([p[0] for p in self.calibration_history])
        actual = np.array([p[1] for p in self.calibration_history])
        
        # Mean calibration error
        calibration_error = np.mean(predicted - actual)
        
        # Mean absolute calibration error
        abs_calibration_error = np.mean(np.abs(predicted - actual))
        
        # Correlation (how well does confidence predict reward?)
        correlation = np.corrcoef(predicted, actual)[0, 1]
        
        return {
            'status': 'calibrated',
            'mean_calibration_error': float(calibration_error),
            'abs_calibration_error': float(abs_calibration_error),
            'confidence_reward_correlation': float(correlation),
            'samples': len(self.calibration_history)
        }
    
    def _save_state(self):
        """Save calibration state"""
        try:
            state = {
                'version': '1.0',
                'calibration_history': self.calibration_history
            }
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save uncertainty state: {e}")
    
    def _load_state(self):
        """Load calibration state"""
        if not self.state_path.exists():
            return
        
        try:
            state = json.loads(self.state_path.read_text())
            self.calibration_history = state.get('calibration_history', [])
            print(f"✓ Loaded uncertainty calibration: {len(self.calibration_history)} samples")
        except Exception as e:
            print(f"Warning: Could not load uncertainty state: {e}")


# ============================================================================
# Integration with ARIA
# ============================================================================

def integrate_with_aria():
    """
    Example: Integrate uncertainty quantifier into aria_main.py
    """
    
    # Initialize quantifier
    quantifier = UncertaintyQuantifier(
        state_path=Path.home() / '.aria_uncertainty.json'
    )
    
    # Mock retrieval result
    query = "How does Thompson Sampling work?"
    chunks = [
        {'text': 'Thompson Sampling uses Bayesian inference...', 'score': 0.85, 'file': '/data/arxiv/bandits.pdf'},
        {'text': 'It samples from posterior distributions...', 'score': 0.78, 'file': '/data/wikipedia/bandits.html'},
        {'text': 'Not sure but I think it uses random sampling...', 'score': 0.55, 'file': '/data/reddit/ml_discussion.txt'}
    ]
    
    # Mock contradictions
    contradictions = []  # Empty for this example
    
    # Mock retrieval metrics
    metrics = {
        'coverage_score': 0.75
    }
    
    # Quantify uncertainty
    breakdown = quantifier.quantify(query, chunks, contradictions, metrics)
    
    print(f"📊 Uncertainty Analysis:")
    print(f"  Overall confidence: {breakdown.confidence:.2%}")
    print(f"  Total uncertainty: {breakdown.total_uncertainty:.2%}")
    print(f"\n  Breakdown:")
    print(f"    Epistemic (knowledge gaps): {breakdown.epistemic:.2%}")
    print(f"    Aleatoric (data noise): {breakdown.aleatoric:.2%}")
    print(f"    Coverage: {breakdown.coverage:.2%}")
    print(f"    Reliability: {breakdown.reliability:.2%}")
    print(f"    Temporal: {breakdown.temporal:.2%}")
    print(f"\n  Explanation: {breakdown.explanation}")
    print(f"  Calibrated: {breakdown.is_calibrated}")
    
    # Simulate updating calibration
    actual_reward = 0.70  # Example: actual quality was 70%
    quantifier.update_calibration(breakdown.confidence, actual_reward)
    
    # Get calibration stats
    stats = quantifier.get_calibration_stats()
    if stats.get('status') == 'calibrated':
        print(f"\n🎯 Calibration Stats:")
        print(f"  Mean error: {stats['mean_calibration_error']:+.3f}")
        print(f"  Absolute error: {stats['abs_calibration_error']:.3f}")
        print(f"  Correlation: {stats['confidence_reward_correlation']:.3f}")


if __name__ == '__main__':
    integrate_with_aria()
