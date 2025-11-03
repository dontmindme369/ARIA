#!/usr/bin/env python3
"""
Contradiction Detector - Wave 2: Advanced Reasoning

Detects contradictions and inconsistencies between retrieved chunks from
different sources.

Integration with Current ARIA:
- Analyzes chunks from aria_retrieval.py
- Uses conversation_state_graph.py claim tracking
- Feeds flags to aria_postfilter.py for filtering
- Integrates with uncertainty_quantifier.py to express doubt

Key Features:
1. Semantic contradiction detection (opposite claims)
2. Factual inconsistency detection (different numbers/dates)
3. Source credibility weighting (trust arxiv over reddit)
4. Confidence scoring (how certain is the contradiction?)
5. Resolution suggestions (which source to trust?)

Contradiction Types:
- Direct negation: "X is Y" vs "X is not Y"
- Numerical: "X = 5" vs "X = 10"
- Temporal: "X happened in 2020" vs "X happened in 2021"
- Causal: "X causes Y" vs "X doesn't cause Y"
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ContradictionType(Enum):
    """Types of contradictions"""
    DIRECT_NEGATION = "direct_negation"  # "X is Y" vs "X is not Y"
    NUMERICAL = "numerical"  # Different numbers
    TEMPORAL = "temporal"  # Different dates/times
    CAUSAL = "causal"  # "X causes Y" vs "X doesn't cause Y"
    ATTRIBUTE = "attribute"  # Different attributes
    SEMANTIC = "semantic"  # Opposite meanings
    UNKNOWN = "unknown"


@dataclass
class Contradiction:
    """A detected contradiction"""
    id: str
    contradiction_type: ContradictionType
    
    # The contradicting chunks
    chunk1: Dict[str, Any]
    chunk2: Dict[str, Any]
    
    # Specific claims that contradict
    claim1: str
    claim2: str
    
    # Source information
    source1: str
    source2: str
    
    # Confidence in contradiction detection
    confidence: float = 0.0
    
    # Which source is more reliable
    preferred_source: Optional[str] = None
    reliability_ratio: float = 1.0  # source1_reliability / source2_reliability
    
    # Explanation
    explanation: str = ""
    
    # Resolution suggestion
    resolution: str = ""


class ContradictionDetector:
    """
    Detect contradictions between retrieved chunks
    
    Process:
    1. Extract claims from chunks
    2. Compare claims pairwise
    3. Detect contradictions using patterns + semantics
    4. Score contradiction confidence
    5. Suggest resolution based on source reliability
    """
    
    def __init__(
        self,
        source_reliability: Optional[Dict[str, float]] = None,
        threshold: float = 0.6
    ):
        """
        Args:
            source_reliability: Dict of {source_name: reliability_score}
            threshold: Minimum confidence to report contradiction
        """
        self.source_reliability = source_reliability or {}
        self.threshold = threshold
        
        # Default source reliability (if not provided)
        self.default_reliability = {
            'arxiv': 0.9,
            'pubmed': 0.9,
            'wikipedia': 0.7,
            'github': 0.6,
            'stackexchange': 0.6,
            'stackoverflow': 0.6,
            'reddit': 0.4,
            'unknown': 0.5
        }
        
        self._build_patterns()
    
    def _build_patterns(self):
        """Build contradiction detection patterns"""
        
        # Negation patterns
        self.negation_patterns = [
            (r'(\w+)\s+is\s+(\w+)', r'\1\s+is\s+not\s+\2'),
            (r'(\w+)\s+can\s+(\w+)', r'\1\s+cannot\s+\2'),
            (r'(\w+)\s+does\s+(\w+)', r'\1\s+does\s+not\s+\2'),
            (r'(\w+)\s+has\s+(\w+)', r'\1\s+does\s+not\s+have\s+\2'),
        ]
        
        # Causal patterns
        self.causal_patterns = [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
        ]
        
        # Attribute patterns
        self.attribute_patterns = [
            r'(\w+)\s+is\s+(?:a|an)\s+(\w+)',
            r'(\w+)\s+has\s+(?:a|an)\s+(\w+)',
        ]
    
    def detect(
        self,
        chunks: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> List[Contradiction]:
        """
        Detect contradictions in chunks
        
        Args:
            chunks: List of retrieved chunks
            query: Optional query for context
        
        Returns:
            List of detected contradictions
        """
        if len(chunks) < 2:
            return []
        
        contradictions = []
        
        # Compare chunks pairwise
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                # Extract text and source
                text1 = chunk1.get('text', '')
                text2 = chunk2.get('text', '')
                source1 = self._extract_source(chunk1)
                source2 = self._extract_source(chunk2)
                
                # Skip if same source
                if source1 == source2:
                    continue
                
                # Detect contradictions
                detected = self._detect_between_texts(
                    text1, text2, 
                    chunk1, chunk2,
                    source1, source2
                )
                
                # Filter by confidence threshold
                contradictions.extend([
                    c for c in detected
                    if c.confidence >= self.threshold
                ])
        
        return contradictions
    
    def _extract_source(self, chunk: Dict[str, Any]) -> str:
        """Extract source name from chunk"""
        file_path = chunk.get('file', chunk.get('path', 'unknown'))
        
        if isinstance(file_path, Path):
            file_path = str(file_path)
        
        # Detect source type from path
        file_lower = file_path.lower()
        
        for source_type in self.default_reliability.keys():
            if source_type in file_lower:
                return source_type
        
        return 'unknown'
    
    def _get_source_reliability(self, source: str) -> float:
        """Get reliability score for source"""
        # Check custom reliability first
        if source in self.source_reliability:
            return self.source_reliability[source]
        
        # Check default reliability
        if source in self.default_reliability:
            return self.default_reliability[source]
        
        return 0.5  # Default
    
    def _detect_between_texts(
        self,
        text1: str,
        text2: str,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any],
        source1: str,
        source2: str
    ) -> List[Contradiction]:
        """Detect contradictions between two texts"""
        contradictions = []
        
        # 1. Direct negation detection
        negation_contrs = self._detect_negations(
            text1, text2, chunk1, chunk2, source1, source2
        )
        contradictions.extend(negation_contrs)
        
        # 2. Numerical contradiction detection
        numerical_contrs = self._detect_numerical(
            text1, text2, chunk1, chunk2, source1, source2
        )
        contradictions.extend(numerical_contrs)
        
        # 3. Temporal contradiction detection
        temporal_contrs = self._detect_temporal(
            text1, text2, chunk1, chunk2, source1, source2
        )
        contradictions.extend(temporal_contrs)
        
        # 4. Causal contradiction detection
        causal_contrs = self._detect_causal(
            text1, text2, chunk1, chunk2, source1, source2
        )
        contradictions.extend(causal_contrs)
        
        return contradictions
    
    def _detect_negations(
        self,
        text1: str,
        text2: str,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any],
        source1: str,
        source2: str
    ) -> List[Contradiction]:
        """Detect direct negation contradictions"""
        contradictions = []
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Check for negation patterns
        for pos_pattern, neg_pattern in self.negation_patterns:
            # Find positive form in text1
            pos_matches = re.finditer(pos_pattern, text1_lower)
            
            for pos_match in pos_matches:
                # Look for negated form in text2
                neg_search = neg_pattern.replace(r'\1', pos_match.group(1)).replace(r'\2', pos_match.group(2))
                
                neg_match = re.search(neg_search, text2_lower)
                if neg_match:
                    # Found contradiction!
                    contradiction = Contradiction(
                        id=f"neg_{len(contradictions)}",
                        contradiction_type=ContradictionType.DIRECT_NEGATION,
                        chunk1=chunk1,
                        chunk2=chunk2,
                        claim1=pos_match.group(0),
                        claim2=neg_match.group(0),
                        source1=source1,
                        source2=source2,
                        confidence=0.85,  # High confidence for direct negation
                        explanation="Direct negation detected"
                    )
                    
                    # Determine preferred source
                    self._add_resolution(contradiction)
                    
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_numerical(
        self,
        text1: str,
        text2: str,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any],
        source1: str,
        source2: str
    ) -> List[Contradiction]:
        """Detect numerical contradictions"""
        contradictions = []
        
        # Extract numbers with context
        num_pattern = r'(\w+(?:\s+\w+)?)\s+(?:is|was|has|had|equals?)\s+(\d+(?:\.\d+)?)\s*(%|percent|degrees?|years?|months?)?'
        
        matches1 = list(re.finditer(num_pattern, text1, re.IGNORECASE))
        matches2 = list(re.finditer(num_pattern, text2, re.IGNORECASE))
        
        # Compare numerical claims about same subject
        for m1 in matches1:
            subject1 = m1.group(1).lower().strip()
            value1 = float(m1.group(2))
            unit1 = m1.group(3) or ''
            
            for m2 in matches2:
                subject2 = m2.group(1).lower().strip()
                value2 = float(m2.group(2))
                unit2 = m2.group(3) or ''
                
                # Check if same subject
                if subject1 == subject2 and unit1 == unit2:
                    # Check if values differ significantly
                    diff_ratio = abs(value1 - value2) / max(value1, value2)
                    
                    if diff_ratio > 0.1:  # >10% difference
                        contradiction = Contradiction(
                            id=f"num_{len(contradictions)}",
                            contradiction_type=ContradictionType.NUMERICAL,
                            chunk1=chunk1,
                            chunk2=chunk2,
                            claim1=m1.group(0),
                            claim2=m2.group(0),
                            source1=source1,
                            source2=source2,
                            confidence=min(0.9, diff_ratio),  # Higher diff = higher confidence
                            explanation=f"Numerical values differ: {value1} vs {value2}"
                        )
                        
                        self._add_resolution(contradiction)
                        contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_temporal(
        self,
        text1: str,
        text2: str,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any],
        source1: str,
        source2: str
    ) -> List[Contradiction]:
        """Detect temporal contradictions"""
        contradictions = []
        
        # Extract years/dates
        year_pattern = r'(\w+(?:\s+\w+)?)\s+(?:in|during|happened|occurred)\s+(\d{4})'
        
        matches1 = list(re.finditer(year_pattern, text1, re.IGNORECASE))
        matches2 = list(re.finditer(year_pattern, text2, re.IGNORECASE))
        
        for m1 in matches1:
            subject1 = m1.group(1).lower().strip()
            year1 = int(m1.group(2))
            
            for m2 in matches2:
                subject2 = m2.group(1).lower().strip()
                year2 = int(m2.group(2))
                
                # Same event, different years
                if subject1 == subject2 and year1 != year2:
                    contradiction = Contradiction(
                        id=f"temp_{len(contradictions)}",
                        contradiction_type=ContradictionType.TEMPORAL,
                        chunk1=chunk1,
                        chunk2=chunk2,
                        claim1=m1.group(0),
                        claim2=m2.group(0),
                        source1=source1,
                        source2=source2,
                        confidence=0.8,
                        explanation=f"Different years: {year1} vs {year2}"
                    )
                    
                    self._add_resolution(contradiction)
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_causal(
        self,
        text1: str,
        text2: str,
        chunk1: Dict[str, Any],
        chunk2: Dict[str, Any],
        source1: str,
        source2: str
    ) -> List[Contradiction]:
        """Detect causal contradictions"""
        contradictions = []
        
        # Extract causal claims
        causal_claims1 = []
        causal_claims2 = []
        
        for pattern in self.causal_patterns:
            causal_claims1.extend(re.finditer(pattern, text1, re.IGNORECASE))
            causal_claims2.extend(re.finditer(pattern, text2, re.IGNORECASE))
        
        # Look for contradictory causal claims
        for m1 in causal_claims1:
            cause1 = m1.group(1).lower()
            effect1 = m1.group(2).lower()
            
            # Check if text2 has negation of this causal claim
            neg_pattern = f"{cause1}.*(?:does not|doesn't|cannot).*{effect1}"
            
            neg_match = re.search(neg_pattern, text2.lower())
            if neg_match:
                contradiction = Contradiction(
                    id=f"caus_{len(contradictions)}",
                    contradiction_type=ContradictionType.CAUSAL,
                    chunk1=chunk1,
                    chunk2=chunk2,
                    claim1=m1.group(0),
                    claim2=neg_match.group(0),
                    source1=source1,
                    source2=source2,
                    confidence=0.75,
                    explanation="Contradictory causal claims"
                )
                
                self._add_resolution(contradiction)
                contradictions.append(contradiction)
        
        return contradictions
    
    def _add_resolution(self, contradiction: Contradiction):
        """Add resolution suggestion based on source reliability"""
        rel1 = self._get_source_reliability(contradiction.source1)
        rel2 = self._get_source_reliability(contradiction.source2)
        
        contradiction.reliability_ratio = rel1 / rel2 if rel2 > 0 else 1.0
        
        if rel1 > rel2 * 1.2:  # 20% more reliable
            contradiction.preferred_source = contradiction.source1
            contradiction.resolution = f"Trust {contradiction.source1} (reliability: {rel1:.2f} vs {rel2:.2f})"
        elif rel2 > rel1 * 1.2:
            contradiction.preferred_source = contradiction.source2
            contradiction.resolution = f"Trust {contradiction.source2} (reliability: {rel2:.2f} vs {rel1:.2f})"
        else:
            contradiction.resolution = "Sources have similar reliability - requires additional verification"
    
    def get_summary(self, contradictions: List[Contradiction]) -> Dict[str, Any]:
        """Get summary of contradictions"""
        if not contradictions:
            return {
                'total': 0,
                'has_contradictions': False
            }
        
        # Count by type
        type_counts = {}
        for c in contradictions:
            type_counts[c.contradiction_type.value] = type_counts.get(c.contradiction_type.value, 0) + 1
        
        # Average confidence
        avg_confidence = sum(c.confidence for c in contradictions) / len(contradictions)
        
        # High confidence contradictions
        high_conf = [c for c in contradictions if c.confidence > 0.8]
        
        return {
            'total': len(contradictions),
            'has_contradictions': True,
            'by_type': type_counts,
            'avg_confidence': avg_confidence,
            'high_confidence_count': len(high_conf),
            'top_contradictions': [
                {
                    'type': c.contradiction_type.value,
                    'claim1': c.claim1,
                    'claim2': c.claim2,
                    'confidence': c.confidence,
                    'resolution': c.resolution
                }
                for c in sorted(contradictions, key=lambda x: x.confidence, reverse=True)[:3]
            ]
        }


# ============================================================================
# Integration with ARIA
# ============================================================================

def integrate_with_aria():
    """
    Example: Integrate contradiction detector into aria_postfilter.py
    """
    
    # Initialize detector
    detector = ContradictionDetector(
        source_reliability={
            'custom_source': 0.95  # Custom reliability scores
        },
        threshold=0.6
    )
    
    # Mock retrieved chunks
    chunks = [
        {
            'text': 'Thompson Sampling is a Bayesian approach to bandits.',
            'file': '/data/arxiv/bandits_paper.pdf',
            'score': 0.9
        },
        {
            'text': 'Thompson Sampling is not a frequentist method.',
            'file': '/data/wikipedia/bandits.html',
            'score': 0.8
        },
        {
            'text': 'The algorithm was proposed in 1933.',
            'file': '/data/arxiv/history_bandits.pdf',
            'score': 0.85
        },
        {
            'text': 'The algorithm was proposed in 1934.',
            'file': '/data/reddit/bandits_discussion.txt',
            'score': 0.7
        }
    ]
    
    # Detect contradictions
    contradictions = detector.detect(chunks)
    
    print(f"Detected {len(contradictions)} contradictions:")
    for c in contradictions:
        print(f"\n  Type: {c.contradiction_type.value}")
        print(f"  Confidence: {c.confidence:.2f}")
        print(f"  Claim 1 ({c.source1}): {c.claim1}")
        print(f"  Claim 2 ({c.source2}): {c.claim2}")
        print(f"  Resolution: {c.resolution}")
    
    # Get summary
    summary = detector.get_summary(contradictions)
    print(f"\n📊 Summary:")
    print(f"  Total contradictions: {summary['total']}")
    print(f"  High confidence: {summary['high_confidence_count']}")
    print(f"  By type: {summary['by_type']}")


if __name__ == '__main__':
    integrate_with_aria()
