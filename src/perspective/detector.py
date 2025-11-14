#!/usr/bin/env python3
"""
Perspective Orientation Detector
Detects document/query perspective orientation based on lexical fingerprints.
Replaces semantic embeddings with geometric orientation vectors.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import Counter
import numpy as np


class PerspectiveOrientationDetector:
    """
    Detects perspective orientation using vocabulary distribution analysis.

    Core insight: Perspective is geometric orientation, not semantic content.
    Different perspectives use different lexical distributions even for same topic.

    Educational: "explain", "understand", "tutorial", "introduction"
    Security: "vulnerability", "exploit", "attack", "malicious"
    Diagnostic: "error", "failure", "debug", "troubleshoot"
    etc.
    """

    def __init__(self, signatures_path: str):
        """
        Initialize with perspective vocabulary signatures.

        Args:
            signatures_path: Path to perspective_signatures.json
        """
        self.signatures_path = Path(signatures_path)
        self.signatures = self._load_signatures()
        self.perspective_names = list(self.signatures.keys())

        # Pre-compile regex patterns for fast matching
        self._compile_patterns()

    def _load_signatures(self) -> Dict:
        """Load perspective vocabulary signatures"""
        with open(self.signatures_path, 'r') as f:
            return json.load(f)

    def _compile_patterns(self):
        """Pre-compile regex patterns for each perspective marker"""
        self.compiled_patterns = {}

        for perspective, signature in self.signatures.items():
            patterns = []

            # Compile patterns for all marker types
            for marker_type in ['primary_markers', 'secondary_markers', 'structural_markers']:
                for marker in signature[marker_type]:
                    # Word boundary matching, case-insensitive
                    pattern = re.compile(r'\b' + re.escape(marker) + r'\b', re.IGNORECASE)
                    patterns.append((marker, marker_type, pattern))

            self.compiled_patterns[perspective] = patterns

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score text across all perspective orientations.

        Returns perspective orientation vector: {perspective: score}
        Higher score = stronger alignment with that perspective.

        Args:
            text: Document or query text to analyze

        Returns:
            Dictionary mapping perspective names to scores (0.0 to 1.0)
        """
        text_lower = text.lower()
        word_count = len(text.split())

        if word_count == 0:
            return {p: 0.0 for p in self.perspective_names}

        scores = {}

        for perspective, patterns in self.compiled_patterns.items():
            primary_hits = 0
            secondary_hits = 0
            structural_hits = 0

            for marker, marker_type, pattern in patterns:
                matches = len(pattern.findall(text_lower))

                if matches > 0:
                    if marker_type == 'primary_markers':
                        primary_hits += matches * 3.0  # Primary markers weighted heavily
                    elif marker_type == 'secondary_markers':
                        secondary_hits += matches * 1.5
                    elif marker_type == 'structural_markers':
                        structural_hits += matches * 1.0

            # Normalize by text length
            raw_score = (primary_hits + secondary_hits + structural_hits) / word_count

            # Apply sigmoid to bound between 0 and 1
            normalized_score = 1 / (1 + np.exp(-5 * (raw_score - 0.5)))

            scores[perspective] = normalized_score

        # Apply softmax with temperature to make scores sum to 1
        # Higher temperature (5.0) sharpens the distribution
        temperature = 5.0
        score_values = np.array([scores[p] for p in self.perspective_names])
        exp_scores = np.exp((score_values - np.max(score_values)) * temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Update scores with softmax-normalized values
        for i, p in enumerate(self.perspective_names):
            scores[p] = softmax_scores[i]

        return scores

    def get_orientation_vector(self, text: str) -> np.ndarray:
        """
        Get geometric orientation vector for text.

        This is the key replacement for semantic embeddings.
        Instead of "what does this say", we measure "from what angle".

        Args:
            text: Document or query text

        Returns:
            8-dimensional orientation vector (one dimension per perspective)
        """
        scores = self.score_text(text)
        return np.array([scores[p] for p in self.perspective_names])

    def get_dominant_perspective(self, text: str, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Get the dominant perspective orientation.

        Args:
            text: Document or query text
            threshold: Minimum score to be considered dominant

        Returns:
            (perspective_name, score) or ("mixed", max_score) if no clear dominant
        """
        scores = self.score_text(text)
        sorted_perspectives = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_perspective, top_score = sorted_perspectives[0]

        if top_score < threshold:
            return ("mixed", top_score)

        # Check if there's a clear leader (2x higher than second)
        if len(sorted_perspectives) > 1:
            second_score = sorted_perspectives[1][1]
            if top_score < second_score * 1.5:
                return ("mixed", top_score)

        return (top_perspective, top_score)

    def detect(self, text: str) -> Dict:
        """
        Unified detection method compatible with aria_core.py API

        Returns comprehensive perspective analysis in standard format.

        Args:
            text: Document or query text to analyze

        Returns:
            Dict with keys: primary, confidence, weights, orientation_vector
        """
        weights = self.score_text(text)
        primary, confidence = self.get_dominant_perspective(text)
        orientation_vector = self.get_orientation_vector(text).tolist()

        return {
            "primary": primary,
            "confidence": float(confidence),
            "weights": {k: float(v) for k, v in weights.items()},
            "orientation_vector": orientation_vector
        }

    def compare_orientations(self, text1: str, text2: str) -> float:
        """
        Compute angular alignment between two texts' perspective orientations.

        This is NOT semantic similarity - it's perspective alignment.
        Two texts can be semantically different but perspectively aligned.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity of orientation vectors (0.0 to 1.0)
        """
        vec1 = self.get_orientation_vector(text1)
        vec2 = self.get_orientation_vector(text2)

        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def detect_perspective_shift(self, text_sequence: List[str]) -> List[Tuple[int, str, str]]:
        """
        Detect perspective shifts in a sequence of texts (e.g., conversation).

        Useful for tracking when conversation switches from educational to
        diagnostic mode, or from theoretical to practical.

        Args:
            text_sequence: List of texts in chronological order

        Returns:
            List of (index, old_perspective, new_perspective) tuples
        """
        shifts = []
        prev_perspective = None

        for i, text in enumerate(text_sequence):
            perspective, score = self.get_dominant_perspective(text)

            if prev_perspective is not None and perspective != prev_perspective:
                shifts.append((i, prev_perspective, perspective))

            prev_perspective = perspective

        return shifts

    def get_perspective_breakdown(self, text: str) -> Dict[str, Dict]:
        """
        Get detailed breakdown of perspective indicators in text.

        Useful for debugging and understanding why a particular
        orientation was detected.

        Returns:
            {perspective: {
                'score': float,
                'primary_hits': [(marker, count)],
                'secondary_hits': [(marker, count)],
                'structural_hits': [(marker, count)]
            }}
        """
        text_lower = text.lower()
        breakdown = {}

        for perspective, patterns in self.compiled_patterns.items():
            primary_hits = []
            secondary_hits = []
            structural_hits = []

            for marker, marker_type, pattern in patterns:
                matches = len(pattern.findall(text_lower))

                if matches > 0:
                    hit_list = {
                        'primary_markers': primary_hits,
                        'secondary_markers': secondary_hits,
                        'structural_markers': structural_hits
                    }[marker_type]

                    hit_list.append((marker, matches))

            breakdown[perspective] = {
                'score': self.score_text(text)[perspective],
                'primary_hits': primary_hits,
                'secondary_hits': secondary_hits,
                'structural_hits': structural_hits
            }

        return breakdown


class PerspectiveSpaceMapper:
    """
    Maps documents into perspective orientation space.

    This is the geometric space where quaternion rotations operate.
    Documents occupy angular positions based on their lexical fingerprints.
    """

    def __init__(self, detector: PerspectiveOrientationDetector):
        self.detector = detector
        self.document_positions = {}  # {doc_id: orientation_vector}

    def add_document(self, doc_id: str, text: str):
        """Add document to perspective space"""
        orientation = self.detector.get_orientation_vector(text)
        self.document_positions[doc_id] = orientation

    def get_document_position(self, doc_id: str) -> Optional[np.ndarray]:
        """Get document's position in perspective space"""
        return self.document_positions.get(doc_id)

    def find_aligned_documents(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find documents with similar perspective orientation to query.

        This replaces semantic similarity search with perspective alignment search.

        Args:
            query_text: Query text
            top_k: Number of aligned documents to return

        Returns:
            List of (doc_id, alignment_score) tuples, sorted by alignment
        """
        query_orientation = self.detector.get_orientation_vector(query_text)

        alignments = []
        for doc_id, doc_orientation in self.document_positions.items():
            # Cosine similarity of orientation vectors
            alignment = np.dot(query_orientation, doc_orientation) / (
                np.linalg.norm(query_orientation) * np.linalg.norm(doc_orientation)
            )
            alignments.append((doc_id, alignment))

        # Sort by alignment score
        alignments.sort(key=lambda x: x[1], reverse=True)

        return alignments[:top_k]

    def get_perspective_clusters(self) -> Dict[str, List[str]]:
        """
        Cluster documents by dominant perspective orientation.

        Returns:
            {perspective_name: [doc_ids]}
        """
        clusters = {p: [] for p in self.detector.perspective_names}
        clusters['mixed'] = []

        for doc_id, orientation in self.document_positions.items():
            # Find dominant dimension
            max_idx = np.argmax(orientation)
            max_val = orientation[max_idx]

            if max_val > 0.3:
                perspective = self.detector.perspective_names[max_idx]
                clusters[perspective].append(doc_id)
            else:
                clusters['mixed'].append(doc_id)

        return clusters

    def visualize_perspective_space(self) -> Dict[str, Any]:
        """
        Generate data for visualizing perspective space.

        Returns 2D projection via PCA for visualization.
        """
        if not self.document_positions:
            return {}

        from sklearn.decomposition import PCA

        doc_ids = list(self.document_positions.keys())
        orientations = np.array([self.document_positions[did] for did in doc_ids])

        # Project to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(orientations)

        return {
            'doc_ids': doc_ids,
            'positions_2d': projected.tolist(),
            'variance_explained': pca.explained_variance_ratio_.tolist()
        }


def demo():
    """Demonstrate perspective orientation detection"""
    from pathlib import Path

    # Find signatures relative to this file
    module_dir = Path(__file__).parent.parent.parent
    signatures_path = module_dir / "data" / "domain_dictionaries" / "perspective_signatures_v2.json"

    print("="*80)
    print("PERSPECTIVE ORIENTATION DETECTOR - DEMO")
    print("="*80)

    detector = PerspectiveOrientationDetector(str(signatures_path))

    # Test texts with different perspectives on same topic (networking)
    test_texts = {
        'educational': """
            This tutorial explains how network routing protocols work.
            Understanding the fundamentals of packet forwarding is essential.
            Let's learn about the basic concepts step-by-step.
            We'll demonstrate how routers make forwarding decisions.
        """,

        'diagnostic': """
            Error: Network routing failure detected. Debugging the issue shows
            packet loss at hop 3. Stack trace indicates timeout exception.
            Troubleshooting steps: check routing table, verify connectivity,
            investigate firewall rules. Root cause appears to be configuration error.
        """,

        'security': """
            Critical vulnerability discovered in routing protocol implementation.
            Attack vector allows unauthorized route injection. Exploit enables
            traffic interception and privilege escalation. CVE-2024-12345 assigned.
            Mitigation: apply security patch immediately. High severity risk.
        """,

        'implementation': """
            Implementing a custom routing protocol in Python. Create the Router class
            with methods for packet forwarding. Build the routing table data structure.
            Develop the algorithm for path computation. Deploy the service and
            integrate with existing network infrastructure.
        """,

        'research': """
            We investigate the performance characteristics of adaptive routing
            algorithms. Our hypothesis is that dynamic load balancing improves
            throughput. Methodology: controlled experiments with varying network
            topologies. Results show 23% improvement. Statistical significance: p<0.01.
        """
    }

    print("\n[Test 1] Detecting perspective orientation for different texts\n")

    for expected_perspective, text in test_texts.items():
        scores = detector.score_text(text)
        detected, score = detector.get_dominant_perspective(text)

        print(f"Expected: {expected_perspective:15s} | Detected: {detected:15s} | Score: {score:.3f}")

        # Show top 3 perspectives
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3: {' | '.join(f'{p}: {s:.3f}' for p, s in sorted_scores)}")
        print()

    print("\n[Test 2] Comparing perspective alignments\n")

    # Compare educational vs educational (should be high)
    alignment_edu = detector.compare_orientations(
        test_texts['educational'],
        "Here's a beginner's guide to understanding network basics. This introduction covers fundamental concepts."
    )
    print(f"Educational vs Educational: {alignment_edu:.3f}")

    # Compare educational vs security (should be low)
    alignment_mixed = detector.compare_orientations(
        test_texts['educational'],
        test_texts['security']
    )
    print(f"Educational vs Security:    {alignment_mixed:.3f}")

    print("\n[Test 3] Detailed perspective breakdown\n")

    breakdown = detector.get_perspective_breakdown(test_texts['diagnostic'])
    print("Diagnostic text markers found:")

    for marker_type in ['primary_hits', 'secondary_hits', 'structural_hits']:
        hits = breakdown['diagnostic'][marker_type]
        if hits:
            print(f"\n  {marker_type}:")
            for marker, count in hits:
                print(f"    - '{marker}': {count}")

    print("\n" + "="*80)
    print("This is how ARIA distinguishes perspective without semantic embeddings!")
    print("="*80)


if __name__ == '__main__':
    demo()
