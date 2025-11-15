#!/usr/bin/env python3
"""
Signature Learner - Learn perspective markers from real corpus data

Analyzes conversation corpus to discover which terms actually indicate
each perspective, then enriches perspective_signatures_v2.json.

Philosophy:
- Learn from successful interactions (positive feedback)
- Weight by feedback quality (high reward = stronger signal)
- Auto-grow signatures over time
- Remove low-confidence markers
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import statistics

# Stopwords (don't learn these as markers)
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "when", "of", "to", "in", "on", "for",
    "with", "as", "by", "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "i", "you", "we", "they", "he", "she", "them", "his", "her",
    "their", "your", "my", "our", "at", "from", "into", "about", "over", "after", "before",
    "than", "so", "not", "no", "yes", "what", "where", "when", "why", "how", "which", "who"
}


@dataclass
class MarkerStats:
    """Statistics for a learned marker"""
    term: str
    frequency: int = 0
    perspectives: Dict[str, int] = field(default_factory=dict)  # perspective -> count
    avg_reward: float = 0.0
    confidence: float = 0.0  # How reliably it indicates a perspective
    dominant_perspective: str = "mixed"


class SignatureLearner:
    """Learn perspective signatures from conversation corpus"""

    def __init__(self, signatures_path: Path, corpus_dir: Path):
        self.signatures_path = signatures_path
        self.corpus_dir = corpus_dir
        self.signatures = self._load_signatures()
        self.marker_stats: Dict[str, MarkerStats] = {}

    def _load_signatures(self) -> Dict:
        """Load existing signatures"""
        if not self.signatures_path.exists():
            return self._default_signatures()

        try:
            with open(self.signatures_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[SignatureLearner] Error loading signatures: {e}")
            return self._default_signatures()

    def _default_signatures(self) -> Dict:
        """Return minimal default signatures"""
        perspectives = [
            "educational", "diagnostic", "security", "implementation",
            "research", "theoretical", "practical", "reference"
        ]
        return {p: {"markers": [], "weight": 1.0} for p in perspectives}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text, remove stopwords"""
        text = text.lower()
        # Remove punctuation except dashes and underscores
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        tokens = text.split()
        # Filter: length 3+, not stopword, not pure digits
        return [
            t for t in tokens
            if len(t) >= 3 and t not in STOP and not t.isdigit()
        ]

    def load_corpus_triples(self) -> List[Dict]:
        """Load all conversation triples from corpus"""
        triples = []

        if not self.corpus_dir.exists():
            print(f"[SignatureLearner] Corpus directory not found: {self.corpus_dir}")
            return []

        for triple_file in self.corpus_dir.glob("*.json"):
            try:
                with open(triple_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract key fields
                triple = {
                    'query': data.get('query', ''),
                    'type': data.get('type', 'conversation'),  # aria_triple or conversation
                    'pack': data.get('pack', {}),
                    'response': data.get('response', ''),
                    'timestamp': data.get('timestamp', ''),
                    'file': triple_file.name
                }

                # Extract perspective if available (from pack metadata)
                pack_meta = triple['pack'].get('meta', {}) if isinstance(triple['pack'], dict) else {}
                perspective_analysis = pack_meta.get('perspective_analysis', {})

                if perspective_analysis:
                    triple['perspective'] = perspective_analysis.get('primary', 'mixed')
                    triple['perspective_confidence'] = perspective_analysis.get('confidence', 0.0)
                else:
                    triple['perspective'] = 'mixed'
                    triple['perspective_confidence'] = 0.0

                # Extract reward if available
                triple['reward'] = pack_meta.get('reward', 0.5)

                triples.append(triple)

            except Exception as e:
                print(f"[SignatureLearner] Error loading {triple_file.name}: {e}")
                continue

        print(f"[SignatureLearner] Loaded {len(triples)} triples from corpus")
        return triples

    def analyze_markers(self, triples: List[Dict], min_frequency: int = 3) -> Dict[str, MarkerStats]:
        """Analyze which terms indicate which perspectives"""

        # Group triples by perspective
        by_perspective: Dict[str, List[Dict]] = defaultdict(list)
        for triple in triples:
            perspective = triple.get('perspective', 'mixed')
            if perspective != 'mixed':  # Only learn from classified queries
                by_perspective[perspective].append(triple)

        print(f"[SignatureLearner] Perspective distribution:")
        for p, trips in by_perspective.items():
            print(f"  {p:15s}: {len(trips):4d} queries")

        # Count term frequencies per perspective
        marker_stats: Dict[str, MarkerStats] = {}

        for perspective, trips in by_perspective.items():
            # Concatenate all queries for this perspective
            all_text = " ".join(t['query'] for t in trips)
            tokens = self._tokenize(all_text)

            # Count frequencies
            freq = Counter(tokens)

            for term, count in freq.items():
                if count < min_frequency:
                    continue

                if term not in marker_stats:
                    marker_stats[term] = MarkerStats(term=term)

                marker_stats[term].frequency += count
                marker_stats[term].perspectives[perspective] = count

                # Weight by average reward
                rewards = [t.get('reward', 0.5) for t in trips if term in t['query'].lower()]
                if rewards:
                    marker_stats[term].avg_reward = statistics.mean(rewards)

        # Compute confidence (how reliably term indicates ONE perspective)
        for term, stats in marker_stats.items():
            if not stats.perspectives:
                continue

            # Total occurrences across all perspectives
            total = sum(stats.perspectives.values())

            # Find dominant perspective
            dominant_perspective = max(stats.perspectives.items(), key=lambda x: x[1])[0]
            dominant_count = stats.perspectives[dominant_perspective]

            # Confidence = (dominant_count / total) weighted by reward
            raw_confidence = dominant_count / total if total > 0 else 0.0
            reward_weight = min(1.0, stats.avg_reward * 1.5)  # Scale up rewards
            stats.confidence = raw_confidence * reward_weight
            stats.dominant_perspective = dominant_perspective

        print(f"[SignatureLearner] Analyzed {len(marker_stats)} unique markers")
        return marker_stats

    def filter_high_confidence(
        self,
        marker_stats: Dict[str, MarkerStats],
        min_confidence: float = 0.6,
        min_frequency: int = 5
    ) -> Dict[str, List[str]]:
        """Filter to high-confidence markers per perspective"""

        by_perspective: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        for term, stats in marker_stats.items():
            if stats.confidence >= min_confidence and stats.frequency >= min_frequency:
                by_perspective[stats.dominant_perspective].append((term, stats.confidence))

        # Sort by confidence (descending)
        result: Dict[str, List[str]] = {}
        for perspective, markers in by_perspective.items():
            markers.sort(key=lambda x: x[1], reverse=True)
            # Keep top 100 markers per perspective
            result[perspective] = [term for term, conf in markers[:100]]

        print(f"[SignatureLearner] High-confidence markers per perspective:")
        for p, markers in result.items():
            print(f"  {p:15s}: {len(markers):3d} markers")

        return result

    def merge_with_existing(
        self,
        learned_markers: Dict[str, List[str]],
        keep_existing: bool = True
    ) -> Dict:
        """Merge learned markers with existing signatures"""

        for perspective, new_markers in learned_markers.items():
            if perspective not in self.signatures:
                self.signatures[perspective] = {"markers": [], "weight": 1.0}

            existing = set(self.signatures[perspective].get("markers", []))

            if keep_existing:
                # Add new markers to existing
                combined = existing | set(new_markers)
            else:
                # Replace with learned markers
                combined = set(new_markers)

            # Convert back to list and sort
            self.signatures[perspective]["markers"] = sorted(list(combined))

        return self.signatures

    def save_signatures(self):
        """Save enriched signatures to file"""
        try:
            self.signatures_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.signatures_path, 'w', encoding='utf-8') as f:
                json.dump(self.signatures, f, indent=2, ensure_ascii=False)
            print(f"[SignatureLearner] Saved signatures to: {self.signatures_path}")
        except Exception as e:
            print(f"[SignatureLearner] Error saving signatures: {e}")

    def learn(
        self,
        min_frequency: int = 5,
        min_confidence: float = 0.6,
        keep_existing: bool = True
    ):
        """Run complete learning pipeline"""

        print("\n" + "=" * 70)
        print("SIGNATURE LEARNER - Learning from Corpus")
        print("=" * 70)

        # 1. Load corpus
        triples = self.load_corpus_triples()
        if not triples:
            print("[SignatureLearner] No triples found in corpus. Skipping learning.")
            return

        # Filter to ARIA triples (ones with packs)
        aria_triples = [t for t in triples if t['type'] == 'aria_triple']
        print(f"[SignatureLearner] Using {len(aria_triples)} ARIA triples (with packs)")

        if len(aria_triples) < 10:
            print("[SignatureLearner] Not enough ARIA triples (<10). Need more data.")
            return

        # 2. Analyze markers
        self.marker_stats = self.analyze_markers(aria_triples, min_frequency=min_frequency)

        # 3. Filter high-confidence
        learned_markers = self.filter_high_confidence(
            self.marker_stats,
            min_confidence=min_confidence,
            min_frequency=min_frequency
        )

        if not any(learned_markers.values()):
            print("[SignatureLearner] No high-confidence markers learned. Try lowering thresholds.")
            return

        # 4. Merge with existing
        self.signatures = self.merge_with_existing(learned_markers, keep_existing=keep_existing)

        # 5. Save
        self.save_signatures()

        # 6. Report
        print("\n" + "=" * 70)
        print("LEARNING COMPLETE")
        print("=" * 70)
        print(f"Total markers per perspective:")
        for perspective, sig in self.signatures.items():
            marker_count = len(sig.get("markers", []))
            print(f"  {perspective:15s}: {marker_count:3d} markers")

        # Show top 10 markers per perspective
        print("\nTop 10 markers per perspective:")
        for perspective, sig in self.signatures.items():
            markers = sig.get("markers", [])[:10]
            if markers:
                print(f"  {perspective:15s}: {', '.join(markers)}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Learn perspective signatures from corpus")
    ap.add_argument(
        "--signatures",
        default="data/domain_dictionaries/perspective_signatures_v2.json",
        help="Path to signatures JSON"
    )
    ap.add_argument(
        "--corpus",
        default="training_data/conversation_corpus",
        help="Path to conversation corpus directory"
    )
    ap.add_argument(
        "--min-frequency",
        type=int,
        default=5,
        help="Minimum term frequency to consider"
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence to include marker (0.0-1.0)"
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing markers instead of merging"
    )

    args = ap.parse_args()

    # Resolve paths relative to project root (aria/)
    # This file is in src/perspective/, so go up 2 levels to reach aria/
    project_root = Path(__file__).parent.parent.parent
    signatures_path = project_root / args.signatures
    corpus_dir = project_root / args.corpus

    print(f"Signatures: {signatures_path}")
    print(f"Corpus: {corpus_dir}")

    # Run learner
    learner = SignatureLearner(signatures_path, corpus_dir)
    learner.learn(
        min_frequency=args.min_frequency,
        min_confidence=args.min_confidence,
        keep_existing=not args.replace
    )


if __name__ == "__main__":
    main()
