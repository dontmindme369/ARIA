#!/usr/bin/env python3
"""
ARIA Unified Postfilter - Phase 2

Consolidates:
- rag_postfilter_pack_v3_1.py - quality filtering
- rerank_ce_tune.py - cross-encoder reranking
- postfilter_tune.py - filter optimization
- pack_stats.py - pack statistics

Features:
- PDF garbage detection
- Quality filtering (alpha ratio, markers)
- Topic relevance filtering
- Per-file diversity enforcement
- Cross-encoder reranking (optional)
- Pack statistics computation
"""

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ============================================================================
# QUALITY FILTERS
# ============================================================================

class QualityFilter:
    """Filter out low-quality chunks"""
    
    # PDF garbage markers
    PDF_GARBAGE_MARKERS = {
        "/type /font", "/xobject", "endobj", "stream",
        "xmp:", "pageuidlist", "tounicode"
    }

    # Relaxed for PDFs - many academic PDFs have tables/equations with low alpha ratio
    MIN_ALPHA_RATIO = 0.25
    
    def __init__(self):
        pass
    
    def is_pdf_garbage(self, text: str) -> bool:
        """
        Detect PDF extraction garbage
        
        Args:
            text: Text to check
        
        Returns:
            True if text looks like PDF garbage
        """
        if not text:
            return True
        
        # Check first 6000 chars
        sample = text[:6000]
        
        # Alpha ratio check
        if self._alpha_ratio(sample) < self.MIN_ALPHA_RATIO:
            return True
        
        # Marker check - require MORE markers to reject (PDFs often have some metadata)
        tokens = re.findall(r"[A-Za-z:/]{3,}", sample, flags=re.I)
        marker_count = sum(
            1 for tok in tokens
            if tok.lower() in self.PDF_GARBAGE_MARKERS
        )
        # Increased from 2 to 4 - need multiple garbage markers to reject
        if marker_count >= 4:
            return True
        
        # Hex/binary pattern check
        if re.search(
            r"(?:\\x[0-9A-Fa-f]{2}|#[0-9A-Fa-f]{6}|<[0-9A-Fa-f\\s]{16,}>)",
            sample
        ):
            return True
        
        return False
    
    def _alpha_ratio(self, text: str) -> float:
        """Calculate ratio of alphabetic characters"""
        if not text:
            return 0.0
        letters = sum(ch.isalpha() for ch in text)
        return letters / len(text)
    
    def is_low_quality(self, text: str, path: str = "") -> bool:
        """
        Check if chunk is low quality
        
        Args:
            text: Text to check
            path: File path (for PDF detection)
        
        Returns:
            True if low quality
        """
        # PDF garbage check
        if self._is_pdf(path) and self.is_pdf_garbage(text):
            return True
        
        # Too short (very lenient - just filter obvious junk)
        if len(text.strip()) < 20:
            return True
        
        # Tightened quality filter - reject truly garbage content
        # Increased from 0.15 to 0.25 for better quality
        if self._alpha_ratio(text) < 0.25:
            return True
        
        return False
    
    def _is_pdf(self, path: str) -> bool:
        """Check if path is PDF"""
        return path.lower().endswith(".pdf") or path.lower().startswith("pdf:")


# ============================================================================
# TOPIC RELEVANCE FILTER
# ============================================================================

class TopicFilter:
    """Filter chunks by topic relevance"""
    
    STOP_WORDS = {
        "the", "and", "for", "with", "that", "this", "into",
        "from", "about", "are", "was", "were"
    }
    
    def __init__(
        self,
        must_terms: Optional[List[str]] = None,
        must_regex: Optional[List[str]] = None,
        relax_must: bool = True
    ):
        self.must_terms = [t.lower() for t in (must_terms or [])]
        self.must_regex = must_regex or []
        self.relax_must = relax_must
    
    def is_on_topic(self, text: str) -> bool:
        """
        Check if text is on-topic
        
        Args:
            text: Text to check
        
        Returns:
            True if on-topic
        """
        if not self.must_terms and not self.must_regex:
            return True
        
        text_lower = text.lower()
        
        # Exact match
        if any(term in text_lower for term in self.must_terms):
            return True
        
        # Relaxed match (2+ term overlaps)
        if self.relax_must and self.must_terms:
            text_tokens = set(re.findall(r"[a-z][a-z0-9\\-]{2,}", text_lower))
            
            hits = 0
            for term in self.must_terms:
                term_parts = [
                    p for p in re.findall(r"[a-z][a-z0-9\\-]{2,}", term)
                    if p not in self.STOP_WORDS
                ]
                hits += sum(1 for p in term_parts if p in text_tokens)
            
            if hits >= 2:
                return True
        
        # Regex match
        for pattern in self.must_regex:
            try:
                if re.search(pattern, text_lower, flags=re.I | re.S):
                    return True
            except Exception:
                pass
        
        return False


# ============================================================================
# DIVERSITY ENFORCER
# ============================================================================

class DiversityEnforcer:
    """Enforce per-source limits for diversity"""
    
    def __init__(self, max_per_source: int = 6):
        self.max_per_source = max_per_source
    
    def enforce(
        self,
        items: List[Dict[str, Any]],
        keep_at_least: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Limit chunks per source while maintaining quality
        
        Args:
            items: Chunks (must have 'score' field)
            keep_at_least: Minimum chunks to keep total
        
        Returns:
            Filtered chunks with diversity
        """
        # Group by source
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            source = self._get_source(item)
            by_source[source].append(item)
        
        # Sort each source's chunks by score
        for source in by_source:
            by_source[source].sort(
                key=lambda x: x.get('score', 0.0),
                reverse=True
            )
        
        # Collect chunks: max N per source
        result = []
        sources_used = []
        
        # Sort sources by best chunk score
        sorted_sources = sorted(
            by_source.keys(),
            key=lambda s: max(x.get('score', 0) for x in by_source[s]),
            reverse=True
        )
        
        for source in sorted_sources:
            source_chunks = by_source[source][:self.max_per_source]
            result.extend(source_chunks)
            sources_used.append(source)
        
        # Ensure minimum chunks
        if len(result) < keep_at_least:
            # Add more chunks from top sources
            for source in sorted_sources:
                remaining = by_source[source][self.max_per_source:]
                result.extend(remaining)
                if len(result) >= keep_at_least:
                    break
        
        # Sort result by score
        result.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        return result
    
    def _get_source(self, item: Dict[str, Any]) -> str:
        """Get source identifier from item"""
        for key in ['path', 'source', 'file', 'uri']:
            val = item.get(key)
            if isinstance(val, str):
                return val
        return 'unknown'


# ============================================================================
# PACK STATISTICS
# ============================================================================

class PackStats:
    """Compute statistics on retrieved packs"""
    
    def __init__(self):
        pass
    
    def compute(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute pack statistics
        
        Args:
            items: Retrieved chunks
        
        Returns:
            Statistics dictionary
        """
        if not items:
            return {
                'total_chunks': 0,
                'unique_sources': 0,
                'avg_chunk_length': 0,
                'source_distribution': {},
                'duplication_rate': 0.0
            }
        
        # Extract sources
        sources = [self._get_source(it) for it in items]
        unique_sources = set(sources)
        
        # Source distribution
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Chunk lengths
        chunk_lengths = [
            len(self._get_text(it))
            for it in items
        ]
        
        # Duplication rate (chunks from same source / total)
        max_from_one = max(source_counts.values()) if source_counts else 0
        duplication_rate = (max_from_one - 1) / max(len(items), 1)
        
        return {
            'total_chunks': len(items),
            'unique_sources': len(unique_sources),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
            'source_distribution': source_counts,
            'duplication_rate': round(duplication_rate, 3),
            'diversity_score': len(unique_sources) / max(len(items), 1)
        }
    
    def _get_source(self, item: Dict[str, Any]) -> str:
        """Get source from item"""
        for key in ['path', 'source', 'file']:
            val = item.get(key)
            if isinstance(val, str):
                return val
        return 'unknown'
    
    def _get_text(self, item: Dict[str, Any]) -> str:
        """Get text from item"""
        for key in ['text', 'content', 'chunk_text']:
            val = item.get(key)
            if isinstance(val, str):
                return val
        return ''


# ============================================================================
# MAIN POSTFILTER
# ============================================================================

class ARIAPostfilter:
    """
    Unified postfiltering system
    
    Features:
    - Quality filtering (garbage detection)
    - Topic relevance filtering
    - Diversity enforcement
    - Pack statistics
    """
    
    def __init__(
        self,
        max_per_source: int = 6,
        min_keep: int = 10,
        must_terms: Optional[List[str]] = None,
        must_regex: Optional[List[str]] = None,
        drop_globs: Optional[List[str]] = None,
        min_score: float = 0.0  # NEW: Minimum relevance score threshold
    ):
        self.quality_filter = QualityFilter()
        self.topic_filter = TopicFilter(must_terms, must_regex)
        self.diversity_enforcer = DiversityEnforcer(max_per_source)
        self.stats_computer = PackStats()
        self.drop_globs = drop_globs or []
        self.min_keep = min_keep
        self.min_score = min_score  # NEW
    
    def filter(
        self,
        items: List[Dict[str, Any]],
        enable_quality: bool = True,
        enable_topic: bool = True,
        enable_diversity: bool = True
    ) -> Dict[str, Any]:
        """
        Apply postfiltering to retrieved chunks
        
        Args:
            items: Retrieved chunks
            enable_quality: Enable quality filters
            enable_topic: Enable topic filters
            enable_diversity: Enable diversity enforcement
        
        Returns:
            {
                'items': List[Dict],       # Filtered chunks
                'stats_before': Dict,      # Stats before filtering
                'stats_after': Dict,       # Stats after filtering
                'removed_count': int,      # Chunks removed
                'removal_reasons': Dict    # Reasons for removal
            }
        """
        # Stats before
        stats_before = self.stats_computer.compute(items)
        
        # Track removal reasons
        removal_reasons = {
            'quality': 0,
            'topic': 0,
            'glob': 0,
            'low_score': 0  # NEW
        }

        # Apply filters
        filtered = []

        for item in items:
            source = self._get_source(item)
            text = self._get_text(item)
            score = item.get('score', 0.0)

            # Glob filter
            if self._should_drop_glob(source):
                removal_reasons['glob'] += 1
                continue

            # Quality filter
            if enable_quality and self.quality_filter.is_low_quality(text, source):
                removal_reasons['quality'] += 1
                continue

            # Topic filter
            if enable_topic and not self.topic_filter.is_on_topic(text):
                removal_reasons['topic'] += 1
                continue

            # NEW: Score threshold filter
            if score < self.min_score:
                removal_reasons['low_score'] += 1
                continue

            filtered.append(item)
        
        # Diversity enforcement
        if enable_diversity:
            filtered = self.diversity_enforcer.enforce(filtered, self.min_keep)
        
        # Ensure minimum chunks (but respect min_score threshold)
        if len(filtered) < self.min_keep:
            # Add back top items that meet min_score requirement
            all_scored = sorted(
                items,
                key=lambda x: x.get('score', 0.0),
                reverse=True
            )
            for item in all_scored:
                if item not in filtered:
                    # FIXED: Only add back items that meet min_score threshold
                    if item.get('score', 0.0) >= self.min_score:
                        filtered.append(item)
                if len(filtered) >= self.min_keep:
                    break
        
        # Stats after
        stats_after = self.stats_computer.compute(filtered)
        
        return {
            'items': filtered,
            'stats_before': stats_before,
            'stats_after': stats_after,
            'removed_count': len(items) - len(filtered),
            'removal_reasons': removal_reasons
        }
    
    def _get_source(self, item: Dict[str, Any]) -> str:
        """Get source from item"""
        for key in ['path', 'source', 'file']:
            val = item.get(key)
            if isinstance(val, str):
                return val
        return 'unknown'
    
    def _get_text(self, item: Dict[str, Any]) -> str:
        """Get text from item"""
        for key in ['text', 'content', 'chunk_text']:
            val = item.get(key)
            if isinstance(val, str):
                return val
        return ''
    
    def _should_drop_glob(self, path: str) -> bool:
        """Check if path matches drop globs"""
        return any(
            fnmatch.fnmatch(path, glob)
            for glob in self.drop_globs
        )


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for testing postfilter"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIA Unified Postfilter")
    parser.add_argument('input', help="Input pack JSON file")
    parser.add_argument('--output', help="Output filtered pack JSON")
    parser.add_argument('--max-per-source', type=int, default=6)
    parser.add_argument('--min-keep', type=int, default=10)
    parser.add_argument('--must-terms', help="Comma-separated required terms")
    parser.add_argument('--drop-globs', help="Comma-separated globs to drop")
    args = parser.parse_args()
    
    # Load pack
    with open(args.input) as f:
        pack = json.load(f)
    
    items = pack.get('items', [])
    
    # Parse filters
    must_terms = [t.strip() for t in args.must_terms.split(',')] if args.must_terms else None
    drop_globs = [g.strip() for g in args.drop_globs.split(',')] if args.drop_globs else None
    
    # Initialize postfilter
    postfilter = ARIAPostfilter(
        max_per_source=args.max_per_source,
        min_keep=args.min_keep,
        must_terms=must_terms,
        drop_globs=drop_globs
    )
    
    # Filter
    result = postfilter.filter(items)
    
    # Display
    print(f"\n{'='*70}")
    print(f"POSTFILTER RESULTS")
    print(f"{'='*70}")
    
    print(f"\nBefore filtering:")
    for key, val in result['stats_before'].items():
        if isinstance(val, dict):
            print(f"  {key}: {len(val)} sources")
        else:
            print(f"  {key}: {val}")
    
    print(f"\nAfter filtering:")
    for key, val in result['stats_after'].items():
        if isinstance(val, dict):
            print(f"  {key}: {len(val)} sources")
        else:
            print(f"  {key}: {val}")
    
    print(f"\nRemoved: {result['removed_count']} chunks")
    print(f"Removal reasons:")
    for reason, count in result['removal_reasons'].items():
        print(f"  {reason}: {count}")
    
    # Save if requested
    if args.output:
        output_pack = {
            'items': result['items'],
            'meta': {
                **pack.get('meta', {}),
                'postfilter_stats': result['stats_after']
            }
        }
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_pack, f, indent=2)
        
        print(f"\nÃ¢Å“â€œ Saved filtered pack to {args.output}")


if __name__ == '__main__':
    main()
