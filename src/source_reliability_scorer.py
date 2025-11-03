#!/usr/bin/env python3
"""
Source Reliability Scorer - Wave 1 Foundation
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Tracks which data sources (files, datasets, domains) perform best over time.

Integration with Current ARIA:
- Analyzes chunks from aria_retrieval.py (file paths, sources)
- Feeds from aria_telemetry.py reward signals
- Used by retrieval to boost reliable sources
- Persists state like other ARIA components

Key Features:
1. Per-source quality tracking (success rate, avg reward)
2. Domain-level reliability (e.g., arxiv papers vs reddit posts)
3. Temporal decay (recent performance weighted higher)
4. Source diversity scoring (penalize over-reliance on single sources)
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class SourceStats:
    """Statistics for a single source"""
    source_id: str  # file path or dataset name
    
    # Performance tracking
    uses: int = 0
    successes: int = 0  # High reward outcomes
    failures: int = 0   # Low reward outcomes
    total_reward: float = 0.0
    
    # Quality metrics
    avg_reward: float = 0.5
    reliability_score: float = 0.5  # Bayesian posterior
    
    # Temporal tracking
    last_used: Optional[str] = None  # ISO timestamp
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Content quality
    avg_chunk_score: float = 0.0  # From exemplar_fit
    coverage_contribution: float = 0.0  # How much this source helps coverage
    
    # Metadata
    domain: str = 'unknown'
    source_type: str = 'unknown'  # 'arxiv', 'github', 'reddit', 'books', etc.


class SourceReliabilityScorer:
    """
    Track and score source reliability over time
    
    Uses Bayesian updating with temporal decay:
    - Prior: Beta(Î±, Î²) distribution over success rate
    - Update: Add successes/failures weighted by recency
    - Score: Posterior mean or UCB
    """
    
    def __init__(
        self,
        state_path: Optional[Path] = None,
        decay_rate: float = 0.99,  # Daily decay
        success_threshold: float = 0.6  # Reward > 0.6 = success
    ):
        """
        Args:
            state_path: Path to save/load state
            decay_rate: Temporal decay per day (0.99 = 1% decay per day)
            success_threshold: Reward threshold for counting as success
        """
        self.state_path = state_path or Path.home() / '.aria_source_reliability.json'
        self.decay_rate = decay_rate
        self.success_threshold = success_threshold
        
        # Source statistics
        self.sources: Dict[str, SourceStats] = {}
        
        # Domain-level aggregation
        self.domain_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'uses': 0, 'total_reward': 0.0, 'avg_reward': 0.5}
        )
        
        # Diversity tracking
        self.query_history: deque = deque(maxlen=100)  # Recent query-source mappings
        
        # Load existing state
        self._load_state()
    
    def extract_source_info(self, chunk: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Extract source information from a chunk
        
        Args:
            chunk: Chunk dict with 'file', 'text', 'score', etc.
        
        Returns:
            (source_id, domain, source_type)
        """
        file_path = chunk.get('file', chunk.get('path', 'unknown'))
        
        # Normalize path
        if isinstance(file_path, Path):
            file_path = str(file_path)
        
        source_id = file_path
        
        # Detect domain from path
        domain = 'unknown'
        source_type = 'unknown'
        
        path_lower = file_path.lower()
        
        # Detect source type
        if '/arxiv/' in path_lower or 'arxiv' in path_lower:
            source_type = 'arxiv'
            domain = 'academic'
        elif '/github/' in path_lower or 'github' in path_lower:
            source_type = 'github'
            domain = 'code'
        elif '/reddit/' in path_lower or 'reddit' in path_lower:
            source_type = 'reddit'
            domain = 'social'
        elif '/pubmed/' in path_lower or 'pubmed' in path_lower:
            source_type = 'pubmed'
            domain = 'medical'
        elif '/books/' in path_lower or '.epub' in path_lower or '.mobi' in path_lower:
            source_type = 'books'
            domain = 'literature'
        elif '/stackexchange/' in path_lower or 'stackoverflow' in path_lower:
            source_type = 'stackexchange'
            domain = 'technical'
        elif '/wikipedia/' in path_lower or 'wiki' in path_lower:
            source_type = 'wikipedia'
            domain = 'reference'
        
        return source_id, domain, source_type
    
    def record_retrieval(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        reward: float,
        chunk_scores: Optional[Dict[str, float]] = None
    ):
        """
        Record a retrieval event and update source statistics
        
        Args:
            query: Query string
            chunks: List of chunks retrieved
            reward: Overall reward for this retrieval (0-1)
            chunk_scores: Optional dict mapping chunk to quality score
        """
        if not chunks:
            return
        
        is_success = reward >= self.success_threshold
        
        # Track which sources were used
        sources_used = set()
        
        for chunk in chunks:
            source_id, domain, source_type = self.extract_source_info(chunk)
            
            # Initialize source if new
            if source_id not in self.sources:
                self.sources[source_id] = SourceStats(
                    source_id=source_id,
                    domain=domain,
                    source_type=source_type
                )
            
            source = self.sources[source_id]
            
            # Update usage
            source.uses += 1
            source.last_used = datetime.now().isoformat()
            
            # Update success/failure
            if is_success:
                source.successes += 1
            else:
                source.failures += 1
            
            # Update reward tracking
            source.total_reward += reward
            source.avg_reward = source.total_reward / max(source.uses, 1)
            source.recent_rewards.append(reward)
            
            # Update chunk score (from exemplar_fit or cross-encoder)
            if chunk_scores and source_id in chunk_scores:
                chunk_score = chunk_scores[source_id]
                # Running average
                source.avg_chunk_score = (
                    0.9 * source.avg_chunk_score + 0.1 * chunk_score
                )
            
            # Bayesian reliability update
            source.reliability_score = self._compute_reliability(source)
            
            # Track domain stats
            self.domain_stats[domain]['uses'] += 1
            self.domain_stats[domain]['total_reward'] += reward
            self.domain_stats[domain]['avg_reward'] = (
                self.domain_stats[domain]['total_reward'] /
                self.domain_stats[domain]['uses']
            )
            
            sources_used.add(source_id)
        
        # Track query-source mapping for diversity
        self.query_history.append({
            'query': query[:100],
            'sources': list(sources_used),
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Periodic save
        if len(self.query_history) % 10 == 0:
            self._save_state()
    
    def _compute_reliability(self, source: SourceStats) -> float:
        """
        Compute Bayesian reliability score
        
        Uses Beta distribution with temporal decay:
        - Prior: Beta(1, 1) = Uniform
        - Posterior: Beta(Î± + successes, Î² + failures)
        - Score: Posterior mean with decay
        """
        # Apply temporal decay to old observations
        decay_factor = self._get_decay_factor(source.last_used)
        
        # Bayesian posterior
        alpha = 1.0 + source.successes * decay_factor
        beta = 1.0 + source.failures * decay_factor
        
        # Posterior mean
        posterior_mean = alpha / (alpha + beta)
        
        # Add bonus for more data (reduce uncertainty)
        n = source.uses
        uncertainty_penalty = 1.0 / (1.0 + n / 10.0)  # Diminishes as n grows
        
        return posterior_mean * (1.0 - 0.1 * uncertainty_penalty)
    
    def _get_decay_factor(self, last_used: Optional[str]) -> float:
        """
        Compute temporal decay factor
        
        Returns:
            decay_factor: 1.0 (very recent) to decay_rate^days (old)
        """
        if not last_used:
            return 1.0
        
        try:
            last_time = datetime.fromisoformat(last_used)
            days_ago = (datetime.now() - last_time).days
            return self.decay_rate ** days_ago
        except:
            return 1.0
    
    def get_source_score(self, source_id: str) -> float:
        """
        Get reliability score for a source
        
        Returns:
            Score in [0, 1], with 0.5 as default for unknown sources
        """
        if source_id not in self.sources:
            return 0.5
        
        return self.sources[source_id].reliability_score
    
    def get_domain_score(self, domain: str) -> float:
        """Get average reliability for a domain"""
        if domain not in self.domain_stats:
            return 0.5
        
        return self.domain_stats[domain]['avg_reward']
    
    def boost_chunks_by_reliability(
        self,
        chunks: List[Dict[str, Any]],
        boost_strength: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Boost chunk scores based on source reliability
        
        Args:
            chunks: List of chunks with 'score' field
            boost_strength: How much to boost (0.1 = up to 10% boost)
        
        Returns:
            Chunks with adjusted scores
        """
        boosted_chunks = []
        
        for chunk in chunks:
            source_id, _, _ = self.extract_source_info(chunk)
            reliability = self.get_source_score(source_id)
            
            # Boost formula: score * (1 + boost_strength * (reliability - 0.5))
            # Reliable sources (reliability > 0.5) get boost
            # Unreliable sources (reliability < 0.5) get penalty
            original_score = chunk.get('score', 0.0)
            boost_factor = 1.0 + boost_strength * (reliability - 0.5)
            boosted_score = original_score * boost_factor
            
            boosted_chunk = chunk.copy()
            boosted_chunk['score'] = boosted_score
            boosted_chunk['reliability'] = reliability
            boosted_chunk['boost_factor'] = boost_factor
            
            boosted_chunks.append(boosted_chunk)
        
        return boosted_chunks
    
    def get_diversity_score(self) -> float:
        """
        Compute diversity score for recent retrievals
        
        Returns:
            Score in [0, 1]: 1.0 = high diversity, 0.0 = using same sources repeatedly
        """
        if len(self.query_history) < 5:
            return 1.0  # Not enough data
        
        # Count unique sources in recent queries
        all_sources = []
        for query_data in list(self.query_history)[-20:]:
            all_sources.extend(query_data['sources'])
        
        if not all_sources:
            return 1.0
        
        unique_sources = len(set(all_sources))
        total_uses = len(all_sources)
        
        # Diversity = unique / total
        diversity = unique_sources / total_uses
        
        return diversity
    
    def get_top_sources(
        self,
        n: int = 10,
        min_uses: int = 3
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Get top N most reliable sources
        
        Args:
            n: Number of top sources to return
            min_uses: Minimum uses required to be considered
        
        Returns:
            List of (source_id, reliability_score, stats_dict)
        """
        # Filter sources with enough data
        candidates = [
            (sid, source.reliability_score, {
                'uses': source.uses,
                'avg_reward': source.avg_reward,
                'domain': source.domain,
                'source_type': source.source_type,
                'recent_avg': np.mean(list(source.recent_rewards)) if source.recent_rewards else 0.0
            })
            for sid, source in self.sources.items()
            if source.uses >= min_uses
        ]
        
        # Sort by reliability
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:n]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get overall statistics summary"""
        if not self.sources:
            return {'total_sources': 0}
        
        all_reliability = [s.reliability_score for s in self.sources.values()]
        
        return {
            'total_sources': len(self.sources),
            'avg_reliability': np.mean(all_reliability),
            'std_reliability': np.std(all_reliability),
            'total_retrievals': len(self.query_history),
            'diversity_score': self.get_diversity_score(),
            'domains': dict(self.domain_stats),
            'top_5_sources': self.get_top_sources(n=5, min_uses=5)
        }
    
    def _save_state(self):
        """Save state to disk"""
        try:
            state = {
                'version': '1.0',
                'sources': {
                    sid: {
                        'uses': s.uses,
                        'successes': s.successes,
                        'failures': s.failures,
                        'total_reward': s.total_reward,
                        'avg_reward': s.avg_reward,
                        'reliability_score': s.reliability_score,
                        'last_used': s.last_used,
                        'recent_rewards': list(s.recent_rewards),
                        'domain': s.domain,
                        'source_type': s.source_type,
                        'avg_chunk_score': s.avg_chunk_score
                    }
                    for sid, s in self.sources.items()
                },
                'domain_stats': dict(self.domain_stats),
                'query_history': list(self.query_history)
            }
            
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save source reliability state: {e}")
    
    def _load_state(self):
        """Load state from disk"""
        if not self.state_path.exists():
            return
        
        try:
            state = json.loads(self.state_path.read_text())
            
            # Restore sources
            for sid, s_data in state.get('sources', {}).items():
                self.sources[sid] = SourceStats(
                    source_id=sid,
                    uses=s_data['uses'],
                    successes=s_data['successes'],
                    failures=s_data['failures'],
                    total_reward=s_data['total_reward'],
                    avg_reward=s_data['avg_reward'],
                    reliability_score=s_data['reliability_score'],
                    last_used=s_data.get('last_used'),
                    domain=s_data.get('domain', 'unknown'),
                    source_type=s_data.get('source_type', 'unknown'),
                    avg_chunk_score=s_data.get('avg_chunk_score', 0.0)
                )
                
                # Restore recent rewards
                for reward in s_data.get('recent_rewards', []):
                    self.sources[sid].recent_rewards.append(reward)
            
            # Restore domain stats
            self.domain_stats.update(state.get('domain_stats', {}))
            
            # Restore query history
            for qh in state.get('query_history', []):
                self.query_history.append(qh)
            
            print(f"âœ“ Loaded source reliability: {len(self.sources)} sources tracked")
        
        except Exception as e:
            print(f"Warning: Could not load source reliability state: {e}")


# ============================================================================
# Integration with Current ARIA
# ============================================================================

def integrate_with_aria_retrieval():
    """
    Example: Integrate source reliability scoring into aria_retrieval.py
    """
    
    # Initialize scorer
    scorer = SourceReliabilityScorer(
        state_path=Path.home() / '.aria_source_reliability.json',
        decay_rate=0.99,
        success_threshold=0.6
    )
    
    # Example: After retrieval in aria_retrieval.py
    query = "How does Thompson Sampling work?"
    
    # Mock retrieved chunks
    chunks = [
        {'file': '/data/arxiv/papers/bandits_2015.pdf', 'text': '...', 'score': 0.85},
        {'file': '/data/github/thompson_sampling.py', 'text': '...', 'score': 0.78},
        {'file': '/data/books/algorithms.epub', 'text': '...', 'score': 0.72},
    ]
    
    # Boost chunks by source reliability
    boosted_chunks = scorer.boost_chunks_by_reliability(
        chunks,
        boost_strength=0.1  # Up to 10% boost/penalty
    )
    
    print("Chunk scores after reliability boost:")
    for chunk in boosted_chunks:
        print(f"  {Path(chunk['file']).name}: {chunk['score']:.3f} (reliability: {chunk['reliability']:.3f})")
    
    # ... use boosted_chunks for postfilter ...
    
    # After getting final reward from aria_telemetry
    reward = 0.82  # Example
    
    # Record the retrieval
    scorer.record_retrieval(query, chunks, reward)
    
    # Get stats
    stats = scorer.get_stats_summary()
    print(f"\nSource Reliability Stats:")
    print(f"  Total sources tracked: {stats['total_sources']}")
    print(f"  Avg reliability: {stats['avg_reliability']:.3f}")
    print(f"  Diversity score: {stats['diversity_score']:.3f}")
    
    print(f"\nTop sources:")
    for source_id, reliability, info in stats['top_5_sources']:
        print(f"  {Path(source_id).name}: {reliability:.3f} ({info['uses']} uses)")


if __name__ == '__main__':
    integrate_with_aria_retrieval()
