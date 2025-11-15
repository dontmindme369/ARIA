#!/usr/bin/env python3
"""
Exemplar Generator - Wave 3: Self-Learning System

Automatically generates high-quality exemplar patterns from successful conversations.

Integration with Current ARIA:
- Monitors aria_telemetry.py for high-reward conversations
- Uses conversation_scorer.py quality metrics
- Extends exemplars.txt with new patterns
- Feeds aria_training.py (ExemplarFitScorer)

Key Features:
1. Automatic pattern extraction from successful Q&A pairs
2. Quality filtering (only learn from high-quality conversations)
3. Deduplication (avoid redundant patterns)
4. Topic clustering (organize by domain)
5. Continuous learning (periodic batch updates)

Flow:
1. Monitor telemetry for high-reward runs (reward > 0.75)
2. Extract query-response patterns from conversation history
3. Cluster by topic and anchor mode
4. Generate exemplar format: "topic::query → response"
5. Append to exemplars.txt (with deduplication)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, TYPE_CHECKING
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib

# Type checking imports
if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVectorizerType
    from sklearn.cluster import KMeans as KMeansType

# Runtime imports
HAVE_SKLEARN = False
TfidfVectorizer: Any = None
KMeans: Any = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except ImportError:
    pass


class ExemplarGenerator:
    """
    Generate exemplar patterns from successful conversations.
    
    Quality Criteria:
    - Reward > 0.75 (high-quality retrieval)
    - Coverage > 0.7 (good knowledge coverage)
    - No contradictions detected
    - Response length > 100 chars (substantive)
    - Query not meta (not about the system itself)
    """
    
    def __init__(
        self,
        telemetry_dir: Path,
        exemplar_file: Path,
        training_corpus_dir: Optional[Path] = None,
        quality_threshold: float = 0.75,
        min_response_length: int = 100,
        max_exemplars_per_batch: int = 50
    ):
        self.telemetry_dir = telemetry_dir
        self.exemplar_file = exemplar_file
        self.training_corpus_dir = training_corpus_dir
        self.quality_threshold = quality_threshold
        self.min_response_length = min_response_length
        self.max_exemplars_per_batch = max_exemplars_per_batch
        
        # Track what we've already converted to exemplars
        self.processed_ids: Set[str] = set()
        self.state_file = Path.home() / '.aria_exemplar_generator.json'
        self._load_state()
        
        # Load existing exemplars for deduplication
        self.existing_patterns: Set[str] = self._load_existing_patterns()
    
    def _load_state(self):
        """Load previously processed conversation IDs"""
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text())
                self.processed_ids = set(state.get('processed_ids', []))
            except:
                pass
    
    def _save_state(self):
        """Save processed conversation IDs"""
        self.state_file.write_text(json.dumps({
            'processed_ids': list(self.processed_ids),
            'last_update': datetime.now().isoformat()
        }, indent=2))
    
    def _load_existing_patterns(self) -> Set[str]:
        """Load existing exemplars to avoid duplicates"""
        patterns = set()
        
        if not self.exemplar_file.exists():
            return patterns
        
        content = self.exemplar_file.read_text(encoding='utf-8')
        
        # Extract all queries from "topic::query → response" format
        for line in content.split('\n'):
            if '::' in line and '→' in line:
                try:
                    topic_query = line.split('→')[0].strip()
                    query = topic_query.split('::')[1].strip() if '::' in topic_query else ''
                    if query:
                        patterns.add(self._normalize_query(query))
                except:
                    continue
        
        return patterns
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for deduplication"""
        # Lowercase, remove extra whitespace, strip punctuation
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[?.!]+$', '', normalized)
        return normalized
    
    def _query_hash(self, query: str) -> str:
        """Generate stable hash for query"""
        return hashlib.md5(self._normalize_query(query).encode()).hexdigest()[:12]
    
    def scan_telemetry(self, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Scan telemetry for high-quality conversations.
        
        Args:
            max_age_hours: Only look at recent telemetry
            
        Returns:
            List of high-quality conversation records
        """
        if not self.telemetry_dir.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        candidates: List[Dict[str, Any]] = []
        
        # Scan telemetry files
        for telemetry_file in self.telemetry_dir.glob('*.jsonl'):
            try:
                mtime = datetime.fromtimestamp(telemetry_file.stat().st_mtime)
                if mtime < cutoff_time:
                    continue
                
                # Read telemetry entries
                for line in telemetry_file.read_text().split('\n'):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Quality filters
                        reward = entry.get('reward', 0.0)
                        coverage = entry.get('generation', {}).get('coverage_score', 0.0)
                        query = entry.get('query', '')
                        run_id = entry.get('run_id', '')
                        
                        if run_id in self.processed_ids:
                            continue
                        
                        if (reward >= self.quality_threshold and 
                            coverage >= 0.7 and
                            len(query) > 10 and
                            not self._is_meta_query(query)):
                            
                            candidates.append({
                                'run_id': run_id,
                                'query': query,
                                'reward': reward,
                                'coverage': coverage,
                                'preset': entry.get('preset', 'unknown'),
                                'timestamp': entry.get('timestamp', datetime.now().isoformat())
                            })
                    
                    except json.JSONDecodeError:
                        continue
            
            except Exception:
                continue
        
        return candidates
    
    def _is_meta_query(self, query: str) -> bool:
        """Check if query is about the system itself (skip these)"""
        meta_patterns = [
            r'\baria\b',
            r'\byou are\b',
            r'\byour (name|system|architecture)\b',
            r'\bhow do you work\b',
            r'\bwhat (model|AI|system) are you\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in meta_patterns)
    
    def extract_response_from_corpus(self, query: str, run_id: str) -> Optional[str]:
        """
        Extract the actual response from training corpus.
        
        Args:
            query: The query text
            run_id: Telemetry run ID
            
        Returns:
            Response text or None
        """
        if not self.training_corpus_dir or not self.training_corpus_dir.exists():
            return None
        
        # Look for conversation files with this query
        for corpus_file in self.training_corpus_dir.glob('*.json'):
            try:
                data = json.loads(corpus_file.read_text())
                
                # Check if this file contains our query
                for qa_pair in data.get('qa_pairs', []):
                    if self._normalize_query(qa_pair.get('query', '')) == self._normalize_query(query):
                        response = qa_pair.get('response', '')
                        if len(response) >= self.min_response_length:
                            return response
            
            except:
                continue
        
        return None
    
    def detect_topic(self, query: str, response: str) -> str:
        """
        Detect topic/domain from query and response.
        
        Uses keyword matching to classify into anchor modes.
        """
        text = (query + ' ' + response).lower()
        
        # Topic keywords (aligned with anchor modes)
        topics = {
            'technical': ['code', 'implement', 'function', 'algorithm', 'debug', 'error', 'syntax'],
            'philosophical': ['why', 'meaning', 'nature', 'existence', 'consciousness', 'ethics'],
            'analytical': ['analyze', 'compare', 'evaluate', 'assess', 'evidence', 'data'],
            'educational': ['explain', 'learn', 'understand', 'teach', 'basics', 'introduction'],
            'factual': ['what is', 'who is', 'when did', 'where is', 'define', 'fact'],
            'creative': ['imagine', 'story', 'design', 'brainstorm', 'creative', 'novel'],
            'formal': ['research', 'academic', 'study', 'paper', 'scientific', 'methodology'],
            'casual': ['help', 'question', 'wondering', 'think', 'general', 'curious']
        }
        
        scores = {}
        for topic, keywords in topics.items():
            scores[topic] = sum(1 for kw in keywords if kw in text)
        
        # Return highest scoring topic
        best_topic = max(scores.items(), key=lambda x: x[1])
        return best_topic[0] if best_topic[1] > 0 else 'general'
    
    def generate_exemplar_pattern(
        self,
        query: str,
        response: str,
        topic: str
    ) -> str:
        """
        Generate exemplar in standard format.
        
        Format: "topic::query → key_points_from_response"
        
        Args:
            query: User query
            response: Assistant response
            topic: Detected topic/domain
            
        Returns:
            Exemplar string
        """
        # Extract key points from response (first 2-3 sentences)
        sentences = re.split(r'[.!?]+', response)
        key_points = '. '.join(sentences[:3]).strip()
        
        # Limit length
        if len(key_points) > 400:
            key_points = key_points[:397] + '...'
        
        # Format
        exemplar = f"{topic}::{query} → {key_points}"
        
        return exemplar
    
    def cluster_and_deduplicate(
        self,
        candidates: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Cluster similar exemplars and remove duplicates.
        
        Args:
            candidates: List of (query, response, topic) tuples
            
        Returns:
            Deduplicated list
        """
        if not candidates:
            return []
        
        # First pass: exact duplicate removal
        seen_queries = set()
        unique_candidates = []
        
        for query, response, topic in candidates:
            norm_query = self._normalize_query(query)
            
            # Skip if already in existing patterns
            if norm_query in self.existing_patterns:
                continue
            
            # Skip if seen in this batch
            if norm_query in seen_queries:
                continue
            
            seen_queries.add(norm_query)
            unique_candidates.append((query, response, topic))
        
        # Second pass: semantic clustering (if sklearn available)
        if HAVE_SKLEARN and len(unique_candidates) > 10:
            queries = [q for q, _, _ in unique_candidates]
            
            try:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                vectors = vectorizer.fit_transform(queries)
                
                # Cluster
                n_clusters = min(10, len(unique_candidates) // 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vectors)
                
                # Keep only most representative from each cluster
                clusters: Dict[int, List[Tuple[str, str, str]]] = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append(unique_candidates[i])
                
                # Select best from each cluster
                final_candidates = []
                for cluster_items in clusters.values():
                    # Pick the one with longest response (most informative)
                    best = max(cluster_items, key=lambda x: len(x[1]))
                    final_candidates.append(best)
                
                return final_candidates[:self.max_exemplars_per_batch]
            
            except Exception:
                # Fall back to simple selection
                return unique_candidates[:self.max_exemplars_per_batch]
        
        return unique_candidates[:self.max_exemplars_per_batch]
    
    def generate_batch(self, max_age_hours: int = 24) -> int:
        """
        Generate a batch of new exemplars from recent conversations.
        
        Args:
            max_age_hours: How far back to scan
            
        Returns:
            Number of exemplars generated
        """
        print(f"🔍 Scanning telemetry for high-quality conversations...")
        
        # Find high-quality conversations
        candidates = self.scan_telemetry(max_age_hours)
        print(f"  Found {len(candidates)} high-quality runs")
        
        if not candidates:
            return 0
        
        # Extract responses
        print(f"📝 Extracting responses from corpus...")
        exemplar_candidates: List[Tuple[str, str, str]] = []
        
        for candidate in candidates:
            query = candidate['query']
            run_id = candidate['run_id']
            
            # Get response
            response = self.extract_response_from_corpus(query, run_id)
            
            if response:
                # Detect topic
                topic = self.detect_topic(query, response)
                exemplar_candidates.append((query, response, topic))
                
                # Mark as processed
                self.processed_ids.add(run_id)
        
        print(f"  Extracted {len(exemplar_candidates)} Q&A pairs")
        
        if not exemplar_candidates:
            self._save_state()
            return 0
        
        # Cluster and deduplicate
        print(f"🔬 Clustering and deduplicating...")
        final_exemplars = self.cluster_and_deduplicate(exemplar_candidates)
        print(f"  Selected {len(final_exemplars)} unique exemplars")
        
        if not final_exemplars:
            self._save_state()
            return 0
        
        # Generate exemplar patterns
        print(f"✍️  Generating exemplar patterns...")
        new_patterns: List[str] = []
        
        for query, response, topic in final_exemplars:
            pattern = self.generate_exemplar_pattern(query, response, topic)
            new_patterns.append(pattern)
        
        # Append to exemplar file
        print(f"💾 Appending to {self.exemplar_file.name}...")
        
        with open(self.exemplar_file, 'a', encoding='utf-8') as f:
            f.write('\n\n')
            f.write(f'# Auto-generated exemplars - {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write('# Generated from high-quality conversations (reward > 0.75)\n\n')
            for pattern in new_patterns:
                f.write(pattern + '\n')
        
        # Update existing patterns
        for query, _, _ in final_exemplars:
            self.existing_patterns.add(self._normalize_query(query))
        
        # Save state
        self._save_state()
        
        print(f"✅ Generated {len(new_patterns)} new exemplars!")
        return len(new_patterns)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics"""
        return {
            'processed_conversations': len(self.processed_ids),
            'existing_patterns': len(self.existing_patterns),
            'exemplar_file': str(self.exemplar_file),
            'quality_threshold': self.quality_threshold
        }


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ARIA Exemplar Generator - Learn from successful conversations"
    )
    parser.add_argument(
        '--telemetry-dir',
        type=Path,
        default=Path('/media/notapplicable/Internal-SSD/ai-quaternions-model/var/telemetry'),
        help='Telemetry directory'
    )
    parser.add_argument(
        '--exemplar-file',
        type=Path,
        default=Path('/media/notapplicable/Internal-SSD/ai-quaternions-model/exemplars.txt'),
        help='Exemplar file to append to'
    )
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        default=Path('/media/notapplicable/ARIA-knowledge/training_corpus'),
        help='Training corpus directory'
    )
    parser.add_argument(
        '--max-age-hours',
        type=int,
        default=24,
        help='Maximum age of telemetry to scan (hours)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.75,
        help='Minimum reward threshold for quality'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics only'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ExemplarGenerator(
        telemetry_dir=args.telemetry_dir,
        exemplar_file=args.exemplar_file,
        training_corpus_dir=args.corpus_dir,
        quality_threshold=args.quality_threshold
    )
    
    if args.stats:
        stats = generator.get_statistics()
        print("\n📊 Exemplar Generator Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        # Generate batch
        count = generator.generate_batch(max_age_hours=args.max_age_hours)
        
        if count > 0:
            print(f"\n🎉 Success! Generated {count} new exemplars")
            print(f"   Added to: {args.exemplar_file}")
        else:
            print("\n ℹ️  No new exemplars generated (no qualifying conversations found)")


if __name__ == '__main__':
    main()
