#!/usr/bin/env python3
"""
Insight Generator - Cross-Domain Connection Discovery with Knowledge Gap Detection

Purpose:
1. Find underlying connections between seemingly disparate domains
2. Build index of potential connections
3. Detect knowledge gaps with natural curiosity to fill them
4. Use high-quality conversation telemetry (score-filtered)
5. Integrate Socratic dialogue for insight generation

Architecture:
- Uses telemetry from scored conversations (conversation_scorer.py)
- Applies Socratic questioning to explore connections
- Employs curiosity engine for gap detection
- Builds cross-domain knowledge graph
- Generates insights for quaternion perspective mapping

Author: dontmindme369
License: CC BY-NC 4.0
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass, asdict
import networkx as nx  # For knowledge graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ARIA paths
ARIA_ROOT = Path("/media/notapplicable/Internal-SSD/ai-quaternions-model")
ARIA_KNOWLEDGE = Path("/media/notapplicable/ARIA-knowledge")
TRAINING_CORPUS = ARIA_KNOWLEDGE / "training_corpus"
TELEMETRY_DIR = Path("/var/telemetry")
INSIGHTS_OUTPUT = ARIA_KNOWLEDGE / "insights"

# Add to path for imports
if str(ARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT))
if str(ARIA_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT / "src"))
if str(ARIA_ROOT / "extra_scripts") not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT / "extra_scripts"))

# Import Socratic dialogue components
try:
    sys.path.insert(0, str(ARIA_ROOT / "extra_scripts"))
    from socratic_dialogue import SocraticDialogueSystem  # type: ignore
except ImportError:
    print("[WARNING] Could not import SocraticDialogueSystem - using fallback", file=sys.stderr)
    SocraticDialogueSystem = None


@dataclass
class KnowledgeGap:
    """Represents a detected knowledge gap"""
    gap_id: str
    domain_a: str
    domain_b: str
    gap_type: str  # 'missing_connection', 'contradiction', 'incomplete', 'boundary'
    description: str
    confidence: float
    evidence: List[str]
    priority: float  # Curiosity-driven priority
    suggested_explorations: List[str]
    timestamp: str


@dataclass
class CrossDomainInsight:
    """Represents a cross-domain connection insight"""
    insight_id: str
    domains: List[str]
    connection_type: str  # 'analogical', 'causal', 'structural', 'conceptual'
    description: str
    confidence: float
    supporting_evidence: List[Dict]
    implications: List[str]
    timestamp: str


class CuriosityDrivenExplorer:
    """
    Implements curiosity-driven knowledge gap exploration
    Similar to curiosity_engine.py but focused on cross-domain connections
    """

    def __init__(self):
        self.curiosity_level = 0.7  # 0-1 scale
        self.exploration_rate = 0.3  # Chance to explore vs exploit
        self.gap_detection_threshold = 0.4
        self.knowledge_gaps: List[KnowledgeGap] = []
        self.investigation_history = []

    def detect_gaps_in_conversation(self, query: str, response: str, domains: List[str]) -> List[KnowledgeGap]:
        """Detect knowledge gaps from a conversation"""
        gaps = []

        # 1. Detect uncertainty markers
        uncertainty_markers = [
            'might be', 'possibly', 'probably', 'unclear', 'uncertain',
            'I think', 'it seems', 'generally', 'usually', 'often'
        ]

        uncertainty_found = any(marker in response.lower() for marker in uncertainty_markers)

        if uncertainty_found and len(domains) >= 2:
            # Potential gap between domains
            gap = KnowledgeGap(
                gap_id=self._generate_gap_id(domains),
                domain_a=domains[0],
                domain_b=domains[1] if len(domains) > 1 else "unknown",
                gap_type='incomplete',
                description=f"Uncertain knowledge connecting {domains[0]} and {domains[1] if len(domains) > 1 else 'unknown'}",
                confidence=0.6,
                evidence=[f"Uncertainty markers in response: {response[:100]}..."],
                priority=self.curiosity_level,
                suggested_explorations=self._generate_exploration_questions(domains),
                timestamp=datetime.now().isoformat()
            )
            gaps.append(gap)

        # 2. Detect missing connections
        if len(domains) >= 2:
            connection_indicators = ['related to', 'similar to', 'like', 'connects to', 'links to']
            has_connection = any(ind in response.lower() for ind in connection_indicators)

            if not has_connection:
                gap = KnowledgeGap(
                    gap_id=self._generate_gap_id(domains),
                    domain_a=domains[0],
                    domain_b=domains[1],
                    gap_type='missing_connection',
                    description=f"Potential unexplored connection between {domains[0]} and {domains[1]}",
                    confidence=0.5,
                    evidence=[f"Query spans multiple domains without explicit connection"],
                    priority=self.curiosity_level * 0.8,
                    suggested_explorations=self._generate_bridge_questions(domains[0], domains[1]),
                    timestamp=datetime.now().isoformat()
                )
                gaps.append(gap)

        # 3. Detect contradictions
        contradiction_markers = ['however', 'but', 'although', 'contrary', 'paradox', 'conflict']
        if any(marker in response.lower() for marker in contradiction_markers):
            gap = KnowledgeGap(
                gap_id=self._generate_gap_id(domains + ['contradiction']),
                domain_a=domains[0] if domains else "unknown",
                domain_b="meta",
                gap_type='contradiction',
                description="Potential contradiction or paradox detected",
                confidence=0.7,
                evidence=[f"Contradiction markers in response"],
                priority=self.curiosity_level * 1.2,  # Higher priority
                suggested_explorations=self._generate_resolution_questions(response),
                timestamp=datetime.now().isoformat()
            )
            gaps.append(gap)

        return gaps

    def _generate_gap_id(self, domains: List[str]) -> str:
        """Generate unique ID for gap"""
        import hashlib
        combined = "_".join(sorted(domains)) + datetime.now().isoformat()
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _generate_exploration_questions(self, domains: List[str]) -> List[str]:
        """Generate questions to explore gaps"""
        if not domains:
            return []

        questions = []

        if len(domains) == 1:
            questions.extend([
                f"What are the fundamental principles underlying {domains[0]}?",
                f"What are the boundaries of {domains[0]}?",
                f"How does {domains[0]} connect to related fields?"
            ])
        else:
            questions.extend([
                f"What deeper principle connects {domains[0]} and {domains[1]}?",
                f"Are there analogies between {domains[0]} and {domains[1]}?",
                f"What would happen if we applied {domains[0]} concepts to {domains[1]}?"
            ])

        return questions

    def _generate_bridge_questions(self, domain_a: str, domain_b: str) -> List[str]:
        """Generate questions to bridge two domains"""
        return [
            f"How does {domain_a} relate to {domain_b}?",
            f"What conceptual structures do {domain_a} and {domain_b} share?",
            f"Can principles from {domain_a} be applied to {domain_b}?",
            f"What would a unified framework for {domain_a} and {domain_b} look like?"
        ]

    def _generate_resolution_questions(self, response: str) -> List[str]:
        """Generate questions to resolve contradictions"""
        return [
            "Under what conditions is each perspective valid?",
            "Is this apparent contradiction actually complementary?",
            "What deeper framework resolves this tension?",
            "Are we asking the right question?"
        ]


class InsightGenerator:
    """
    Main class for generating cross-domain insights from conversation telemetry
    """

    def __init__(self,
                 telemetry_dir: Path = TELEMETRY_DIR,
                 training_corpus_dir: Path = TRAINING_CORPUS,
                 min_quality_score: float = 0.6,
                 enable_socratic: bool = True):

        self.telemetry_dir = Path(telemetry_dir)
        self.training_corpus_dir = Path(training_corpus_dir)
        self.min_quality_score = min_quality_score
        self.enable_socratic = enable_socratic

        # Initialize components
        self.curiosity_explorer = CuriosityDrivenExplorer()
        self.socratic_system = None

        # Knowledge graph
        self.knowledge_graph = nx.Graph()

        # Storage
        self.insights: List[CrossDomainInsight] = []
        self.knowledge_gaps: List[KnowledgeGap] = []
        self.domain_connections = defaultdict(list)

        # TF-IDF for content similarity
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

        # Domain taxonomy (from COMPREHENSIVE_DOMAIN_LIST)
        self.domain_keywords = self._load_domain_taxonomy()

        print(f"[INFO] InsightGenerator initialized", file=sys.stderr)
        print(f"[INFO] Telemetry dir: {self.telemetry_dir}", file=sys.stderr)
        print(f"[INFO] Min quality score: {self.min_quality_score}", file=sys.stderr)

    def _load_domain_taxonomy(self) -> Dict[str, List[str]]:
        """Load domain taxonomy from COMPREHENSIVE_DOMAIN_LIST"""
        # Simplified version - in production, load from the actual file
        return {
            'quantum_physics': ['quantum', 'qubit', 'superposition', 'entanglement', 'wave function'],
            'ai_ml': ['neural network', 'machine learning', 'training', 'model', 'embedding', 'transformer'],
            'consciousness': ['consciousness', 'qualia', 'awareness', 'subjective', 'phenomenal'],
            'mathematics': ['theorem', 'proof', 'equation', 'formula', 'algebra', 'calculus'],
            'information_theory': ['entropy', 'information', 'Shannon', 'compression', 'channel'],
            'philosophy': ['epistemology', 'metaphysics', 'ontology', 'phenomenology', 'ethics'],
            'systems_theory': ['emergence', 'complexity', 'feedback', 'self-organization', 'homeostasis'],
            'computer_science': ['algorithm', 'data structure', 'complexity', 'computation', 'recursion'],
            'engineering': ['circuit', 'signal', 'system', 'design', 'optimization'],
            'biology': ['evolution', 'organism', 'cell', 'gene', 'adaptation']
        }

    def load_high_quality_telemetry(self) -> List[Dict]:
        """Load high-quality conversations from telemetry and training corpus"""
        conversations = []

        # 1. Load from telemetry directory
        if self.telemetry_dir.exists():
            for telemetry_file in self.telemetry_dir.glob("*.jsonl"):
                try:
                    with open(telemetry_file, 'r') as f:
                        for line in f:
                            entry = json.loads(line.strip())

                            # Filter by quality score
                            if 'reward' in entry and entry['reward'] >= self.min_quality_score:
                                conversations.append(entry)
                            elif 'generation' in entry:
                                coverage = entry['generation'].get('coverage_score', 0)
                                if coverage >= self.min_quality_score:
                                    conversations.append(entry)
                except Exception as e:
                    print(f"[WARNING] Error reading {telemetry_file}: {e}", file=sys.stderr)

        # 2. Load from training corpus (already high-quality)
        if self.training_corpus_dir.exists():
            for corpus_file in self.training_corpus_dir.glob("qa_*.json"):
                try:
                    with open(corpus_file, 'r') as f:
                        entry = json.load(f)
                        if entry.get('quality_score', 0) >= self.min_quality_score:
                            conversations.append({
                                'query': entry.get('query'),
                                'response': entry.get('response'),
                                'reward': entry.get('quality_score'),
                                'preset': entry.get('preset'),
                                'timestamp': entry.get('timestamp')
                            })
                except Exception as e:
                    print(f"[WARNING] Error reading {corpus_file}: {e}", file=sys.stderr)

        print(f"[INFO] Loaded {len(conversations)} high-quality conversations", file=sys.stderr)
        return conversations

    def detect_domains_in_text(self, text: str) -> List[Tuple[str, float]]:
        """Detect which domains are present in text"""
        text_lower = text.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by position (earlier = more important)
                    position = text_lower.index(keyword) / len(text_lower)
                    weight = 1.0 - (position * 0.3)
                    # Weight by frequency
                    freq = text_lower.count(keyword)
                    score += weight * freq * 0.2

            domain_scores[domain] = min(score, 1.0)

        # Return domains above threshold
        threshold = 0.15
        detected = [(domain, score) for domain, score in domain_scores.items() if score >= threshold]
        detected.sort(key=lambda x: x[1], reverse=True)

        return detected

    def find_cross_domain_connections(self, conversations: List[Dict]) -> List[CrossDomainInsight]:
        """Find connections between different domains"""
        insights = []

        # Group conversations by domain pairs
        domain_pair_conversations = defaultdict(list)

        for conv in conversations:
            query = conv.get('query', '')
            response = conv.get('response', '')
            combined_text = query + " " + response

            # Detect domains
            domains = self.detect_domains_in_text(combined_text)
            domain_names = [d[0] for d in domains[:3]]  # Top 3 domains

            # If multiple domains detected, this is a cross-domain conversation
            if len(domain_names) >= 2:
                domain_pair = tuple(sorted(domain_names[:2]))
                domain_pair_conversations[domain_pair].append({
                    'query': query,
                    'response': response,
                    'domains': domain_names,
                    'reward': conv.get('reward', 0)
                })

        # Analyze each domain pair
        for domain_pair, convs in domain_pair_conversations.items():
            if len(convs) < 2:  # Need at least 2 conversations
                continue

            # Extract common themes using TF-IDF
            all_text = [c['query'] + " " + c['response'] for c in convs]

            try:
                tfidf_matrix = self.vectorizer.fit_transform(all_text)
                feature_names = self.vectorizer.get_feature_names_out()

                # Get top terms
                avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)  # type: ignore
                top_indices = avg_tfidf.argsort()[-10:][::-1]
                top_terms = [str(feature_names[i]) for i in top_indices]

                # Create insight
                insight = CrossDomainInsight(
                    insight_id=self._generate_insight_id(domain_pair),
                    domains=list(domain_pair),
                    connection_type=self._infer_connection_type(convs, top_terms),
                    description=f"Connection between {domain_pair[0]} and {domain_pair[1]} through: {', '.join(top_terms[:5])}",
                    confidence=self._calculate_connection_confidence(convs),
                    supporting_evidence=[
                        {'query': c['query'][:100], 'reward': c['reward']} for c in convs[:3]
                    ],
                    implications=self._generate_implications(domain_pair, top_terms),
                    timestamp=datetime.now().isoformat()
                )

                insights.append(insight)

                # Add to knowledge graph
                self.knowledge_graph.add_edge(
                    domain_pair[0],
                    domain_pair[1],
                    weight=insight.confidence,
                    connection_type=insight.connection_type,
                    key_terms=top_terms[:5]
                )

            except Exception as e:
                print(f"[WARNING] Error analyzing domain pair {domain_pair}: {e}", file=sys.stderr)
                continue

        return insights

    def _generate_insight_id(self, domain_pair: Tuple[str, str]) -> str:
        """Generate unique insight ID"""
        import hashlib
        combined = "_".join(domain_pair) + datetime.now().isoformat()
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _infer_connection_type(self, conversations: List[Dict], top_terms: List[str]) -> str:
        """Infer the type of connection between domains"""
        # Analyze terms and conversation patterns
        combined_text = " ".join([c['query'] + " " + c['response'] for c in conversations]).lower()

        if any(term in combined_text for term in ['similar', 'like', 'analogous', 'comparable']):
            return 'analogical'
        elif any(term in combined_text for term in ['causes', 'leads to', 'results in', 'because']):
            return 'causal'
        elif any(term in combined_text for term in ['structure', 'pattern', 'architecture', 'framework']):
            return 'structural'
        else:
            return 'conceptual'

    def _calculate_connection_confidence(self, conversations: List[Dict]) -> float:
        """Calculate confidence in the connection"""
        # Based on:
        # 1. Number of supporting conversations
        # 2. Average quality score
        # 3. Consistency of themes

        count_score = min(len(conversations) / 10.0, 1.0)  # Cap at 10 conversations
        avg_quality = np.mean([c.get('reward', 0.5) for c in conversations])

        return float((count_score * 0.4) + (avg_quality * 0.6))

    def _generate_implications(self, domain_pair: Tuple[str, str], key_terms: List[str]) -> List[str]:
        """Generate implications of the connection"""
        return [
            f"Principles from {domain_pair[0]} may apply to {domain_pair[1]}",
            f"Shared concepts: {', '.join(key_terms[:3])}",
            f"Potential for unified framework bridging {domain_pair[0]} and {domain_pair[1]}",
            f"Cross-pollination opportunities in research and application"
        ]

    def detect_knowledge_gaps(self, conversations: List[Dict]) -> List[KnowledgeGap]:
        """Detect knowledge gaps from conversations"""
        all_gaps = []

        for conv in conversations:
            query = conv.get('query', '')
            response = conv.get('response', '')

            # Detect domains
            domains = self.detect_domains_in_text(query + " " + response)
            domain_names = [d[0] for d in domains[:3]]

            # Use curiosity explorer to detect gaps
            gaps = self.curiosity_explorer.detect_gaps_in_conversation(query, response, domain_names)
            all_gaps.extend(gaps)

        return all_gaps

    def generate_insights(self) -> Dict[str, Any]:
        """Main method to generate all insights"""
        print("\n" + "="*70)
        print("ARIA Insight Generator - Cross-Domain Connection Discovery")
        print("="*70 + "\n")

        # 1. Load high-quality conversations
        print("📥 Loading high-quality conversations...")
        conversations = self.load_high_quality_telemetry()

        if not conversations:
            print("⚠️ No high-quality conversations found")
            return {'insights': [], 'gaps': [], 'graph': None}

        # 2. Find cross-domain connections
        print(f"\n🔍 Analyzing {len(conversations)} conversations for cross-domain connections...")
        insights = self.find_cross_domain_connections(conversations)
        print(f"✓ Found {len(insights)} cross-domain insights")

        # 3. Detect knowledge gaps
        print("\n🕳️ Detecting knowledge gaps...")
        gaps = self.detect_knowledge_gaps(conversations)
        print(f"✓ Detected {len(gaps)} knowledge gaps")

        # 4. Build summary statistics
        print("\n📊 Building knowledge graph statistics...")
        graph_stats = {
            'nodes': self.knowledge_graph.number_of_nodes(),
            'edges': self.knowledge_graph.number_of_edges(),
            'density': nx.density(self.knowledge_graph) if self.knowledge_graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_connected_components(self.knowledge_graph)
        }

        print(f"   Nodes (domains): {graph_stats['nodes']}")
        print(f"   Edges (connections): {graph_stats['edges']}")
        print(f"   Graph density: {graph_stats['density']:.3f}")
        print(f"   Connected components: {graph_stats['connected_components']}")

        # 5. Prioritize gaps by curiosity
        gaps.sort(key=lambda g: g.priority, reverse=True)

        # 6. Store results
        self.insights = insights
        self.knowledge_gaps = gaps

        # 7. Save outputs
        self._save_insights()
        self._save_knowledge_gaps()
        self._save_knowledge_graph()

        print("\n" + "="*70)
        print("Insight Generation Complete")
        print("="*70 + "\n")

        return {
            'insights': insights,
            'gaps': gaps,
            'graph_stats': graph_stats,
            'conversations_analyzed': len(conversations)
        }

    def _save_insights(self):
        """Save insights to JSON"""
        output_dir = INSIGHTS_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)

        insights_file = output_dir / f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        insights_data = [asdict(insight) for insight in self.insights]

        with open(insights_file, 'w') as f:
            json.dump(insights_data, f, indent=2)

        print(f"💾 Insights saved to: {insights_file}")

    def _save_knowledge_gaps(self):
        """Save knowledge gaps to JSON"""
        output_dir = INSIGHTS_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)

        gaps_file = output_dir / f"knowledge_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        gaps_data = [asdict(gap) for gap in self.knowledge_gaps]

        with open(gaps_file, 'w') as f:
            json.dump(gaps_data, f, indent=2)

        print(f"💾 Knowledge gaps saved to: {gaps_file}")

    def _save_knowledge_graph(self):
        """Save knowledge graph"""
        output_dir = INSIGHTS_OUTPUT
        output_dir.mkdir(parents=True, exist_ok=True)

        graph_file = output_dir / f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert graph to JSON-serializable format
        graph_data = {
            'nodes': list(self.knowledge_graph.nodes()),
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': data.get('weight', 0),
                    'connection_type': data.get('connection_type', 'unknown'),
                    'key_terms': data.get('key_terms', [])
                }
                for u, v, data in self.knowledge_graph.edges(data=True)
            ]
        }

        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)

        print(f"💾 Knowledge graph saved to: {graph_file}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Generate cross-domain insights from conversation telemetry")
    ap.add_argument("--telemetry-dir", default="/var/telemetry", help="Telemetry directory")
    ap.add_argument("--corpus-dir", default=str(TRAINING_CORPUS), help="Training corpus directory")
    ap.add_argument("--min-score", type=float, default=0.6, help="Minimum quality score (0.0-1.0)")
    ap.add_argument("--enable-socratic", action="store_true", default=True, help="Enable Socratic dialogue")

    args = ap.parse_args()

    generator = InsightGenerator(
        telemetry_dir=Path(args.telemetry_dir),
        training_corpus_dir=Path(args.corpus_dir),
        min_quality_score=args.min_score,
        enable_socratic=args.enable_socratic
    )

    results = generator.generate_insights()

    print(f"\nGenerated {len(results['insights'])} insights and detected {len(results['gaps'])} knowledge gaps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
