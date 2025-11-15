#!/usr/bin/env python3
"""
ARIA Curiosity Engine - Unified Gap Detection, Socratic Questioning & Synthesis

Consolidates:
- curiosity_engine.py - gap detection
- socratic_dialogue.py - questioning patterns  
- answer_synth.py - response generation
- answer_synth_socratic.py - enhanced synthesis
- curiosity_training_prompts.py - exemplar learning

Flow:
1. Gap Detection: What's missing from retrieved context?
2. Socratic Questions: Generate probing questions (internal + external)
3. Internal Resolution: Re-query retrieval for answerable gaps
4. Synthesis: Weave everything into coherent response
5. Learning: Train from high-quality exchanges
"""

import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from datetime import datetime
import numpy as np


# ============================================================================
# EXEMPLAR LIBRARY - Learn from High-Quality Exchanges
# ============================================================================

class ExemplarLibrary:
    """Load and learn from high-quality Q&A exemplars"""
    
    def __init__(self, exemplar_path: Optional[Path] = None):
        self.exemplars: List[Dict[str, Any]] = []
        self.patterns: Dict[str, List[str]] = defaultdict(list)
        self.quality_threshold = 0.7
        
        if exemplar_path and exemplar_path.exists():
            self._load_exemplars(exemplar_path)
    
    def _load_exemplars(self, path: Path):
        """Load exemplars from file (supports both formats)"""
        content = path.read_text(encoding='utf-8')
        
        # Format: topic::query → response
        for line in content.split('\n'):
            if '::' not in line or '→' not in line:
                continue
            
            try:
                topic_query, response = line.split('→', 1)
                topic, query = topic_query.split('::', 1)
                
                self.exemplars.append({
                    'topic': topic.strip(),
                    'query': query.strip(),
                    'response': response.strip(),
                    'quality': 1.0  # All exemplars are high quality
                })
                
                self.patterns[topic.strip()].append(query.strip())
            except:
                continue
    
    def find_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find k most similar exemplars to query"""
        if not self.exemplars:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for ex in self.exemplars:
            ex_words = set(ex['query'].lower().split())
            overlap = len(query_words & ex_words) / max(len(query_words), 1)
            scored.append((overlap, ex))
        
        scored.sort(reverse=True)
        return [ex for score, ex in scored[:k] if score > 0.1]
    
    def get_topic_patterns(self, topic: str) -> List[str]:
        """Get example queries for a topic"""
        return self.patterns.get(topic, [])
    
    def add_successful_exchange(self, query: str, response: str, quality: float):
        """Learn from successful exchanges"""
        if quality < self.quality_threshold:
            return
        
        self.exemplars.append({
            'topic': 'learned',
            'query': query,
            'response': response,
            'quality': quality,
            'timestamp': datetime.now().isoformat()
        })


# ============================================================================
# GAP DETECTOR - Identify Missing Information
# ============================================================================

class GapDetector:
    """Detect semantic, factual, and logical gaps in retrieved context"""
    
    def __init__(self):
        # Uncertainty markers indicating gaps
        self.uncertainty_markers = [
            'might be', 'possibly', 'probably', 'unclear', 'unsure',
            'approximately', 'roughly', 'generally', 'usually', 'often',
            'sometimes', 'may', 'could be', 'appears to', 'seems like'
        ]
        
        # Patterns indicating missing information
        self.missing_patterns = [
            r"(?:I\s+)?don't (?:have|know)(?: about| regarding)?",
            r"(?:unclear|unknown|unsure) (?:what|why|how|when|where)",
            r"need(?: more)? (?:information|context|details)",
            r"(?:lacking|missing) (?:data|information|context)",
            r"would need to (?:know|understand|learn)",
            r"not enough (?:information|data|context)",
            r"insufficient (?:information|data|evidence)"
        ]
    
    def analyze_context_gaps(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]],
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze retrieved context for gaps relative to query
        
        Returns:
            {
                'has_gaps': bool,
                'gap_score': float,  # 0-1, higher = more gaps
                'semantic_gaps': List[str],
                'factual_gaps': List[str],
                'logical_gaps': List[str],
                'confidence': float
            }
        """
        gaps = {
            'has_gaps': False,
            'gap_score': 0.0,
            'semantic_gaps': [],
            'factual_gaps': [],
            'logical_gaps': [],
            'confidence': confidence
        }
        
        # Extract text from chunks
        context_text = ' '.join(
            chunk.get('text', '') for chunk in retrieved_chunks
        ).lower()
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        context_words = set(re.findall(r'\w+', context_text))
        
        # 1. SEMANTIC GAPS: Query terms not in context
        missing_terms = query_words - context_words - {'what', 'how', 'why', 'when', 'where', 'the', 'a', 'an', 'is', 'are', 'be'}
        if missing_terms:
            gaps['semantic_gaps'] = list(missing_terms)
            gaps['gap_score'] += 0.3
        
        # 2. FACTUAL GAPS: Uncertainty markers in context
        uncertainty_count = sum(
            context_text.count(marker) for marker in self.uncertainty_markers
        )
        if uncertainty_count > 0:
            gaps['factual_gaps'].append(f"Context contains {uncertainty_count} uncertainty markers")
            gaps['gap_score'] += 0.2 * min(uncertainty_count / 3, 1.0)
        
        # 3. LOGICAL GAPS: Missing information patterns
        for pattern in self.missing_patterns:
            if re.search(pattern, context_text, re.IGNORECASE):
                gaps['logical_gaps'].append(f"Pattern: {pattern}")
                gaps['gap_score'] += 0.15
        
        # 4. LOW CONFIDENCE: Inherently indicates gaps
        if confidence < 0.7:
            gaps['confidence'] = confidence
            gaps['gap_score'] += (0.7 - confidence) * 0.5
        
        # 5. INSUFFICIENT CONTEXT: Too little retrieved
        if len(retrieved_chunks) < 3:
            gaps['logical_gaps'].append("Insufficient context retrieved")
            gaps['gap_score'] += 0.2
        
        # Normalize gap score
        gaps['gap_score'] = min(gaps['gap_score'], 1.0)
        gaps['has_gaps'] = gaps['gap_score'] > 0.3
        
        return gaps


# ============================================================================
# SOCRATIC QUESTION GENERATOR
# ============================================================================

class SocraticGenerator:
    """Generate probing questions to fill knowledge gaps"""
    
    def __init__(self, exemplar_library: Optional[ExemplarLibrary] = None):
        self.exemplars = exemplar_library
        
        # Question templates by category
        self.patterns = {
            'clarification': [
                "What specifically is meant by {term}?",
                "Can you provide an example of {concept}?",
                "How does {this} relate to {that}?",
                "Could you elaborate on {topic}?",
            ],
            'assumptions': [
                "What assumptions underlie {concept}?",
                "What if {assumption} weren't true?",
                "Is {claim} always the case?",
                "What evidence supports {assertion}?",
            ],
            'implications': [
                "What are the implications of {finding}?",
                "How does {this} affect {that}?",
                "What follows from {premise}?",
                "What would happen if {scenario}?",
            ],
            'alternatives': [
                "Are there alternative explanations for {phenomenon}?",
                "Could {alternative} also be true?",
                "What other factors might influence {outcome}?",
                "Have we considered {perspective}?",
            ],
            'deeper_inquiry': [
                "Why is {aspect} important?",
                "What causes {effect}?",
                "How can we verify {claim}?",
                "What is the underlying mechanism of {process}?",
            ]
        }
    
    def generate_questions(
        self,
        query: str,
        gaps: Dict[str, Any],
        max_questions: int = 3
    ) -> Dict[str, List[str]]:
        """
        Generate Socratic questions based on detected gaps
        
        Returns:
            {
                'internal': List[str],  # Questions retrieval might answer
                'external': List[str]   # Questions needing human input
            }
        """
        questions: Dict[str, List[str]] = {'internal': [], 'external': []}
        
        # Generate questions from semantic gaps
        for term in gaps.get('semantic_gaps', [])[:3]:
            if len(term) > 3:  # Skip very short terms
                q = f"What is {term} in the context of {query.split('?')[0]}?"
                if len(questions['internal']) < max_questions:
                    questions['internal'].append(q)
        
        # Generate questions from factual gaps
        if gaps.get('factual_gaps'):
            q = "What additional details would clarify the uncertainty in this context?"
            questions['external'].append(q)
        
        # Generate questions from logical gaps
        if gaps.get('logical_gaps'):
            q = "What information is needed to fully address this question?"
            if len(questions['internal']) < max_questions:
                questions['internal'].append(q)
        
        # Limit output
        questions['internal'] = questions['internal'][:max_questions]
        questions['external'] = questions['external'][:max_questions]
        
        return questions
    
    def refine_questions(
        self,
        questions: List[str],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Filter out questions already answered by retrieved context"""
        refined = []
        context_text = ' '.join(c.get('text', '') for c in retrieved_chunks).lower()
        
        for q in questions:
            # Simple heuristic: if question terms mostly appear in context, likely answered
            q_terms = set(re.findall(r'\w+', q.lower()))
            context_terms = set(re.findall(r'\w+', context_text))
            overlap = len(q_terms & context_terms) / len(q_terms) if q_terms else 0
            
            if overlap < 0.7:  # Question likely not answered
                refined.append(q)
        
        return refined


# ============================================================================
# RESPONSE SYNTHESIZER
# ============================================================================

class ResponseSynthesizer:
    """Synthesize coherent responses from chunks, gaps, and Socratic questions"""
    
    def __init__(self, personality: int = 7):
        """
        Args:
            personality: 0-10, controls response depth/style
                0-3: Brief, factual
                4-6: Balanced
                7-10: Deep, exploratory
        """
        self.personality = max(0, min(10, personality))
    
    def synthesize(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        mode: str,
        gaps: Optional[Dict[str, Any]] = None,
        socratic_questions: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Synthesize response based on mode and personality
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            mode: 'speed', 'depth', or 'adaptive'
            gaps: Gap analysis results
            socratic_questions: Generated questions
        
        Returns:
            Synthesized response text
        """
        if not chunks:
            return "No relevant information found."
        
        # Determine synthesis approach
        if mode == 'speed':
            return self._speed_synthesis(query, chunks)
        elif mode == 'depth':
            return self._depth_synthesis(query, chunks, gaps, socratic_questions)
        else:  # adaptive
            # Choose based on personality and gap score
            gap_score = gaps.get('gap_score', 0) if gaps else 0
            if gap_score > 0.5 or self.personality >= 7:
                return self._depth_synthesis(query, chunks, gaps, socratic_questions)
            else:
                return self._speed_synthesis(query, chunks)
    
    def _speed_synthesis(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Fast, concise response"""
        # Take top 3 chunks
        top_chunks = chunks[:3]
        
        # Extract key sentences
        sentences = []
        for chunk in top_chunks:
            text = chunk.get('text', '')
            # Get first 2 sentences
            sents = re.split(r'[.!?]\s+', text)
            sentences.extend(sents[:2])
        
        # Combine and truncate
        response = ' '.join(sentences)
        if len(response) > 500:
            response = response[:500] + '...'
        
        return response
    
    def _depth_synthesis(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        gaps: Optional[Dict[str, Any]],
        socratic_questions: Optional[Dict[str, List[str]]]
    ) -> str:
        """Deep, layered exploration"""
        sections = []
        
        # 1. DIRECT ANSWER from chunks
        if chunks:
            primary = self._extract_primary_info(query, chunks[:5])
            sections.append(primary)
        
        # 2. DEEPER CONTEXT from more chunks
        if len(chunks) > 5:
            context = self._build_context(chunks[5:10])
            if context:
                sections.append(f"\nAdditional context: {context}")
        
        # 3. IDENTIFIED GAPS (if any)
        if gaps and gaps.get('has_gaps'):
            gap_note = self._format_gaps(gaps)
            sections.append(f"\n{gap_note}")
        
        # 4. SOCRATIC FOLLOW-UPS (if personality allows)
        if socratic_questions and self.personality >= 5:
            external_qs = socratic_questions.get('external', [])
            if external_qs:
                sections.append(f"\nTo explore further: {external_qs[0]}")
        
        return '\n'.join(sections)
    
    def _extract_primary_info(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Extract most relevant information"""
        # Handle empty chunks
        if not chunks:
            return "No information available."
        
        # Score chunks by query term overlap - FIX: Use enumerate to prevent dict comparison
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        # Create scored list with index to ensure unique sorting
        scored_chunks = [
            (len(query_terms & set(re.findall(r'\w+', chunk.get('text', '').lower()))), i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        
        # Sort by score (desc), then by index (for deterministic ordering)
        scored_chunks.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        
        # Get best chunk
        best_chunk = scored_chunks[0][2]
        
        text = best_chunk.get('text', '')
        # Take first paragraph
        para = text.split('\n\n')[0] if '\n\n' in text else text[:400]
        return para
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build supporting context"""
        snippets = []
        for chunk in chunks:
            text = chunk.get('text', '')
            # First sentence only
            sent = re.split(r'[.!?]\s+', text)[0]
            if sent and len(sent) > 20:
                snippets.append(sent)
        
        return ' '.join(snippets[:3])
    
    def _format_gaps(self, gaps: Dict[str, Any]) -> str:
        """Format gap information"""
        if not gaps.get('has_gaps'):
            return ""
        
        notes = []
        if gaps.get('semantic_gaps'):
            terms = ', '.join(gaps['semantic_gaps'][:3])
            notes.append(f"Some terms need clarification: {terms}")
        
        if gaps['gap_score'] > 0.6:
            notes.append("The available information may be incomplete.")
        
        return ' '.join(notes) if notes else ""


# ============================================================================
# MAIN CURIOSITY ENGINE - Orchestrates Everything
# ============================================================================

class ARIACuriosity:
    """
    Unified Curiosity Engine
    
    Orchestrates gap detection, Socratic questioning, synthesis, and learning
    """
    
    def __init__(
        self,
        exemplar_path: Optional[Path] = None,
        personality: int = 7,
        gap_threshold: float = 0.3
    ):
        self.exemplars = ExemplarLibrary(exemplar_path)
        self.gap_detector = GapDetector()
        self.socratic_gen = SocraticGenerator(self.exemplars)
        self.synthesizer = ResponseSynthesizer(personality)
        
        self.gap_threshold = gap_threshold
        self.curiosity_history: List[Dict[str, Any]] = []
    
    async def process(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence: float = 1.0,
        mode: str = 'adaptive',  # 'speed', 'depth', 'adaptive'
        enable_socratic: bool = True
    ) -> Dict[str, Any]:
        """
        Main processing pipeline
        
        Returns:
            {
                'response': str,
                'gaps': Dict,
                'socratic_questions': Dict,
                'needs_followup': bool,
                'curiosity_level': float
            }
        """
        # 1. DETECT GAPS
        gaps = self.gap_detector.analyze_context_gaps(
            query, retrieved_chunks, confidence
        )
        
        # 2. GENERATE SOCRATIC QUESTIONS (if gaps exist)
        socratic_questions = {'internal': [], 'external': []}
        if enable_socratic and gaps['has_gaps'] and gaps['gap_score'] > self.gap_threshold:
            socratic_questions = self.socratic_gen.generate_questions(
                query, gaps, max_questions=3
            )
        
        # 3. SYNTHESIZE RESPONSE
        response = self.synthesizer.synthesize(
            query, retrieved_chunks, mode, gaps, socratic_questions
        )
        
        # 4. DETERMINE IF FOLLOW-UP NEEDED
        needs_followup = (
            gaps['gap_score'] > 0.5 and 
            len(socratic_questions.get('external', [])) > 0
        )
        
        # 5. LOG INTERACTION
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'gap_score': gaps['gap_score'],
            'needs_followup': needs_followup,
            'mode': mode
        }
        self.curiosity_history.append(interaction)
        
        return {
            'response': response,
            'gaps': gaps,
            'socratic_questions': socratic_questions,
            'needs_followup': needs_followup,
            'curiosity_level': gaps['gap_score']
        }
    
    async def resolve_internal_questions(
        self,
        questions: List[str],
        retrieval_fn  # Callback to re-query retrieval
    ) -> List[Dict[str, Any]]:
        """
        Re-query retrieval to answer internal Socratic questions
        
        Args:
            questions: List of questions to answer
            retrieval_fn: Async function(query) -> chunks
        
        Returns:
            List of {question, answer_chunks}
        """
        results = []
        for q in questions:
            chunks = await retrieval_fn(q)
            results.append({'question': q, 'answer_chunks': chunks})
        return results
    
    def learn_from_exchange(
        self,
        query: str,
        response: str,
        quality_score: float
    ):
        """Learn from successful exchanges"""
        self.exemplars.add_successful_exchange(query, response, quality_score)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curiosity statistics"""
        if not self.curiosity_history:
            return {}
        
        recent = self.curiosity_history[-100:]
        
        return {
            'total_interactions': len(self.curiosity_history),
            'avg_gap_score': sum(x['gap_score'] for x in recent) / len(recent),
            'followup_rate': sum(1 for x in recent if x['needs_followup']) / len(recent),
            'mode_distribution': {
                mode: sum(1 for x in recent if x.get('mode') == mode) / len(recent)
                for mode in ['speed', 'depth', 'adaptive']
            },
            'learned_exemplars': len([e for e in self.exemplars.exemplars if e.get('topic') == 'learned'])
        }


# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

def main():
    """Test curiosity engine"""
    import sys
    
    # Initialize
    exemplar_path = Path("data/exemplars.txt")
    if not exemplar_path.exists():
        exemplar_path = None
    
    curiosity = ARIACuriosity(exemplar_path, personality=7)
    
    # Mock retrieved chunks
    mock_chunks = [
        {'text': 'Machine learning is a subset of AI that enables computers to learn from data.', 'score': 0.9},
        {'text': 'Deep learning uses neural networks with multiple layers.', 'score': 0.8},
    ]
    
    # Test query
    query = sys.argv[1] if len(sys.argv) > 1 else "How does machine learning work?"
    
    # Process
    async def test():
        result = await curiosity.process(query, mock_chunks, confidence=0.6)
        
        print("\n=== CURIOSITY ENGINE TEST ===")
        print(f"\nQuery: {query}")
        print(f"\nResponse:\n{result['response']}")
        print(f"\nGap Score: {result['curiosity_level']:.2f}")
        print(f"Needs Follow-up: {result['needs_followup']}")
        
        if result['socratic_questions']['external']:
            print(f"\nSocratic Questions:")
            for q in result['socratic_questions']['external']:
                print(f"  - {q}")
        
        print(f"\n{curiosity.get_stats()}")
    
    asyncio.run(test())


if __name__ == '__main__':
    main()
