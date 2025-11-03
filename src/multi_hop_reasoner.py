#!/usr/bin/env python3
"""
Multi-Hop Reasoner - Wave 2: Advanced Reasoning

Decomposes complex queries requiring multiple retrieval steps into sub-queries,
executes them in order, and synthesizes a final answer.

Integration with Current ARIA:
- Wraps aria_retrieval.py for multi-step retrieval
- Uses aria_curiosity.py for gap-driven sub-query generation
- Feeds into conversation_state_graph.py to track reasoning chains
- Integrates with aria_session.py for context-aware decomposition

Key Features:
1. Query complexity detection (is multi-hop needed?)
2. Intelligent decomposition (comparison, causal, compositional)
3. Dependency-aware execution (sub-query ordering)
4. Incremental context building (each answer informs next query)
5. Answer synthesis (combine sub-answers coherently)

Query Types Handled:
- Comparison: "Compare X and Y"
- Causal: "Why does X cause Y?"
- Compositional: "What is X and how does it relate to Y?"
- Conditional: "If X then Y, but what about Z?"
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class QueryType(Enum):
    """Types of complex queries"""
    SIMPLE = "simple"  # Single-hop, no decomposition needed
    COMPARISON = "comparison"  # Compare A vs B
    CAUSAL = "causal"  # Why X? How X causes Y?
    COMPOSITIONAL = "compositional"  # X + Y + Z together
    CONDITIONAL = "conditional"  # If X then Y
    SEQUENTIAL = "sequential"  # First X, then Y
    MULTI_ASPECT = "multi_aspect"  # Multiple facets of X


@dataclass
class SubQuery:
    """A sub-query in a multi-hop reasoning chain"""
    id: str
    text: str
    query_type: QueryType
    
    # Dependencies (which sub-queries must be answered first)
    depends_on: List[str] = field(default_factory=list)
    
    # Execution state
    executed: bool = False
    answer: Optional[str] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Why this sub-query was generated
    reasoning: str = ""


@dataclass
class ReasoningChain:
    """Complete multi-hop reasoning chain"""
    original_query: str
    query_type: QueryType
    
    # Sub-queries in dependency order
    sub_queries: List[SubQuery] = field(default_factory=list)
    
    # Final synthesized answer
    final_answer: Optional[str] = None
    synthesis_confidence: float = 0.0
    
    # Execution metadata
    total_retrievals: int = 0
    execution_time: float = 0.0


class MultiHopReasoner:
    """
    Multi-hop query decomposition and reasoning
    
    Process:
    1. Detect if query is multi-hop
    2. Classify query type
    3. Decompose into sub-queries with dependencies
    4. Execute sub-queries in order (respecting dependencies)
    5. Synthesize final answer from sub-answers
    """
    
    def __init__(
        self,
        retrieval_fn: Optional[Callable] = None,
        max_hops: int = 5
    ):
        """
        Args:
            retrieval_fn: Function(query: str) -> List[chunks]
            max_hops: Maximum number of sub-queries
        """
        self.retrieval_fn = retrieval_fn
        self.max_hops = max_hops
        
        # Patterns for query classification
        self._build_patterns()
    
    def _build_patterns(self):
        """Build patterns for query classification"""
        
        self.comparison_patterns = [
            r'\b(compare|contrast|versus|vs\.?|difference between)\b',
            r'\b(better|worse|advantages?|disadvantages?)\b.*\b(than|over)\b',
            r'\bwhich\b.*(better|best|preferable)',
        ]
        
        self.causal_patterns = [
            r'\b(why|how come)\b',
            r'\b(cause|reason|lead to|result in)\b',
            r'\bwhat makes?\b',
        ]
        
        self.compositional_patterns = [
            r'\band\b.*\band\b',  # Multiple "and"s
            r'\b(relate|connection|relationship) between\b',
            r'\b(combine|integrate|together)\b',
        ]
        
        self.conditional_patterns = [
            r'\b(if|when|suppose|assuming)\b',
            r'\b(then|otherwise|else)\b',
        ]
        
        self.sequential_patterns = [
            r'\b(first|then|next|finally)\b',
            r'\b(step by step|stages?|phases?)\b',
            r'\b(process|procedure|workflow)\b',
        ]
    
    def detect_multi_hop(self, query: str) -> bool:
        """
        Detect if query requires multi-hop reasoning
        
        Returns:
            True if multi-hop needed
        """
        query_lower = query.lower()
        
        # Indicators of complexity
        indicators = [
            ' and ' in query_lower and query_lower.count(' and ') > 1,
            ' but ' in query_lower,
            ' however ' in query_lower,
            'compare' in query_lower,
            'versus' in query_lower or ' vs ' in query_lower,
            'why' in query_lower and len(query.split()) > 8,
            'how' in query_lower and 'relate' in query_lower,
            query_lower.count('?') > 1,  # Multiple questions
        ]
        
        return any(indicators)
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type
        
        Returns:
            QueryType enum
        """
        query_lower = query.lower()
        
        # Check patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON
        
        for pattern in self.causal_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CAUSAL
        
        for pattern in self.compositional_patterns:
            if re.search(pattern, query_lower):
                return QueryType.COMPOSITIONAL
        
        for pattern in self.conditional_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CONDITIONAL
        
        for pattern in self.sequential_patterns:
            if re.search(pattern, query_lower):
                return QueryType.SEQUENTIAL
        
        # Check for multi-aspect (multiple unrelated questions)
        if query.count('?') > 1:
            return QueryType.MULTI_ASPECT
        
        return QueryType.SIMPLE
    
    def decompose(self, query: str) -> ReasoningChain:
        """
        Decompose query into sub-queries
        
        Returns:
            ReasoningChain with sub-queries
        """
        query_type = self.classify_query(query)
        
        if query_type == QueryType.SIMPLE:
            # No decomposition needed
            return ReasoningChain(
                original_query=query,
                query_type=query_type
            )
        
        # Decompose based on type
        if query_type == QueryType.COMPARISON:
            sub_queries = self._decompose_comparison(query)
        elif query_type == QueryType.CAUSAL:
            sub_queries = self._decompose_causal(query)
        elif query_type == QueryType.COMPOSITIONAL:
            sub_queries = self._decompose_compositional(query)
        elif query_type == QueryType.CONDITIONAL:
            sub_queries = self._decompose_conditional(query)
        elif query_type == QueryType.SEQUENTIAL:
            sub_queries = self._decompose_sequential(query)
        elif query_type == QueryType.MULTI_ASPECT:
            sub_queries = self._decompose_multi_aspect(query)
        else:
            sub_queries = []
        
        return ReasoningChain(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries[:self.max_hops]
        )
    
    def _decompose_comparison(self, query: str) -> List[SubQuery]:
        """
        Decompose comparison query
        
        Example: "Compare Thompson Sampling vs UCB"
        → Sub-queries:
          1. "What is Thompson Sampling?"
          2. "What is UCB?"
          3. "What are advantages of Thompson Sampling?"
          4. "What are advantages of UCB?"
        """
        sub_queries = []
        
        # Extract entities being compared
        # Pattern: "Compare X and Y" or "X vs Y"
        entities = []
        
        vs_match = re.search(r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
        if vs_match:
            entities = [vs_match.group(1).strip(), vs_match.group(2).strip()]
        else:
            compare_match = re.search(r'compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
            if compare_match:
                entities = [compare_match.group(1).strip(), compare_match.group(2).strip()]
        
        if len(entities) >= 2:
            # Sub-query 1: What is X?
            sub_queries.append(SubQuery(
                id="cmp_1",
                text=f"What is {entities[0]}?",
                query_type=QueryType.SIMPLE,
                reasoning="Understand first entity"
            ))
            
            # Sub-query 2: What is Y?
            sub_queries.append(SubQuery(
                id="cmp_2",
                text=f"What is {entities[1]}?",
                query_type=QueryType.SIMPLE,
                reasoning="Understand second entity"
            ))
            
            # Sub-query 3: Advantages of X
            sub_queries.append(SubQuery(
                id="cmp_3",
                text=f"What are the advantages of {entities[0]}?",
                query_type=QueryType.SIMPLE,
                depends_on=["cmp_1"],
                reasoning="Identify strengths of first entity"
            ))
            
            # Sub-query 4: Advantages of Y
            sub_queries.append(SubQuery(
                id="cmp_4",
                text=f"What are the advantages of {entities[1]}?",
                query_type=QueryType.SIMPLE,
                depends_on=["cmp_2"],
                reasoning="Identify strengths of second entity"
            ))
        
        return sub_queries
    
    def _decompose_causal(self, query: str) -> List[SubQuery]:
        """
        Decompose causal query
        
        Example: "Why does Thompson Sampling work well?"
        → Sub-queries:
          1. "What is Thompson Sampling?"
          2. "What are the key mechanisms of Thompson Sampling?"
          3. "What problem does Thompson Sampling solve?"
        """
        sub_queries = []
        
        # Extract main subject
        subject = re.sub(r'\b(why|how come|what makes)\b', '', query, flags=re.IGNORECASE).strip('? ')
        
        # Sub-query 1: What is it?
        sub_queries.append(SubQuery(
            id="caus_1",
            text=f"What is {subject}?",
            query_type=QueryType.SIMPLE,
            reasoning="Understand the subject"
        ))
        
        # Sub-query 2: How does it work?
        sub_queries.append(SubQuery(
            id="caus_2",
            text=f"How does {subject} work?",
            query_type=QueryType.SIMPLE,
            depends_on=["caus_1"],
            reasoning="Understand mechanisms"
        ))
        
        # Sub-query 3: What problem does it solve?
        sub_queries.append(SubQuery(
            id="caus_3",
            text=f"What problem does {subject} solve?",
            query_type=QueryType.SIMPLE,
            depends_on=["caus_1"],
            reasoning="Understand purpose"
        ))
        
        return sub_queries
    
    def _decompose_compositional(self, query: str) -> List[SubQuery]:
        """
        Decompose compositional query
        
        Example: "How do bandits relate to reinforcement learning and optimization?"
        → Sub-queries for each component + relationship query
        """
        sub_queries = []
        
        # Extract components (words connected by "and")
        components = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
        
        for i, component in enumerate(components[:3]):  # Max 3 components
            component = component.strip('? ')
            sub_queries.append(SubQuery(
                id=f"comp_{i+1}",
                text=f"What is {component}?",
                query_type=QueryType.SIMPLE,
                reasoning=f"Understand component {i+1}"
            ))
        
        # Final query about relationships
        if len(components) >= 2:
            sub_queries.append(SubQuery(
                id="comp_rel",
                text=f"How do {' and '.join(components[:3])} relate?",
                query_type=QueryType.SIMPLE,
                depends_on=[f"comp_{i+1}" for i in range(min(3, len(components)))],
                reasoning="Understand relationships between components"
            ))
        
        return sub_queries
    
    def _decompose_conditional(self, query: str) -> List[SubQuery]:
        """Decompose conditional query"""
        sub_queries = []
        
        # Split on "if" and "then"
        parts = re.split(r'\b(if|then|when|otherwise)\b', query, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            part = part.strip('? ')
            if len(part) > 5 and part.lower() not in ['if', 'then', 'when', 'otherwise']:
                sub_queries.append(SubQuery(
                    id=f"cond_{i+1}",
                    text=part + "?",
                    query_type=QueryType.SIMPLE,
                    reasoning=f"Evaluate condition/consequence {i+1}"
                ))
        
        return sub_queries
    
    def _decompose_sequential(self, query: str) -> List[SubQuery]:
        """Decompose sequential/process query"""
        sub_queries = []
        
        # Look for steps
        step_markers = re.finditer(r'\b(step|stage|phase)\s+(\d+)', query, re.IGNORECASE)
        
        for match in step_markers:
            step_num = match.group(2)
            # Extract text around this step
            sub_queries.append(SubQuery(
                id=f"seq_{step_num}",
                text=f"What is step {step_num}?",
                query_type=QueryType.SIMPLE,
                reasoning=f"Understand step {step_num}"
            ))
        
        if not sub_queries:
            # Generic decomposition
            sub_queries.append(SubQuery(
                id="seq_1",
                text=f"What are the main steps of {query.replace('?', '')}?",
                query_type=QueryType.SIMPLE,
                reasoning="Identify overall process"
            ))
        
        return sub_queries
    
    def _decompose_multi_aspect(self, query: str) -> List[SubQuery]:
        """Decompose multi-aspect query (multiple questions)"""
        sub_queries = []
        
        # Split on question marks
        questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
        
        for i, question in enumerate(questions[:self.max_hops]):
            sub_queries.append(SubQuery(
                id=f"asp_{i+1}",
                text=question,
                query_type=QueryType.SIMPLE,
                reasoning=f"Answer aspect {i+1}"
            ))
        
        return sub_queries
    
    def execute(
        self,
        chain: ReasoningChain,
        retrieval_fn: Optional[Callable] = None
    ) -> ReasoningChain:
        """
        Execute reasoning chain by running sub-queries in dependency order
        
        Args:
            chain: ReasoningChain to execute
            retrieval_fn: Optional retrieval function override
        
        Returns:
            Completed ReasoningChain with answers
        """
        start_time = datetime.now()
        retrieval_fn = retrieval_fn or self.retrieval_fn
        
        if not retrieval_fn:
            raise ValueError("No retrieval function provided")
        
        # If simple query, execute directly
        if chain.query_type == QueryType.SIMPLE or not chain.sub_queries:
            chunks = retrieval_fn(chain.original_query)
            chain.final_answer = self._synthesize_simple(chunks)
            chain.total_retrievals = 1
            chain.execution_time = (datetime.now() - start_time).total_seconds()
            return chain
        
        # Execute sub-queries in dependency order
        executed_count = 0
        max_iterations = len(chain.sub_queries) * 2  # Prevent infinite loops
        
        for _ in range(max_iterations):
            if executed_count >= len(chain.sub_queries):
                break
            
            # Find next executable sub-query (dependencies satisfied)
            for sub_q in chain.sub_queries:
                if sub_q.executed:
                    continue
                
                # Check dependencies
                deps_satisfied = all(
                    any(sq.id == dep_id and sq.executed for sq in chain.sub_queries)
                    for dep_id in sub_q.depends_on
                )
                
                if not deps_satisfied and len(sub_q.depends_on) > 0:
                    continue
                
                # Execute this sub-query
                # Build context from previous answers
                context = self._build_context(chain, sub_q)
                query_with_context = sub_q.text
                if context:
                    query_with_context = f"{context}\n\nQuestion: {sub_q.text}"
                
                chunks = retrieval_fn(query_with_context)
                sub_q.chunks = chunks
                sub_q.answer = self._synthesize_simple(chunks)
                sub_q.executed = True
                sub_q.confidence = self._estimate_confidence(chunks)
                
                executed_count += 1
                chain.total_retrievals += 1
        
        # Synthesize final answer from all sub-answers
        chain.final_answer = self._synthesize_multi_hop(chain)
        chain.synthesis_confidence = self._estimate_chain_confidence(chain)
        chain.execution_time = (datetime.now() - start_time).total_seconds()
        
        return chain
    
    def _build_context(self, chain: ReasoningChain, current_sub_q: SubQuery) -> str:
        """Build context from previously executed sub-queries"""
        context_parts = []
        
        # Include answers from dependencies
        for dep_id in current_sub_q.depends_on:
            for sq in chain.sub_queries:
                if sq.id == dep_id and sq.executed and sq.answer:
                    context_parts.append(f"Previous finding: {sq.answer}")
        
        return "\n".join(context_parts)
    
    def _synthesize_simple(self, chunks: List[Dict[str, Any]]) -> str:
        """Synthesize answer from chunks for simple query"""
        if not chunks:
            return "No relevant information found."
        
        # Take top 3 chunks
        top_chunks = chunks[:3]
        texts = [c.get('text', '') for c in top_chunks]
        
        # Simple concatenation (in real system, use LLM for synthesis)
        return " ".join(texts[:2])  # First 2 chunks
    
    def _synthesize_multi_hop(self, chain: ReasoningChain) -> str:
        """Synthesize final answer from all sub-answers"""
        if not chain.sub_queries:
            return "Unable to decompose query."
        
        # Collect all sub-answers
        answers = [
            f"{sq.text} {sq.answer}"
            for sq in chain.sub_queries
            if sq.executed and sq.answer
        ]
        
        if not answers:
            return "Unable to answer sub-queries."
        
        # Combine answers (in real system, use LLM for coherent synthesis)
        return "\n\n".join(answers)
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Estimate confidence in answer from chunks"""
        if not chunks:
            return 0.0
        
        # Simple heuristic: average of top 3 chunk scores
        scores = [c.get('score', 0.0) for c in chunks[:3]]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _estimate_chain_confidence(self, chain: ReasoningChain) -> float:
        """Estimate overall chain confidence"""
        if not chain.sub_queries:
            return 0.0
        
        executed = [sq for sq in chain.sub_queries if sq.executed]
        if not executed:
            return 0.0
        
        # Average of sub-query confidences
        return sum(sq.confidence for sq in executed) / len(executed)


# ============================================================================
# Integration with ARIA
# ============================================================================

def integrate_with_aria():
    """
    Example: Integrate multi-hop reasoner into aria_main.py
    """
    
    # Mock retrieval function (would be aria_retrieval.py in real system)
    def mock_retrieval(query: str) -> List[Dict[str, Any]]:
        return [
            {'text': f'Result for: {query}', 'score': 0.8},
            {'text': f'More info on: {query}', 'score': 0.7}
        ]
    
    # Initialize reasoner
    reasoner = MultiHopReasoner(
        retrieval_fn=mock_retrieval,
        max_hops=5
    )
    
    # Example: Complex comparison query
    query = "Compare Thompson Sampling versus UCB in multi-armed bandits"
    
    # Check if multi-hop needed
    is_multi_hop = reasoner.detect_multi_hop(query)
    print(f"Query: {query}")
    print(f"Multi-hop needed: {is_multi_hop}")
    
    # Decompose
    chain = reasoner.decompose(query)
    print(f"\nQuery type: {chain.query_type.value}")
    print(f"Sub-queries generated: {len(chain.sub_queries)}")
    
    for sq in chain.sub_queries:
        print(f"  [{sq.id}] {sq.text}")
        if sq.depends_on:
            print(f"       Depends on: {sq.depends_on}")
        print(f"       Reasoning: {sq.reasoning}")
    
    # Execute chain
    chain = reasoner.execute(chain)
    
    print(f"\nExecution complete:")
    print(f"  Total retrievals: {chain.total_retrievals}")
    print(f"  Execution time: {chain.execution_time:.2f}s")
    print(f"  Confidence: {chain.synthesis_confidence:.2f}")
    print(f"\nFinal answer:\n{chain.final_answer}")


if __name__ == '__main__':
    integrate_with_aria()
