#!/usr/bin/env python3
"""
Anchor Mode Selector - Detects query type and selects appropriate reasoning anchor
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Uses exemplar patterns and query features to classify queries into modes:
- formal: Academic/research queries
- casual: Conversational questions
- technical: Code/implementation queries
- educational: Learning/teaching queries
- philosophical: Deep inquiry/existence questions
- analytical: Problem-solving/analysis
- factual: Direct fact lookup
- creative: Brainstorming/exploration
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class AnchorSelector:
    """Select appropriate reasoning anchor based on query characteristics"""
    
    def __init__(self, exemplar_path: Optional[Path] = None):
        self.mode_patterns = self._build_patterns()
        self.exemplar_path = exemplar_path
        
    def _build_patterns(self) -> Dict[str, Dict[str, List]]:
        """Define patterns for each mode based on exemplar analysis"""
        return {
            'formal': {
                'keywords': [
                    'research', 'study', 'paper', 'theory', 'hypothesis', 'evidence',
                    'methodology', 'analysis', 'empirical', 'peer-reviewed', 'citation',
                    'systematic', 'meta-analysis', 'framework', 'paradigm', 'literature'
                ],
                'patterns': [
                    r'\bet al\.',
                    r'\b\d{4}\b.*\b(study|research|paper)\b',
                    r'\b(according to|based on).*\b(research|studies|literature)\b',
                    r'\b(theoretical|empirical|methodological)\b',
                ],
                'indicators': ['academic', 'scholarly', 'rigorous', 'formal']
            },
            
            'casual': {
                'keywords': [
                    'basically', 'pretty much', 'kind of', 'sort of', 'tell me',
                    'curious', 'wondering', 'heard', 'friend', 'chat', 'discuss'
                ],
                'patterns': [
                    r"^(hey|hi|hello|yo)",
                    r"(can you|could you|would you) (just|simply|quickly)",
                    r"(tell me|explain).*(in simple terms|simply|casually)",
                    r"(what's|how's|why's|when's)",
                ],
                'indicators': ['conversational', 'informal', 'friendly']
            },
            
            'technical': {
                'keywords': [
                    'code', 'function', 'class', 'method', 'api', 'debug', 'error',
                    'implement', 'algorithm', 'syntax', 'compile', 'deploy', 'stack',
                    'framework', 'library', 'package', 'module', 'variable', 'type',
                    'async', 'await', 'promise', 'callback', 'memory leak', 'binary search'
                ],
                'patterns': [
                    r'```',  # Code blocks
                    r'\b\w+\([^)]*\)',  # Function calls
                    r'\b[A-Z][a-zA-Z]*Error\b',  # Error types
                    r'\b(def|class|import|from|return)\b',  # Python keywords
                    r'\.(py|js|ts|java|cpp|go|rs)\b',  # File extensions
                    r'\b(how (to|do I)) (implement|code|write|fix|debug|build)\b',
                    r'\b(implement|build|create|write) (a|an) \w+ (in|using|with)\b',
                    r'\b(async|await|promise|callback|memory leak|stack trace)\b',
                ],
                'indicators': ['programming', 'development', 'implementation', 'coding']
            },
            
            'educational': {
                'keywords': [
                    'explain', 'understand', 'learn', 'teach', 'eli5', 'simple',
                    'beginner', 'basics', 'introduction', 'help me understand',
                    'confused', 'dont get', 'make sense', 'clarify', 'breakdown'
                ],
                'patterns': [
                    r'\b(eli5|explain like|simple terms|layman)\b',
                    r'\b(I.?m|I am) (new to|learning|trying to understand|confused)',
                    r'\b(help me|teach me|show me how)\b',
                    r'\b(what (is|are)|how (does|do))',
                    r'\b(break (it|this) down|step by step|from scratch)\b',
                ],
                'indicators': ['learning', 'teaching', 'educational', 'tutorial']
            },
            
            'philosophical': {
                'keywords': [
                    'consciousness', 'existence', 'reality', 'meaning', 'purpose',
                    'metaphysics', 'epistemology', 'ontology', 'phenomenology',
                    'existential', 'being', 'truth', 'knowledge', 'mind', 'free will',
                    'subjective', 'objective', 'essence', 'nature of', 'fundamental',
                    'meaning of life', 'meaning of existence'
                ],
                'patterns': [
                    r'\b(what is|nature of) (the )?(consciousness|existence|reality|truth|being)\b',
                    r'\b(why do (we|I)|what is the meaning of|what is the purpose)\b',
                    r'\b(philosophical|metaphysical|existential|ontological)\b',
                    r'\b(can we (really|truly) know|is it possible to)\b',
                    r'\bmeaning of (life|existence)\b',
                    r'\b(do we have|is there) free will\b',
                ],
                'indicators': ['philosophy', 'deep', 'existential', 'metaphysical']
            },
            
            'analytical': {
                'keywords': [
                    'analyze', 'evaluate', 'compare', 'assess', 'determine',
                    'solve', 'approach', 'strategy', 'optimization', 'trade-off',
                    'decision', 'choose', 'best', 'optimal', 'pros and cons',
                    'framework', 'methodology', 'systematic', 'structured'
                ],
                'patterns': [
                    r'\b(how (should|can|do) (I|we) (solve|approach|tackle|handle))',
                    r'\b(what.?s the best (way|approach|method|strategy))',
                    r'\b(compare|contrast|versus|vs\.?|difference between)',
                    r'\b(analyze|evaluate|assess|determine|calculate)',
                    r'\b(pros and cons|advantages and disadvantages|trade-?offs)',
                ],
                'indicators': ['analysis', 'problem-solving', 'systematic', 'strategic']
            },
            
            'factual': {
                'keywords': [
                    'what is', 'who is', 'when did', 'where is', 'how many',
                    'definition', 'date', 'fact', 'number', 'statistic', 'data',
                    'capital', 'population', 'founded', 'invented', 'discovered'
                ],
                'patterns': [
                    r'^\s*(what|who|when|where|which) (is|are|was|were|did)',
                    r'\b(how (many|much|long|far|high|deep|old))',
                    r'\b(define|definition of|meaning of)\b',
                    r'^\s*(give me|tell me|show me) (the|a) (fact|number|date)',
                ],
                'indicators': ['lookup', 'fact', 'direct', 'quick']
            },
            
            'creative': {
                'keywords': [
                    'brainstorm', 'ideas', 'creative', 'imagine', 'what if',
                    'innovative', 'novel', 'original', 'explore', 'possibilities',
                    'invent', 'design', 'create', 'generate', 'envision', 'dream'
                ],
                'patterns': [
                    r'\b(what if|imagine|suppose|let.?s say)',
                    r'\b(brainstorm|generate|come up with|think of) (ideas|ways|approaches)',
                    r'\b(creative|innovative|novel|unique|original|unconventional)',
                    r'\b(how (could|might) we|what are some ways to)',
                    r'\b(explore|envision|dream up|invent)\b',
                ],
                'indicators': ['ideation', 'exploration', 'speculation', 'imagination']
            }
        }
    
    def select_mode(self, query: str) -> str:
        """
        Select the most appropriate reasoning anchor mode for the query
        
        Args:
            query: User's query text
            
        Returns:
            Mode name: 'formal', 'casual', 'technical', 'educational', 
                      'philosophical', 'analytical', 'factual', or 'creative'
        """
        query_lower = query.lower()
        
        # Score each mode
        scores = {}
        for mode, patterns in self.mode_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 2
            
            # Regex pattern matching (higher weight)
            for pattern in patterns['patterns']:
                if re.search(pattern, query_lower):
                    score += 4  # Increased from 3
            
            # Indicator words
            for indicator in patterns['indicators']:
                if indicator in query_lower:
                    score += 1
            
            scores[mode] = score
        
        # Apply mode-specific boosting
        # DOMINANCE RULE: Domain-specific terms override conversational tone
        
        # Check for domain-specific terms (technical/research/philosophical)
        domain_terms_present = False
        
        # Technical domain terms (including proper nouns for technical systems)
        technical_indicators = ['implement', 'code', 'debug', 'async', 'javascript', 'python', 
                                'react', 'api', 'function', 'class', 'error', 'aria', 'algorithm',
                                'architecture', 'system', 'neural', 'model']
        if any(term in query_lower for term in technical_indicators):
            scores['technical'] = scores.get('technical', 0) + 10  # ADD points instead of multiply
            domain_terms_present = True
        
        # Research/academic terms
        research_terms = ['research', 'study', 'paper', 'theory', 'analysis', 'empirical',
                          'methodology', 'hypothesis', 'evidence', 'literature']
        if any(term in query_lower for term in research_terms):
            scores['formal'] = scores.get('formal', 0) + 10  # ADD points instead of multiply
            domain_terms_present = True
        
        # Philosophical/existential terms  
        existential_terms = ['meaning of', 'nature of', 'consciousness', 'existence', 'reality',
                             'free will', 'purpose of', 'epistemology', 'ontology', 'metaphysics']
        if any(term in query_lower for term in existential_terms):
            scores['philosophical'] = scores.get('philosophical', 0) + 10  # ADD points instead of multiply
            domain_terms_present = True
        
        # Analytical terms
        analytical_terms = ['compare', 'analyze', 'evaluate', 'assess', 'trade-off', 
                           'pros and cons', 'versus', 'vs', 'optimization', 'tie together', 'relate']
        if any(term in query_lower for term in analytical_terms):
            scores['analytical'] = scores.get('analytical', 0) + 10  # ADD points instead of multiply
            domain_terms_present = True
        
        # DOMINANCE: If domain terms present, deprioritize casual mode
        if domain_terms_present and 'casual' in scores:
            scores['casual'] = scores['casual'] * 0.3  # Reduce casual more aggressively
        
        # Get mode with highest score
        if max(scores.values()) == 0:
            # No strong match - use heuristics
            return self._heuristic_selection(query)
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _heuristic_selection(self, query: str) -> str:
        """Fallback heuristics when pattern matching doesn't give clear signal"""
        query_lower = query.lower()
        
        # Check for technical terms first (highest priority)
        technical_terms = ['code', 'implement', 'debug', 'error', 'function', 'api',
                          'javascript', 'python', 'react', 'async']
        if any(term in query_lower for term in technical_terms):
            return 'technical'
        
        # Check for philosophical/existential terms
        philosophical_terms = ['meaning', 'existence', 'consciousness', 'reality', 
                              'purpose', 'free will', 'nature of']
        if any(term in query_lower for term in philosophical_terms):
            return 'philosophical'
        
        # Very short queries are usually factual
        if len(query.split()) <= 5:
            return 'factual'
        
        # Questions starting with "why" are often philosophical or educational
        if query_lower.startswith('why'):
            if any(word in query_lower for word in ['should', 'do we', 'is it']):
                return 'philosophical'
            return 'educational'
        
        # Questions starting with "how" depend on context
        if query_lower.startswith('how'):
            if any(word in query_lower for word in ['code', 'implement', 'build', 'fix']):
                return 'technical'
            if any(word in query_lower for word in ['solve', 'approach', 'optimize']):
                return 'analytical'
            return 'educational'
        
        # Presence of question words usually means educational or factual
        if any(q in query_lower for q in ['what', 'who', 'when', 'where']):
            # Check if it's asking about meaning/nature (philosophical)
            if 'meaning of' in query_lower or 'nature of' in query_lower:
                return 'philosophical'
            # Longer explanatory queries are educational
            if len(query.split()) > 10:
                return 'educational'
            return 'factual'
        
        # Default to casual for ambiguous queries
        return 'casual'
    
    def get_mode_info(self, mode: str) -> Dict[str, str]:
        """Get information about a mode"""
        info = {
            'formal': {
                'name': 'Formal/Academic',
                'description': 'Research-grade analysis with citations and scholarly rigor',
                'best_for': 'Academic questions, research topics, scholarly analysis'
            },
            'casual': {
                'name': 'Casual/Conversational',
                'description': 'Friendly dialogue like talking to a knowledgeable friend',
                'best_for': 'General questions, casual learning, friendly conversation'
            },
            'technical': {
                'name': 'Technical/Developer',
                'description': 'Code-level precision with implementation details',
                'best_for': 'Programming, debugging, technical implementation'
            },
            'educational': {
                'name': 'Educational/ELI5',
                'description': 'Clear teaching with examples and progressive complexity',
                'best_for': 'Learning new concepts, understanding fundamentals'
            },
            'philosophical': {
                'name': 'Philosophical/Deep Inquiry',
                'description': 'Exploration of fundamental questions and multiple perspectives',
                'best_for': 'Existential questions, deep conceptual analysis'
            },
            'analytical': {
                'name': 'Analytical/Problem-Solving',
                'description': 'Systematic reasoning with structured decomposition',
                'best_for': 'Problem-solving, decision-making, strategic analysis'
            },
            'factual': {
                'name': 'Factual/Direct',
                'description': 'Concise direct answers to factual questions',
                'best_for': 'Quick facts, definitions, direct information lookup'
            },
            'creative': {
                'name': 'Creative/Exploratory',
                'description': 'Open-ended ideation and imaginative exploration',
                'best_for': 'Brainstorming, innovative thinking, speculation'
            }
        }
        return info.get(mode, {})


def main():
    """Test the anchor selector"""
    import sys
    
    selector = AnchorSelector()
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Explain quantum entanglement like I'm five",
        "How do I implement a binary search tree in Python?",
        "What is the nature of consciousness?",
        "Compare the pros and cons of microservices vs monolithic architecture",
        "According to recent research, what are the effects of sleep deprivation?",
        "Hey, tell me about machine learning",
        "What if we could travel faster than light?",
    ]
    
    if len(sys.argv) > 1:
        # Test with command line argument
        query = ' '.join(sys.argv[1:])
        mode = selector.select_mode(query)
        info = selector.get_mode_info(mode)
        print(f"Query: {query}")
        print(f"Mode: {info['name']}")
        print(f"Description: {info['description']}")
    else:
        # Test with predefined queries
        print("Testing Anchor Selector:\n")
        for query in test_queries:
            mode = selector.select_mode(query)
            info = selector.get_mode_info(mode)
            print(f"Query: {query}")
            print(f"  â†’ Mode: {info['name']}")
            print()


if __name__ == '__main__':
    main()
