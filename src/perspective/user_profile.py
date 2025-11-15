#!/usr/bin/env python3
"""
Build user profile from conversation history for personalized ARIA responses
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List
import statistics

# Find project root (aria/) relative to this file (src/perspective/)
# Go up 2 levels: src/perspective/ -> src/ -> aria/
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONVERSATIONS_DIR = Path.home() / '.lmstudio' / 'conversations'
OUTPUT_FILE = PROJECT_ROOT / 'var' / 'user_profile.json'

# Technical depth indicators
TECHNICAL_INDICATORS = {
    'expert': [
        r'\b(algorithm|implementation|architecture|optimization|complexity)\b',
        r'\b(async|concurrency|parallelism|threading)\b',
        r'\b(protocol|RFC|specification|standard)\b',
        r'\bO\([nlogn]+\)',
        r'\b(tensor|gradient|backprop|inference)\b'
    ],
    'advanced': [
        r'\b(function|class|method|variable|parameter)\b',
        r'\b(design pattern|best practice|refactor)\b',
        r'\b(api|endpoint|request|response)\b',
        r'\b(query|database|index|schema)\b'
    ],
    'intermediate': [
        r'\b(code|program|script|syntax)\b',
        r'\b(error|bug|debug|fix)\b',
        r'\b(install|setup|configure)\b',
        r'\b(file|folder|path|directory)\b'
    ],
    'basic': [
        r'\b(how to|what is|explain|help me)\b',
        r'\b(simple|easy|basic|beginner)\b',
        r'\b(tutorial|guide|example|show me)\b'
    ]
}

# Domain indicators
DOMAIN_KEYWORDS = {
    'ai_ml': ['machine learning', 'neural network', 'model', 'training', 'dataset', 'ai', 'llm'],
    'physics': ['quantum', 'energy', 'electromagnetic', 'resonance', 'frequency', 'particle'],
    'philosophy': ['consciousness', 'qualia', 'awareness', 'metaphysics', 'epistemology'],
    'engineering': ['circuit', 'design', 'system', 'hardware', 'signal', 'component'],
    'programming': ['python', 'javascript', 'code', 'function', 'algorithm', 'debugging'],
    'mathematics': ['equation', 'theorem', 'proof', 'calculate', 'formula', 'derivative']
}

def load_conversations() -> List[Dict]:
    """Load all conversation JSON files"""
    conversations = []
    
    if not CONVERSATIONS_DIR.exists():
        print(f"Warning: Conversations directory not found: {CONVERSATIONS_DIR}")
        return []
    
    for conv_file in CONVERSATIONS_DIR.glob('*.json'):
        try:
            with open(conv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract messages
                messages = data.get('messages', [])
                if isinstance(messages, list):
                    conversations.append({
                        'file': conv_file.name,
                        'messages': messages
                    })
        except Exception as e:
            print(f"Error loading {conv_file}: {e}")
            continue
    
    return conversations

def extract_user_queries(conversations: List[Dict]) -> List[str]:
    """Extract all user queries from conversations"""
    queries = []
    
    for conv in conversations:
        for msg in conv['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                if content and len(content) > 10:  # Skip very short messages
                    queries.append(content)
    
    return queries

def analyze_technical_depth(queries: List[str]) -> str:
    """Determine user's technical depth from queries"""
    depth_scores = Counter()
    
    for query in queries:
        q_lower = query.lower()
        
        for depth, patterns in TECHNICAL_INDICATORS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, q_lower))
            depth_scores[depth] += score
    
    if not depth_scores:
        return 'intermediate'
    
    # Weight higher depths more
    weights = {'basic': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
    weighted_total = sum(depth_scores[d] * weights[d] for d in depth_scores)
    total_matches = sum(depth_scores.values())
    
    if total_matches == 0:
        return 'intermediate'
    
    avg_weight = weighted_total / total_matches
    
    if avg_weight >= 3.5:
        return 'expert'
    elif avg_weight >= 2.5:
        return 'advanced'
    elif avg_weight >= 1.5:
        return 'intermediate'
    else:
        return 'basic'

def analyze_domains(queries: List[str]) -> List[str]:
    """Identify user's domains of expertise/interest"""
    domain_scores = Counter()
    
    for query in queries:
        q_lower = query.lower()
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in q_lower)
            domain_scores[domain] += score
    
    # Return top domains with significant presence
    threshold = max(3, len(queries) * 0.1)  # At least 3 or 10% of queries
    significant_domains = [
        domain for domain, score in domain_scores.most_common()
        if score >= threshold
    ]
    
    return significant_domains[:5]  # Top 5 domains

def analyze_communication_patterns(queries: List[str]) -> Dict:
    """Analyze how user communicates"""
    query_lengths = [len(q.split()) for q in queries]
    
    has_code = sum(1 for q in queries if '```' in q or 'code' in q.lower())
    has_followups = sum(1 for q in queries if any(word in q.lower() for word in ['also', 'furthermore', 'additionally', 'what about']))
    
    return {
        'avg_query_length': round(statistics.mean(query_lengths)) if query_lengths else 15,
        'prefers_code_examples': (has_code / len(queries)) > 0.3 if queries else False,
        'asks_followup_questions': (has_followups / len(queries)) > 0.2 if queries else False
    }

def determine_explanation_style(queries: List[str]) -> str:
    """Determine preferred explanation style"""
    style_indicators = {
        'concise': ['brief', 'quick', 'short', 'summary', 'tldr'],
        'detailed': ['detailed', 'thorough', 'comprehensive', 'explain', 'deep dive'],
        'examples': ['example', 'show me', 'demonstrate', 'sample'],
        'analogies': ['like', 'analogy', 'similar to', 'compared to', 'metaphor']
    }
    
    style_scores = Counter()
    
    for query in queries:
        q_lower = query.lower()
        for style, indicators in style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in q_lower)
            style_scores[style] += score
    
    if not style_scores:
        return 'detailed'
    
    return style_scores.most_common(1)[0][0]

def build_profile() -> Dict:
    """Build complete user profile"""
    print("Loading conversations...")
    conversations = load_conversations()
    
    if not conversations:
        print("No conversations found. Using default profile.")
        return {
            'technical_depth': 'intermediate',
            'preferred_explanation_style': 'detailed',
            'domains_of_expertise': [],
            'communication_patterns': {
                'avg_query_length': 15,
                'prefers_code_examples': False,
                'asks_followup_questions': False
            }
        }
    
    print(f"Analyzing {len(conversations)} conversations...")
    queries = extract_user_queries(conversations)
    print(f"Found {len(queries)} user queries")
    
    print("Determining technical depth...")
    technical_depth = analyze_technical_depth(queries)
    
    print("Identifying domains of expertise...")
    domains = analyze_domains(queries)
    
    print("Analyzing communication patterns...")
    comm_patterns = analyze_communication_patterns(queries)
    
    print("Determining explanation style...")
    explanation_style = determine_explanation_style(queries)
    
    profile = {
        'technical_depth': technical_depth,
        'preferred_explanation_style': explanation_style,
        'domains_of_expertise': domains,
        'communication_patterns': comm_patterns,
        'generated_at': Path(CONVERSATIONS_DIR).stat().st_mtime if CONVERSATIONS_DIR.exists() else 0,
        'based_on_queries': len(queries)
    }
    
    return profile

def main():
    print("=" * 60)
    print("ARIA User Profile Builder")
    print("=" * 60)
    print()
    
    profile = build_profile()
    
    print()
    print("Profile Summary:")
    print(f"  Technical Depth: {profile['technical_depth']}")
    print(f"  Explanation Style: {profile['preferred_explanation_style']}")
    print(f"  Domains: {', '.join(profile['domains_of_expertise']) or 'None detected'}")
    print(f"  Avg Query Length: {profile['communication_patterns']['avg_query_length']} words")
    print(f"  Prefers Code: {profile['communication_patterns']['prefers_code_examples']}")
    print(f"  Asks Followups: {profile['communication_patterns']['asks_followup_questions']}")
    print()
    
    # Save profile
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)
    
    print(f"✓ Profile saved to: {OUTPUT_FILE}")
    print()
    print("This profile will be used by ARIA to personalize responses.")

if __name__ == '__main__':
    main()
