#!/usr/bin/env python3
"""
Test Script for Hybrid Multi-Anchor System
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Demonstrates mode detection and anchor selection for various query types.
"""

from pathlib import Path
import sys

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from anchor_selector import AnchorSelector


def test_mode_detection():
    """Test the anchor selector with various query types"""
    
    selector = AnchorSelector()
    
    # Test cases: (query, expected_mode)
    test_cases = [
        # Factual queries
        ("What is the capital of France?", "factual"),
        ("When was the iPhone invented?", "factual"),
        ("How many planets are in the solar system?", "factual"),
        
        # Educational queries
        ("Explain photosynthesis like I'm five", "educational"),
        ("Help me understand neural networks", "educational"),
        ("I'm confused about recursion", "educational"),
        
        # Technical queries
        ("How do I implement a binary search tree in Python?", "technical"),
        ("Debug this error: IndexError list index out of range", "technical"),
        ("What's the best way to handle async in JavaScript?", "technical"),
        
        # Philosophical queries
        ("What is the nature of consciousness?", "philosophical"),
        ("Do we have free will?", "philosophical"),
        ("What is the meaning of existence?", "philosophical"),
        
        # Analytical queries
        ("Compare microservices vs monolithic architecture", "analytical"),
        ("How should I approach this optimization problem?", "analytical"),
        ("Analyze the trade-offs between SQL and NoSQL", "analytical"),
        
        # Formal queries
        ("According to recent research, what are the effects of sleep deprivation?", "formal"),
        ("What does the literature say about climate change?", "formal"),
        ("Summarize the empirical evidence for meditation benefits", "formal"),
        
        # Casual queries
        ("Hey, tell me about machine learning", "casual"),
        ("What's the deal with quantum computers?", "casual"),
        ("I'm curious about black holes", "casual"),
        
        # Creative queries
        ("Brainstorm ideas for a science fiction story", "creative"),
        ("What if we could upload consciousness to computers?", "creative"),
        ("Generate creative solutions for traffic congestion", "creative"),
    ]
    
    print("=" * 80)
    print("HYBRID MULTI-ANCHOR SYSTEM - MODE DETECTION TEST")
    print("=" * 80)
    print()
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        detected = selector.select_mode(query)
        is_correct = detected == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"   Expected: {expected:15} | Detected: {detected:15}")
        
        if not is_correct:
            print(f"   ⚠️  MISMATCH!")
        print()
    
    accuracy = (correct / total) * 100
    
    print("=" * 80)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("=" * 80)
    print()
    
    if accuracy >= 90:
        print("🎉 Excellent! Mode detection is working very well.")
    elif accuracy >= 75:
        print("✅ Good! Mode detection is working reliably.")
    elif accuracy >= 60:
        print("⚠️  Acceptable, but could use some tuning.")
    else:
        print("❌ Needs improvement. Check pattern definitions.")
    
    return accuracy


def demonstrate_anchor_selection():
    """Show how different query types get different anchors"""
    
    selector = AnchorSelector()
    
    print("\n" + "=" * 80)
    print("ANCHOR SELECTION DEMONSTRATION")
    print("=" * 80)
    print()
    
    demo_queries = {
        "Factual": "What is the speed of light?",
        "Educational": "Explain how blockchain works in simple terms",
        "Technical": "How do I fix a memory leak in my React app?",
        "Philosophical": "What is the nature of reality?",
        "Analytical": "Compare the pros and cons of remote work",
        "Formal": "What does the research say about coffee consumption?",
        "Casual": "Tell me about the James Webb telescope",
        "Creative": "Brainstorm ideas for a sustainable city"
    }
    
    for category, query in demo_queries.items():
        mode = selector.select_mode(query)
        info = selector.get_mode_info(mode)
        
        print(f"Category: {category}")
        print(f"Query: \"{query}\"")
        print(f"→ Selected Mode: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print()
    
    print("=" * 80)


def test_custom_query():
    """Allow testing with custom query from command line"""
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        selector = AnchorSelector()
        mode = selector.select_mode(query)
        info = selector.get_mode_info(mode)
        
        print("\n" + "=" * 80)
        print("CUSTOM QUERY TEST")
        print("=" * 80)
        print()
        print(f"Query: \"{query}\"")
        print()
        print(f"Selected Mode: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Best for: {info['best_for']}")
        print()
        print("=" * 80)
        
        return True
    
    return False


def main():
    """Run all tests"""
    
    # Test with custom query if provided
    if test_custom_query():
        return
    
    # Run full test suite
    accuracy = test_mode_detection()
    
    # Show demonstration
    demonstrate_anchor_selection()
    
    # Final summary
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print("To test with your own query, run:")
    print("  python test_anchor_system.py \"your query here\"")
    print()


if __name__ == '__main__':
    main()
