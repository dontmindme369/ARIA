#!/usr/bin/env python3
"""
Score Recent LM Studio Conversations and Update Bandit

Reads your actual LM Studio conversations, scores quality,
and updates ARIA's bandit rewards for closed-loop learning.
"""

import sys
from pathlib import Path

# Add ARIA to path
ARIA_ROOT = Path("/media/notapplicable/Internal-SSD/ai-quaternions-model")
sys.path.insert(0, str(ARIA_ROOT))
sys.path.insert(0, str(ARIA_ROOT / "src"))

from conversation_scorer import ConversationScorer

def main():
    print("=" * 80)
    print("ARIA Conversation Scorer - Production Run".center(80))
    print("=" * 80)
    print()
    
    # Initialize scorer with verbose output
    print("📊 Initializing ConversationScorer...")
    scorer = ConversationScorer(
        verbose=True,
        save_to_corpus=True  # Save good conversations to training corpus
    )
    print()
    
    # Get recent conversations (last 24 hours)
    print("🔍 Scanning LM Studio conversations (last 24 hours)...")
    conversations = scorer.get_recent_conversations(hours=24)
    
    print(f"✅ Found {len(conversations)} recent conversations")
    print()
    
    if not conversations:
        print("⚠️  No recent conversations found in ~/.lmstudio/conversations")
        print("   This is normal if you haven't used LM Studio recently")
        return
    
    # Score and update bandit
    print("🎯 Scoring conversations and updating bandit rewards...")
    print("-" * 80)
    scorer.update_bandit_from_conversations(conversations)
    print("-" * 80)
    print()
    
    # Show statistics
    print("📈 Scoring Statistics:")
    print(f"   Total conversations: {scorer.stats['total_conversations']}")
    print(f"   Total Q&A pairs: {scorer.stats['total_qa_pairs']}")
    print(f"   Successfully scored: {scorer.stats['total_scored']}")
    print(f"   Bandit updates: {scorer.stats['bandit_updates']}")
    print(f"   Corpus saves: {scorer.stats['corpus_saves']}")
    print(f"   Average score: {scorer.stats['avg_score']:.3f}")
    print(f"   Preset matches: {scorer.stats['preset_matches']}")
    print(f"   Preset misses: {scorer.stats['preset_misses']}")
    print()
    
    print("=" * 80)
    print("✅ Closed-Loop Learning Update Complete!".center(80))
    print("=" * 80)
    print()
    print("Your ARIA system learned from these conversations and updated")
    print("its bandit strategy to get better at choosing the right preset.")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
