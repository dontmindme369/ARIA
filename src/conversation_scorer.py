#!/usr/bin/env python3
"""
Conversation Quality Scorer for ARIA Learning Loop

Reads LM Studio conversation JSONs, scores quality, updates ARIA bandit rewards.
This creates the closed-loop learning system.

Configuration:
  Set these paths via environment variables:
  - ARIA_ROOT: Root directory of ARIA installation (default: current directory)
  - ARIA_DATA_DIR: Data directory (default: ./data)
  - ARIA_OUTPUT_DIR: Output directory (default: ./output)
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Protocol
from datetime import datetime
import re

# ARIA paths - configurable via environment variables
ARIA_ROOT = Path(os.getenv('ARIA_ROOT', Path.cwd()))
DATA_DIR = Path(os.getenv('ARIA_DATA_DIR', ARIA_ROOT / 'data'))
OUTPUT_DIR = Path(os.getenv('ARIA_OUTPUT_DIR', ARIA_ROOT / 'output'))
TRAINING_CORPUS = DATA_DIR / "training_corpus"
ARIA_OUTPUT_DIR = OUTPUT_DIR / "rag_runs" / "aria"

# Add ARIA to path for imports
if str(ARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT))
if str(ARIA_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT / "src"))

# Protocol defines the function signature with optional parameter
class GiveRewardFunc(Protocol):
    """Type for give_reward function"""
    def __call__(self, preset_name: str, reward: float, state_path: str = "") -> None: ...

# Fallback implementation (always defined)
def give_reward(preset_name: str, reward: float, state_path: str = "") -> None:
    """Stub function when bandit_context is not available"""
    pass

# Try to import real implementation
BANDIT_AVAILABLE = False
try:
    from bandit_context import give_reward  # type: ignore[no-redef]
    BANDIT_AVAILABLE = True
except ImportError:
    print("[WARNING] bandit_context not available - scores won't update bandit", file=sys.stderr)


class ConversationScorer:
    """Score conversation quality and update ARIA's learning system"""
    
    LMSTUDIO_CONVOS = Path.home() / ".lmstudio/conversations"
    
    def __init__(self, 
                 bandit_state_path: str = "~/.rag_bandit_state.json",
                 save_to_corpus: bool = True,
                 verbose: bool = False):
        self.bandit_state_path = Path(bandit_state_path).expanduser()
        self.save_to_corpus = save_to_corpus
        self.verbose = verbose
        
        # Ensure training corpus directory exists
        if self.save_to_corpus:
            TRAINING_CORPUS.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.stats = {
            "total_conversations": 0,
            "total_qa_pairs": 0,
            "total_scored": 0,
            "bandit_updates": 0,
            "corpus_saves": 0,
            "avg_score": 0.0,
            "preset_matches": 0,
            "preset_misses": 0
        }
        
        if self.verbose:
            print(f"[INFO] Initializing ConversationScorer", file=sys.stderr)
            print(f"[INFO] LM Studio conversations: {self.LMSTUDIO_CONVOS}", file=sys.stderr)
            print(f"[INFO] ARIA output dir: {ARIA_OUTPUT_DIR}", file=sys.stderr)
            print(f"[INFO] Training corpus: {TRAINING_CORPUS}", file=sys.stderr)
            print(f"[INFO] Bandit state: {self.bandit_state_path}", file=sys.stderr)
        
    def score_response(self, user_query: str, assistant_response: str, 
                      context_used: bool = False) -> float:
        """
        Score response quality (0.0 to 1.0)
        
        High scores for:
        - Natural conversation flow
        - Synthesis of context + knowledge
        - Answering the actual question
        - Appropriate length
        
        Low scores for:
        - Just regurgitating context
        - Not answering the question
        - Generic/unhelpful responses
        """
        score = 0.5  # baseline
        
        response_lower = assistant_response.lower()
        query_lower = user_query.lower()
        
        # POSITIVE SIGNALS
        
        # 1. Actually answers question (not just "here's what I found")
        if any(phrase in response_lower for phrase in [
            "based on", "according to", "this means", "in other words",
            "essentially", "to answer your question", "yes", "no",
            "because", "the reason", "this is"
        ]):
            score += 0.15
        
        # 2. Synthesizes (uses connector words)
        synthesis_markers = [
            "building on", "additionally", "furthermore", "however",
            "in contrast", "similarly", "this relates to", "connecting"
        ]
        if any(marker in response_lower for marker in synthesis_markers):
            score += 0.15
        
        # 3. Conversational (not just data dump)
        if len(assistant_response.split(".")) > 2:  # Multiple sentences
            score += 0.1
        
        # 4. Appropriate length (not too short or too long)
        response_len = len(assistant_response.split())
        if 50 < response_len < 500:
            score += 0.1
        elif response_len > 1000:  # Penalize extremely long responses
            score -= 0.05
        
        # 5. References specific concepts from query
        query_keywords = set(re.findall(r'\b\w{4,}\b', query_lower))
        response_keywords = set(re.findall(r'\b\w{4,}\b', response_lower))
        keyword_overlap = len(query_keywords & response_keywords) / max(len(query_keywords), 1)
        score += keyword_overlap * 0.1
        
        # NEGATIVE SIGNALS
        
        # 1. Just presenting options/lists without answering
        if any(phrase in response_lower for phrase in [
            "are you looking for:", "would you like to know about:",
            "i can help with:", "please specify", "what specifically"
        ]):
            score -= 0.2
        
        # 2. Too generic/unhelpful
        generic_phrases = [
            "i don't have enough information",
            "could you provide more details",
            "i'm not sure what you're asking"
        ]
        if any(phrase in response_lower for phrase in generic_phrases):
            score -= 0.15
        
        # 3. Just listing context items without synthesis
        if response_lower.count("##") > 3 or response_lower.count("- ") > 10:
            score -= 0.15  # Probably just dumping pack contents
        
        # 4. No connection to query
        if keyword_overlap < 0.1:
            score -= 0.2
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def get_recent_conversations(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent conversations from LM Studio"""
        if not self.LMSTUDIO_CONVOS.exists():
            print(f"[ERROR] LM Studio conversations directory not found: {self.LMSTUDIO_CONVOS}", file=sys.stderr)
            return []
        
        cutoff = datetime.now().timestamp() - (hours * 3600)
        recent = []
        
        for conv_file in self.LMSTUDIO_CONVOS.glob("*.json"):
            if conv_file.stat().st_mtime < cutoff:
                continue
            
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    convo = json.load(f)
                    recent.append({
                        "file": conv_file.name,
                        "modified": conv_file.stat().st_mtime,
                        "data": convo
                    })
            except Exception as e:
                print(f"[WARNING] Error reading {conv_file.name}: {e}", file=sys.stderr)
        
        # Sort by modification time (most recent first)
        recent.sort(key=lambda x: x["modified"], reverse=True)
        
        return recent
    
    def extract_qa_pairs(self, conversation: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Extract (user_query, assistant_response) pairs from LM Studio conversation.
        
        LM Studio structure: message → versions[] → content[] → {type: 'text', text: '...'}
        """
        messages = conversation.get("messages", [])
        
        if self.verbose:
            print(f"[DEBUG] Conversation has {len(messages)} messages", file=sys.stderr)
            if messages and len(messages) > 0:
                print(f"[DEBUG] Message structure: {list(messages[0].keys())}", file=sys.stderr)
        
        pairs = []
        
        for i in range(len(messages) - 1):
            msg = messages[i]
            next_msg = messages[i + 1]
            
            # Extract current message
            versions = msg.get('versions', [])
            if not versions:
                if self.verbose and i < 3:
                    print(f"[DEBUG] Message {i}: No versions array", file=sys.stderr)
                continue
            
            # Get currently selected version (usually index 0)
            selected_idx = msg.get('currentlySelected', 0)
            if selected_idx >= len(versions):
                selected_idx = 0
            
            version = versions[selected_idx]
            role = version.get('role', '')
            
            if self.verbose and i < 3:
                print(f"[DEBUG] Message {i}: role={role}", file=sys.stderr)
            
            # Skip if not a user message
            if role != 'user':
                continue
            
            # Extract user query from content blocks
            content = version.get('content', [])
            query_parts = []
            
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '').strip()
                        if text:
                            query_parts.append(text)
            elif isinstance(content, str):
                query_parts.append(content)
            
            query = ' '.join(query_parts).strip()
            
            if not query:
                if self.verbose:
                    print(f"[DEBUG] Message {i}: Empty query after extraction", file=sys.stderr)
                continue
            
            # Extract assistant response from next message
            next_versions = next_msg.get('versions', [])
            if not next_versions:
                if self.verbose:
                    print(f"[DEBUG] Message {i+1}: No versions (incomplete?)", file=sys.stderr)
                continue
            
            next_selected_idx = next_msg.get('currentlySelected', 0)
            if next_selected_idx >= len(next_versions):
                next_selected_idx = 0
            
            next_version = next_versions[next_selected_idx]
            next_role = next_version.get('role', '')
            
            # Skip if next message is not assistant
            if next_role != 'assistant':
                if self.verbose and i < 3:
                    print(f"[DEBUG] Message {i+1}: Not assistant (role={next_role})", file=sys.stderr)
                continue
            
            # Extract assistant response from content blocks
            next_content = next_version.get('content', [])
            response_parts = []
            
            if isinstance(next_content, list):
                for block in next_content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '').strip()
                        if text:
                            response_parts.append(text)
            elif isinstance(next_content, str):
                response_parts.append(next_content)
            
            # For multi-step responses, also check steps array
            if next_version.get('type') == 'multiStep':
                steps = next_version.get('steps', [])
                for step in steps:
                    if step.get('type') == 'contentBlock':
                        # Skip thinking blocks
                        style = step.get('style', {})
                        if style.get('type') == 'thinking':
                            continue
                        
                        step_content = step.get('content', [])
                        for block in step_content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '').strip()
                                if text and not block.get('isStructural', False):
                                    response_parts.append(text)
            
            response = ' '.join(response_parts).strip()
            
            if not response:
                if self.verbose:
                    print(f"[DEBUG] Message {i+1}: Empty response after extraction", file=sys.stderr)
                continue
            
            # Successfully extracted a QA pair
            pairs.append((query, response))
            
            if self.verbose and len(pairs) <= 3:
                print(f"[DEBUG] Extracted pair {len(pairs)}:", file=sys.stderr)
                print(f"[DEBUG]   Q: {query[:80]}...", file=sys.stderr)
                print(f"[DEBUG]   A: {response[:80]}...", file=sys.stderr)
        
        if self.verbose:
            print(f"[DEBUG] Total QA pairs extracted: {len(pairs)}", file=sys.stderr)
        
        return pairs
    
    def find_preset_for_query(self, query: str, timestamp: float) -> Optional[str]:
        """
        Try to find which preset was used for a query by looking at ARIA telemetry
        
        Searches pack.stats.txt files in ARIA output directory for matching queries
        """
        if not ARIA_OUTPUT_DIR.exists():
            print(f"[DEBUG] ARIA output directory not found: {ARIA_OUTPUT_DIR}", file=sys.stderr)
            return None
        
        # Look for telemetry files within 10 minutes of the conversation
        time_window = 600  # 10 minutes in seconds
        
        # Normalize query for matching
        query_normalized = query.strip().lower()[:100]
        query_words = set(re.findall(r'\b\w{4,}\b', query_normalized))
        
        best_match_preset = None
        best_match_score = 0.0
        checked_files = 0
        
        # Search through recent run directories
        for stats_file in ARIA_OUTPUT_DIR.rglob("pack.stats.txt"):
            file_time = stats_file.stat().st_mtime
            
            # Skip files outside time window
            if abs(file_time - timestamp) > time_window:
                continue
            
            checked_files += 1
            
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    
                    # Parse telemetry JSON
                    data = json.loads(content)
                    
                    telemetry_query = data.get('query', '').strip().lower()[:100]
                    preset = data.get('preset', '')
                    
                    if not preset:
                        continue
                    
                    # Calculate match score
                    telemetry_words = set(re.findall(r'\b\w{4,}\b', telemetry_query))
                    overlap = len(query_words & telemetry_words)
                    score = overlap / max(len(query_words), 1) if query_words else 0
                    
                    # Exact match
                    if query_normalized == telemetry_query:
                        return preset
                    
                    # Track best fuzzy match
                    if score > best_match_score and score > 0.6:
                        best_match_score = score
                        best_match_preset = preset
                        
            except Exception as e:
                continue
        
        if checked_files == 0:
            print(f"[DEBUG] No telemetry files found in time window around {datetime.fromtimestamp(timestamp)}", file=sys.stderr)
        
        return best_match_preset
    
    def save_to_training_corpus(self, qa_pair: Tuple[str, str], score: float, preset: Optional[str]):
        """Save high-quality QA pairs to training corpus"""
        if not self.save_to_corpus or score < 0.6:
            return  # Only save good responses
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = TRAINING_CORPUS / f"scored_{timestamp}.jsonl"
            
            entry = {
                "query": qa_pair[0],
                "response": qa_pair[1],
                "score": score,
                "preset": preset,
                "timestamp": datetime.now().isoformat()
            }
            
            # Append to JSONL file
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            
            self.stats["corpus_saves"] += 1
            
        except Exception as e:
            print(f"[WARNING] Failed to save to corpus: {e}", file=sys.stderr)
    
    def update_bandit_from_conversations(self, conversations: List[Dict[str, Any]]):
        """Score conversations and update bandit rewards"""
        if not conversations:
            print("No conversations to score")
            return
        
        self.stats["total_conversations"] = len(conversations)
        total_scored = 0
        sum_scores = 0.0
        preset_scores = {}  # Track scores per preset
        
        print(f"\n{'='*70}")
        print(f"ARIA CONVERSATION SCORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        if self.verbose and conversations:
            print(f"[DEBUG] Sample conversation structure:", file=sys.stderr)
            sample = conversations[0]
            print(f"[DEBUG]   Keys: {list(sample.keys())}", file=sys.stderr)
            if "data" in sample:
                print(f"[DEBUG]   Data keys: {list(sample['data'].keys())}", file=sys.stderr)
            print(file=sys.stderr)
        
        for conv_entry in conversations:
            conv = conv_entry["data"]
            conv_time = conv_entry["modified"]
            
            pairs = self.extract_qa_pairs(conv)
            self.stats["total_qa_pairs"] += len(pairs)
            
            for user_query, assistant_response in pairs:
                # Score the response
                score = self.score_response(user_query, assistant_response)
                
                # Try to find which preset was used
                preset = self.find_preset_for_query(user_query, conv_time)
                if not preset:
                    preset = "hybrid_explore"  # Default fallback
                    self.stats["preset_misses"] += 1
                    if self.verbose:
                        print(f"[DEBUG] No preset match found for query, using fallback", file=sys.stderr)
                else:
                    self.stats["preset_matches"] += 1
                
                # Track scores per preset
                if preset not in preset_scores:
                    preset_scores[preset] = []
                preset_scores[preset].append(score)
                
                # Update bandit if available
                if BANDIT_AVAILABLE:
                    try:
                        give_reward(
                            preset_name=preset,
                            reward=score,
                            state_path=str(self.bandit_state_path)
                        )
                        self.stats["bandit_updates"] += 1
                    except Exception as e:
                        print(f"[WARNING] Bandit update failed: {e}", file=sys.stderr)
                
                # Save high-quality responses to corpus
                self.save_to_training_corpus((user_query, assistant_response), score, preset)
                
                # Print scored item
                print(f"[{preset:15s}] Score: {score:.3f}")
                print(f"  Q: {user_query[:80]}{'...' if len(user_query) > 80 else ''}")
                print(f"  A: {assistant_response[:80]}{'...' if len(assistant_response) > 80 else ''}")
                print()
                
                total_scored += 1
                sum_scores += score
        
        # Calculate statistics
        if total_scored > 0:
            self.stats["total_scored"] = total_scored
            self.stats["avg_score"] = sum_scores / total_scored
            
            print(f"\n{'='*70}")
            print(f"SCORING SUMMARY")
            print(f"{'='*70}")
            print(f"Conversations processed: {self.stats['total_conversations']}")
            print(f"QA pairs found: {self.stats['total_qa_pairs']}")
            print(f"Responses scored: {self.stats['total_scored']}")
            print(f"Average quality: {self.stats['avg_score']:.3f}")
            print(f"Bandit updates: {self.stats['bandit_updates']}")
            print(f"Corpus saves: {self.stats['corpus_saves']}")
            print(f"Preset matches: {self.stats['preset_matches']}")
            print(f"Preset misses: {self.stats['preset_misses']}")
            
            if self.stats['preset_misses'] > self.stats['preset_matches']:
                print(f"\n[WARNING] More preset misses than matches!")
                print(f"[WARNING] Check that ARIA is saving telemetry to: {ARIA_OUTPUT_DIR}")
                print(f"[WARNING] Use --verbose flag for more details")
            
            if preset_scores:
                print(f"\nScores by preset:")
                for preset, scores in sorted(preset_scores.items()):
                    avg = sum(scores) / len(scores)
                    print(f"  {preset:20s}: {avg:.3f} ({len(scores)} samples)")
            
            print(f"{'='*70}\n")
        else:
            # No responses were scored - show why
            print(f"\n{'='*70}")
            print(f"ERROR: NO RESPONSES SCORED")
            print(f"{'='*70}")
            print(f"Conversations processed: {self.stats['total_conversations']}")
            print(f"QA pairs found: {self.stats['total_qa_pairs']}")
            print()
            
            if self.stats['total_qa_pairs'] == 0:
                print("❌ ISSUE: No QA pairs extracted from conversations")
                print()
                print("Possible causes:")
                print("  1. Conversations don't have 'messages' field")
                print("  2. Messages don't have 'role' and 'content' fields")
                print("  3. No user/assistant message pairs found")
                print()
                print("Run with --verbose to see conversation structure:")
                print("  python3 conversation_scorer.py --verbose")
            else:
                print("❌ ISSUE: QA pairs found but scoring failed")
                print()
                print("This suggests an error in the scoring loop.")
                print("Run with --verbose to see more details:")
                print("  python3 conversation_scorer.py --verbose")
            
            print(f"{'='*70}\n")
    
    def run_scoring_loop(self, hours: int = 24):
        """Main scoring loop - reads conversations and updates learning"""
        print(f"🔍 Scoring conversations from last {hours} hours...")
        
        conversations = self.get_recent_conversations(hours)
        print(f"📊 Found {len(conversations)} recent conversations")
        
        if conversations:
            self.update_bandit_from_conversations(conversations)
        else:
            print("⚠️  No recent conversations to score")
        
        return self.stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Score conversations for ARIA learning")
    parser.add_argument("--hours", type=int, default=24, 
                       help="Score conversations from last N hours")
    parser.add_argument("--state", type=str, default="~/.rag_bandit_state.json",
                       help="Bandit state file")
    parser.add_argument("--no-corpus", action="store_true",
                       help="Don't save to training corpus")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug output")
    
    args = parser.parse_args()
    
    scorer = ConversationScorer(
        bandit_state_path=args.state,
        save_to_corpus=not args.no_corpus,
        verbose=args.verbose
    )
    stats = scorer.run_scoring_loop(hours=args.hours)
    
    # Exit with error code if no conversations were scored
    if stats["total_scored"] == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
