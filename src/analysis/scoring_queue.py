#!/usr/bin/env python3
"""
ARIA Scoring Queue - Lightweight Conversation Marker

Minimal overhead function to mark conversations for async scoring.
Called by aria_main.py after each response.

Design:
- Write single JSON line to queue file (~50 bytes)
- No scoring computation (< 1ms overhead)
- Daemon picks up and processes asynchronously
"""

import json
import time
from pathlib import Path
from typing import Optional

# Queue file location
ARIA_VAR = Path("/media/notapplicable/Internal-SSD/ai-quaternions-model/var")
SCORING_QUEUE = ARIA_VAR / "scoring_queue.jsonl"

# Ensure directory exists
ARIA_VAR.mkdir(parents=True, exist_ok=True)


def mark_conversation_for_scoring(
    query: str,
    preset: str,
    conversation_id: Optional[str] = None
) -> None:
    """
    Mark a conversation for async scoring (< 1ms overhead).
    
    Args:
        query: User query text
        preset: ARIA preset used
        conversation_id: LM Studio conversation ID (if available)
    
    This creates a lightweight marker that the scoring daemon will pick up.
    No scoring computation happens here - keeps query latency minimal.
    """
    try:
        # Create marker entry
        marker = {
            'timestamp': time.time(),
            'query': query[:200],  # First 200 chars only
            'preset': preset,
            'conversation_id': conversation_id
        }
        
        # Append to queue file (atomic write)
        with open(SCORING_QUEUE, 'a') as f:
            f.write(json.dumps(marker) + '\n')
            
    except Exception:
        # Silently fail - don't break queries if scoring fails
        pass


def get_queue_size() -> int:
    """Get number of items in scoring queue"""
    if not SCORING_QUEUE.exists():
        return 0
    try:
        with open(SCORING_QUEUE) as f:
            return sum(1 for _ in f)
    except:
        return 0


def clear_queue() -> None:
    """Clear the scoring queue"""
    try:
        if SCORING_QUEUE.exists():
            SCORING_QUEUE.unlink()
    except:
        pass
