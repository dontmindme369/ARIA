#!/usr/bin/env python3
"""
Real-Time Feedback Detector for ARIA

Detects positive and negative feedback signals from user messages to enable
immediate bandit updates and response correction.

This is the KEY to making ARIA adapt in real-time, not just batch scoring.
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback detected"""
    NEGATIVE = -1.0
    NEUTRAL = 0.0
    POSITIVE = 1.0


@dataclass
class FeedbackSignal:
    """Detected feedback from user message"""
    type: FeedbackType
    confidence: float  # 0.0 to 1.0
    reason: str  # What triggered the detection
    keywords_matched: list
    should_trigger_correction: bool  # Should we trigger feedback anchor?


class FeedbackDetector:
    """
    Detect positive/negative feedback from user messages
    
    This runs on EVERY user query to detect if they're giving feedback
    about the previous response.
    """
    
    # NEGATIVE FEEDBACK PATTERNS
    # These indicate the user is unhappy with the previous response
    NEGATIVE_KEYWORDS = {
        # Direct rejection
        'no': 0.3,
        'nope': 0.4,
        'wrong': 0.9,
        'incorrect': 0.9,
        'not what i asked': 1.0,
        'not what i meant': 1.0,
        'that\'s not': 0.7,
        'thats not': 0.7,
        
        # Misunderstanding indicators
        'you misunderstood': 1.0,
        'you missed': 0.8,
        'you didn\'t': 0.7,
        'didnt understand': 0.8,
        'didn\'t get': 0.7,
        'not understanding': 0.7,
        
        # Complexity issues
        'too simple': 0.9,
        'too simplistic': 0.9,
        'oversimplified': 0.9,
        'too complex': 0.8,
        'too complicated': 0.8,
        'too technical': 0.7,
        'too basic': 0.8,
        'eli5': 0.6,  # Asking for simpler after getting complex
        
        # Tone/approach issues
        'not helpful': 0.9,
        'unhelpful': 0.9,
        'useless': 0.9,
        'missed the point': 1.0,
        'off topic': 0.8,
        'not relevant': 0.8,
        'doesn\'t answer': 0.9,
        'didnt answer': 0.9,
        
        # Clarification needed
        'let me clarify': 0.7,
        'let me rephrase': 0.6,
        'what i meant was': 0.8,
        'i meant': 0.5,
        'actually i': 0.5,
        
        # Frustration
        'ugh': 0.6,
        'seriously': 0.4,
        'come on': 0.5,
        'for real': 0.5,
    }
    
    NEGATIVE_PATTERNS = [
        # Explicit corrections
        (r'\bno[,.]?\s+(i|that)', 0.7),
        (r'actually[,.]?\s+(i|it|that)', 0.6),
        (r'(i said|i asked)[^.]{0,30}(not|didn)', 0.8),
        
        # Rejection with reason
        (r'(wrong|incorrect|not right) because', 0.9),
        (r'that.{0,20}(wrong|incorrect|not (what|how))', 0.8),
        
        # Redirection
        (r'(but|however) (i|what i|my question)', 0.6),
        (r'(instead|rather)[,.]?\s+(i|my)', 0.5),
        
        # Strong negation
        (r'(no|nope|wrong)[.!]{1,3}$', 0.8),
        (r'^(no|nope|wrong)\s', 0.7),
    ]
    
    # POSITIVE FEEDBACK PATTERNS
    # These indicate the user is happy with the response
    POSITIVE_KEYWORDS = {
        # Strong positive
        'perfect': 1.0,
        'exactly': 1.0,
        'precisely': 0.9,
        'spot on': 1.0,
        'brilliant': 0.9,
        'excellent': 0.9,
        'outstanding': 0.9,
        
        # Gratitude
        'thank you': 0.7,
        'thanks': 0.6,
        'appreciate': 0.7,
        'helpful': 0.8,
        'very helpful': 0.9,
        'super helpful': 0.9,
        
        # Agreement
        'makes sense': 0.7,
        'got it': 0.6,
        'understand now': 0.8,
        'i see': 0.5,
        'ah i see': 0.7,
        'clear now': 0.8,
        'that clarifies': 0.8,
        
        # Approval
        'great': 0.7,
        'awesome': 0.8,
        'nice': 0.6,
        'cool': 0.5,
        'good': 0.5,
        'works': 0.6,
        'that works': 0.7,
        
        # Satisfaction
        'exactly what i needed': 1.0,
        'just what i wanted': 0.9,
        'that\'s it': 0.7,
        'thats it': 0.7,
    }
    
    POSITIVE_PATTERNS = [
        # Enthusiastic agreement
        (r'(yes|yeah|yep)[!.]{1,3}', 0.7),
        (r'(perfect|exactly|precisely)[!.]{1,3}', 1.0),
        
        # Gratitude with emphasis
        (r'thank(s| you)(\s+(so )?much)?[!.]{1,3}', 0.8),
        (r'really appreciate', 0.8),
        
        # Understanding achieved
        (r'(now|finally) (i )?(get it|understand|see)', 0.8),
        (r'makes (total|complete|perfect) sense', 0.9),
    ]
    
    # CONTEXT INDICATORS
    # These help determine if feedback is about the previous response
    FEEDBACK_CONTEXT_INDICATORS = [
        'that',
        'your answer',
        'your response',
        'your explanation',
        'what you said',
        'you mentioned',
        'you explained',
        'the previous',
        'above',
    ]
    
    def __init__(
        self, 
        verbose: bool = False,
        training_corpus_path: str = "/media/notapplicable/ARIA-knowledge/training_corpus/"
    ):
        self.verbose = verbose
        self.training_corpus_path = Path(training_corpus_path)
        self.training_corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'positive_captured': 0,
            'negative_captured': 0,
            'neutral_skipped': 0,
            'total_detections': 0
        }
    
    def _write_to_corpus(
        self, 
        user_message: str, 
        previous_response: str, 
        signal: FeedbackSignal,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Write feedback detection to training corpus (JSONL format)
        
        Args:
            user_message: Current user message containing feedback
            previous_response: Previous assistant response being evaluated
            signal: Detected feedback signal
            metadata: Additional metadata (reasoning_mode, timestamp, etc.)
        """
        # Skip neutral feedback (not useful for training)
        if signal.type == FeedbackType.NEUTRAL:
            self.stats['neutral_skipped'] += 1
            return
        
        # Generate safe filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_msg = "".join(c if c.isalnum() else "_" for c in user_message[:30])
        feedback_type = signal.type.name.lower()
        filename = f"feedback_{feedback_type}_{timestamp}_{safe_msg}.json"
        
        # Create corpus entry
        corpus_data = {
            "type": "feedback",
            "feedback_type": signal.type.name,
            "confidence": signal.confidence,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "previous_response": previous_response[:1000],  # Truncate to 1000 chars
            "keywords_matched": signal.keywords_matched[:5],  # Top 5 keywords
            "reason": signal.reason,
            "should_trigger_correction": signal.should_trigger_correction,
            "metadata": metadata or {}
        }
        
        # Write to corpus
        try:
            corpus_path = self.training_corpus_path / filename
            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, indent=2, ensure_ascii=False)
            
            # Update stats
            if signal.type == FeedbackType.POSITIVE:
                self.stats['positive_captured'] += 1
            elif signal.type == FeedbackType.NEGATIVE:
                self.stats['negative_captured'] += 1
            
            if self.verbose:
                print(f"[FeedbackDetector] ✅ Wrote {feedback_type} feedback to corpus: {filename}")
        
        except Exception as e:
            if self.verbose:
                print(f"[FeedbackDetector] ⚠️  Failed to write corpus: {e}")
    
    def detect(self, user_message: str, previous_response: Optional[str] = None, metadata: Optional[Dict] = None) -> FeedbackSignal:
        """
        Detect feedback in user's message and optionally write to training corpus
        
        Args:
            user_message: Current user message
            previous_response: Previous assistant response (for context and corpus capture)
            metadata: Additional metadata to store in corpus (reasoning_mode, timestamp, etc.)
        
        Returns:
            FeedbackSignal with type, confidence, and reason
        """
        self.stats['total_detections'] += 1
        msg_lower = user_message.lower()
        
        # Check if this message is likely feedback about previous response
        is_feedback_context = any(ind in msg_lower for ind in self.FEEDBACK_CONTEXT_INDICATORS)
        
        # Very short messages (< 10 chars) are usually not feedback
        if len(user_message.strip()) < 10:
            return FeedbackSignal(
                type=FeedbackType.NEUTRAL,
                confidence=1.0,
                reason="Message too short to contain feedback",
                keywords_matched=[],
                should_trigger_correction=False
            )
        
        # Check negative feedback first (more important)
        negative_score, neg_keywords = self._check_negative(msg_lower)
        
        # Check positive feedback
        positive_score, pos_keywords = self._check_positive(msg_lower)
        
        # Determine feedback type
        if negative_score > 0.5:  # Strong negative signal
            # Boost confidence if feedback context indicators present
            confidence = min(1.0, negative_score * (1.3 if is_feedback_context else 1.0))
            
            # Should trigger correction if confidence > 0.7
            should_correct = confidence >= 0.7
            
            signal = FeedbackSignal(
                type=FeedbackType.NEGATIVE,
                confidence=confidence,
                reason=f"Negative feedback detected (keywords: {', '.join(neg_keywords)})",
                keywords_matched=neg_keywords,
                should_trigger_correction=should_correct
            )
            
            # Write to corpus if we have previous response
            if previous_response:
                self._write_to_corpus(user_message, previous_response, signal, metadata)
            
            return signal
        
        elif positive_score > 0.5:  # Strong positive signal
            confidence = min(1.0, positive_score * (1.2 if is_feedback_context else 1.0))
            
            signal = FeedbackSignal(
                type=FeedbackType.POSITIVE,
                confidence=confidence,
                reason=f"Positive feedback detected (keywords: {', '.join(pos_keywords)})",
                keywords_matched=pos_keywords,
                should_trigger_correction=False
            )
            
            # Write to corpus if we have previous response
            if previous_response:
                self._write_to_corpus(user_message, previous_response, signal, metadata)
            
            return signal
        
        else:  # Neutral or ambiguous
            return FeedbackSignal(
                type=FeedbackType.NEUTRAL,
                confidence=1.0,
                reason="No clear feedback signal detected",
                keywords_matched=[],
                should_trigger_correction=False
            )
    
    def _check_negative(self, msg_lower: str) -> Tuple[float, list]:
        """
        Check for negative feedback patterns
        
        Returns:
            (score, keywords_matched)
        """
        score = 0.0
        keywords_matched = []
        
        # Check keywords
        for keyword, weight in self.NEGATIVE_KEYWORDS.items():
            if keyword in msg_lower:
                score = max(score, weight)
                keywords_matched.append(keyword)
        
        # Check regex patterns
        for pattern, weight in self.NEGATIVE_PATTERNS:
            if re.search(pattern, msg_lower):
                score = max(score, weight)
                keywords_matched.append(f"pattern:{pattern[:20]}")
        
        return score, keywords_matched
    
    def _check_positive(self, msg_lower: str) -> Tuple[float, list]:
        """
        Check for positive feedback patterns
        
        Returns:
            (score, keywords_matched)
        """
        score = 0.0
        keywords_matched = []
        
        # Check keywords
        for keyword, weight in self.POSITIVE_KEYWORDS.items():
            if keyword in msg_lower:
                score = max(score, weight)
                keywords_matched.append(keyword)
        
        # Check regex patterns
        for pattern, weight in self.POSITIVE_PATTERNS:
            if re.search(pattern, msg_lower):
                score = max(score, weight)
                keywords_matched.append(f"pattern:{pattern[:20]}")
        
        return score, keywords_matched
    
    def get_feedback_stats(self) -> Dict[str, int]:
        """Return counts of feedback types (for monitoring)"""
        return {
            'negative_keywords': len(self.NEGATIVE_KEYWORDS),
            'negative_patterns': len(self.NEGATIVE_PATTERNS),
            'positive_keywords': len(self.POSITIVE_KEYWORDS),
            'positive_patterns': len(self.POSITIVE_PATTERNS),
        }


# ============================================================================
# TESTING
# ============================================================================

def test_feedback_detector():
    """Test the feedback detector with various messages"""
    detector = FeedbackDetector(verbose=True)
    
    test_cases = [
        # Negative feedback
        "No, that's not what I asked",
        "You misunderstood my question",
        "That's too simple, I need technical details",
        "That's too complex, explain it simply",
        "Wrong, that's not how it works",
        "You missed the point completely",
        "Not helpful at all",
        "Let me clarify - I meant something different",
        
        # Positive feedback
        "Perfect! Exactly what I needed",
        "Thank you so much, that's very helpful",
        "Great explanation, makes sense now",
        "That's exactly right",
        "Brilliant answer",
        "I understand now, thanks!",
        
        # Neutral (actual queries)
        "What is machine learning?",
        "How do I implement this?",
        "Tell me more about that",
        "Can you elaborate?",
    ]
    
    print("=" * 70)
    print("FEEDBACK DETECTOR TEST")
    print("=" * 70)
    print()
    
    for test in test_cases:
        signal = detector.detect(test)
        
        color = {
            FeedbackType.NEGATIVE: '\033[91m',  # Red
            FeedbackType.POSITIVE: '\033[92m',  # Green
            FeedbackType.NEUTRAL: '\033[93m',   # Yellow
        }.get(signal.type, '')
        reset = '\033[0m'
        
        print(f"Message: \"{test}\"")
        print(f"  Type: {color}{signal.type.name}{reset}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Trigger correction: {signal.should_trigger_correction}")
        print(f"  Reason: {signal.reason}")
        print()


if __name__ == "__main__":
    test_feedback_detector()
