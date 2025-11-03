#!/usr/bin/env python3
"""
Temporal Tracker - Wave 2: Advanced Reasoning (FINAL FILE)

Tracks topic evolution, drift, and conversation arc across multi-turn interactions.

Integration with Current ARIA:
- Extends aria_session.py with temporal analysis
- Uses conversation_state_graph.py entity tracking
- Informs multi_hop_reasoner.py about conversation context
- Feeds into aria_telemetry.py for conversation metrics

Key Features:
1. Topic drift detection (when conversation shifts)
2. Conversation arc tracking (introduction -> exploration -> conclusion)
3. Topic coherence scoring (how connected are topics?)
4. Attention span modeling (when is user losing interest?)
5. Context window management (what to keep/discard)

Tracking Dimensions:
- Topic centrality (main vs tangential topics)
- Temporal decay (older topics less relevant)
- Transition smoothness (abrupt vs smooth topic changes)
- Conversation depth (surface vs deep exploration)
"""

import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class ConversationPhase(Enum):
    """Phases of conversation"""
    OPENING = "opening"  # Initial question
    EXPLORATION = "exploration"  # Deep dive
    TANGENT = "tangent"  # Off-topic
    CLARIFICATION = "clarification"  # Follow-up questions
    SYNTHESIS = "synthesis"  # Pulling together
    CLOSING = "closing"  # Winding down


@dataclass
class Topic:
    """A topic discussed in conversation"""
    id: str
    name: str
    
    # When introduced/discussed
    first_turn: int
    last_turn: int
    turns_active: List[int] = field(default_factory=list)
    
    # Centrality (how important is this topic?)
    centrality: float = 0.5  # 0 = tangential, 1 = core
    
    # Related entities from conversation graph
    entities: Set[str] = field(default_factory=set)
    
    # Intensity (how deeply explored?)
    intensity: float = 0.0  # Based on turn count + depth
    
    # Is this topic still active?
    active: bool = True


@dataclass
class TopicTransition:
    """Transition between topics"""
    from_topic: str
    to_topic: str
    turn_index: int
    
    # Transition type
    is_smooth: bool = True  # False = abrupt shift
    is_tangent: bool = False  # Going off-topic
    is_return: bool = False  # Returning to previous topic
    
    # Confidence in transition detection
    confidence: float = 1.0


@dataclass
class ConversationArc:
    """Overall conversation trajectory"""
    start_time: datetime
    current_turn: int
    
    # Topics discussed
    topics: Dict[str, Topic] = field(default_factory=dict)
    
    # Topic transitions
    transitions: List[TopicTransition] = field(default_factory=list)
    
    # Current phase
    phase: ConversationPhase = ConversationPhase.OPENING
    
    # Metrics
    coherence_score: float = 1.0  # How connected are topics?
    depth_score: float = 0.0  # How deep is exploration?
    drift_score: float = 0.0  # How much drift?
    
    # Context window (what's currently relevant?)
    active_topics: List[str] = field(default_factory=list)
    context_size: int = 3  # Number of topics to keep active


class TemporalTracker:
    """
    Track conversation evolution over time
    
    Process:
    1. Extract topics from each turn
    2. Detect topic transitions
    3. Update topic centrality scores
    4. Identify conversation phase
    5. Manage active context window
    6. Detect drift and recommend interventions
    """
    
    def __init__(
        self,
        context_window_size: int = 3,
        drift_threshold: float = 0.7,
        state_path: Optional[Path] = None
    ):
        """
        Args:
            context_window_size: Number of topics to keep active
            drift_threshold: Threshold for drift detection (0-1)
            state_path: Path to save/load state
        """
        self.context_window_size = context_window_size
        self.drift_threshold = drift_threshold
        self.state_path = state_path or Path.home() / '.aria_temporal_tracker.json'
        
        # Current conversation arc
        self.arc: Optional[ConversationArc] = None
        
        # Topic extraction patterns
        self._build_patterns()
    
    def _build_patterns(self):
        """Build patterns for topic extraction"""
        # Technical terms (likely topics)
        self.topic_patterns = [
            r'\b[A-Z][a-zA-Z]{3,}\b',  # Capitalized words
            r'\b[a-z]+(?:-[a-z]+)+\b',  # Hyphenated terms
            r'\b(?:algorithm|method|approach|technique|system|model|framework)\b',
        ]
    
    def start_conversation(self):
        """Start tracking a new conversation"""
        self.arc = ConversationArc(
            start_time=datetime.now(),
            current_turn=0,
            phase=ConversationPhase.OPENING
        )
    
    def add_turn(
        self,
        turn_index: int,
        user_message: str,
        assistant_response: str,
        entities: Optional[List[str]] = None
    ):
        """
        Process a conversation turn
        
        Args:
            turn_index: Turn number
            user_message: User's message
            assistant_response: Assistant's response
            entities: Optional entities from conversation_state_graph
        """
        if self.arc is None:
            self.start_conversation()
        
        assert self.arc is not None, "Arc should be initialized"
        self.arc.current_turn = turn_index
        
        # Extract topics from turn
        turn_topics = self._extract_topics(user_message + " " + assistant_response)
        
        # Add entities as topics too
        if entities:
            turn_topics.extend(entities)
        
        # Update existing topics or create new ones
        for topic_name in turn_topics:
            self._update_topic(topic_name, turn_index)
        
        # Detect topic transitions
        if turn_index > 0:
            self._detect_transitions(turn_index, turn_topics)
        
        # Update topic centrality
        self._update_centrality()
        
        # Update conversation phase
        self._update_phase()
        
        # Manage active context window
        self._update_context_window()
        
        # Compute metrics
        self._compute_metrics()
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from text
        
        Returns:
            List of topic names
        """
        if not self.arc:
            return []
        
        topics = []
        
        # Extract using patterns
        for pattern in self.topic_patterns:
            matches = set(m.lower() for m in re.findall(pattern, text))
            topics.extend(matches)
        
        # Simple noun extraction (words with capital letters or 5+ chars)
        words = text.split()
        for word in words:
            word_clean = word.strip('.,!?').lower()
            if len(word_clean) >= 5 and word_clean not in topics:
                topics.append(word_clean)
        
        # Deduplicate
        return list(set(topics))[:10]  # Max 10 topics per turn
    
    def _update_topic(self, topic_name: str, turn_index: int):
        """Update or create a topic"""
        if not self.arc:
            return
        
        topic_id = topic_name.lower().replace(' ', '_')
        
        if topic_id in self.arc.topics:
            # Update existing topic
            topic = self.arc.topics[topic_id]
            topic.last_turn = turn_index
            topic.turns_active.append(turn_index)
            topic.intensity += 1.0 / max(turn_index - topic.first_turn + 1, 1)
            topic.active = True
        else:
            # Create new topic
            self.arc.topics[topic_id] = Topic(
                id=topic_id,
                name=topic_name,
                first_turn=turn_index,
                last_turn=turn_index,
                turns_active=[turn_index],
                centrality=0.5  # Initial centrality
            )
    
    def _detect_transitions(self, turn_index: int, current_topics: List[str]):
        """Detect topic transitions"""
        if not self.arc:
            return
        
        if not self.arc.transitions and turn_index == 1:
            # First transition - always smooth
            for topic in current_topics:
                self.arc.transitions.append(TopicTransition(
                    from_topic="__start__",
                    to_topic=topic,
                    turn_index=turn_index,
                    is_smooth=True
                ))
            return
        
        # Get previous turn's topics
        prev_topics = [
            t.name for t in self.arc.topics.values()
            if turn_index - 1 in t.turns_active
        ]
        
        # Find new topics (not in previous turn)
        new_topics = [t for t in current_topics if t not in [pt.lower() for pt in prev_topics]]
        
        if new_topics:
            # Determine if transition is smooth or abrupt
            overlap = len(set(current_topics) & set([pt.lower() for pt in prev_topics]))
            is_smooth = overlap > 0 or len(new_topics) == 1
            
            # Check if returning to older topic
            older_topics = [t.name for t in self.arc.topics.values() if t.first_turn < turn_index - 1]
            is_return = any(nt in [ot.lower() for ot in older_topics] for nt in new_topics)
            
            # Create transition
            for new_topic in new_topics:
                from_topic = prev_topics[0] if prev_topics else "__unknown__"
                
                self.arc.transitions.append(TopicTransition(
                    from_topic=from_topic,
                    to_topic=new_topic,
                    turn_index=turn_index,
                    is_smooth=is_smooth,
                    is_return=is_return,
                    confidence=0.8 if is_smooth else 0.6
                ))
    
    def _update_centrality(self):
        """
        Update topic centrality scores
        
        Central topics: discussed frequently, connected to many others
        """
        if not self.arc or not self.arc.topics:
            return
        
        # Frequency component
        max_mentions = max(len(t.turns_active) for t in self.arc.topics.values())
        
        for topic in self.arc.topics.values():
            freq_score = len(topic.turns_active) / max(max_mentions, 1)
            
            # Recency component (recent = more central)
            recency_score = 1.0 - (self.arc.current_turn - topic.last_turn) / max(self.arc.current_turn, 1)
            
            # Combine
            topic.centrality = 0.6 * freq_score + 0.4 * recency_score
    
    def _update_phase(self):
        """Update conversation phase"""
        if not self.arc:
            return
        
        if self.arc.current_turn == 0:
            self.arc.phase = ConversationPhase.OPENING
        elif self.arc.current_turn <= 2:
            self.arc.phase = ConversationPhase.EXPLORATION
        else:
            # Check for patterns
            recent_transitions = self.arc.transitions[-3:] if len(self.arc.transitions) >= 3 else self.arc.transitions
            
            # Many abrupt transitions = tangent phase
            abrupt_count = sum(1 for t in recent_transitions if not t.is_smooth)
            if abrupt_count >= 2:
                self.arc.phase = ConversationPhase.TANGENT
            # Returning to old topics = synthesis
            elif any(t.is_return for t in recent_transitions):
                self.arc.phase = ConversationPhase.SYNTHESIS
            else:
                self.arc.phase = ConversationPhase.EXPLORATION
    
    def _update_context_window(self):
        """Update active context window"""
        if not self.arc:
            return
        
        # Sort topics by centrality
        sorted_topics = sorted(
            self.arc.topics.values(),
            key=lambda t: t.centrality,
            reverse=True
        )
        
        # Keep top N most central topics active
        self.arc.active_topics = [
            t.id for t in sorted_topics[:self.context_window_size]
        ]
        
        # Mark topics as active/inactive
        for topic in self.arc.topics.values():
            topic.active = topic.id in self.arc.active_topics
    
    def _compute_metrics(self):
        """Compute conversation metrics"""
        if not self.arc or not self.arc.topics:
            return
        
        # Coherence: how connected are topics?
        # High coherence = smooth transitions, low topic count
        smooth_ratio = sum(1 for t in self.arc.transitions if t.is_smooth) / max(len(self.arc.transitions), 1)
        topic_spread = len(self.arc.topics) / max(self.arc.current_turn + 1, 1)
        self.arc.coherence_score = smooth_ratio * (1.0 - min(topic_spread, 1.0))
        
        # Depth: how deeply are topics explored?
        # High depth = topics discussed across multiple turns
        avg_topic_intensity = float(np.mean([t.intensity for t in self.arc.topics.values()]))
        self.arc.depth_score = min(1.0, avg_topic_intensity / 3.0)
        
        # Drift: how much are we moving away from original topic?
        if self.arc.topics:
            first_topic = min(self.arc.topics.values(), key=lambda t: t.first_turn)
            current_centrality = first_topic.centrality
            self.arc.drift_score = 1.0 - current_centrality
    
    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect if conversation has drifted significantly
        
        Returns:
            Dict with drift info and recommendations
        """
        if not self.arc:
            return {'drift_detected': False}
        
        drift_detected = self.arc.drift_score > self.drift_threshold
        
        result = {
            'drift_detected': drift_detected,
            'drift_score': self.arc.drift_score,
            'coherence_score': self.arc.coherence_score,
            'current_phase': self.arc.phase.value
        }
        
        if drift_detected:
            # Find original topic
            original_topic = min(self.arc.topics.values(), key=lambda t: t.first_turn)
            
            result['original_topic'] = original_topic.name
            result['recommendation'] = f"Consider refocusing on '{original_topic.name}'"
            
            # Find current central topic
            current_central = max(self.arc.topics.values(), key=lambda t: t.centrality)
            result['current_topic'] = current_central.name
        
        return result
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get comprehensive conversation summary"""
        if not self.arc:
            return {'status': 'no_conversation'}
        
        # Top topics
        top_topics = sorted(
            self.arc.topics.values(),
            key=lambda t: t.centrality,
            reverse=True
        )[:5]
        
        # Transition analysis
        transition_types = {
            'smooth': sum(1 for t in self.arc.transitions if t.is_smooth),
            'abrupt': sum(1 for t in self.arc.transitions if not t.is_smooth),
            'tangent': sum(1 for t in self.arc.transitions if t.is_tangent),
            'return': sum(1 for t in self.arc.transitions if t.is_return)
        }
        
        return {
            'status': 'active',
            'turns': self.arc.current_turn + 1,
            'phase': self.arc.phase.value,
            'total_topics': len(self.arc.topics),
            'active_topics': len(self.arc.active_topics),
            'coherence': self.arc.coherence_score,
            'depth': self.arc.depth_score,
            'drift': self.arc.drift_score,
            'top_topics': [
                {'name': t.name, 'centrality': t.centrality, 'turns': len(t.turns_active)}
                for t in top_topics
            ],
            'transition_analysis': transition_types
        }
    
    def get_context_for_retrieval(self) -> str:
        """
        Get contextual information for next retrieval
        
        Returns:
            String with relevant context from active topics
        """
        if not self.arc or not self.arc.active_topics:
            return ""
        
        # Get active topic names
        active_topic_names = [
            self.arc.topics[tid].name
            for tid in self.arc.active_topics
            if tid in self.arc.topics
        ]
        
        context = f"Previous topics discussed: {', '.join(active_topic_names)}"
        
        return context


# ============================================================================
# Integration with ARIA
# ============================================================================

def integrate_with_aria():
    """
    Example: Integrate temporal tracker with aria_session.py
    """
    
    # Initialize tracker
    tracker = TemporalTracker(
        context_window_size=3,
        drift_threshold=0.7
    )
    
    # Start conversation
    tracker.start_conversation()
    
    # Example conversation turns
    turns = [
        ("What is Thompson Sampling?", "Thompson Sampling is a Bayesian approach..."),
        ("How does it compare to UCB?", "UCB uses upper confidence bounds while Thompson Sampling..."),
        ("What about epsilon-greedy?", "Epsilon-greedy is simpler..."),
        ("Can you explain neural networks?", "Neural networks are composed of layers..."),  # Topic shift!
    ]
    
    for i, (user_msg, asst_resp) in enumerate(turns):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_msg}")
        
        tracker.add_turn(i, user_msg, asst_resp)
        
        # Get conversation summary
        summary = tracker.get_conversation_summary()
        print(f"Phase: {summary['phase']}")
        print(f"Coherence: {summary['coherence']:.2f}")
        print(f"Depth: {summary['depth']:.2f}")
        print(f"Drift: {summary['drift']:.2f}")
        
        # Check for drift
        drift_info = tracker.detect_drift()
        if drift_info['drift_detected']:
            print(f"âš ï¸  Drift detected!")
            print(f"   Original: {drift_info['original_topic']}")
            print(f"   Current: {drift_info['current_topic']}")
            print(f"   {drift_info['recommendation']}")
    
    # Final summary
    print("\n" + "="*60)
    final_summary = tracker.get_conversation_summary()
    print("ðŸ“Š Final Conversation Summary:")
    print(f"  Total turns: {final_summary['turns']}")
    print(f"  Topics discussed: {final_summary['total_topics']}")
    print(f"  Coherence: {final_summary['coherence']:.2%}")
    print(f"  Depth: {final_summary['depth']:.2%}")
    print(f"\n  Top topics:")
    for topic in final_summary['top_topics']:
        print(f"    â€¢ {topic['name']} (centrality: {topic['centrality']:.2f}, turns: {topic['turns']})")


if __name__ == '__main__':
    import re  # Need this for regex
    integrate_with_aria()
