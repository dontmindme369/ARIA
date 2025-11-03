#!/usr/bin/env python3
"""
ARIA Session Manager v45 - Multi-Turn Conversation Memory

Phase 3 of v45 Integration:
- Tracks conversation history across queries
- Detects follow-up questions
- Syncs with LM Studio conversations
- Enables context-aware retrieval

Combines:
- aria_conversation_watcher.py (conversation tracking)
- lmstudio_chat_sync.py (LM Studio sync)
- conversation_scorer.py (quality scoring)
"""
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field


def extract_message_content(message: Dict[str, Any]) -> str:
    """
    Extract text content from LM Studio's nested message structure.
    
    LM Studio structure: message → versions[] → content[] → text
    Falls back to direct 'content' field for compatibility.
    """
    try:
        # Try nested structure first (LM Studio's actual format)
        versions = message.get('versions', [])
        if versions and isinstance(versions, list):
            # Get the latest version (usually last one)
            latest_version = versions[-1] if versions else {}
            
            content_array = latest_version.get('content', [])
            if content_array and isinstance(content_array, list):
                # Extract text from all content blocks
                texts = []
                for content_block in content_array:
                    if isinstance(content_block, dict):
                        text = content_block.get('text', '')
                        if text:
                            texts.append(str(text))
                
                if texts:
                    return '\n'.join(texts).strip()
        
        # Fallback: try direct content field
        direct_content = message.get('content', '')
        if direct_content:
            if isinstance(direct_content, list):
                # Handle list of content blocks
                text_parts = [
                    block.get('text', '')
                    for block in direct_content
                    if isinstance(block, dict) and block.get('type') == 'text'
                ]
                return '\n'.join(text_parts).strip()
            return str(direct_content).strip()
        
        return ''
    
    except Exception:
        return ''


@dataclass
class Message:
    """Single conversation message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.message_id:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:12]
            self.message_id = f"{self.role}_{int(self.timestamp)}_{content_hash}"


@dataclass
class Session:
    """Conversation session with history"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the session"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.last_updated = time.time()
        return msg
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get last N messages"""
        return self.messages[-n:]
    
    def get_context_window(self, max_chars: int = 4000) -> str:
        """Get recent conversation context as text"""
        context_parts = []
        total_chars = 0
        
        # Work backwards from most recent
        for msg in reversed(self.messages):
            msg_text = f"{msg.role.upper()}: {msg.content}"
            msg_len = len(msg_text)
            
            if total_chars + msg_len > max_chars:
                break
                
            context_parts.insert(0, msg_text)
            total_chars += msg_len
        
        return "\n\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        """Serialize to dict"""
        return {
            'session_id': self.session_id,
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp,
                    'message_id': m.message_id
                }
                for m in self.messages
            ],
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'metadata': self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'Session':
        """Deserialize from dict"""
        session = Session(
            session_id=data['session_id'],
            created_at=data.get('created_at', time.time()),
            last_updated=data.get('last_updated', time.time()),
            metadata=data.get('metadata', {})
        )
        
        for msg_data in data.get('messages', []):
            msg = Message(
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=msg_data.get('timestamp', time.time()),
                message_id=msg_data.get('message_id')
            )
            session.messages.append(msg)
        
        return session


class SessionManager:
    """
    Manages multi-turn conversation sessions
    
    Features:
    - Session tracking with history
    - Follow-up query detection
    - LM Studio conversation sync
    - Context window management
    """
    
    def __init__(self, state_dir: Path = Path("./var/sessions")):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, Session] = {}
        self.active_session_id: Optional[str] = None
        
        # LM Studio paths
        self.lmstudio_conversations = Path.home() / ".lmstudio" / "conversations"
        self.processed_messages: Set[str] = set()
        
        self._load_state()
    
    def _load_state(self):
        """Load persisted sessions"""
        state_file = self.state_dir / "sessions.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
                
            self.active_session_id = data.get('active_session_id')
            self.processed_messages = set(data.get('processed_messages', []))
            
            for session_data in data.get('sessions', []):
                session = Session.from_dict(session_data)
                self.sessions[session.session_id] = session
        except Exception as e:
            print(f"[SessionManager] Error loading state: {e}")
    
    def _save_state(self):
        """Persist sessions"""
        state_file = self.state_dir / "sessions.json"
        
        try:
            data = {
                'active_session_id': self.active_session_id,
                'processed_messages': list(self.processed_messages),
                'sessions': [s.to_dict() for s in self.sessions.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SessionManager] Error saving state: {e}")
    
    def create_session(self) -> str:
        """Create a new session"""
        session_id = f"session_{int(time.time())}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        self.active_session_id = session_id
        self._save_state()
        return session_id
    
    def get_or_create_active_session(self) -> Session:
        """Get active session or create new one"""
        if self.active_session_id and self.active_session_id in self.sessions:
            return self.sessions[self.active_session_id]
        
        session_id = self.create_session()
        return self.sessions[session_id]
    
    def add_message(self, role: str, content: str, session_id: Optional[str] = None) -> Message:
        """Add a message to a session"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = self.get_or_create_active_session()
        
        msg = session.add_message(role, content)
        self._save_state()
        return msg
    
    def is_follow_up(self, query: str, session_id: Optional[str] = None, lookback_seconds: int = 300) -> bool:
        """
        Detect if query is a follow-up to recent conversation
        
        Follow-up indicators:
        - Short query referencing recent context
        - Pronouns (it, this, that, they)
        - Comparative language (compared to, vs, versus)
        - Question words about previous topic
        """
        if session_id:
            session = self.sessions.get(session_id)
        else:
            session = self.get_or_create_active_session()
        
        if not session or len(session.messages) < 2:
            return False
        
        # Check if there's recent conversation activity
        last_msg = session.messages[-1]
        time_since_last = time.time() - last_msg.timestamp
        
        if time_since_last > lookback_seconds:
            return False
        
        # Follow-up patterns
        query_lower = query.lower()
        
        # Pronouns
        pronouns = ['it', 'this', 'that', 'they', 'them', 'those', 'these']
        if any(query_lower.startswith(p + ' ') or f' {p} ' in query_lower for p in pronouns):
            return True
        
        # Comparative
        comparatives = ['compared to', 'versus', ' vs ', 'compared with', 'relative to']
        if any(comp in query_lower for comp in comparatives):
            return True
        
        # Question follow-ups
        question_starters = ['how does', 'what about', 'why does', 'can you explain', 'tell me more']
        if any(query_lower.startswith(q) for q in question_starters):
            # Short queries are likely follow-ups
            if len(query.split()) < 10:
                return True
        
        # Very short queries are often follow-ups
        if len(query.split()) < 5:
            return True
        
        return False
    
    def get_recent_context(self, session_id: Optional[str] = None, max_chars: int = 4000) -> str:
        """Get recent conversation context"""
        if session_id:
            session = self.sessions.get(session_id)
        else:
            session = self.get_or_create_active_session()
        
        if not session:
            return ""
        
        return session.get_context_window(max_chars)
    
    def sync_from_lmstudio(self) -> int:
        """
        Sync conversations from LM Studio
        Returns number of new messages synced
        """
        if not self.lmstudio_conversations.exists():
            return 0
        
        synced_count = 0
        
        try:
            for conv_file in self.lmstudio_conversations.glob("*.json"):
                with open(conv_file, 'r') as f:
                    conv_data = json.load(f)
                
                # Get session ID from filename
                session_id = f"lmstudio_{conv_file.stem}"
                
                # Create session if needed
                if session_id not in self.sessions:
                    self.sessions[session_id] = Session(session_id=session_id)
                
                session = self.sessions[session_id]
                
                # Extract messages
                messages = conv_data.get('messages', [])
                if isinstance(conv_data.get('conversation'), dict):
                    messages = conv_data['conversation'].get('messages', [])
                
                for msg in messages:
                    role = msg.get('role', 'user')
                    
                    # Use enhanced extraction for nested LM Studio structure
                    content = extract_message_content(msg)
                    
                    if not content:
                        continue
                    
                    msg_id = f"{session_id}:{msg.get('id', hashlib.sha256(content.encode()).hexdigest()[:12])}"
                    
                    if msg_id not in self.processed_messages:
                        session.add_message(role, content)
                        self.processed_messages.add(msg_id)
                        synced_count += 1
            
            if synced_count > 0:
                self._save_state()
        
        except Exception as e:
            print(f"[SessionManager] Error syncing from LM Studio: {e}")
        
        return synced_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        total_user = sum(len([m for m in s.messages if m.role == 'user']) for s in self.sessions.values())
        total_assistant = sum(len([m for m in s.messages if m.role == 'assistant']) for s in self.sessions.values())
        
        return {
            'total_sessions': len(self.sessions),
            'active_session': self.active_session_id,
            'total_messages': total_messages,
            'user_messages': total_user,
            'assistant_messages': total_assistant,
            'avg_messages_per_session': total_messages / max(len(self.sessions), 1)
        }
    
    def cleanup_old_sessions(self, days: int = 7):
        """Remove sessions older than N days"""
        cutoff = time.time() - (days * 86400)
        removed = []
        
        for session_id, session in list(self.sessions.items()):
            if session.last_updated < cutoff:
                del self.sessions[session_id]
                removed.append(session_id)
        
        if removed:
            self._save_state()
        
        return removed


# Convenience functions for Phase 3 integration
def create_session_manager(state_dir: Path = Path("./var/sessions")) -> SessionManager:
    """Create a session manager"""
    return SessionManager(state_dir)


def detect_follow_up(query: str, session_manager: SessionManager) -> bool:
    """Convenience function to detect follow-ups"""
    return session_manager.is_follow_up(query)


def get_conversation_context(session_manager: SessionManager, max_chars: int = 4000) -> str:
    """Get recent conversation context"""
    return session_manager.get_recent_context(max_chars=max_chars)


if __name__ == "__main__":
    # Test Phase 3 functionality
    print("=" * 60)
    print("Phase 3: Multi-Turn Memory Test")
    print("=" * 60)
    
    manager = SessionManager(state_dir=Path("./var/sessions"))
    
    # Create session
    sid = manager.create_session()
    print(f"\nâœ… Created session: {sid}")
    
    # Add conversation
    manager.add_message("user", "What is quantum entanglement?")
    manager.add_message("assistant", "Quantum entanglement is a phenomenon where particles become correlated...")
    manager.add_message("user", "How does it work?")
    
    # Test follow-up detection
    is_followup = manager.is_follow_up("How does it work?")
    print(f"âœ… Follow-up detected: {is_followup}")
    
    # Get context
    context = manager.get_recent_context()
    print(f"\nðŸ“ Recent context ({len(context)} chars):")
    print(context[:200] + "...")
    
    # Statistics
    stats = manager.get_statistics()
    print(f"\nðŸ“Š Statistics:")
    print(json.dumps(stats, indent=2))
    
    print("\n" + "=" * 60)
    print("âœ… Phase 3 Multi-Turn Memory - READY")
    print("=" * 60)
