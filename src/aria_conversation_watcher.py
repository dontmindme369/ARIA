#!/usr/bin/env python3
"""
ARIA Conversation Watcher v3 - The Universal Observer
Captures ALL LM Studio conversations to learn cross-domain synthesis patterns.
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



PHILOSOPHY:
- Capture EVERYTHING, not just ARIA matches
- Learn meta-patterns across ALL domains
- Observe how intelligence synthesizes across topics
- Build corpus of thinking patterns, not just retrieval patterns
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Set, Any, List, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path("{DATA_DIR}")
LMSTUDIO_CONVERSATIONS = Path.home() / ".lmstudio" / "conversations"
CORPUS_DIR = Path("{DATA_DIR}")
RAG_RUNS_DIR = Path("{DATA_DIR}")
LOG_FILE = ROOT / "var" / "telemetry" / "conversation_watcher.log"

PACK_LOOKBACK_MINUTES = 15
POLL_INTERVAL_SECONDS = 30

# Create directories
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('aria.watcher')

# ============================================================================
# STATE TRACKING
# ============================================================================

class WatcherState:
    """Track processed messages per file + message count"""
    def __init__(self) -> None:
        self.processed_messages: Set[str] = set()
        self.file_message_counts: Dict[str, int] = {}  # Track how many messages processed per file
        self.state_file: Path = ROOT / "var" / "watcher_state.json"
        self.lock = threading.Lock()
        self.load_state()
    
    def load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data: Dict[str, Any] = json.load(f)
                self.processed_messages = set(data.get('processed_messages', []))
                self.file_message_counts = data.get('file_message_counts', {})
                logger.info(f"Loaded state: {len(self.processed_messages)} messages, {len(self.file_message_counts)} files tracked")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save_state(self) -> None:
        try:
            with self.lock:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'processed_messages': list(self.processed_messages),
                        'file_message_counts': self.file_message_counts,
                        'last_save': datetime.now().isoformat()
                    }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state: {e}")
    
    def is_message_processed(self, message_id: str) -> bool:
        with self.lock:
            return message_id in self.processed_messages
    
    def mark_message_processed(self, message_id: str) -> None:
        with self.lock:
            self.processed_messages.add(message_id)
            if len(self.processed_messages) % 10 == 0:
                self.save_state()
    
    def get_processed_count(self, filepath: str) -> int:
        """Get how many messages we've processed from this file"""
        with self.lock:
            return self.file_message_counts.get(filepath, 0)
    
    def update_processed_count(self, filepath: str, count: int) -> None:
        """Update the message count for this file"""
        with self.lock:
            self.file_message_counts[filepath] = count
            self.save_state()

state = WatcherState()

# ============================================================================
# LM STUDIO MESSAGE EXTRACTION (FIXED FOR NESTED STRUCTURE)
# ============================================================================

def extract_message_content(message: Dict[str, Any]) -> str:
    """
    Extract text content from LM Studio's nested message structure.
    
    Structure: message â†’ versions[] â†’ content[] â†’ text
    
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
            return str(direct_content).strip()
        
        return ''
    
    except Exception as e:
        logger.debug(f"Error extracting message content: {e}")
        return ''

def extract_qa_pairs(conversation: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Extract ALL (query, response, message_id) pairs with proper nested extraction"""
    pairs: List[Tuple[str, str, str]] = []
    
    try:
        messages: List[Dict[str, Any]] = conversation.get('messages', [])
        if not messages:
            logger.debug("No messages array found in conversation")
            return pairs
        
        logger.debug(f"Processing {len(messages)} messages")
        
        for i in range(len(messages) - 1):
            msg_role = messages[i].get('role', '')
            next_role = messages[i+1].get('role', '')
            
            if msg_role == 'user' and next_role == 'assistant':
                # Extract using nested structure parser
                user_content = extract_message_content(messages[i])
                assistant_content = extract_message_content(messages[i+1])
                
                # Get message ID
                msg_id: str = messages[i+1].get('id', '')
                if not msg_id:
                    msg_id = f"{i}_{hash(user_content + assistant_content)}"
                
                # Validate content
                if user_content and assistant_content and len(user_content) > 3 and len(assistant_content) > 10:
                    pairs.append((user_content, assistant_content, msg_id))
                    logger.debug(f"Extracted pair: query={len(user_content)} chars, response={len(assistant_content)} chars")
                else:
                    logger.debug(f"Skipped pair: user={len(user_content)}, assistant={len(assistant_content)}")
        
        return pairs
    
    except Exception as e:
        logger.warning(f"Error extracting QA pairs: {e}", exc_info=True)
        return pairs

# ============================================================================
# PACK MATCHING (Optional Enhancement)
# ============================================================================

def find_matching_pack(query: str, max_age_minutes: int = PACK_LOOKBACK_MINUTES) -> Optional[Dict[str, Any]]:
    """Find pack matching the query - returns None if no match (that's OK!)"""
    
    if not RAG_RUNS_DIR.exists():
        return None
    
    cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
    run_dirs: List[Path] = []
    
    for timestamp_dir in RAG_RUNS_DIR.iterdir():
        if not timestamp_dir.is_dir():
            continue
        for slug_dir in timestamp_dir.iterdir():
            if not slug_dir.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(slug_dir.stat().st_mtime)
                if mtime > cutoff_time:
                    run_dirs.append(slug_dir)
            except Exception:
                continue
    
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    query_lower = query.lower().strip()
    best_match: Optional[Dict[str, Any]] = None
    best_similarity = 0.0
    
    for run_dir in run_dirs[:50]:
        pack_files = list(run_dir.glob("last_pack*.json"))
        
        for pack_file in pack_files:
            try:
                with open(pack_file, 'r', encoding='utf-8') as f:
                    pack_data: Dict[str, Any] = json.load(f)
                
                pack_query: str = pack_data.get('meta', {}).get('query', '')
                if not pack_query:
                    pack_query = pack_data.get('query', '')
                
                if not pack_query:
                    continue
                
                pack_query_lower = pack_query.lower().strip()
                words_query = set(query_lower.split())
                words_pack = set(pack_query_lower.split())
                
                if not words_query or not words_pack:
                    continue
                
                intersection = len(words_query & words_pack)
                union = len(words_query | words_pack)
                similarity = intersection / union if union > 0 else 0.0
                
                if query_lower == pack_query_lower:
                    similarity = 1.0
                
                if similarity > best_similarity and similarity >= 0.7:
                    best_similarity = similarity
                    best_match = {
                        'pack_file': str(pack_file),
                        'pack_data': pack_data,
                        'similarity': similarity,
                        'pack_query': pack_query,
                        'pack_time': datetime.fromtimestamp(pack_file.stat().st_mtime)
                    }
                    
                    if similarity == 1.0:
                        break
            
            except Exception as e:
                logger.debug(f"Error reading pack {pack_file}: {e}")
                continue
        
        if best_similarity == 1.0:
            break
    
    return best_match

# ============================================================================
# CONVERSATION PROCESSING
# ============================================================================

def safe_read_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Safely read JSON file with retries"""
    
    if not filepath.exists():
        return None
    
    try:
        size = filepath.stat().st_size
        if size < 10:
            logger.debug(f"File too small: {filepath.name}")
            return None
    except Exception:
        return None
    
    for attempt in range(3):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.startswith('\ufeff'):
                content = content[1:]
            
            data: Any = json.loads(content)
            
            if not isinstance(data, dict):
                logger.debug(f"Not a dict: {filepath.name}")
                return None
            
            return data
        
        except json.JSONDecodeError as e:
            if attempt < 2:
                logger.debug(f"JSON decode error (attempt {attempt+1}): {e}")
                time.sleep(1.0)
            else:
                logger.debug(f"Failed to parse after 3 attempts: {filepath.name}")
                return None
        
        except Exception as e:
            logger.debug(f"Read error: {e}")
            return None
    
    return None

def process_conversation_file(filepath: Path) -> int:
    """
    Process NEW QA pairs from conversation (only messages added since last check).
    Capture with pack if available, capture without if not.
    Returns count of new captures.
    """
    
    captured_count = 0
    
    try:
        conversation = safe_read_json(filepath)
        if not conversation:
            logger.debug(f"Could not read conversation: {filepath.name}")
            return 0
        
        qa_pairs = extract_qa_pairs(conversation)
        if not qa_pairs:
            logger.debug(f"No QA pairs found in {filepath.name}")
            return 0
        
        # Check how many pairs we've already processed from this file
        filepath_str = str(filepath)
        previously_processed = state.get_processed_count(filepath_str)
        
        # Only process NEW pairs (beyond what we've seen before)
        new_pairs = qa_pairs[previously_processed:]
        
        if not new_pairs:
            logger.debug(f"No new messages in {filepath.name} (already processed {previously_processed})")
            return 0
        
        logger.debug(f"Found {len(new_pairs)} NEW QA pairs in {filepath.name} (total: {len(qa_pairs)}, prev: {previously_processed})")
        
        for user_query, assistant_response, message_id in new_pairs:
            
            if state.is_message_processed(message_id):
                continue
            
            # Try to find matching pack (but don't require it!)
            pack_match = find_matching_pack(user_query)
            
            # Capture regardless of pack match
            if pack_match:
                # ARIA Triple: query + pack + response
                capture_type = "aria_triple"
                capture_data: Dict[str, Any] = {
                    'type': capture_type,
                    'timestamp': datetime.now().isoformat(),
                    'message_id': message_id,
                    'query': user_query,
                    'pack': pack_match['pack_data'],
                    'pack_file': pack_match['pack_file'],
                    'pack_similarity': pack_match['similarity'],
                    'pack_age_minutes': (datetime.now() - pack_match['pack_time']).total_seconds() / 60,
                    'response': assistant_response,
                    'conversation_file': str(filepath)
                }
                logger.info(f"âœ… ARIA Triple captured")
                logger.info(f"  Pack age: {capture_data['pack_age_minutes']:.1f} minutes")
            else:
                # Pure Conversation: query + response (no pack)
                capture_type = "conversation"
                capture_data = {
                    'type': capture_type,
                    'timestamp': datetime.now().isoformat(),
                    'message_id': message_id,
                    'query': user_query,
                    'response': assistant_response,
                    'conversation_file': str(filepath)
                }
                logger.info(f"âœ… Conversation captured (no pack)")
            
            # Save capture
            safe_query = "".join(c if c.isalnum() else "_" for c in user_query[:30])
            filename = f"{capture_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_query}.json"
            capture_path = CORPUS_DIR / filename
            
            with open(capture_path, 'w', encoding='utf-8') as f:
                json.dump(capture_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  Query: {user_query[:60]}...")
            logger.info(f"  Response: {len(assistant_response)} chars")
            logger.info(f"  File: {filename}")
            
            state.mark_message_processed(message_id)
            captured_count += 1
        
        # Update the count of processed messages for this file
        state.update_processed_count(filepath_str, len(qa_pairs))
        
        return captured_count
    
    except Exception as e:
        logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)
        return 0

# ============================================================================
# FILE SYSTEM EVENT HANDLER
# ============================================================================

class ConversationHandler(FileSystemEventHandler):
    """Handle file system events"""
    
    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        
        filepath = Path(str(event.src_path))
        
        if not filepath.name.endswith('.conversation.json'):
            return
        
        logger.debug(f"Detected change: {filepath.name}")
        time.sleep(1.0)
        
        captured = process_conversation_file(filepath)
        if captured > 0:
            logger.info(f"ðŸ”” Event captured {captured} new entry(ies)")
    
    def on_created(self, event: FileSystemEvent) -> None:
        self.on_modified(event)

# ============================================================================
# PERIODIC POLLING
# ============================================================================

def periodic_poll(stop_event: threading.Event) -> None:
    """Periodically check all conversation files"""
    
    logger.info(f"Started periodic polling (every {POLL_INTERVAL_SECONDS}s)")
    
    while not stop_event.is_set():
        try:
            if not LMSTUDIO_CONVERSATIONS.exists():
                stop_event.wait(POLL_INTERVAL_SECONDS)
                continue
            
            conv_files = list(LMSTUDIO_CONVERSATIONS.glob("*.conversation.json"))
            
            if conv_files:
                logger.debug(f"Polling {len(conv_files)} conversation files...")
                
                total_captured = 0
                for conv_file in conv_files:
                    captured = process_conversation_file(conv_file)
                    total_captured += captured
                
                if total_captured > 0:
                    logger.info(f"ðŸ”” Polling captured {total_captured} new entry(ies)")
                
                state.save_state()
        
        except Exception as e:
            logger.error(f"Error in periodic poll: {e}", exc_info=True)
        
        stop_event.wait(POLL_INTERVAL_SECONDS)
    
    logger.info("Stopped periodic polling")

# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    logger.info("=" * 60)
    logger.info("ARIA Universal Observer v3 - Watch Everything")
    logger.info("=" * 60)
    logger.info(f"LM Studio path: {LMSTUDIO_CONVERSATIONS}")
    logger.info(f"Corpus dir: {CORPUS_DIR}")
    logger.info(f"Capture mode: ALL conversations (ARIA + non-ARIA)")
    logger.info(f"Pack lookback: {PACK_LOOKBACK_MINUTES} minutes")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS} seconds")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)
    
    if not LMSTUDIO_CONVERSATIONS.exists():
        logger.error(f"LM Studio conversations directory not found: {LMSTUDIO_CONVERSATIONS}")
        return 1
    
    # Start periodic polling
    stop_event = threading.Event()
    poll_thread = threading.Thread(target=periodic_poll, args=(stop_event,), daemon=True)
    poll_thread.start()
    
    # Set up file watcher
    event_handler = ConversationHandler()
    observer = Observer()
    observer.schedule(event_handler, str(LMSTUDIO_CONVERSATIONS), recursive=False)
    
    try:
        observer.start()
        logger.info("ðŸŒ€ Observing ALL conversations... (Ctrl+C to stop)")
        logger.info("")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping observer...")
        stop_event.set()
        observer.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        stop_event.set()
        observer.stop()
        return 1
    
    observer.join()
    poll_thread.join(timeout=5)
    state.save_state()
    
    logger.info("Observer stopped.")
    return 0

if __name__ == "__main__":
    exit(main())
