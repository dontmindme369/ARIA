#!/usr/bin/env python3
"""
ARIA Scoring Daemon - Background Conversation Quality Scorer

Hybrid Learning Architecture:
- Watches scoring queue for new conversations to score
- Scores conversations asynchronously (no query latency)
- Updates bandit rewards in real-time
- Maintains scoring statistics and status

Usage:
    # Start daemon
    python3 aria_scoring_daemon.py start
    
    # Stop daemon
    python3 aria_scoring_daemon.py stop
    
    # Check status
    python3 aria_scoring_daemon.py status
"""

import sys
import os
import json
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict

# ARIA paths
ARIA_ROOT = Path("/media/notapplicable/Internal-SSD/ai-quaternions-model")
ARIA_VAR = ARIA_ROOT / "var"
SCORING_QUEUE = ARIA_VAR / "scoring_queue.jsonl"
DAEMON_STATUS = ARIA_VAR / "scoring_daemon_status.json"
DAEMON_PID = ARIA_VAR / "scoring_daemon.pid"
DAEMON_LOG = ARIA_VAR / "telemetry" / "scoring_daemon.log"

# Ensure directories exist
ARIA_VAR.mkdir(parents=True, exist_ok=True)
DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)

# Add ARIA to path
if str(ARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT))
if str(ARIA_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ARIA_ROOT / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(DAEMON_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('aria.scoring_daemon')


@dataclass
class DaemonStatus:
    """Daemon status information"""
    running: bool = False
    pid: Optional[int] = None
    started_at: Optional[float] = None
    last_activity: Optional[float] = None
    conversations_scored: int = 0
    total_score_sum: float = 0.0
    bandit_updates: int = 0
    errors: int = 0
    queue_size: int = 0
    
    @property
    def avg_score(self) -> float:
        """Average conversation score"""
        if self.conversations_scored == 0:
            return 0.0
        return self.total_score_sum / self.conversations_scored
    
    @property
    def uptime_hours(self) -> float:
        """Hours since daemon started"""
        if not self.started_at:
            return 0.0
        return (time.time() - self.started_at) / 3600
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        d = asdict(self)
        d['avg_score'] = self.avg_score
        d['uptime_hours'] = self.uptime_hours
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DaemonStatus':
        """Load from dict"""
        return cls(
            running=data.get('running', False),
            pid=data.get('pid'),
            started_at=data.get('started_at'),
            last_activity=data.get('last_activity'),
            conversations_scored=data.get('conversations_scored', 0),
            total_score_sum=data.get('total_score_sum', 0.0),
            bandit_updates=data.get('bandit_updates', 0),
            errors=data.get('errors', 0),
            queue_size=data.get('queue_size', 0)
        )


class ScoringDaemon:
    """Background scoring daemon"""
    
    def __init__(self):
        self.status = self.load_status()
        self.running = False
        self.seen_conversations: Set[str] = set()
        
        # Import scorer (lazy to avoid import errors during status checks)
        try:
            from analysis.conversation_scorer import ConversationScorer
            self.scorer = ConversationScorer(save_to_corpus=True, verbose=False)
        except ImportError as e:
            logger.error(f"Failed to import ConversationScorer: {e}")
            self.scorer = None
    
    def load_status(self) -> DaemonStatus:
        """Load daemon status from file"""
        if DAEMON_STATUS.exists():
            try:
                with open(DAEMON_STATUS) as f:
                    data = json.load(f)
                return DaemonStatus.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load status: {e}")
        return DaemonStatus()
    
    def save_status(self):
        """Save daemon status to file"""
        try:
            with open(DAEMON_STATUS, 'w') as f:
                json.dump(self.status.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def write_pid(self):
        """Write PID file"""
        try:
            DAEMON_PID.write_text(str(self.status.pid))
        except Exception as e:
            logger.error(f"Failed to write PID: {e}")
    
    def remove_pid(self):
        """Remove PID file"""
        try:
            if DAEMON_PID.exists():
                DAEMON_PID.unlink()
        except Exception as e:
            logger.error(f"Failed to remove PID: {e}")
    
    def check_queue_size(self) -> int:
        """Count items in scoring queue"""
        if not SCORING_QUEUE.exists():
            return 0
        try:
            with open(SCORING_QUEUE) as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def read_queue(self) -> List[Dict]:
        """Read all items from queue"""
        if not SCORING_QUEUE.exists():
            return []
        
        items = []
        try:
            with open(SCORING_QUEUE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in queue: {line[:50]}...")
        except Exception as e:
            logger.error(f"Failed to read queue: {e}")
        
        return items
    
    def clear_queue(self):
        """Clear the scoring queue"""
        try:
            if SCORING_QUEUE.exists():
                SCORING_QUEUE.unlink()
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
    
    def score_queued_conversations(self):
        """Process all queued conversations"""
        if not self.scorer:
            logger.error("Scorer not available")
            return
        
        items = self.read_queue()
        if not items:
            return
        
        logger.info(f"Processing {len(items)} queued conversation markers")
        
        # Get unique conversation IDs
        conversation_ids = set()
        for item in items:
            conv_id = item.get('conversation_id')
            if conv_id and conv_id not in self.seen_conversations:
                conversation_ids.add(conv_id)
        
        # Score each conversation
        for conv_id in conversation_ids:
            try:
                # Get conversation file
                conv_path = Path.home() / ".lmstudio" / "conversations" / f"{conv_id}.json"
                
                if not conv_path.exists():
                    logger.warning(f"Conversation file not found: {conv_path}")
                    continue
                
                # Load conversation
                with open(conv_path) as f:
                    conversation = json.load(f)
                
                # Extract Q&A pairs
                qa_pairs = self.scorer.extract_qa_pairs(conversation)
                
                if not qa_pairs:
                    logger.debug(f"No Q&A pairs in conversation {conv_id}")
                    continue
                
                # Score each pair
                scores = []
                for query, response in qa_pairs:
                    score = self.scorer.score_response(query, response, context_used=True)
                    scores.append(score)
                
                # Calculate average
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Try to find preset used and extract query features
                preset, query_features = self.scorer.find_preset_for_query(
                    qa_pairs[0][0],  # First query
                    conversation.get('createdAt', time.time()) / 1000  # Convert to seconds
                )

                # Update bandit if preset found
                if preset:
                    try:
                        from bandit_context import give_reward
                        give_reward(preset, avg_score, features=query_features)  # Pass features for LinUCB
                        self.status.bandit_updates += 1
                        logger.info(f"Updated bandit for preset '{preset}' with score {avg_score:.3f}")
                    except Exception as e:
                        logger.error(f"Failed to update bandit: {e}")
                
                # Update stats
                self.status.conversations_scored += 1
                self.status.total_score_sum += avg_score
                self.status.last_activity = time.time()
                
                # Mark as seen
                self.seen_conversations.add(conv_id)
                
                logger.info(f"Scored conversation {conv_id}: {avg_score:.3f} ({len(qa_pairs)} pairs)")
                
            except Exception as e:
                logger.error(f"Failed to score conversation {conv_id}: {e}")
                self.status.errors += 1
        
        # Clear queue after processing
        self.clear_queue()
        
        # Save updated status
        self.save_status()
    
    def run(self):
        """Main daemon loop"""
        logger.info("ARIA Scoring Daemon starting...")
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)
        
        # Update status
        self.status.running = True
        self.status.pid = os.getpid()
        self.status.started_at = time.time()
        self.status.last_activity = time.time()
        self.save_status()
        self.write_pid()
        
        self.running = True
        logger.info(f"Daemon started (PID: {self.status.pid})")
        
        # Main loop
        check_interval = 10  # Check queue every 10 seconds
        
        while self.running:
            try:
                # Check queue size
                self.status.queue_size = self.check_queue_size()
                
                # Process queue if not empty
                if self.status.queue_size > 0:
                    logger.info(f"Queue has {self.status.queue_size} items")
                    self.score_queued_conversations()
                
                # Save status periodically
                self.save_status()
                
                # Sleep
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
                self.status.errors += 1
                self.save_status()
                time.sleep(check_interval)
        
        # Cleanup on shutdown
        self.shutdown()
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Daemon shutting down...")
        
        # Update status
        self.status.running = False
        self.status.pid = None
        self.save_status()
        
        # Remove PID file
        self.remove_pid()
        
        logger.info("Daemon stopped")


# ============================================================================
# CLI Commands
# ============================================================================

def start_daemon():
    """Start the scoring daemon"""
    # Check if already running
    if DAEMON_PID.exists():
        try:
            pid = int(DAEMON_PID.read_text().strip())
            # Check if process is actually running
            import psutil
            if psutil.pid_exists(pid):
                print(f"✗ Daemon already running (PID: {pid})")
                return 1
        except:
            # PID file exists but process is dead, clean it up
            DAEMON_PID.unlink()
    
    print("Starting ARIA Scoring Daemon...")
    
    # Fork to background
    if os.fork() > 0:
        # Parent process
        time.sleep(1)  # Give daemon time to start
        
        if DAEMON_PID.exists():
            pid = int(DAEMON_PID.read_text().strip())
            print(f"✓ Daemon started (PID: {pid})")
            print(f"  Log: {DAEMON_LOG}")
            print(f"  Status: {DAEMON_STATUS}")
            return 0
        else:
            print("✗ Failed to start daemon")
            return 1
    
    # Child process (daemon)
    os.setsid()  # Create new session
    
    # Redirect stdio
    sys.stdin.close()
    sys.stdout = open(DAEMON_LOG.parent / "daemon_stdout.log", 'a')
    sys.stderr = open(DAEMON_LOG.parent / "daemon_stderr.log", 'a')
    
    # Run daemon
    daemon = ScoringDaemon()
    daemon.run()
    
    return 0


def stop_daemon():
    """Stop the scoring daemon"""
    if not DAEMON_PID.exists():
        print("✗ Daemon is not running")
        return 1
    
    try:
        pid = int(DAEMON_PID.read_text().strip())
        print(f"Stopping daemon (PID: {pid})...")
        
        # Send SIGTERM
        import os
        os.kill(pid, signal.SIGTERM)
        
        # Wait for shutdown
        for _ in range(10):
            time.sleep(0.5)
            if not DAEMON_PID.exists():
                print("✓ Daemon stopped")
                return 0
        
        # Force kill if still running
        try:
            os.kill(pid, signal.SIGKILL)
            print("✓ Daemon forcefully stopped")
        except:
            pass
        
        # Clean up PID file
        if DAEMON_PID.exists():
            DAEMON_PID.unlink()
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to stop daemon: {e}")
        return 1


def daemon_status():
    """Show daemon status"""
    status = ScoringDaemon().load_status()
    
    print("=" * 60)
    print("ARIA Scoring Daemon Status".center(60))
    print("=" * 60)
    print()
    
    if status.running and status.pid:
        # Check if process actually exists
        try:
            import psutil
            if psutil.pid_exists(status.pid):
                print(f"  Status: ✓ RUNNING")
                print(f"  PID: {status.pid}")
            else:
                print(f"  Status: ✗ STOPPED (stale PID file)")
                print(f"  PID: {status.pid} (not found)")
        except ImportError:
            print(f"  Status: RUNNING (PID: {status.pid})")
    else:
        print(f"  Status: ✗ STOPPED")
    
    print()
    
    if status.started_at:
        started = datetime.fromtimestamp(status.started_at)
        print(f"  Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Uptime: {status.uptime_hours:.1f} hours")
    
    if status.last_activity:
        last = datetime.fromtimestamp(status.last_activity)
        print(f"  Last Activity: {last.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print()
    print("  Statistics:")
    print(f"    Conversations Scored: {status.conversations_scored}")
    print(f"    Average Score: {status.avg_score:.3f}")
    print(f"    Bandit Updates: {status.bandit_updates}")
    print(f"    Errors: {status.errors}")
    print(f"    Queue Size: {status.queue_size}")
    print()
    print(f"  Files:")
    print(f"    Status: {DAEMON_STATUS}")
    print(f"    Log: {DAEMON_LOG}")
    print(f"    Queue: {SCORING_QUEUE}")
    print()
    print("=" * 60)
    
    return 0


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: aria_scoring_daemon.py {start|stop|status|restart}")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        return start_daemon()
    elif command == 'stop':
        return stop_daemon()
    elif command == 'status':
        return daemon_status()
    elif command == 'restart':
        stop_daemon()
        time.sleep(1)
        return start_daemon()
    else:
        print(f"Unknown command: {command}")
        print("Usage: aria_scoring_daemon.py {start|stop|status|restart}")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())
