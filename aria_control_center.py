#!/usr/bin/env python3
"""
ARIA Unified Control Center
Command center for Teacher & Student ARIA with integrated corpus learning
"""

import os
import sys
import json
import subprocess
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# ============================================================================
# COLORS & STYLING
# ============================================================================

class C:
    """ANSI color codes"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear():
    """Clear terminal"""
    os.system('clear' if os.name != 'nt' else 'cls')

def color(text: str, code: str) -> str:
    """Apply color to text"""
    if sys.stdout.isatty():
        return f"{code}{text}{C.END}"
    return text

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    def __init__(self):
        self.root = Path(__file__).parent
        self.src = self.root / "src"

        # Teacher ARIA
        self.aria_packs = self.root / "aria_packs"
        self.datasets = self.root.parent / "datasets"

        # Student ARIA
        self.corpus_dir = self.root.parent / "training_data" / "conversation_corpus"
        self.lmstudio_conversations = Path.home() / ".lmstudio" / "conversations"

        # State
        self.bandit_state = Path.home() / ".aria" / "bandit_state.json"
        self.watcher_state = self.root.parent / "var" / "watcher_state.json"

        # Telemetry
        self.telemetry = self.root.parent / "var" / "telemetry"

    def create_dirs(self):
        """Create required directories"""
        self.aria_packs.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry.mkdir(parents=True, exist_ok=True)
        self.bandit_state.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEACHER ARIA - Query Interface
# ============================================================================

class TeacherARIA:
    """Teacher ARIA query interface"""

    def __init__(self, config: Config):
        self.config = config
        self.aria_core = None

    def initialize(self) -> bool:
        """Initialize ARIA core"""
        try:
            sys.path.insert(0, str(self.config.src))
            from core.aria_core import ARIA

            self.aria_core = ARIA(
                index_roots=[str(self.config.datasets)],
                out_root=str(self.config.aria_packs),
                state_path=str(self.config.bandit_state),
                enforce_session=False
            )
            return True
        except Exception as e:
            print(color(f"✗ Failed to initialize ARIA: {e}", C.RED))
            return False

    def query(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Execute query"""
        if not self.aria_core:
            if not self.initialize():
                return None

        if not self.aria_core:  # Type checker satisfaction
            return None

        try:
            result = self.aria_core.query(query_text)
            return result
        except Exception as e:
            print(color(f"✗ Query failed: {e}", C.RED))
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get system stats"""
        stats = {
            "packs_created": 0,
            "total_chunks": 0,
            "bandit_pulls": 0
        }

        # Count packs
        if self.config.aria_packs.exists():
            packs = list(self.config.aria_packs.glob("*/*/last_pack.json"))
            stats["packs_created"] = len(packs)

        # Bandit stats
        if self.config.bandit_state.exists():
            try:
                with open(self.config.bandit_state) as f:
                    bandit = json.load(f)
                    stats["bandit_pulls"] = bandit.get("total_pulls", 0)
            except:
                pass

        return stats

# ============================================================================
# STUDENT ARIA - Corpus Learning
# ============================================================================

class StudentARIA:
    """Student ARIA corpus learning interface"""

    def __init__(self, config: Config):
        self.config = config
        self.watcher_process = None

    def is_watcher_running(self) -> bool:
        """Check if conversation watcher is running"""
        if self.watcher_process and self.watcher_process.poll() is None:
            return True
        return False

    def start_watcher(self) -> bool:
        """Start conversation watcher"""
        if self.is_watcher_running():
            print(color("✓ Watcher already running", C.YELLOW))
            return True

        watcher_script = self.config.root / "conversation_watcher.py"

        if not watcher_script.exists():
            print(color(f"✗ Watcher script not found: {watcher_script}", C.RED))
            print(color(f"  Expected at: {watcher_script}", C.DIM))
            return False

        try:
            self.watcher_process = subprocess.Popen(
                ["python3", str(watcher_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            time.sleep(1)

            if self.watcher_process.poll() is None:
                print(color("✓ Conversation watcher started", C.GREEN))
                return True
            else:
                print(color("✗ Watcher failed to start", C.RED))
                return False
        except Exception as e:
            print(color(f"✗ Failed to start watcher: {e}", C.RED))
            return False

    def stop_watcher(self):
        """Stop conversation watcher"""
        if self.watcher_process and self.watcher_process.poll() is None:
            self.watcher_process.terminate()
            try:
                self.watcher_process.wait(timeout=5)
                print(color("✓ Watcher stopped", C.GREEN))
            except:
                self.watcher_process.kill()
                print(color("✓ Watcher killed", C.YELLOW))
        else:
            print(color("✗ Watcher not running", C.YELLOW))

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        stats = {
            "conversations_captured": 0,
            "total_messages": 0,
            "corpus_size_mb": 0.0
        }

        if self.config.corpus_dir.exists():
            conversations = list(self.config.corpus_dir.glob("*.json"))
            stats["conversations_captured"] = len(conversations)

            total_size = 0
            total_messages = 0

            for conv_file in conversations:
                total_size += conv_file.stat().st_size
                try:
                    with open(conv_file) as f:
                        data = json.load(f)
                        total_messages += len(data.get("messages", []))
                except:
                    pass

            stats["total_messages"] = total_messages
            stats["corpus_size_mb"] = total_size / (1024 * 1024)

        return stats

# ============================================================================
# UNIFIED CONTROL CENTER
# ============================================================================

class ControlCenter:
    """Unified ARIA control center"""

    def __init__(self):
        self.config = Config()
        self.config.create_dirs()
        self.teacher = TeacherARIA(self.config)
        self.student = StudentARIA(self.config)
        self.running = True

    def print_header(self):
        """Print header"""
        clear()
        print(color("╔" + "═" * 68 + "╗", C.CYAN))
        print(color("║", C.CYAN) + color("  🌀 ARIA - Adaptive Resonant Intelligent Architecture", C.BOLD + C.MAGENTA) + " " * 8 + color("║", C.CYAN))
        print(color("╠" + "═" * 68 + "╣", C.CYAN))
        print(color("║", C.CYAN) + f"  {color('👨‍🏫 Teacher', C.GREEN)}: Query & Retrieval      " +
              f"{color('🎓 Student', C.BLUE)}: Corpus Learning      " + color("║", C.CYAN))
        print(color("╚" + "═" * 68 + "╝", C.CYAN))
        print()

    def print_status(self):
        """Print system status"""
        print(color("📊 SYSTEM STATUS", C.BOLD + C.BLUE))
        print(color("─" * 68, C.DIM))

        # Teacher stats
        teacher_stats = self.teacher.get_stats()
        print(f"{color('👨‍🏫 Teacher ARIA:', C.GREEN)}")
        print(f"  • Packs created: {color(str(teacher_stats['packs_created']), C.CYAN)}")
        print(f"  • Bandit pulls: {color(str(teacher_stats['bandit_pulls']), C.CYAN)}")

        # Student stats
        student_stats = self.student.get_corpus_stats()
        watcher_status = "🟢 Running" if self.student.is_watcher_running() else "🔴 Stopped"
        print(f"\n{color('🎓 Student ARIA:', C.BLUE)}")
        print(f"  • Watcher: {watcher_status}")
        print(f"  • Conversations: {color(str(student_stats['conversations_captured']), C.CYAN)}")
        print(f"  • Messages: {color(str(student_stats['total_messages']), C.CYAN)}")
        corpus_size = f"{student_stats['corpus_size_mb']:.2f} MB"
        print(f"  • Corpus size: {color(corpus_size, C.CYAN)}")
        print()

    def print_menu(self):
        """Print main menu"""
        print(color("MAIN MENU", C.BOLD + C.YELLOW))
        print(color("─" * 68, C.DIM))
        print(f"{color('[1]', C.CYAN)} Query Teacher ARIA")
        print(f"{color('[2]', C.CYAN)} Start Student Watcher")
        print(f"{color('[3]', C.CYAN)} Stop Student Watcher")
        print(f"{color('[4]', C.CYAN)} View Corpus Stats")
        print(f"{color('[5]', C.CYAN)} Run Flywheel Test")
        print(f"{color('[6]', C.CYAN)} View Telemetry")
        print(f"{color('[r]', C.CYAN)} Refresh Status")
        print(f"{color('[q]', C.CYAN)} Quit")
        print()

    def query_teacher(self):
        """Interactive query interface"""
        clear()
        self.print_header()
        print(color("👨‍🏫 TEACHER ARIA - QUERY MODE", C.BOLD + C.GREEN))
        print(color("─" * 68, C.DIM))
        print(color("Type 'back' to return to main menu\n", C.DIM))

        while True:
            query = input(color("Query: ", C.CYAN))

            if query.lower() in ['back', 'exit', 'quit']:
                break

            if not query.strip():
                continue

            print(color("\n⏳ Processing...", C.YELLOW))
            start = time.time()

            result = self.teacher.query(query)
            elapsed = time.time() - start

            if result:
                print(color(f"\n✓ Query completed in {elapsed:.2f}s", C.GREEN))
                print(color(f"  • Preset: {result.get('preset', 'unknown')}", C.CYAN))
                print(color(f"  • Run dir: {result.get('run_dir', 'unknown')}", C.DIM))

                # Show pack path
                if "pack" in result:
                    pack_path = Path(result["pack"])
                    if pack_path.exists():
                        try:
                            with open(pack_path) as f:
                                pack = json.load(f)
                                chunks = len(pack.get("items", []))
                                print(color(f"  • Chunks retrieved: {chunks}", C.CYAN))
                        except:
                            pass
            else:
                print(color(f"\n✗ Query failed", C.RED))

            print()

    def view_corpus_stats(self):
        """View detailed corpus statistics"""
        clear()
        self.print_header()
        print(color("🎓 STUDENT ARIA - CORPUS STATISTICS", C.BOLD + C.BLUE))
        print(color("─" * 68, C.DIM))

        stats = self.student.get_corpus_stats()

        print(f"Conversations captured: {color(str(stats['conversations_captured']), C.CYAN)}")
        print(f"Total messages: {color(str(stats['total_messages']), C.CYAN)}")
        corpus_size = f"{stats['corpus_size_mb']:.2f} MB"
        print(f"Corpus size: {color(corpus_size, C.CYAN)}")

        if stats['conversations_captured'] > 0:
            avg_msgs = stats['total_messages'] / stats['conversations_captured']
            print(f"Avg messages per conversation: {color(f'{avg_msgs:.1f}', C.CYAN)}")

        print(f"\nCorpus directory: {color(str(self.config.corpus_dir), C.DIM)}")

        # Show recent conversations
        if self.config.corpus_dir.exists():
            recent = sorted(self.config.corpus_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            if recent:
                print(f"\n{color('Recent conversations:', C.YELLOW)}")
                for i, conv_file in enumerate(recent, 1):
                    timestamp = datetime.fromtimestamp(conv_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"  {i}. {conv_file.name} ({timestamp})")

        input(color("\nPress Enter to continue...", C.DIM))

    def run_flywheel_test(self):
        """Run flywheel test"""
        clear()
        self.print_header()
        print(color("🔥 FLYWHEEL TEST", C.BOLD + C.YELLOW))
        print(color("─" * 68, C.DIM))

        test_script = self.config.root.parent / "aria_systems_test_and_analysis" / "stress_tests" / "test_real_data_flywheel.py"

        if not test_script.exists():
            print(color(f"✗ Test script not found: {test_script}", C.RED))
            input(color("\nPress Enter to continue...", C.DIM))
            return

        print(color("Running flywheel test...\n", C.CYAN))

        try:
            subprocess.run(["python3", str(test_script)], check=True)
        except subprocess.CalledProcessError:
            print(color("\n✗ Test failed", C.RED))
        except KeyboardInterrupt:
            print(color("\n\n⚠ Test interrupted", C.YELLOW))

        input(color("\nPress Enter to continue...", C.DIM))

    def view_telemetry(self):
        """View telemetry logs"""
        clear()
        self.print_header()
        print(color("📈 TELEMETRY", C.BOLD + C.MAGENTA))
        print(color("─" * 68, C.DIM))

        if not self.config.telemetry.exists():
            print(color("No telemetry data available", C.YELLOW))
            input(color("\nPress Enter to continue...", C.DIM))
            return

        log_files = list(self.config.telemetry.glob("*.log"))

        if not log_files:
            print(color("No log files found", C.YELLOW))
        else:
            print(f"Found {len(log_files)} log file(s):\n")
            for i, log in enumerate(log_files, 1):
                size = log.stat().st_size / 1024
                print(f"{i}. {log.name} ({size:.1f} KB)")

        input(color("\nPress Enter to continue...", C.DIM))

    def run(self):
        """Main control loop"""
        # Setup signal handlers
        def signal_handler(sig, frame):
            self.running = False
            print(color("\n\nShutting down...", C.YELLOW))
            self.student.stop_watcher()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            self.print_header()
            self.print_status()
            self.print_menu()

            choice = input(color("Select option: ", C.CYAN)).strip().lower()

            if choice == '1':
                self.query_teacher()
            elif choice == '2':
                self.student.start_watcher()
                time.sleep(2)
            elif choice == '3':
                self.student.stop_watcher()
                time.sleep(1)
            elif choice == '4':
                self.view_corpus_stats()
            elif choice == '5':
                self.run_flywheel_test()
            elif choice == '6':
                self.view_telemetry()
            elif choice == 'r':
                continue
            elif choice == 'q':
                self.running = False
                print(color("\nShutting down...", C.YELLOW))
                self.student.stop_watcher()
                break
            else:
                print(color("Invalid option", C.RED))
                time.sleep(1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point"""
    try:
        center = ControlCenter()
        center.run()
    except KeyboardInterrupt:
        print(color("\n\nInterrupted by user", C.YELLOW))
        sys.exit(0)
    except Exception as e:
        print(color(f"\n\nFatal error: {e}", C.RED))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
