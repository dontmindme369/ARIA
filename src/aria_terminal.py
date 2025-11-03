#!/usr/bin/env python3
"""
ARIA Terminal Display - Comprehensive rich output with colors
Shows every component, chunk, metric, and resource usage as ARIA runs
Can be toggled on/off via config
"""
import sys
import time
import psutil
import os
from typing import Dict, Any, List, Union


class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    # Semantic colors
    STAGE = BRIGHT_CYAN
    SUCCESS = BRIGHT_GREEN
    WARNING = BRIGHT_YELLOW
    ERROR = BRIGHT_RED
    INFO = BRIGHT_BLUE
    QUERY = BRIGHT_MAGENTA
    CHUNK = CYAN
    SOURCE = YELLOW
    METRIC = BRIGHT_WHITE
    GOOD = GREEN
    BAD = RED


class NoOpTerminal:
    """No-op terminal for when display is disabled"""
    
    def show_query(self, query: str) -> None:
        pass
    
    def show_bandit_selection(self, preset: str, reason: str, phase: str, feats: Dict) -> None:
        pass
    
    def show_retrieval_start(self, index_roots: List[str], flags: List[str]) -> None:
        pass
    
    def show_chunks_summary(self, chunks: List[Dict], show_all: bool = False) -> None:
        pass
    
    def show_postfilter_start(self, mode: str) -> None:
        pass
    
    def show_postfilter_results(self, raw_count: int, filtered_count: int) -> None:
        pass
    
    def show_metrics(self, metrics: Dict) -> None:
        pass
    
    def show_resources(self) -> None:
        pass
    
    def show_output_paths(self, pack: str, filtered: str) -> None:
        pass
    
    def show_final_summary(self, reward: float, elapsed: float) -> None:
        pass
    
    def show_error(self, stage: str, error: str) -> None:
        pass


class ARIATerminal:
    """Comprehensive terminal visualization for ARIA"""
    
    def __init__(self):
        self.stage_times = {}
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        
    def _hr(self, char="═", color=Colors.DIM):
        """Horizontal rule"""
        print(f"{color}{char * 80}{Colors.RESET}", file=sys.stderr)
    
    def _section(self, title: str, color=Colors.STAGE, icon="▶"):
        """Section header with icon"""
        self._hr(char="═", color=color)
        print(f"{color}{Colors.BOLD}{icon} {title}{Colors.RESET}", file=sys.stderr)
        self._hr(char="─", color=Colors.DIM)
    
    def _subsection(self, title: str, icon="•"):
        """Subsection header"""
        print(f"\n{Colors.INFO}{Colors.BOLD}  {icon} {title}{Colors.RESET}", file=sys.stderr)
    
    def _kv(self, key: str, value: Any, indent=2, key_color=Colors.DIM):
        """Key-value pair"""
        spaces = " " * indent
        print(f"{spaces}{key_color}{key}:{Colors.RESET} {value}", file=sys.stderr)
    
    def _metric(self, key: str, value: Any, unit: str = "", indent=2, good_threshold=None):
        """Metric display with color coding"""
        spaces = " " * indent
        
        # Color code based on value if threshold provided
        value_color = Colors.METRIC
        if good_threshold is not None and isinstance(value, (int, float)):
            value_color = Colors.GOOD if value >= good_threshold else Colors.BAD
        
        print(f"{spaces}{Colors.DIM}{key}:{Colors.RESET} {value_color}{Colors.BOLD}{value}{unit}{Colors.RESET}", file=sys.stderr)
    
    def get_resources(self) -> Dict[str, Any]:
        """Get current CPU and RAM usage"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            
            return {
                "cpu_percent": f"{cpu_percent:.1f}%",
                "ram_mb": f"{mem_mb:.1f}MB"
            }
        except Exception:
            return {"cpu_percent": "N/A", "ram_mb": "N/A"}
    
    def show_query(self, query: str) -> None:
        """Display the query"""
        self._section("QUERY", Colors.QUERY, "🔍")
        
        # Wrap long queries
        max_width = 76
        words = query.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > max_width:
                print(f"  {Colors.BRIGHT_WHITE}{line}{Colors.RESET}", file=sys.stderr)
                line = word
            else:
                line += (" " if line else "") + word
        if line:
            print(f"  {Colors.BRIGHT_WHITE}{line}{Colors.RESET}", file=sys.stderr)
        print(file=sys.stderr)
    
    def show_bandit_selection(self, preset: str, reason: str, phase: str, feats: Dict) -> None:
        """Display bandit selection details"""
        self._section("BANDIT SELECTION", Colors.INFO, "🎯")
        self._kv("preset", f"{Colors.BRIGHT_CYAN}{preset}{Colors.RESET}")
        self._kv("reason", reason)
        self._kv("phase", phase)
        
        # Show key features
        if feats:
            print(f"\n  {Colors.DIM}Query Features:{Colors.RESET}", file=sys.stderr)
            for key in ["length", "has_question", "complexity"]:
                if key in feats:
                    self._kv(key, feats[key], indent=4)
        print(file=sys.stderr)
    
    def show_retrieval_start(self, index_roots: List[str], flags: List[str]) -> None:
        """Display retrieval start"""
        self._section("RETRIEVAL", Colors.STAGE, "📚")
        self._subsection("Index Roots")
        for root in index_roots:
            print(f"    {Colors.SOURCE}{root}{Colors.RESET}", file=sys.stderr)
        print(file=sys.stderr)
    
    def show_chunks_summary(self, chunks: List[Dict], show_all: bool = False) -> None:
        """Display chunk summary"""
        self._subsection(f"Retrieved {len(chunks)} chunks")
        
        if not chunks:
            print(f"    {Colors.WARNING}No chunks retrieved{Colors.RESET}", file=sys.stderr)
            return
        
        # Show first 3 chunks
        display_count = len(chunks) if show_all else min(3, len(chunks))
        for i, chunk in enumerate(chunks[:display_count]):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")[:80]
            
            print(f"    {Colors.CHUNK}[{i+1}]{Colors.RESET} {Colors.SOURCE}{source}{Colors.RESET}", file=sys.stderr)
            print(f"        score: {Colors.METRIC}{score:.3f}{Colors.RESET}", file=sys.stderr)
            print(f"        {Colors.DIM}{text}...{Colors.RESET}", file=sys.stderr)
        
        if len(chunks) > display_count:
            print(f"    {Colors.DIM}...and {len(chunks) - display_count} more{Colors.RESET}", file=sys.stderr)
        print(file=sys.stderr)
    
    def show_postfilter_start(self, mode: str) -> None:
        """Display postfilter start"""
        self._section("POSTFILTER", Colors.WARNING, "🔬")
        self._kv("mode", mode)
        print(file=sys.stderr)
    
    def show_postfilter_results(self, raw_count: int, filtered_count: int) -> None:
        """Display postfilter results"""
        removed = raw_count - filtered_count
        pct = (removed / raw_count * 100) if raw_count > 0 else 0
        
        self._metric("raw chunks", raw_count)
        self._metric("filtered chunks", filtered_count, good_threshold=10)
        self._metric("removed", f"{removed} ({pct:.1f}%)")
        print(file=sys.stderr)
    
    def show_metrics(self, metrics: Dict) -> None:
        """Display comprehensive metrics"""
        self._section("METRICS", Colors.SUCCESS, "📊")
        
        # Retrieval stats
        if "retrieval" in metrics:
            r = metrics["retrieval"]
            self._subsection("Retrieval")
            # Handle both 'total_chunks' and 'total' key names
            total = r.get("total_chunks") or r.get("total", 0)
            self._metric("total chunks", total, indent=4, good_threshold=10)
            self._metric("unique sources", r.get("unique_sources", 0), indent=4, good_threshold=3)
            # Handle both 'diversity_mm' and 'diversity' key names
            diversity = r.get("diversity_mm") or r.get("diversity", 0)
            self._metric("diversity (MM)", f"{diversity:.2f}", indent=4)
            # Handle both 'mean_score' and 'avg_score' key names  
            avg_score = r.get("mean_score") or r.get("avg_score", 0)
            self._metric("avg score", f"{avg_score:.3f}", indent=4)
        
        # Generation stats
        if "generation" in metrics:
            g = metrics["generation"]
            self._subsection("Generation")
            self._metric("coverage", f"{g.get('coverage_score', 0):.2%}", indent=4, good_threshold=0.5)
            self._metric("exemplar fit", f"{g.get('exemplar_fit', 0):.3f}", indent=4, good_threshold=0.7)
        
        # Timing
        if "timing" in metrics:
            t = metrics["timing"]
            self._subsection("Timing")
            self._metric("retrieval", f"{t.get('retrieval_sec', 0):.2f}s", indent=4)
            self._metric("postfilter", f"{t.get('postfilter_sec', 0):.2f}s", indent=4)
            self._metric("total", f"{t.get('total_sec', 0):.2f}s", indent=4)
        
        # Issues
        if "issues" in metrics and metrics["issues"]:
            self._subsection("⚠️  Issues")
            for issue in metrics["issues"]:
                print(f"    {Colors.WARNING}• {issue}{Colors.RESET}", file=sys.stderr)
        
        # Reward
        if "reward" in metrics:
            reward = metrics["reward"]
            self._subsection("Reward")
            self._metric("final reward", f"{reward:.3f}", indent=4, good_threshold=0.7)
        
        print(file=sys.stderr)
    
    def show_resources(self) -> None:
        """Display resource usage"""
        resources = self.get_resources()
        self._subsection("Resources")
        self._kv("CPU", resources["cpu_percent"], indent=4)
        self._kv("RAM", resources["ram_mb"], indent=4)
        print(file=sys.stderr)
    
    def show_output_paths(self, pack: str, filtered: str) -> None:
        """Display output file paths"""
        self._subsection("Output Files")
        self._kv("pack", pack, indent=4, key_color=Colors.SOURCE)
        if pack != filtered:
            self._kv("filtered", filtered, indent=4, key_color=Colors.SOURCE)
        print(file=sys.stderr)
    
    def show_final_summary(self, reward: float, elapsed: float) -> None:
        """Display final summary"""
        self._section("COMPLETE", Colors.SUCCESS, "✓")
        
        reward_color = Colors.GOOD if reward >= 0.7 else Colors.WARNING if reward >= 0.4 else Colors.BAD
        
        print(f"  {Colors.BOLD}Reward:{Colors.RESET} {reward_color}{reward:.3f}{Colors.RESET}", file=sys.stderr)
        print(f"  {Colors.BOLD}Total Time:{Colors.RESET} {Colors.METRIC}{elapsed:.2f}s{Colors.RESET}", file=sys.stderr)
        self._hr(char="═", color=Colors.SUCCESS)
        print(file=sys.stderr)
    
    def show_error(self, stage: str, error: str) -> None:
        """Display error message"""
        self._section(f"ERROR: {stage}", Colors.ERROR, "✗")
        
        # Split long error messages
        max_width = 76
        words = error.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > max_width:
                print(f"  {Colors.ERROR}{line}{Colors.RESET}", file=sys.stderr)
                line = word
            else:
                line += (" " if line else "") + word
        if line:
            print(f"  {Colors.ERROR}{line}{Colors.RESET}", file=sys.stderr)
        
        self._hr(char="═", color=Colors.ERROR)
        print(file=sys.stderr)


# Global terminal instances
_terminal_enabled = None
_terminal_disabled = None


def get_terminal(enabled: bool = True) -> Union[ARIATerminal, NoOpTerminal]:
    """
    Get terminal instance (real or no-op based on enabled flag)
    
    Args:
        enabled: If False, returns NoOpTerminal that does nothing
    
    Returns:
        ARIATerminal or NoOpTerminal instance
    """
    global _terminal_enabled, _terminal_disabled
    
    if enabled:
        if _terminal_enabled is None:
            _terminal_enabled = ARIATerminal()
        return _terminal_enabled
    else:
        if _terminal_disabled is None:
            _terminal_disabled = NoOpTerminal()
        return _terminal_disabled
