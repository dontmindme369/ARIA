#!/usr/bin/env python3
"""
aria_telemetry.py â€” Unified telemetry, metrics, and evaluation system
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Consolidates:
- metrics_utils.py (core metrics dataclasses and utilities)
- rag_telemetry.py (telemetry logging)
- telemetry_collect.py (telemetry collection)
- rag_eval_v2.py (comparative evaluation)

Features:
- Flexible metric dataclasses with full keyword support
- Telemetry logging with reward tracking
- Run collection and aggregation
- Comparative evaluation (RR@K, Jaccard, unique docs)
- Issue detection and diagnostics
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, re, statistics, subprocess, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, overload

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Core Metrics Dataclasses (from metrics_utils.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _tok3(s: str) -> set[str]:
    """Extract 3+ char alphanumeric tokens."""
    return set(re.findall(r"[a-zA-Z0-9]{3,}", s.lower()))

def _to_list(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Convert input to list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return [str(t) for t in x]
    except Exception:
        return [str(x)]

@dataclass
class TimingStats:
    """Timing statistics for retrieval operations."""
    total_s: float = 0.0
    scan_s: float = 0.0
    retrieve_s: float = 0.0
    postfilter_s: float = 0.0
    ts: float = field(default_factory=lambda: time.time())
    
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "ts"):
            self.ts = time.time()

@dataclass
class GenerationStats:
    """Statistics about retrieved context."""
    total: int = 0
    unique_sources: int = 0
    mean_len: float = 0.0
    diversity: float = 0.0
    diversity_mm: float = 0.0
    dup_ratio: float = 0.0
    coverage_score: float = 0.0
    exemplar_fit: float = 0.0
    
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

@dataclass
class RunMetrics:
    """Complete metrics for a single retrieval run."""
    exemplar_fit: float = 0.0
    coverage_score: float = 0.0
    semantic_recall: float = 0.0
    reward: float = 0.0
    run_id: str = ""
    query: str = ""
    model: str = ""
    preset: str = ""
    retrieval: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.reward = (
            0.5 * float(getattr(self, "exemplar_fit", 0.0)) +
            0.3 * float(getattr(self, "coverage_score", 0.0)) +
            0.2 * float(getattr(self, "semantic_recall", 0.0))
        )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Metrics Utilities
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def coverage_score(
    exemplars: Optional[Union[str, Sequence[str]]],
    retrieved: Optional[Union[str, Sequence[str]]]
) -> float:
    """
    Calculate coverage score: fraction of exemplar tokens found in retrieved text.
    
    Args:
        exemplars: Query or exemplar text(s)
        retrieved: Retrieved text(s)
    
    Returns:
        Coverage score between 0.0 and 1.0
    """
    ex, rv = _to_list(exemplars), _to_list(retrieved)
    if not ex or not rv:
        return 0.0
    R = _tok3(" ".join(rv))
    vals = [len(_tok3(e) & R) / max(1, len(_tok3(e))) for e in ex]
    return float(statistics.mean(vals)) if vals else 0.0

def compute_retrieval_stats(chunks: List[Mapping[str, Any]]) -> GenerationStats:
    """
    Compute retrieval statistics from chunks.
    
    Args:
        chunks: List of retrieved chunk dictionaries
    
    Returns:
        GenerationStats with diversity and duplicate metrics
    """
    if not chunks:
        return GenerationStats()
    total = len(chunks)
    # Support both v7 (path) and older scripts (source)
    uniq = len({c.get("path") or c.get("source") for c in chunks})
    mean_len = statistics.mean(len(str(c.get("text", ""))) for c in chunks)
    diversity = uniq / max(1, total)
    return GenerationStats(
        total=total,
        unique_sources=uniq,
        mean_len=float(mean_len),
        diversity=float(diversity),
        diversity_mm=float(diversity),
        dup_ratio=float(1.0 - diversity),
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Telemetry Line Formatting
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@overload
def telemetry_line() -> str: ...

@overload
def telemetry_line(
    run_id: str = "",
    *,
    preset: str = "",
    metrics: Optional[Mapping[str, Any]] = None,
    version: str = "",
    scope: str = "",
    note: Optional[str] = None
) -> str: ...

@overload
def telemetry_line(
    run_id: RunMetrics,
    *,
    preset: str = "",
    metrics: Optional[Mapping[str, Any]] = None,
    version: str = "",
    scope: str = "",
    note: Optional[str] = None
) -> str: ...

def telemetry_line(
    run_id: Union[str, RunMetrics, None] = None,
    *,
    preset: str = "",
    metrics: Optional[Mapping[str, Any]] = None,
    version: str = "",
    scope: str = "",
    note: Optional[str] = None
) -> str:
    """
    Format telemetry line as JSON.
    
    Accepts:
    - No args (empty telemetry)
    - String run_id with keyword args
    - RunMetrics instance
    
    Returns:
        JSON string for logging
    """
    if isinstance(run_id, RunMetrics):
        data = asdict(run_id)
    else:
        data = {
            "timestamp": time.time(),
            "run_id": run_id or "",
            "preset": preset,
            "version": version or os.environ.get("RAG_SESSION_VERSION", ""),
            "scope": scope or os.environ.get("RAG_SESSION_SCOPE", ""),
            "metrics": dict(metrics or {}),
        }
        if note:
            data["note"] = note
    return json.dumps(data, ensure_ascii=False)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Issue Detection & Diagnostics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_nested(d: Mapping[str, Any], *path: str, default: Any = 0) -> Any:
    """Extract nested value from dictionary."""
    cur: Any = d
    for k in path:
        if isinstance(cur, Mapping) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def detect_issues(
    metrics: Union[Mapping[str, Any], RunMetrics],
    policy: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """
    Detect quality issues in retrieval metrics.
    
    Args:
        metrics: Flat or nested metrics dict, or RunMetrics instance
        policy: Thresholds dict with keys:
            - slow_seconds: 8.0
            - dup_ratio: 0.6
            - min_unique_sources: 3
            - min_coverage: 0.2
    
    Returns:
        List of issue descriptions
    """
    try:
        as_dict: Dict[str, Any] = asdict(metrics) if isinstance(metrics, RunMetrics) else dict(metrics)
    except Exception:
        as_dict = dict(metrics) if isinstance(metrics, Mapping) else {}
    
    p = {
        "slow_seconds": 8.0,
        "dup_ratio": 0.6,
        "min_unique_sources": 3,
        "min_coverage": 0.2,
    }
    if policy:
        p.update({k: policy[k] for k in policy.keys() if k in p})
    
    # Extract values (nested or flat)
    cov = float(_get_nested(as_dict, "generation", "coverage_score", default=as_dict.get("coverage_score", 0.0)))
    uniq = int(_get_nested(as_dict, "retrieval", "unique_sources", default=as_dict.get("unique_sources", 0)))
    dup = float(_get_nested(as_dict, "retrieval", "dup_ratio", default=as_dict.get("dup_ratio", 0.0)))
    total_s = float(_get_nested(as_dict, "timing", "total_s", default=as_dict.get("total_s", 0.0)))
    
    issues: List[str] = []
    if cov < float(p["min_coverage"]):
        issues.append("Low coverage")
    if uniq < int(p["min_unique_sources"]):
        issues.append("Too few unique sources")
    if dup > float(p["dup_ratio"]):
        issues.append("High duplicate ratio")
    if total_s > float(p["slow_seconds"]):
        issues.append("Slow run")
    return issues

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Telemetry Logging (from rag_telemetry.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _root() -> Path:
    return Path(os.environ.get("ROOT") or os.getcwd())

def _telemetry_dir(base: Optional[str] = None) -> Path:
    if base:
        p = Path(os.path.expanduser(base))
    else:
        p = _root() / "var" / "telemetry"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _write_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _shape_reward(metrics: Mapping[str, Any]) -> float:
    """Calculate composite reward from metrics."""
    ef = float(metrics.get("exemplar_fit", 0.0) or 0.0)
    cov = float(metrics.get("coverage_score", 0.0) or 0.0)
    sem = float(metrics.get("semantic_recall", 0.0) or 0.0)
    return 0.5 * ef + 0.3 * cov + 0.2 * sem

class TelemetryLogger:
    """Telemetry logging with reward state tracking."""
    
    def __init__(self, telemetry_dir: Optional[str] = None):
        self.tdir = _telemetry_dir(telemetry_dir)
        self.log_path = self.tdir / "telemetry_log.jsonl"
        self.state_path = self.tdir / "reward_state.json"
    
    def log_run(
        self,
        run_id: str,
        query: str,
        version: str,
        scope: str,
        preset_name: str,
        metrics: Mapping[str, Any],
        profile: str = "default",
        extra: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log telemetry record and update reward state.
        
        Args:
            run_id: Unique run identifier
            query: Query text
            version: System version
            scope: Session scope
            preset_name: Preset name used
            metrics: Metrics dictionary
            profile: User profile
            extra: Additional fields
        
        Returns:
            Snapshot with log path, state path, reward, and state
        """
        now = time.time()
        reward = _shape_reward(metrics)
        rec = {
            "run_id": run_id,
            "query": query,
            "version": version,
            "scope": scope,
            "preset_name": preset_name,
            "metrics": dict(metrics),
            "profile": profile,
            "reward": reward,
            "timestamp": now,
            **(dict(extra or {}))
        }
        _write_jsonl(self.log_path, rec)
        
        st = _read_json(self.state_path) or {"presets": {}}
        p = st.setdefault("presets", {}).setdefault(preset_name, {"mean_reward": 0.0, "n": 0})
        alpha = float(os.environ.get("RAG_REWARD_ALPHA", "0.20"))
        p["mean_reward"] = (1.0 - alpha) * float(p.get("mean_reward", 0.0)) + alpha * float(reward)
        p["n"] = int(p.get("n", 0)) + 1
        st["updated_at"] = now
        self.state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"log_path": str(self.log_path), "state_path": str(self.state_path), "reward": reward, "state": st}
    
    def show(self, tail: int = 5) -> None:
        """Show recent telemetry and reward state."""
        print(f"[dir] {self.tdir}")
        if self.log_path.exists():
            with self.log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-tail:]
            print("== tail telemetry ==")
            for ln in lines:
                print(ln.rstrip())
        else:
            print("[no telemetry_log.jsonl]")
        if self.state_path.exists():
            print("== reward_state.json ==")
            print(self.state_path.read_text(encoding="utf-8"))
        else:
            print("[no reward_state.json]")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Evaluation (from rag_eval_v2.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_pack(path: str) -> Tuple[str, Dict[str, Any], List[Tuple[str, float, Dict[str, Any]]]]:
    """Load pack file and extract query, metadata, and scored items."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    q = str(data.get("query", "")).strip()
    items = []
    for it in data.get("items", []):
        doc_id = f"{it.get('path','')}#{it.get('chunk_index',0)}"
        score = float(it.get("score", 0.0))
        items.append((doc_id, score, it))
    meta = data.get("metadata", {})
    return q, meta, items

def parse_runs(run_args: Sequence[str]) -> Dict[str, List[str]]:
    """Parse run specifications into name -> paths mapping."""
    out: Dict[str, List[str]] = {}
    for spec in run_args:
        if "=" not in spec:
            raise SystemExit(f"--run expects name=path, got: {spec}")
        name, path = spec.split("=", 1)
        path = path.strip()
        name = name.strip()
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")
        paths: List[str] = []
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    if fn.endswith(".json"):
                        paths.append(os.path.join(root, fn))
            if not paths and os.path.isfile(os.path.join(path, "last_pack.json")):
                paths.append(os.path.join(path, "last_pack.json"))
        else:
            paths.append(path)
        out[name] = sorted(paths)
    return out

def topk(items: List[Tuple[str, float, Dict[str, Any]]], k: int) -> List[str]:
    """Extract top-k document IDs."""
    return [doc for doc, _, _ in items[:k]]

def jaccard(a: List[str], b: List[str]) -> float:
    """Calculate Jaccard similarity between two lists."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / float(len(A | B)) if (A | B) else 0.0

def rel_recall(run_docs: List[str], gt_docs: List[str]) -> float:
    """Calculate relative recall against ground truth."""
    if not gt_docs:
        return 0.0
    return len(set(run_docs) & set(gt_docs)) / float(len(set(gt_docs)))

def is_pdf_noise(item: Dict[str, Any]) -> bool:
    """Detect PDF garbage in text."""
    txt = str(item.get("text", ""))[:200]
    return bool(re.search(r"%PDF-\d\.\d|xref|obj\s+\d+\s+\d+\s+obj", txt, re.I))

def is_self(doc_id: str) -> bool:
    """Check if document is self-referential (from runs/)."""
    return "/runs/" in doc_id.replace("\\", "/")

class Evaluator:
    """Comparative evaluation across multiple runs."""
    
    def __init__(self, out_dir: str = "./runs/eval_report_v2"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def eval_runs(
        self,
        run_specs: Sequence[str],
        k: int,
        baseline: str
    ) -> None:
        """
        Evaluate multiple runs with comparative metrics.
        
        Args:
            run_specs: List of "name=path" specifications
            k: Top-k cutoff for metrics
            baseline: Baseline run name for comparison
        """
        run_to_packs = parse_runs(run_specs)
        
        # Group by query
        groups: Dict[str, Dict[str, List[Tuple[str, float, Dict[str, Any]]]]] = {}
        for rn, paths in run_to_packs.items():
            for p in paths:
                try:
                    q, meta, items = load_pack(p)
                except Exception as e:
                    print(f"[warn] skip {p}: {e}")
                    continue
                groups.setdefault(q, {})[rn] = items
        
        rows: List[Dict[str, Any]] = []
        for q, rmap in groups.items():
            base_name = baseline if baseline in rmap else next(iter(rmap))
            base_docs = topk(rmap[base_name], k)
            universe = sorted(set(d for rn, items in rmap.items() for d in topk(items, k)))
            
            for rn, items in rmap.items():
                docs = topk(items, k)
                rr = rel_recall(docs, universe)
                jac = jaccard(docs, base_docs)
                uniq = sorted(set(docs) - set(base_docs))
                self_k = sum(1 for d in docs if is_self(d))
                pdf_k = 0
                for d, s, it in items[:k]:
                    if is_pdf_noise(it):
                        pdf_k += 1
                rows.append({
                    "query": q,
                    "run": rn,
                    f"RR@{k}": round(rr, 3),
                    f"Jaccard@{k}_vs_{base_name}": round(jac, 3),
                    f"Unique@{k}_vs_{base_name}": len(uniq),
                    "Self@K": self_k,
                    "PDF@K": pdf_k
                })
            
            # Per-query diff
            safe_q = hashlib.sha1(q.encode("utf-8")).hexdigest()[:10]
            diff_path = self.out_dir / f"diff_{safe_q}.md"
            lines = [f"# Query: {q}\n", f"Baseline: {base_name}\n"]
            for rn, items in rmap.items():
                docs = topk(items, k)
                uniq = sorted(set(docs) - set(base_docs))
                lines.append(f"## {rn}\n- Unique vs {base_name} ({len(uniq)}):")
                for u in uniq[:20]:
                    lines.append(f"  - {u}")
                lines.append("")
            diff_path.write_text("\n".join(lines), encoding="utf-8")
        
        # Summary CSV
        csv_path = self.out_dir / "summary.csv"
        if rows:
            cols = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                [w.writerow(r) for r in rows]
        print(f"[eval] wrote {csv_path} for {len(groups)} queries")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    ap = argparse.ArgumentParser(description="ARIA Telemetry - metrics, logging, evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # Log command
    log_ap = sub.add_parser("log", help="Log telemetry record")
    log_ap.add_argument("--run-id", required=True)
    log_ap.add_argument("--query", default="")
    log_ap.add_argument("--version", default=os.environ.get("RAG_SESSION_VERSION", "unknown"))
    log_ap.add_argument("--scope", default=os.environ.get("RAG_SESSION_SCOPE", "unknown"))
    log_ap.add_argument("--preset", required=True)
    log_ap.add_argument("--metrics", default="")
    log_ap.add_argument("--profile", default="default")
    log_ap.add_argument("--dir", default=None)
    
    # Show command
    show_ap = sub.add_parser("show", help="Show recent telemetry")
    show_ap.add_argument("--dir", default=None)
    show_ap.add_argument("--tail", type=int, default=5)
    
    # Eval command
    eval_ap = sub.add_parser("eval", help="Evaluate runs comparatively")
    eval_ap.add_argument("--run", action="append", required=True, help="name=path_to_pack_or_dir")
    eval_ap.add_argument("--k", type=int, default=10)
    eval_ap.add_argument("--baseline", type=str, required=True)
    eval_ap.add_argument("--out-dir", type=str, default="./runs/eval_report_v2")
    
    args = ap.parse_args()
    
    if args.cmd == "log":
        metrics: Dict[str, Any] = {}
        if args.metrics:
            try:
                metrics = json.loads(args.metrics)
            except Exception:
                pass
        logger = TelemetryLogger(args.dir)
        snap = logger.log_run(
            run_id=args.run_id,
            query=args.query,
            version=args.version,
            scope=args.scope,
            preset_name=args.preset,
            metrics=metrics,
            profile=args.profile
        )
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    
    elif args.cmd == "show":
        logger = TelemetryLogger(args.dir)
        logger.show(args.tail)
    
    elif args.cmd == "eval":
        evaluator = Evaluator(args.out_dir)
        evaluator.eval_runs(args.run, args.k, args.baseline)

if __name__ == "__main__":
    main()
