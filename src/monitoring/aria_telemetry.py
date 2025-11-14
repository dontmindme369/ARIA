#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aria_telemetry.py — Metrics and telemetry utilities for ARIA.
"""
from __future__ import annotations

import os, json, re, time, statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, overload

# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────
def _tok3(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{3,}", s.lower()))

def _to_list(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return [str(t) for t in x]  # type: ignore
    except Exception:
        return [str(x)]

# ─────────────────────────────────────────────
# dataclasses with full keyword flexibility
# ─────────────────────────────────────────────
@dataclass
class TimingStats:
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

# ─────────────────────────────────────────────
# metrics & telemetry utilities
# ─────────────────────────────────────────────
def coverage_score(exemplars: Optional[Union[str, Sequence[str]]],
                   retrieved: Optional[Union[str, Sequence[str]]]) -> float:
    ex, rv = _to_list(exemplars), _to_list(retrieved)
    if not ex or not rv:
        return 0.0
    R = _tok3(" ".join(rv))
    vals = [len(_tok3(e) & R) / max(1, len(_tok3(e))) for e in ex]
    return float(statistics.mean(vals)) if vals else 0.0

def compute_retrieval_stats(chunks: List[Mapping[str, Any]]) -> GenerationStats:
    if not chunks:
        return GenerationStats()
    total = len(chunks)
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

# ─────────────────────────────────────────────
# telemetry_line overloads (fully tolerant)
# ─────────────────────────────────────────────
@overload
def telemetry_line() -> str: ...
@overload
def telemetry_line(run_id: str = "", *, preset: str = "",
                   metrics: Optional[Mapping[str, Any]] = None,
                   version: str = "", scope: str = "",
                   note: Optional[str] = None) -> str: ...
@overload
def telemetry_line(run_id: RunMetrics, *, preset: str = "",
                   metrics: Optional[Mapping[str, Any]] = None,
                   version: str = "", scope: str = "",
                   note: Optional[str] = None) -> str: ...

def telemetry_line(run_id: Union[str, RunMetrics, None] = None, *, preset: str = "",
                   metrics: Optional[Mapping[str, Any]] = None,
                   version: str = "", scope: str = "",
                   note: Optional[str] = None) -> str:
    """Accepts 0 args, a string run_id, or a RunMetrics instance."""
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

# ─────────────────────────────────────────────
# diagnostics — now with policy argument
# ─────────────────────────────────────────────
def _get_nested(d: Mapping[str, Any], *path: str, default: Any = 0) -> Any:
    cur: Any = d
    for k in path:
        if isinstance(cur, Mapping) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def detect_issues(metrics: Union[Mapping[str, Any], RunMetrics],
                  policy: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Accepts either a flat metrics dict/RunMetrics or a nested run record.
    policy keys (with defaults):
      - slow_seconds: 8.0
      - dup_ratio: 0.6
      - min_unique_sources: 3
      - min_coverage: 0.2
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

    # pull from nested maps if present, else from flat
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
