#!/usr/bin/env python3
"""
metrics_utils.py - Compatibility stub

Re-exports telemetry functions for backward compatibility
"""

from .aria_telemetry import (
    compute_retrieval_stats,
    coverage_score,
    telemetry_line,
    RunMetrics,
    GenerationStats,
    TimingStats,
    detect_issues,
)

__all__ = [
    'compute_retrieval_stats',
    'coverage_score',
    'telemetry_line',
    'RunMetrics',
    'GenerationStats',
    'TimingStats',
    'detect_issues',
]
