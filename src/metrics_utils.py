#!/usr/bin/env python3
"""
metrics_utils.py - Compatibility stub
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Re-exports telemetry functions for backward compatibility
"""

from aria_telemetry import (
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
