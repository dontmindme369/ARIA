#!/usr/bin/env python3
"""
ARIA CLI - Main entry point for command-line usage

Simplified, portable version for GitHub release.
Core functionality: Query → Retrieval → Pack Generation
Optional features can be added incrementally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Core imports
from utils.config_loader import load_config, get_config_value
from utils.paths import get_project_root, ensure_dir

# Import ARIA core (will be ported next)
try:
    from core.aria_core import ARIA
    ARIA_CORE_AVAILABLE = True
except ImportError as e:
    print(f"[ARIA] ERROR: Core ARIA class not available: {e}", file=sys.stderr)
    print(f"[ARIA] Make sure aria_core.py is ported to src/core/", file=sys.stderr)
    ARIA_CORE_AVAILABLE = False


def main():
    """
    Main CLI entry point for ARIA.

    Usage:
        aria_main.py "Your question here" [options]
    """
    parser = argparse.ArgumentParser(
        description="ARIA - Adaptive Retrieval with Intelligent Anchoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python aria_main.py "How does gradient descent work?"

  # With custom config
  python aria_main.py "What is quantum mechanics?" --config my_config.yaml

  # With custom output directory
  python aria_main.py "Explain neural networks" --output ./my_outputs

  # With anchor mode
  python aria_main.py "What is recursion?" --with-anchor

  # With specific preset
  python aria_main.py "History of AI" --preset deep
        """
    )

    # Required: Query
    parser.add_argument(
        "query",
        nargs="+",
        help="Query string (multiple words automatically joined)"
    )

    # Optional: Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ARIA config file (default: aria_config.yaml in project root)"
    )

    # Optional: Enable anchor modes
    parser.add_argument(
        "--with-anchor",
        action="store_true",
        help="Enable 16-anchor reasoning system"
    )

    # Optional: Preset selection
    parser.add_argument(
        "--preset",
        type=str,
        choices=["balanced", "diverse", "focused", "creative", "fast", "thorough", "deep"],
        default=None,
        help="Manual preset override (bypasses bandit selection)"
    )

    # Optional: Output directory
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory (from config)"
    )

    args = parser.parse_args()

    # Join query words
    query = " ".join(args.query)

    # Check if ARIA core is available
    if not ARIA_CORE_AVAILABLE:
        print("\n❌ ERROR: ARIA core not available")
        print("Please port aria_core.py to src/core/ first")
        return 1

    # Load config
    try:
        if args.config:
            config = load_config(args.config)
            print(f"[ARIA] Using config: {args.config}")
        else:
            config = load_config()
            print(f"[ARIA] Using default config: {PROJECT_ROOT / 'aria_config.yaml'}")
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nHint: Create aria_config.yaml in project root")
        print("      or specify --config path/to/config.yaml")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR loading config: {e}")
        return 1

    # Get index roots from config
    index_roots = config['paths']['index_roots']

    # Get output directory (from args or config)
    if args.output:
        from utils.paths import expand_path
        output_dir = expand_path(args.output)
    else:
        output_dir = config['paths']['output_dir']

    # Ensure output directory exists
    output_dir = ensure_dir(output_dir)

    print(f"\n{'='*70}")
    print(f"🎯 ARIA Query")
    print(f"{'='*70}")
    print(f"Query: {query}")
    print(f"Index roots: {index_roots}")
    print(f"Output: {output_dir}")
    print(f"Anchor mode: {'enabled' if args.with_anchor else 'disabled'}")
    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"{'='*70}\n")

    # Create ARIA instance
    try:
        aria = ARIA(
            index_roots=[str(r) for r in index_roots],
            out_root=str(output_dir),
            enforce_session=False,  # No session requirement for CLI
        )
    except Exception as e:
        print(f"\n❌ ERROR initializing ARIA: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run query
    try:
        print("[ARIA] Running query...")
        result = aria.query(
            query,
            with_anchor=args.with_anchor,
            preset_override=args.preset,
        )

        # Print results
        print(f"\n{'='*70}")
        print(f"✅ Query Complete")
        print(f"{'='*70}")

        # Output JSON result
        print(json.dumps(result, indent=2, default=str))

        return 0

    except Exception as e:
        print(f"\n❌ ERROR during query: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
