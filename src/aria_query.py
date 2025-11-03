#!/usr/bin/env python3
"""
aria_query.py – LM Studio Backend Interface with Exploration

Uses consolidated ARIA modules to retrieve relevant context for queries.
Called by LM Studio's TypeScript preprocessor.

NEW: Includes quaternion state exploration for semantic reranking

Usage:
    python3 aria_query.py "your query here" [--index path/to/data] [--top-k 32]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "src").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import consolidated ARIA modules
from aria_retrieval import ARIARetrieval
from aria_postfilter import ARIAPostfilter

# Import exploration components (optional)
try:
    from aria_exploration import ExplorationManager
    EXPLORATION_AVAILABLE = True
except ImportError:
    EXPLORATION_AVAILABLE = False
    ExplorationManager = None


def format_for_lmstudio(retrieval_result: Dict[str, Any], filtered_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format ARIA results for LM Studio consumption"""
    stats_after = filtered_result.get('stats_after', {})
    
    return {
        "query": retrieval_result['query'],
        "items": filtered_result['items'],
        "meta": {
            "retrieved": retrieval_result['meta']['total_chunks'],
            "filtered": len(filtered_result['items']),
            "unique_sources": stats_after.get('unique_sources', 0),
            "avg_quality": stats_after.get('avg_quality', 0.0),
            "total_tokens": stats_after.get('total_tokens', 0),
        },
        "pack_file": filtered_result.get('pack_file', ''),
        "exploration": filtered_result.get('exploration', None)
    }


def main():
    parser = argparse.ArgumentParser(description="ARIA Query Interface for LM Studio")
    parser.add_argument('query', help="Query text")
    parser.add_argument('--index', default='./data', 
                       help="Index roots (colon-separated)")
    parser.add_argument('--top-k', type=int, default=32, help="Number of chunks to retrieve")
    parser.add_argument('--max-per-file', type=int, default=6, help="Max chunks per file")
    parser.add_argument('--prefer', default='datasets,_ingested', help="Boost these directories")
    parser.add_argument('--out-dir', default='./output',
                       help="Output directory for pack files")
    parser.add_argument('--enable-exploration', action='store_true',
                       help="Enable quaternion state exploration")
    parser.add_argument('--exploration-state-dir', default='./var/exploration',
                       help="Directory for exploration state")
    
    args = parser.parse_args()
    
    try:
        # Parse index roots
        roots = [Path(r.strip()).expanduser() for r in args.index.replace(',', ':').split(':') if r.strip()]
        roots = [r for r in roots if r.exists()]
        
        if not roots:
            print(json.dumps({"error": "No valid index roots found"}), file=sys.stderr)
            sys.exit(1)
        
        # Initialize retriever
        retriever = ARIARetrieval(
            index_roots=roots,
            prefer_dirs=args.prefer.split(','),
            max_per_file=args.max_per_file
        )
        
        # Retrieve chunks
        retrieval_result = retriever.retrieve(args.query, top_k=args.top_k)
        
        # EXPLORATION: Apply quaternion state exploration if enabled
        exploration_metadata: Optional[Dict[str, Any]] = None
        if args.enable_exploration and EXPLORATION_AVAILABLE and ExplorationManager is not None:
            try:
                # Initialize exploration manager
                exploration_state_dir = Path(args.exploration_state_dir)
                exploration_state_dir.mkdir(parents=True, exist_ok=True)
                
                explorer = ExplorationManager(
                    state_dir=exploration_state_dir,
                    enable_quaternion=True,
                    enable_pca=False
                )
                
                # Fit semantic model if needed (first query)
                if explorer.quaternion_mgr is not None:
                    # Check if model is fitted using the is_fitted property
                    if not explorer.quaternion_mgr.is_fitted:
                        fit_texts = [item.get('text', '') for item in retrieval_result['items'][:100]]
                        explorer.fit_semantic_model(fit_texts)
                        print(f"[ARIA] Fitted 4D semantic model on {len(fit_texts)} documents", file=sys.stderr)
                
                # Apply exploration
                result_with_exploration = explorer.explore(
                    query=args.query,
                    retrieval_results=retrieval_result,
                    strategy="golden_ratio_spiral",
                    momentum_weight=0.25,
                    recall_similar=True,
                    save_state=True
                )
                
                # Update retrieval result with explored items
                retrieval_result = result_with_exploration
                exploration_metadata = result_with_exploration.get('exploration', {})
                
                if exploration_metadata and exploration_metadata.get('applied'):
                    print(f"[ARIA] Quaternion exploration applied: {exploration_metadata.get('message', '')}", file=sys.stderr)
                    
            except Exception as e:
                print(f"[ARIA] Warning: Exploration failed: {e}", file=sys.stderr)
        
        # Initialize postfilter
        postfilter = ARIAPostfilter(
            max_per_source=args.max_per_file,
            min_keep=10
        )
        
        # Filter chunks
        filtered_result = postfilter.filter(
            items=retrieval_result['items'],
            enable_quality=True,
            enable_topic=True,
            enable_diversity=True
        )
        
        # Add exploration metadata to filtered result
        if exploration_metadata:
            filtered_result['exploration'] = exploration_metadata
        
        # Save pack file
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        pack_file = out_dir / f"pack_{timestamp}.json"
        
        pack_data = {
            'query': args.query,
            'items': filtered_result['items'],
            'meta': {
                **retrieval_result['meta'],
                'postfilter_stats': filtered_result['stats_after']
            },
            'exploration': exploration_metadata,
            'timestamp': timestamp
        }
        
        with open(pack_file, 'w') as f:
            json.dump(pack_data, f, indent=2)
        
        # Also save as last_pack for compatibility
        last_pack = out_dir / "last_pack.filtered.json"
        with open(last_pack, 'w') as f:
            json.dump(pack_data, f, indent=2)
        
        # Add pack_file to result
        filtered_result['pack_file'] = str(pack_file)
        
        # Format for LM Studio
        result = format_for_lmstudio(retrieval_result, filtered_result)
        
        # Output JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "query": args.query,
            "items": [],
            "meta": {}
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
