#!/usr/bin/env python3
"""
ARIA v7 Hybrid Retrieval - Lexical + Semantic with Quaternion Rotation

Combines:
1. Lexical BM25 retrieval (fast, keyword-based)
2. Semantic embeddings (sentence-transformers, optional)
3. Quaternion rotation exploration (perspective-aware)
4. Hybrid score fusion

This is the COMPLETE pipeline that actually uses the quaternion math.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import subprocess
import json
import numpy as np

# Add intelligence to path for quaternion access
SCRIPT_DIR = Path(__file__).parent
ARIA_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ARIA_ROOT / "src"))

try:
    from intelligence.aria_exploration import QuaternionExplorer
    HAVE_QUATERNION = True
except ImportError:
    HAVE_QUATERNION = False
    print("[ARIA] Warning: Quaternion exploration not available", file=sys.stderr)

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False


class HybridRetriever:
    """
    Hybrid retrieval combining lexical (BM25) and semantic (embeddings + quaternions)
    """

    def __init__(
        self,
        use_semantic: bool = True,
        use_quaternion: bool = True,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid retriever

        Args:
            use_semantic: Enable semantic embeddings
            use_quaternion: Enable quaternion rotation
            model_name: Sentence-transformer model name
        """
        self.use_semantic = use_semantic and HAVE_EMBEDDINGS
        self.use_quaternion = use_quaternion and HAVE_QUATERNION

        # Load embedding model if available
        self.encoder = None
        if self.use_semantic:
            try:
                self.encoder = SentenceTransformer(model_name)
                print(f"[ARIA] Loaded embedding model: {model_name}", file=sys.stderr)
            except Exception as e:
                print(f"[ARIA] Failed to load embeddings: {e}", file=sys.stderr)
                self.use_semantic = False

        # Initialize quaternion explorer
        self.explorer = None
        if self.use_quaternion and HAVE_QUATERNION:
            self.explorer = QuaternionExplorer(embedding_dim=384)  # MiniLM dimension
            print("[ARIA] Quaternion exploration enabled", file=sys.stderr)

    def retrieve_hybrid(
        self,
        query: str,
        pack_path: str,
        num_rotations: int = 0,
        rotation_angle: float = 30.0,
        rotation_subspace: int = 0,
        alpha_lexical: float = 0.7,
        alpha_semantic: float = 0.3,
    ) -> Dict:
        """
        Hybrid retrieval with optional quaternion rotation

        Args:
            query: Search query
            pack_path: Path to lexical pack JSON from v7
            num_rotations: Number of quaternion rotations (0 = disable)
            rotation_angle: Rotation angle per step (degrees)
            rotation_subspace: Subspace index for rotation axis
            alpha_lexical: Weight for lexical scores
            alpha_semantic: Weight for semantic scores

        Returns:
            Enhanced pack with hybrid scores
        """
        # Load lexical results
        try:
            with open(pack_path, 'r') as f:
                pack = json.load(f)
        except Exception as e:
            print(f"[ARIA] Failed to load pack: {e}", file=sys.stderr)
            return {}

        items = pack.get("items", [])
        if not items:
            return pack

        # If semantic disabled or not available, return lexical-only
        if not self.use_semantic or not self.encoder:
            print("[ARIA] Using lexical-only retrieval", file=sys.stderr)
            return pack

        # Extract texts
        texts = [item.get("text", "") for item in items]

        # Encode query and documents
        try:
            query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
            doc_embs = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            print(f"[ARIA] Encoding failed: {e}", file=sys.stderr)
            return pack

        # Apply quaternion rotation if enabled
        if self.use_quaternion and self.explorer and num_rotations > 0:
            print(f"[ARIA] Applying {num_rotations} quaternion rotations ({rotation_angle}°, subspace={rotation_subspace})", file=sys.stderr)

            # Multi-rotation exploration returns (doc_idx, semantic_score) tuples
            rotation_results = self.explorer.multi_rotation_exploration(
                query_embedding=query_emb,
                document_embeddings=doc_embs,
                num_rotations=num_rotations,
                angle_per_rotation=rotation_angle,
                subspace_index=rotation_subspace
            )

            # Extract semantic scores
            semantic_scores = np.zeros(len(items))
            for doc_idx, score in rotation_results:
                if 0 <= doc_idx < len(items):
                    semantic_scores[doc_idx] = float(score)
        else:
            # No rotation: simple cosine similarity
            query_norm = np.linalg.norm(query_emb)
            doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            doc_norms[doc_norms < 1e-12] = 1.0

            query_normalized = query_emb / query_norm if query_norm > 0 else query_emb
            docs_normalized = doc_embs / doc_norms

            semantic_scores = np.dot(docs_normalized, query_normalized)

        # Normalize lexical scores to 0-1 range
        lexical_scores = np.array([item.get("score", 0.0) for item in items])
        if lexical_scores.max() > 0:
            lexical_scores = lexical_scores / lexical_scores.max()

        # Hybrid fusion
        hybrid_scores = alpha_lexical * lexical_scores + alpha_semantic * semantic_scores

        # Update items with hybrid scores
        for i, item in enumerate(items):
            item["score_lexical"] = float(lexical_scores[i])
            item["score_semantic"] = float(semantic_scores[i])
            item["score"] = float(hybrid_scores[i])  # Override with hybrid

        # Re-sort by hybrid score
        items.sort(key=lambda x: x["score"], reverse=True)
        pack["items"] = items

        # Update metadata
        pack["meta"]["semantic_enabled"] = self.use_semantic
        pack["meta"]["quaternion_enabled"] = self.use_quaternion and num_rotations > 0
        pack["meta"]["num_rotations"] = num_rotations
        pack["meta"]["rotation_angle"] = rotation_angle
        pack["meta"]["rotation_subspace"] = rotation_subspace
        pack["meta"]["hybrid_weights"] = {"lexical": alpha_lexical, "semantic": alpha_semantic}

        print(f"[ARIA] Hybrid retrieval complete: {len(items)} items ranked", file=sys.stderr)

        return pack


def main():
    """
    Wrapper script that calls lexical v7, then enhances with semantic+quaternion
    """
    import argparse

    ap = argparse.ArgumentParser(description="ARIA v7 Hybrid Retrieval")
    ap.add_argument("--query", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--rotations", type=int, default=0)
    ap.add_argument("--rotation-angle", type=float, default=30.0)
    ap.add_argument("--rotation-subspace", type=int, default=0)
    ap.add_argument("--alpha-lexical", type=float, default=0.7)
    ap.add_argument("--alpha-semantic", type=float, default=0.3)
    ap.add_argument("--use-semantic", action="store_true", default=True)
    ap.add_argument("--use-quaternion", action="store_true", default=True)

    args = ap.parse_args()

    # Step 1: Run lexical v7 to get baseline results
    lexical_script = SCRIPT_DIR / "local_rag_context_v7_guided_exploration.py"
    temp_pack = Path(args.out).parent / "temp_lexical_pack.json"

    print(f"[ARIA] Step 1: Lexical retrieval...", file=sys.stderr)
    try:
        subprocess.run([
            "python3", str(lexical_script),
            "--query", args.query,
            "--index", args.index,
            "--out", str(temp_pack),
            "--top-k", str(args.top_k * 2),  # Get 2x for re-ranking
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"[ARIA] Lexical retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Enhance with semantic+quaternion
    print(f"[ARIA] Step 2: Semantic+quaternion enhancement...", file=sys.stderr)
    retriever = HybridRetriever(
        use_semantic=args.use_semantic,
        use_quaternion=args.use_quaternion
    )

    hybrid_pack = retriever.retrieve_hybrid(
        query=args.query,
        pack_path=str(temp_pack),
        num_rotations=args.rotations,
        rotation_angle=args.rotation_angle,
        rotation_subspace=args.rotation_subspace,
        alpha_lexical=args.alpha_lexical,
        alpha_semantic=args.alpha_semantic,
    )

    # Trim to top-k
    if "items" in hybrid_pack:
        hybrid_pack["items"] = hybrid_pack["items"][:args.top_k]

    # Save final pack
    with open(args.out, 'w') as f:
        json.dump(hybrid_pack, f, ensure_ascii=False, indent=2)

    # Cleanup temp file
    temp_pack.unlink(missing_ok=True)

    print(f"[ARIA] Hybrid pack saved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
