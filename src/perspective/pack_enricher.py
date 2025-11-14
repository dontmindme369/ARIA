#!/usr/bin/env python3
"""
Pack Enricher - Add perspective metadata to pack JSON files

After retrieval and postfiltering, enriches pack with:
- Perspective analysis (primary, confidence, weights, orientation vector)
- Anchor selected
- Anchor-perspective alignment score
- Perspective coherence metrics
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional


def enrich_pack_with_perspective(
    pack_path: Path,
    perspective_analysis: Dict[str, Any],
    anchor_selected: Optional[str] = None,
    anchor_perspective_alignment: float = 0.0
) -> bool:
    """
    Enrich pack JSON with perspective metadata

    Args:
        pack_path: Path to last_pack.json or last_pack.filtered.json
        perspective_analysis: Dict from perspective_detector.detect()
        anchor_selected: Name of anchor selected (e.g., "educational")
        anchor_perspective_alignment: How well anchor matches perspective (0.0-1.0)

    Returns:
        True if successful, False otherwise
    """
    if not pack_path.exists():
        return False

    try:
        # Load existing pack
        with open(pack_path, 'r', encoding='utf-8') as f:
            pack = json.load(f)

        # Ensure meta exists
        if "meta" not in pack:
            pack["meta"] = {}

        # Add perspective metadata
        pack["meta"]["perspective_analysis"] = {
            "primary": perspective_analysis.get("primary", "mixed"),
            "confidence": float(perspective_analysis.get("confidence", 0.0)),
            "weights": perspective_analysis.get("weights", {}),
            "orientation_vector": perspective_analysis.get("orientation_vector", [])
        }

        # Add anchor info if provided
        if anchor_selected:
            pack["meta"]["anchor_selected"] = anchor_selected
            pack["meta"]["anchor_perspective_alignment"] = float(anchor_perspective_alignment)

        # Write back
        with open(pack_path, 'w', encoding='utf-8') as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"[PackEnricher] Error enriching {pack_path}: {e}")
        return False


def compute_perspective_coherence(pack_path: Path, perspective_analysis: Dict[str, Any]) -> float:
    """
    Compute how well pack chunks match detected perspective

    Returns coherence score 0.0-1.0
    """
    if not pack_path.exists():
        return 0.0

    try:
        with open(pack_path, 'r', encoding='utf-8') as f:
            pack = json.load(f)

        items = pack.get("items", [])
        if not items:
            return 0.0

        primary_perspective = perspective_analysis.get("primary", "mixed")

        # Simple heuristic: count how many chunks contain perspective markers
        # (In future, use chunk-level perspective detection)

        # For now, return confidence as placeholder
        # TODO: Implement proper chunk-level coherence scoring
        return float(perspective_analysis.get("confidence", 0.5))

    except Exception:
        return 0.0


def add_perspective_distribution(pack_path: Path) -> bool:
    """
    Analyze pack and add perspective distribution to metadata

    Counts how many chunks align with each perspective
    """
    if not pack_path.exists():
        return False

    try:
        with open(pack_path, 'r', encoding='utf-8') as f:
            pack = json.load(f)

        items = pack.get("items", [])
        if not items:
            return False

        # TODO: Implement chunk-level perspective detection
        # For now, just record total count
        if "meta" not in pack:
            pack["meta"] = {}

        pack["meta"]["chunk_count"] = len(items)

        with open(pack_path, 'w', encoding='utf-8') as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"[PackEnricher] Error adding distribution to {pack_path}: {e}")
        return False
