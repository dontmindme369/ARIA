#!/usr/bin/env python3
"""
User Profile Loader - Hardware-anchored perspective preferences

Loads user-specific perspective adjustments based on hardware signature.
Enables long-term learning of user preferences.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def get_hardware_signature() -> str:
    """
    Get hardware signature for user profile identification

    Uses machine-id as primary, falls back to hostname
    """
    try:
        # Try /etc/machine-id first (most reliable)
        machine_id_path = Path("/etc/machine-id")
        if machine_id_path.exists():
            machine_id = machine_id_path.read_text().strip()
            return f"machine_{hashlib.sha256(machine_id.encode()).hexdigest()[:16]}"
    except Exception:
        pass

    try:
        # Fallback: hostname
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        return f"host_{hashlib.sha256(hostname.encode()).hexdigest()[:16]}"
    except Exception:
        pass

    # Last resort: use "default"
    return "default"


def get_profile_path(profile_dir: Optional[Path] = None) -> Path:
    """Get path to user profile JSON"""
    if profile_dir is None:
        profile_dir = Path.home() / ".aria" / "profiles"

    profile_dir.mkdir(parents=True, exist_ok=True)

    hardware_sig = get_hardware_signature()
    return profile_dir / f"{hardware_sig}.json"


def load_user_profile(profile_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load user profile with perspective preferences

    Returns profile with:
    - technical_depth: "basic", "intermediate", "advanced", "expert"
    - domains_of_expertise: ["ai_ml", "physics", ...]
    - preferred_explanation_style: "concise", "detailed", "examples", "analogies"
    - communication_patterns: {...}
    - perspective_adjustments: {"educational": +0.2, "technical": -0.1, ...}
    """
    profile_path = get_profile_path(profile_dir)

    # Default profile
    default_profile = {
        "hardware_signature": get_hardware_signature(),
        "technical_depth": "intermediate",
        "preferred_explanation_style": "detailed",
        "domains_of_expertise": [],
        "communication_patterns": {
            "avg_query_length": 15,
            "prefers_code_examples": False,
            "asks_followup_questions": False
        },
        "perspective_adjustments": {},  # Empty = no adjustments
        "created_at": None,
        "last_updated": None,
        "total_interactions": 0
    }

    if not profile_path.exists():
        return default_profile

    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)

        # Ensure all default fields exist
        for key, value in default_profile.items():
            if key not in profile:
                profile[key] = value

        return profile

    except Exception as e:
        print(f"[ProfileLoader] Error loading profile: {e}")
        return default_profile


def save_user_profile(profile: Dict[str, Any], profile_dir: Optional[Path] = None) -> bool:
    """Save user profile to disk"""
    try:
        profile_path = get_profile_path(profile_dir)
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"[ProfileLoader] Error saving profile: {e}")
        return False


def update_perspective_adjustments(
    profile: Dict[str, Any],
    perspective: str,
    adjustment: float
) -> Dict[str, Any]:
    """
    Update perspective adjustment for user

    Args:
        profile: User profile dict
        perspective: Perspective name (e.g., "educational")
        adjustment: Adjustment value (e.g., +0.1 for more, -0.1 for less)

    Returns:
        Updated profile
    """
    if "perspective_adjustments" not in profile:
        profile["perspective_adjustments"] = {}

    current = profile["perspective_adjustments"].get(perspective, 0.0)
    new_value = max(-1.0, min(1.0, current + adjustment))  # Clamp to [-1, 1]

    profile["perspective_adjustments"][perspective] = new_value
    profile["total_interactions"] = profile.get("total_interactions", 0) + 1

    return profile


def get_perspective_bias(profile: Dict[str, Any], perspective: str) -> float:
    """
    Get perspective bias for user

    Returns adjustment value in range [-1.0, 1.0]
    - Positive: User prefers MORE of this perspective
    - Negative: User prefers LESS of this perspective
    - Zero: No preference
    """
    return profile.get("perspective_adjustments", {}).get(perspective, 0.0)


def update_profile_from_feedback(
    perspective_detected: str,
    reward: float,
    user_response: Optional[str] = None
) -> bool:
    """
    Update user profile based on feedback signals (NEW)

    Args:
        perspective_detected: The perspective that was detected ("educational", "diagnostic", etc.)
        reward: Reward score (0.0-1.0) from telemetry
        user_response: Optional explicit user feedback

    Returns:
        True if profile was updated successfully
    """
    try:
        profile = load_user_profile()

        # Update interaction count
        profile["total_interactions"] = profile.get("total_interactions", 0) + 1

        # Adjust perspective bias based on reward
        # High reward (>0.7) → positive adjustment
        # Low reward (<0.4) → negative adjustment
        if reward > 0.7:
            adjustment = +0.05  # Small positive boost
        elif reward < 0.4:
            adjustment = -0.05  # Small negative adjustment
        else:
            adjustment = 0.0  # Neutral

        profile = update_perspective_adjustments(profile, perspective_detected, adjustment)

        # Parse explicit feedback if provided
        if user_response:
            response_lower = user_response.lower()

            # Positive feedback indicators
            if any(word in response_lower for word in ["great", "perfect", "excellent", "thanks"]):
                profile = update_perspective_adjustments(profile, perspective_detected, +0.1)

            # Negative feedback indicators
            elif any(word in response_lower for word in ["wrong", "not helpful", "irrelevant", "bad"]):
                profile = update_perspective_adjustments(profile, perspective_detected, -0.1)

            # Request for more detail
            if any(phrase in response_lower for phrase in ["more detail", "explain more", "elaborate"]):
                if profile.get("preferred_explanation_style") == "concise":
                    profile["preferred_explanation_style"] = "detailed"

            # Request for less detail
            elif any(phrase in response_lower for phrase in ["too long", "shorter", "summarize"]):
                if profile.get("preferred_explanation_style") == "detailed":
                    profile["preferred_explanation_style"] = "concise"

        # Save updated profile
        return save_user_profile(profile)

    except Exception as e:
        print(f"[ProfileUpdate] Failed to update from feedback: {e}", file=sys.stderr)
        return False


def infer_socratic_personality(profile: Dict[str, Any]) -> int:
    """
    Infer Socratic personality level (0-10) from user profile

    - 0: Pure factual
    - 2: Minimal engagement
    - 5: Balanced (default)
    - 7: Curious/inquisitive
    - 10: Fully Socratic
    """
    depth = profile.get("technical_depth", "intermediate")
    style = profile.get("preferred_explanation_style", "detailed")
    asks_followups = profile.get("communication_patterns", {}).get("asks_followup_questions", False)

    # Expert + concise → Low personality (factual)
    if depth == "expert" and style == "concise":
        return 2

    # Basic + detailed + asks followups → High personality (conversational)
    elif depth == "basic" and style in ["detailed", "examples"] and asks_followups:
        return 8

    # Basic + examples → Moderate-high
    elif depth == "basic" and style in ["detailed", "examples"]:
        return 7

    # Advanced/expert + analogies → Moderate
    elif depth in ["advanced", "expert"] and style == "analogies":
        return 6

    # Asks followups → Slightly higher
    elif asks_followups:
        return 6

    # Default: balanced
    else:
        return 5


# ============================================================================
# CLI for profile management
# ============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Manage ARIA user profiles")
    ap.add_argument("--show", action="store_true", help="Show current profile")
    ap.add_argument("--reset", action="store_true", help="Reset to default profile")
    ap.add_argument("--adjust", metavar="PERSPECTIVE:VALUE", help="Adjust perspective (e.g., educational:+0.1)")

    args = ap.parse_args()

    if args.show:
        profile = load_user_profile()
        print(json.dumps(profile, indent=2))

    elif args.reset:
        profile_path = get_profile_path()
        if profile_path.exists():
            profile_path.unlink()
            print(f"Profile reset: {profile_path}")
        else:
            print("No profile to reset")

    elif args.adjust:
        try:
            perspective, value = args.adjust.split(":")
            adjustment = float(value)

            profile = load_user_profile()
            profile = update_perspective_adjustments(profile, perspective, adjustment)
            save_user_profile(profile)

            print(f"Updated {perspective}: {profile['perspective_adjustments'][perspective]}")
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: --adjust PERSPECTIVE:VALUE (e.g., educational:+0.1)")

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
