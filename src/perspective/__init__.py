"""
Perspective Detection & Management

8-dimensional perspective understanding:
1. Educational - Teaching, learning, explanation
2. Diagnostic - Troubleshooting, debugging
3. Security - Vulnerabilities, threat analysis
4. Implementation - Building, coding
5. Research - Investigation, analysis
6. Theoretical - Conceptual, mathematical
7. Practical - Applied, hands-on
8. Reference - Facts, definitions

Components:
- detector.py - Hybrid perspective detection (lexical + neural)
- rotator.py - Quaternion rotation for perspective-biased search
- signature_learner.py - Learn perspective markers from corpus
- pack_enricher.py - Add perspective metadata to packs
- user_profile.py - Hardware-anchored user preferences
"""

from .detector import *
from .rotator import *
from .signature_learner import SignatureLearner
from .pack_enricher import enrich_pack_with_perspective, compute_perspective_coherence
from .profile_loader import (
    load_user_profile,
    save_user_profile,
    update_perspective_adjustments,
    get_perspective_bias,
    update_profile_from_feedback,
    infer_socratic_personality
)

__all__ = [
    'detector',
    'rotator',
    'SignatureLearner',
    'enrich_pack_with_perspective',
    'compute_perspective_coherence',
    'load_user_profile',
    'save_user_profile',
    'update_perspective_adjustments',
    'get_perspective_bias',
    'update_profile_from_feedback',
    'infer_socratic_personality'
]
