#!/usr/bin/env python3
"""
aria_core.py — Unified ARIA orchestration with integrated session security

Consolidates:
- aria_main.py (orchestration, bandit selection, reward calculation)
- session_cli.py (session management, security, hardware anchoring)

Features:
- Multi-armed bandit preset selection
- Hardware-anchored session security
- Retrieval + postfiltering pipeline
- Exemplar fit scoring
- Comprehensive telemetry
"""
from __future__ import annotations

import argparse, dataclasses, getpass, hashlib, inspect, json, os, platform
import re, shutil, socket, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

# ─────────────────────────────────────────────────────────────────────────────
# Path setup - PORTABLE
# ─────────────────────────────────────────────────────────────────────────────
# Import portable path utilities
try:
    from utils.paths import get_project_root, expand_path, ensure_dir
    from utils.config_loader import load_config, get_config_value
    PROJECT_ROOT = get_project_root()
except ImportError:
    # Fallback for development
    _current = Path(__file__).resolve().parent
    _src = _current.parent
    _project = _src.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from utils.paths import get_project_root, expand_path, ensure_dir
    from utils.config_loader import load_config, get_config_value
    PROJECT_ROOT = get_project_root()

# Ensure src/ is in path
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Session Management (from session_cli.py)
# ─────────────────────────────────────────────────────────────────────────────
CLI_VERSION = "2025-10-30-core"

def _now() -> float:
    return time.time()

def _truthy(v: str | None) -> bool:
    return bool(v and v.strip().lower() in ("1", "true", "yes", "on"))

def _read_json(p: Path) -> dict | None:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _read_text(path: str) -> str:
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""

def _hash_pass(pass_text: str, salt_hex: str | None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else b""
    h = hashlib.blake2b(salt=salt, digest_size=32)
    h.update(pass_text.encode("utf-8"))
    return h.hexdigest()

# Hardware anchors (ALWAYS ON)
def _linux_hw_anchors() -> List[str]:
    anchors: List[str] = []
    for cand in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        t = _read_text(cand)
        if t:
            anchors.append(f"machine-id:{t}")
    dmi_dir = Path("/sys/class/dmi/id")
    for name in ["product_uuid", "product_serial", "board_serial", "board_name", "product_name"]:
        p = dmi_dir / name
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    anchors.append(f"dmi:{name}:{txt}")
            except Exception:
                pass
    cpuinfo = _read_text("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("serial"):
                _, _, val = line.partition(":")
                val = val.strip()
                if val:
                    anchors.append(f"cpu-serial:{val}")
                break
    tpm_file = os.environ.get("RAG_TPM_ATTEST_FILE") or "security/tpm_attest.txt"
    tpm = _read_text(tpm_file)
    if tpm:
        anchors.append(f"tpm:{hashlib.blake2b(tpm.encode(), digest_size=20).hexdigest()}")
    return anchors

def _darwin_hw_anchors() -> List[str]:
    anchors: List[str] = []
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        )
        for line in out.stdout.splitlines():
            if "IOPlatformUUID" in line:
                val = line.split("=")[-1].strip().strip('"')
                if val:
                    anchors.append(f"platform-uuid:{val}")
                break
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3
        )
        for line in out.stdout.splitlines():
            if "Serial Number" in line:
                anchors.append(f"serial:{line.split(':')[-1].strip()}")
                break
    except Exception:
        pass
    return anchors

def _windows_hw_anchors() -> List[str]:
    anchors: List[str] = []
    try:
        import importlib
        winreg = importlib.import_module("winreg")
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography")
        val, _ = winreg.QueryValueEx(key, "MachineGuid")
        if val:
            anchors.append(f"machine-guid:{val}")
    except Exception:
        pass
    return anchors

def _hw_anchors() -> List[str]:
    sysname = platform.system().lower()
    if sysname == "linux":
        return _linux_hw_anchors()
    if sysname == "darwin":
        return _darwin_hw_anchors()
    if sysname == "windows":
        return _windows_hw_anchors()
    return []

def _whohash(identity_file: Path) -> str:
    """Pin session to user@host + identity fingerprint + hardware anchors."""
    user = os.environ.get("RAG_USER") or getpass.getuser()
    host = os.environ.get("RAG_HOST") or socket.gethostname()
    ident = _read_json(identity_file) or {}
    fid = str(ident.get("pgp_fingerprint") or ident.get("fingerprint") or "")
    salt = str(ident.get("session_salt") or ident.get("passkey_salt") or "")
    parts = [f"user@host:{user}@{host}", f"fid:{fid}", f"salt:{salt}"]
    anchors = [a for a in _hw_anchors() if a]
    if anchors:
        h = hashlib.blake2b(digest_size=20)
        for a in sorted(set(anchors)):
            h.update(a.encode("utf-8"))
        parts.append(f"hw:{h.hexdigest()}")
    basis = "|".join(parts)
    h2 = hashlib.blake2b(digest_size=16)
    h2.update(basis.encode("utf-8"))
    return h2.hexdigest()

class SessionManager:
    """Secure session management with hardware anchoring."""

    def __init__(
        self,
        session_file: Optional[str] = None,
        identity_file: Optional[str] = None,
        require_passkey: bool = True,
        ttl_hours: float = 24.0
    ):
        self.session_file = self._resolve_session_file(session_file)
        self.identity_file = self._resolve_identity_file(identity_file)
        self.require_passkey = require_passkey
        self.ttl_hours = ttl_hours

    def _resolve_session_file(self, override: Optional[str]) -> Path:
        """Resolve session file path - PORTABLE"""
        if override:
            return expand_path(override)

        env = os.environ.get("RAG_SESSION_FILE")
        if env:
            return expand_path(env)

        # Default: .aria/session.json in user home
        p = Path.home() / ".aria" / "session.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _resolve_identity_file(self, override: Optional[str]) -> Path:
        """Resolve identity file path - PORTABLE"""
        if override:
            return expand_path(override)

        env = os.environ.get("RAG_IDENTITY_FILE")
        if env:
            return expand_path(env)

        # Check multiple locations
        # 1. Project security dir
        p1 = PROJECT_ROOT / "security" / ".rag_identity.json"
        if p1.exists():
            return p1

        # 2. User home .aria dir
        p2 = Path.home() / ".aria" / "identity.json"
        if p2.exists():
            return p2

        return p1  # Default to project security dir

    def _load_expected_pass(self) -> Tuple[str | None, str | None, bool]:
        """Return (expected, salt_hex, is_hashed)."""
        env_plain = os.environ.get("RAG_PASSKEY")
        if env_plain:
            return env_plain, None, False
        ident = _read_json(self.identity_file) or {}
        if ident.get("passkey_hash"):
            return str(ident.get("passkey_hash")), str(ident.get("passkey_salt") or ""), True
        if ident.get("passkey"):
            return str(ident.get("passkey")), None, False
        return None, None, False

    def _check_passkey(self, cached_ok: bool, who_ok: bool) -> bool:
        if not self.require_passkey:
            return True
        if cached_ok and who_ok:
            return True
        exp, salt, is_hashed = self._load_expected_pass()
        if not exp:
            sys.stderr.write("[SECURITY] passkey required but no expected value configured.\n")
            return False
        provided = os.environ.get("RAG_PASSKEY_INPUT")
        if not provided:
            try:
                provided = getpass.getpass("Passphrase: ")
            except Exception:
                sys.stderr.write("[SECURITY] falling back to reading passphrase from stdin...\n")
                provided = sys.stdin.readline().rstrip("\n")
        ok = (_hash_pass(provided, salt) == exp) if is_hashed else (provided == exp)
        return ok

    def ensure_session(self) -> bool:
        """Ensure valid session exists. Returns True if valid."""
        if not self.identity_file.exists():
            sys.stderr.write(f"[SECURITY] identity missing at {self.identity_file}\n")
            return False

        now = _now()
        s = _read_json(self.session_file) or {}
        active = bool(s and now < float(s.get("expires_at", 0)))
        cached_ok = bool(s.get("passkey_ok"))
        current_who = _whohash(self.identity_file)
        who_ok = (s.get("whohash") == current_who)

        if not active:
            s = {}

        if not self._check_passkey(cached_ok, who_ok):
            sys.stderr.write("[SECURITY] cognitive check failed.\n")
            return False

        s["created_at"] = s.get("created_at", now)
        s["updated_at"] = now
        s["expires_at"] = now + self.ttl_hours * 3600.0
        s["ttl_hours"] = self.ttl_hours
        s["passkey_ok"] = True
        s["whohash"] = current_who
        s["scope"] = "user@host+hw"
        _write_json(self.session_file, s)
        return True

    def is_valid(self) -> bool:
        """Check if current session is valid."""
        s = _read_json(self.session_file)
        return bool(s and _now() < float(s.get("expires_at", 0)))

    def clear(self) -> None:
        """Clear session file."""
        try:
            self.session_file.unlink(missing_ok=True)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# ARIA Orchestration (from aria_main.py)
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATE_V7 = [
    "local_rag_context_v7_guided_exploration.py",  # Lexical BM25 (stable, primary)
    "aria_v7_hybrid_semantic.py",  # Hybrid lexical+semantic+quaternion (optional)
    "rag_v7_multirot.py",
    "local_rag_context_v6_9k_adaptive_learn.py",
]
CANDIDATE_POST = [
    "aria_postfilter.py",
    "rag_postfilter_pack_v3_1.py",
    "rag_postfilter_v2.py",
    "rag_postfilter.py",
]

def _slug(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = "-".join(s.split())[:60]
    h = hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()[:6]
    return f"{s or 'query'}-{h}"

def _which_script(candidates: List[str]) -> Optional[str]:
    """Find script in project locations - PORTABLE"""
    locations = [
        PROJECT_ROOT,
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "src" / "retrieval",
        PROJECT_ROOT / "retrieval",
        Path.cwd(),
    ]
    for loc in locations:
        for name in candidates:
            p = (loc / name).resolve()
            if p.exists():
                return str(p)
    for name in candidates:
        p = shutil.which(name)
        if p:
            return p
    return None

def _feats_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        d = obj.to_dict()
        return d if isinstance(d, dict) else {}
    if dataclasses.is_dataclass(obj) and not inspect.isclass(obj):
        try:
            return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        except Exception:
            return {}
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if not k.startswith("_")}
    return {}

def _load_pack_chunks(pack_path: str) -> List[Dict[str, Any]]:
    """Load chunks from pack file (supports JSON and TXT formats)."""
    chunks = []
    try:
        if pack_path.endswith('.txt'):
            with open(pack_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if '[' in text and ']' in text:
                    sections = text.split('\n\n')
                    for section in sections:
                        if section.strip():
                            chunks.append({"text": section.strip()})
                else:
                    chunks.append({"text": text})
        else:
            with open(pack_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data.get("items") or data.get("sources") or data.get("chunks") or []
    except Exception as e:
        print(f"[ARIA] Failed to load pack: {e}", file=sys.stderr)
    return chunks

def _load_exemplars(path: str) -> List[str]:
    """Load exemplars from file into list of strings."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []

class ARIA:
    """
    Adaptive Resonant Intelligent Architecture orchestrator.

    Features:
    - Multi-armed bandit preset selection
    - Retrieval + postfiltering pipeline
    - Exemplar fit scoring
    - Session security enforcement
    - Comprehensive telemetry
    """

    def __init__(
        self,
        index_roots: List[str] | None = None,
        out_root: str | None = None,
        state_path: str | None = None,
        exemplars_path: Optional[str] = None,
        session_manager: Optional[SessionManager] = None,
        enforce_session: bool = False,  # Default False for CLI portability
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize ARIA with portable paths.

        Args:
            index_roots: List of knowledge base directories (or None to use config)
            out_root: Output directory (or None to use config)
            state_path: Bandit state file path (or None to use config)
            exemplars_path: Path to exemplars file (or None to use config)
            session_manager: Custom session manager (optional)
            enforce_session: Enable session security (default False for portability)
            config: Pre-loaded config dict (optional)
        """
        # Load config if not provided
        if config is None:
            try:
                config = load_config()
            except Exception:
                config = {}

        # Resolve paths from config
        self.index_roots = index_roots or [str(r) for r in config.get('paths', {}).get('index_roots', [])]
        if not self.index_roots:
            # Fallback to environment or default
            self.index_roots = [os.getenv("RAG_INDEX_ROOT", str(PROJECT_ROOT / "data"))]

        # Output directory
        if out_root:
            self.out_root = ensure_dir(expand_path(out_root))
        else:
            out_cfg = config.get('paths', {}).get('output_dir')
            if out_cfg:
                self.out_root = ensure_dir(out_cfg)
            else:
                self.out_root = ensure_dir(PROJECT_ROOT / "aria_packs")

        # State file
        if state_path:
            self.state_path = str(expand_path(state_path))
        else:
            state_cfg = config.get('paths', {}).get('bandit_state')
            if state_cfg:
                self.state_path = str(state_cfg)
            else:
                # Store in current working directory for portability
                self.state_path = str(Path.cwd() / ".aria_contextual_bandit.json")

        # Exemplars
        if exemplars_path:
            self.exemplars_path = str(expand_path(exemplars_path))
        else:
            ex_cfg = config.get('paths', {}).get('exemplars')
            if ex_cfg:
                self.exemplars_path = str(ex_cfg)
            else:
                self.exemplars_path = str(PROJECT_ROOT / "data" / "exemplars.txt")

        self.enforce_session = enforce_session

        # Initialize session manager
        self.session = session_manager or SessionManager()

        # Import dependencies
        try:
            from retrieval.query_features import QueryFeatureExtractor
            from utils.presets import preset_to_cli_args, preset_to_postfilter_args
            from monitoring.metrics_utils import (
                compute_retrieval_stats,
                coverage_score,
                telemetry_line,
                RunMetrics,
                GenerationStats,
                TimingStats,
                detect_issues,
            )
            from anchors.exemplar_fit import ExemplarFitScorer
            from intelligence.bandit_context import select_preset as bandit_select, give_reward as bandit_update

            self.feat_ex = QueryFeatureExtractor()
            self.preset_to_cli_args = preset_to_cli_args
            self.preset_to_postfilter_args = preset_to_postfilter_args
            self.compute_retrieval_stats = compute_retrieval_stats
            self.coverage_score = coverage_score
            self.telemetry_line = telemetry_line
            self.RunMetrics = RunMetrics
            self.GenerationStats = GenerationStats
            self.TimingStats = TimingStats
            self.detect_issues = detect_issues
            self.bandit_select = bandit_select
            self.bandit_update = bandit_update

            # Load exemplars
            exemplars_list = _load_exemplars(self.exemplars_path) if Path(self.exemplars_path).exists() else None
            self.exemplar_scorer = ExemplarFitScorer(exemplars_list) if exemplars_list else None
        except ImportError as e:
            sys.stderr.write(f"[ARIA] Failed to import dependencies: {e}\n")
            raise

    def _enforce_session(self) -> None:
        """Ensure valid session exists."""
        if not self.enforce_session:
            return
        if os.environ.get("RAG_ENFORCE_SESSION", "0") not in ("1", "true", "TRUE", "Yes", "yes"):
            return
        if not self.session.ensure_session():
            raise RuntimeError("[SECURITY] Session enforcement failed")

    def _detect_perspective(self, query_text: str) -> Dict[str, Any]:
        """
        Detect perspective for query (NEW - perspective integration)

        Returns dict with:
        - primary: str (perspective name)
        - confidence: float (0-1)
        - weights: dict (all 8 perspectives)
        - orientation_vector: list (8D vector)
        """
        try:
            from perspective.detector import PerspectiveOrientationDetector

            # Lazy-load detector with fallback path resolution
            if not hasattr(self, '_perspective_detector'):
                # Try multiple possible locations for signature file
                possible_paths = [
                    PROJECT_ROOT / "data" / "domain_dictionaries" / "perspective_signatures_v2.json",
                    PROJECT_ROOT / "data" / "perspective_signatures" / "anchor_perspective_signatures_v2.json",
                    PROJECT_ROOT / "data" / "domain_dictionaries" / "perspective_signatures.json",  # Non-v2 fallback
                ]

                signatures_path = None
                for path in possible_paths:
                    if path.exists():
                        signatures_path = path
                        break

                if signatures_path is None:
                    raise FileNotFoundError(
                        f"Perspective signatures file not found. Tried:\n" +
                        "\n".join(f"  - {p}" for p in possible_paths)
                    )

                self._perspective_detector = PerspectiveOrientationDetector(str(signatures_path))
                print(f"[ARIA] Loaded perspective detector from: {signatures_path.name}")

            result = self._perspective_detector.detect(query_text)
            return result

        except FileNotFoundError as e:
            print(f"[ARIA ERROR] Perspective signatures file not found: {e}", file=sys.stderr)
            print(f"[ARIA ERROR] Perspective detection disabled - using fallback mixed mode", file=sys.stderr)
            # Fallback to mixed
            return {
                "primary": "mixed",
                "confidence": 0.0,
                "weights": {},
                "orientation_vector": [0.125] * 8  # Uniform distribution
            }
        except Exception as e:
            print(f"[ARIA WARNING] Perspective detection failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fallback to mixed
            return {
                "primary": "mixed",
                "confidence": 0.0,
                "weights": {},
                "orientation_vector": [0.125] * 8  # Uniform distribution
            }

    def _load_user_profile(self) -> Dict[str, Any]:
        """Load hardware-anchored user profile (NEW)"""
        try:
            from perspective.profile_loader import load_user_profile
            return load_user_profile()
        except Exception as e:
            print(f"[ARIA] Profile loading failed: {e}")
            return {
                "technical_depth": "intermediate",
                "preferred_explanation_style": "detailed",
                "perspective_adjustments": {}
            }

    def _enrich_pack_with_perspective(
        self,
        pack_path: Path,
        perspective_analysis: Dict[str, Any],
        user_profile: Dict[str, Any],
        preset_name: str = "unknown",
        anchor_mode: Optional[str] = None,
        anchor_alignment: float = 0.0
    ) -> None:
        """
        Enrich pack JSON with perspective metadata after retrieval (NEW)

        Adds:
        - perspective_analysis (primary, confidence, weights, orientation_vector)
        - anchor_selected (anchor mode name if with_anchor=True)
        - anchor_perspective_alignment score

        Args:
            pack_path: Path to last_pack.json
            perspective_analysis: Dict from _detect_perspective()
            user_profile: Dict from _load_user_profile()
            preset_name: Name of selected preset/anchor
            anchor_mode: Selected anchor mode (e.g., "technical", "code")
            anchor_alignment: Anchor-perspective alignment score (0-1)
        """
        try:
            from perspective.pack_enricher import enrich_pack_with_perspective

            primary = perspective_analysis.get("primary", "mixed")
            confidence = perspective_analysis.get("confidence", 0.0)

            # Use provided anchor alignment if available, otherwise compute from preset
            if anchor_mode and anchor_alignment > 0:
                # Anchor mode was selected, use its alignment score
                final_anchor_name = anchor_mode
                final_alignment = anchor_alignment
            else:
                # No anchor mode, compute alignment from preset name
                preset_lower = preset_name.lower()
                primary_lower = primary.lower()

                # Direct match: preset contains perspective keyword
                if primary_lower in preset_lower:
                    final_alignment = 0.9 + (confidence * 0.1)
                # Semantic matches (educational ↔ tutorial, diagnostic ↔ debug, etc.)
                elif any(pair in [(primary_lower, preset_lower), (preset_lower, primary_lower)]
                         for pair in [
                             ("educational", "tutorial"), ("educational", "teaching"),
                             ("diagnostic", "debug"), ("diagnostic", "troubleshoot"),
                             ("security", "audit"), ("security", "vulnerability"),
                             ("implementation", "code"), ("implementation", "build"),
                             ("research", "analysis"), ("research", "investigation"),
                             ("theoretical", "math"), ("theoretical", "concept"),
                             ("practical", "hands-on"), ("practical", "applied"),
                             ("reference", "docs"), ("reference", "documentation")
                         ]):
                    final_alignment = 0.8 + (confidence * 0.2)
                # Moderate confidence: use confidence as base
                elif confidence > 0.5:
                    final_alignment = 0.5 + (confidence * 0.3)
                else:
                    final_alignment = 0.5  # Neutral default

                final_anchor_name = preset_name

            # Enrich the pack
            success = enrich_pack_with_perspective(
                pack_path=pack_path,
                perspective_analysis=perspective_analysis,
                anchor_selected=final_anchor_name,
                anchor_perspective_alignment=final_alignment
            )

            if not success:
                print(f"[ARIA] Pack enrichment failed for {pack_path}", file=sys.stderr)

        except Exception as e:
            print(f"[ARIA] Pack enrichment error: {e}", file=sys.stderr)

    def _compute_rotation_params(self, perspective_analysis: Dict[str, Any], user_profile: Dict[str, Any]) -> List[str]:
        """
        Compute rotation parameters from perspective and user profile (NEW)

        Maps detected perspective to rotation angle and subspace for v7 retrieval.
        Applies user profile adjustments for personalization.

        Args:
            perspective_analysis: Dict from _detect_perspective()
            user_profile: Dict from _load_user_profile()

        Returns:
            List of CLI args: ["--rotation-angle", "X", "--rotation-subspace", "Y"]
        """
        try:
            # Map perspectives to rotation angles (degrees) and subspace indices
            perspective_map = {
                "educational": {"angle": 30.0, "subspace": 0},
                "diagnostic": {"angle": 90.0, "subspace": 1},
                "security": {"angle": 45.0, "subspace": 2},
                "implementation": {"angle": 60.0, "subspace": 3},
                "research": {"angle": 120.0, "subspace": 4},
                "theoretical": {"angle": 75.0, "subspace": 5},
                "practical": {"angle": 50.0, "subspace": 6},
                "reference": {"angle": 15.0, "subspace": 7},
                "mixed": {"angle": 0.0, "subspace": -1}  # No rotation for mixed
            }

            primary = perspective_analysis.get("primary", "mixed")
            confidence = float(perspective_analysis.get("confidence", 0.0))

            # Get base rotation params
            params = perspective_map.get(primary, perspective_map["mixed"])
            angle = params["angle"]
            subspace = params["subspace"]

            # Apply user profile adjustments
            adjustments = user_profile.get("perspective_adjustments", {})
            if primary in adjustments:
                # Adjustments are stored as floats (e.g., -0.15 or 0.3)
                # Convert to multiplier: adjustment of 0.3 = 1.3x, -0.15 = 0.85x
                adjustment_value = float(adjustments[primary])
                angle_multiplier = 1.0 + adjustment_value
                angle *= angle_multiplier

            # Scale by confidence (low confidence = less aggressive rotation)
            angle *= confidence

            # Don't rotate if confidence too low or mixed perspective
            if confidence < 0.3 or subspace == -1:
                return []  # No rotation params

            # Return CLI args for v7
            return [
                "--rotation-angle", f"{angle:.1f}",
                "--rotation-subspace", str(subspace)
            ]

        except Exception as e:
            print(f"[ARIA] Rotation params computation failed: {e}")
            return []  # Safe fallback: no rotation

    def query(
        self,
        query_text: str,
        with_anchor: bool = False,
        preset_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute retrieval query with bandit-selected preset.

        Args:
            query_text: User query string
            with_anchor: Enable 16-anchor reasoning mode
            preset_override: Manual preset selection (bypasses bandit)

        Returns:
            Dict with preset, metrics, pack paths, and run directory
        """
        self._enforce_session()

        slug = _slug(query_text)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.out_root / ts / slug
        run_dir.mkdir(parents=True, exist_ok=True)

        # Extract features and select preset
        feats = _feats_to_dict(self.feat_ex.extract(query_text))

        if preset_override:
            # Manual preset override (for testing)
            from intelligence.bandit_context import BanditState
            state = BanditState()
            preset = next((p for p in state.presets if p.name == preset_override), state.presets[1])
            reason = f"manual override: {preset_override}"
            meta = {"phase": "manual"}
        else:
            preset, reason, meta = self.bandit_select(feats, state_path=self.state_path)

        phase = meta.get("phase", "unknown")

        # Detect perspective (NEW - perspective integration)
        perspective_analysis = self._detect_perspective(query_text)

        # Load user profile (NEW)
        user_profile = self._load_user_profile()

        # Select anchor mode (NEW - anchor integration)
        anchor_mode = None
        anchor_alignment = 0.0
        anchor_template = None

        if with_anchor:
            try:
                from anchors.anchor_selector import AnchorSelector

                # Lazy-load anchor selector
                if not hasattr(self, '_anchor_selector'):
                    self._anchor_selector = AnchorSelector()

                # Select anchor mode based on query
                anchor_mode = self._anchor_selector.select_mode(query_text)

                # Calculate alignment based on perspective
                if perspective_analysis:
                    # Simple alignment: check if perspective matches anchor mode
                    anchor_alignment = 0.8 if perspective_analysis.get('mode') == anchor_mode else 0.5
                else:
                    anchor_alignment = 0.5

                print(f"[ARIA] Anchor selected: {anchor_mode} (alignment: {anchor_alignment:.2f})")

            except Exception as e:
                print(f"[ARIA] Anchor selection failed: {e}", file=sys.stderr)
                anchor_mode = "casual"  # Fallback to casual
                anchor_alignment = 0.0

        # Find scripts with validation
        v7_script = _which_script(CANDIDATE_V7)
        post_script = _which_script(CANDIDATE_POST)

        if not v7_script:
            print(f"[ARIA ERROR] Retrieval script not found. Tried: {', '.join(CANDIDATE_V7)}", file=sys.stderr)
            return {
                "preset": "default_fallback",
                "reason": "v7 retrieval script not found",
                "result": {"ok": False, "reason": "retrieval script not found"},
                "run_dir": str(run_dir),
            }

        # Validate script is executable
        v7_path = Path(v7_script)
        if not v7_path.exists():
            print(f"[ARIA ERROR] Script path invalid: {v7_script}", file=sys.stderr)
            return {
                "preset": preset.name,
                "reason": reason,
                "result": {"ok": False, "reason": "script path invalid"},
                "run_dir": str(run_dir),
            }

        print(f"[ARIA] Using retrieval script: {v7_path.name}")

        # Prepare retrieval
        flags = self.preset_to_cli_args(preset)
        pack_path = run_dir / "last_pack.json"
        filtered_path = run_dir / "last_pack.filtered.json"

        # Compute perspective-biased rotation params (NEW)
        rotation_params = self._compute_rotation_params(perspective_analysis, user_profile)

        start = time.time()

        # Call v7 retrieval with retry logic
        idx_arg = ":".join(str(p) for p in self.index_roots)
        cmd = [
            "python3", v7_script,
            "--query", query_text,
            "--index", idx_arg,
            "--out-dir", str(run_dir),
            *flags,
            *rotation_params  # Add perspective-biased rotation
        ]

        # Retry configuration
        max_retries = 2
        timeout_seconds = 60  # Increased from 30s for large knowledge bases
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                break  # Success, exit retry loop

            except subprocess.TimeoutExpired as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"[ARIA WARNING] Retrieval timeout (attempt {retry_count}/{max_retries + 1}), retrying...", file=sys.stderr)
                    time.sleep(1)  # Brief pause before retry
                else:
                    print(f"[ARIA ERROR] Retrieval timeout after {max_retries + 1} attempts", file=sys.stderr)
                    return {
                        "preset": preset.name,
                        "reason": reason,
                        "result": {"ok": False, "error": f"timeout after {timeout_seconds}s ({max_retries + 1} attempts)"},
                        "run_dir": str(run_dir),
                    }

            except subprocess.CalledProcessError as e:
                last_error = e
                print(f"[ARIA ERROR] Retrieval script failed with code {e.returncode}", file=sys.stderr)
                print(f"[ARIA ERROR] stdout: {e.stdout[:500]}", file=sys.stderr)
                print(f"[ARIA ERROR] stderr: {e.stderr[:500]}", file=sys.stderr)
                return {
                    "preset": preset.name,
                    "reason": reason,
                    "result": {"ok": False, "error": str(e), "returncode": e.returncode},
                    "run_dir": str(run_dir),
                }

            except Exception as e:
                last_error = e
                print(f"[ARIA ERROR] Unexpected retrieval error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                return {
                    "preset": preset.name,
                    "reason": reason,
                    "result": {"ok": False, "error": str(e)},
                    "run_dir": str(run_dir),
                }

        if not pack_path.exists():
            return {
                "preset": preset.name,
                "reason": reason,
                "result": {"ok": False, "error": "v7 script did not create pack file"},
            }

        # Enrich pack with perspective metadata BEFORE postfilter (NEW)
        # This ensures perspective info is available for postfilter scoring
        self._enrich_pack_with_perspective(
            pack_path=pack_path,
            perspective_analysis=perspective_analysis,
            user_profile=user_profile,
            preset_name=preset.name,
            anchor_mode=anchor_mode,
            anchor_alignment=anchor_alignment
        )

        # Postfilter
        filtered_json = run_dir / "last_pack.filtered.json"
        if post_script and pack_path.exists():
            try:
                # Build postfilter command with preset-specific parameters
                postfilter_args = self.preset_to_postfilter_args(preset)
                cmd = [
                    "python3", post_script,
                    str(pack_path),
                    "--output", str(filtered_json)
                ] + postfilter_args

                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)

                if filtered_json.exists():
                    filtered_path = filtered_json
                else:
                    filtered_path = pack_path
            except Exception as e:
                print(f"[ARIA] Postfilter failed: {e}", file=sys.stderr)
                filtered_path = pack_path
        else:
            filtered_path = pack_path

        elapsed = time.time() - start

        # Load pack JSON for semantic metrics
        pack_json = json.loads(pack_path.read_text(encoding='utf-8'))
        semantic_recall = pack_json.get("semantic_recall", 0.0)

        # Load and compute stats
        raw_chunks = _load_pack_chunks(str(pack_path))
        chunks_mapped = cast(List[Mapping[str, Any]], raw_chunks)
        r = self.compute_retrieval_stats(chunks_mapped)

        # Extract text for coverage
        if filtered_path.suffix == '.txt':
            retrieved_text = filtered_path.read_text(encoding='utf-8')
        else:
            chunk_texts = [chunk.get("text", "") for chunk in raw_chunks if chunk.get("text")]
            retrieved_text = "\n\n".join(chunk_texts)

        cov_score = self.coverage_score(query_text, retrieved_text)

        gen = self.GenerationStats(coverage_score=cov_score)
        timing = self.TimingStats(total_s=elapsed)

        # Extract flags
        if hasattr(preset, "args") and isinstance(preset.args, dict):
            flags_dict = {
                "top_k": preset.args.get("top_k", 32),
                "sem_limit": preset.args.get("sem_limit", 128),
                "rotations": preset.args.get("rotations", 1),
                "rotation_subspace": preset.args.get("rotation_subspace", 4),
            }
        else:
            flags_dict = {
                "top_k": getattr(preset, "top_k", 32),
                "sem_limit": getattr(preset, "sem_limit", 128),
                "rotations": getattr(preset, "rotations", 1),
                "rotation_subspace": getattr(preset, "rotation_subspace", 4),
            }

        # Build metrics (temporary with reward=0)
        run_metrics = self.RunMetrics(
            retrieval=dataclasses.asdict(r),
            generation=dataclasses.asdict(gen),
            timing=dataclasses.asdict(timing),
            issues=[],
            reward=0.0,  # Temporary, will update below
            query=query_text,
            preset=preset.name,
            flags=flags_dict,
            semantic_recall=semantic_recall,
        )

        # Detect issues FIRST
        run_metrics.issues = self.detect_issues(run_metrics, {
            "slow_seconds": 8.0,
            "dup_ratio": 0.6,
            "min_unique_sources": 3,
            "min_coverage": 0.2,
        })

        # Calculate compound reward from multiple signals
        # Configurable weights for multi-objective optimization
        EXEMPLAR_WEIGHT = 0.35  # Quality/relevance to anchor
        COVERAGE_WEIGHT = 0.35  # Semantic coverage of query
        DIVERSITY_WEIGHT = 0.20  # Source diversity
        SEMANTIC_RECALL_WEIGHT = 0.10  # BM25 recall

        reward = 0.0
        exemplar_fit = 0.0

        if self.exemplar_scorer and raw_chunks:
            try:
                exemplar_fit, _ = self.exemplar_scorer.calculate_fit(retrieved_text[:5000], cov_score)
            except Exception as e:
                print(f"[ARIA WARNING] Exemplar scoring failed: {e}", file=sys.stderr)
                exemplar_fit = cov_score  # Fallback to coverage as proxy

        # Compound reward with explicit weights (should sum to 1.0)
        reward_components = {
            "exemplar_fit": exemplar_fit * EXEMPLAR_WEIGHT,
            "coverage": cov_score * COVERAGE_WEIGHT,
            "diversity": r.diversity_mm * DIVERSITY_WEIGHT,
            "semantic_recall": semantic_recall * SEMANTIC_RECALL_WEIGHT,
        }

        # Base reward (before penalties)
        base_reward = sum(reward_components.values())

        # Apply penalties for quality issues (additive penalty, not multiplicative)
        issue_penalty = 0.0
        if run_metrics.issues:
            # Each issue type gets a specific penalty
            for issue in run_metrics.issues:
                if "slow" in issue.lower():
                    issue_penalty += 0.05  # 5% penalty for slow queries
                elif "duplicate" in issue.lower():
                    issue_penalty += 0.10  # 10% penalty for high duplication
                elif "coverage" in issue.lower():
                    issue_penalty += 0.15  # 15% penalty for low coverage
                else:
                    issue_penalty += 0.05  # 5% for other issues

            # Cap total penalty at 30%
            issue_penalty = min(issue_penalty, 0.30)

        # Final reward = base - penalties
        reward = base_reward - issue_penalty

        # Clip to [0, 1] range
        reward = max(0.0, min(1.0, reward))

        # Log reward breakdown for debugging (can disable in production)
        if os.environ.get("ARIA_DEBUG_REWARD"):
            print(f"[ARIA REWARD] Base: {base_reward:.3f}, Components: {reward_components}, Penalty: {issue_penalty:.3f}, Final: {reward:.3f}")

        # Update metrics with final reward
        run_metrics.reward = reward

        # Update bandit
        try:
            bandit_metrics = {
                "coverage_score": cov_score,
                "exemplar_fit": reward,
                "diversity": r.diversity_mm,
            }
            self.bandit_update(
                preset_name=preset.name,
                reward=reward,
                state_path=self.state_path,
                features=feats,  # Pass features for LinUCB learning
            )
        except Exception as e:
            print(f"[ARIA] Bandit update failed: {e}", file=sys.stderr)

        # Save metadata (including query features for LinUCB)
        # Convert any numpy arrays in feats dict to lists for JSON serialization
        def _serialize_feats(f: Dict[str, Any]) -> Dict[str, Any]:
            """Convert numpy arrays to lists for JSON serialization"""
            result = {}
            for k, v in f.items():
                if hasattr(v, 'tolist'):  # numpy array
                    result[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    result[k] = [x.tolist() if hasattr(x, 'tolist') else x for x in v]
                else:
                    result[k] = v
            return result

        (run_dir / "run.meta.json").write_text(
            json.dumps({
                "query": query_text,
                "preset": preset.name,
                "flags": flags_dict,
                "reward": reward,
                "phase": phase,
                "query_features": _serialize_feats(feats),
            }, indent=2),
            encoding="utf-8",
        )
        (run_dir / "pack.stats.txt").write_text(
            self.telemetry_line(run_metrics) + "\n", encoding="utf-8"
        )

        result = {
            "preset": preset.name,
            "reason": reason,
            "flags": flags,
            "pack": str(pack_path),
            "filtered": str(filtered_path),
            "postfiltered": filtered_path != pack_path,
            "metrics": {
                "retrieval": dataclasses.asdict(r),
                "generation": dataclasses.asdict(gen),
                "timing": dataclasses.asdict(timing),
                "issues": run_metrics.issues,
                "reward": reward,
            },
            "run_dir": str(run_dir),
        }

        # Add anchor information if available (NEW)
        if anchor_mode:
            result["anchor"] = {
                "mode": anchor_mode,
                "alignment": anchor_alignment,
                "template": anchor_template,  # Full template text for LLM
                "template_length": len(anchor_template) if anchor_template else 0,
            }

            # Also save anchor template to run directory for reference
            if anchor_template:
                anchor_file = run_dir / f"anchor_{anchor_mode}.md"
                anchor_file.write_text(anchor_template, encoding="utf-8")
                result["anchor"]["template_file"] = str(anchor_file)

        # Add perspective information to result (NEW)
        result["perspective"] = {
            "primary": perspective_analysis.get("primary", "mixed"),
            "confidence": perspective_analysis.get("confidence", 0.0),
            "weights": perspective_analysis.get("weights", {}),
        }

        return result

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ARIA Core - Adaptive Resonant Intelligent Architecture")
    ap.add_argument("query", nargs="+", help="Query text")
    ap.add_argument("--index", default=None, help="Index roots (colon-separated)")
    ap.add_argument("--out-dir", default=None, help="Output directory")
    ap.add_argument("--state", default=None, help="Bandit state file")
    ap.add_argument("--exemplars", default=None, help="Exemplars file")
    ap.add_argument("--no-session", action="store_true", help="Disable session enforcement")
    ap.add_argument("--preset", default=None, help="Manual preset override")
    ap.add_argument("--with-anchor", action="store_true", help="Enable 16-anchor reasoning system")
    args = ap.parse_args()

    # Parse index roots
    index_roots = None
    if args.index:
        index_roots = [p.strip() for p in args.index.replace(",", ":").split(":") if p.strip()]

    aria = ARIA(
        index_roots=index_roots,
        out_root=args.out_dir,
        state_path=args.state,
        exemplars_path=args.exemplars,
        enforce_session=not args.no_session,
    )
    res = aria.query(
        " ".join(args.query),
        with_anchor=getattr(args, 'with_anchor', False),
        preset_override=args.preset
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    raise SystemExit(main())
