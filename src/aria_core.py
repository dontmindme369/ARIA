#!/usr/bin/env python3
"""
aria_core.py — Unified ARIA orchestration with integrated session security
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



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
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT if (MODULE_ROOT / "src").exists() else MODULE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

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
        if override:
            p = Path(override).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        env = os.environ.get("RAG_SESSION_FILE")
        if env:
            p = Path(env).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p = (REPO_ROOT / ".lmstudio" / "session.json").resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    
    def _resolve_identity_file(self, override: Optional[str]) -> Path:
        if override:
            return Path(override).expanduser().resolve()
        env = os.environ.get("RAG_IDENTITY_FILE")
        if env:
            return Path(env).expanduser().resolve()
        repo_sec = REPO_ROOT / "security" / ".rag_identity.json"
        if repo_sec.exists():
            return repo_sec.resolve()
        alt = (REPO_ROOT / ".." / "security" / ".rag_identity.json")
        return alt.resolve()
    
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
    "local_rag_context_v7_guided_exploration.py",
    "rag_v7_multirot.py",
    "local_rag_context_v6_9k_adaptive_learn.py",
]
CANDIDATE_POST = [
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
    locations = [REPO_ROOT, Path.cwd(), REPO_ROOT / "src"]
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
        out_root: str = "./rag_runs/aria",
        state_path: str = "~/.rag_bandit_state.json",
        exemplars_path: Optional[str] = None,
        session_manager: Optional[SessionManager] = None,
        enforce_session: bool = True,
    ) -> None:
        self.index_roots = index_roots or [os.getenv("RAG_INDEX_ROOT", "./data")]
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.state_path = os.path.expanduser(state_path)
        self.exemplars_path = exemplars_path
        self.enforce_session = enforce_session
        
        # Initialize session manager
        self.session = session_manager or SessionManager()
        
        # Import dependencies
        try:
            from query_features import QueryFeatureExtractor
            from presets import preset_to_cli_args
            from metrics_utils import (
                compute_retrieval_stats,
                coverage_score,
                telemetry_line,
                RunMetrics,
                GenerationStats,
                TimingStats,
                detect_issues,
            )
            from exemplar_fit import ExemplarFitScorer
            from bandit_context import select_preset as bandit_select, give_reward as bandit_update
            
            self.feat_ex = QueryFeatureExtractor()
            self.preset_to_cli_args = preset_to_cli_args
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
            exemplars_list = _load_exemplars(exemplars_path) if exemplars_path else None
            self.exemplar_scorer = ExemplarFitScorer(exemplars_list) if exemplars_list else None
        except ImportError as e:
            sys.stderr.write(f"[ARIA] Failed to import dependencies: {e}\n")
            raise
    
    def _enforce_session(self) -> None:
        """Ensure valid session exists."""
        if not self.enforce_session:
            return
        if os.environ.get("RAG_ENFORCE_SESSION", "1") not in ("1", "true", "TRUE", "Yes", "yes"):
            return
        if not self.session.ensure_session():
            raise RuntimeError("[SECURITY] Session enforcement failed")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Execute retrieval query with bandit-selected preset.
        
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
        preset, reason, meta = self.bandit_select(feats, state_path=self.state_path)
        phase = meta.get("phase", "unknown")
        
        # Find scripts
        v7_script = _which_script(CANDIDATE_V7)
        post_script = _which_script(CANDIDATE_POST)
        
        if not v7_script:
            return {
                "preset": "default_fallback",
                "reason": "v7 script not found",
                "result": {"ok": False, "reason": "retrieval script not found"},
            }
        
        # Prepare retrieval
        flags = self.preset_to_cli_args(preset)
        pack_path = run_dir / "last_pack.json"
        filtered_path = run_dir / "last_pack.filtered.json"
        
        start = time.time()
        
        # Call v7 retrieval
        idx_arg = ":".join(str(p) for p in self.index_roots)
        cmd = [
            "python3", v7_script,
            "--query", query_text,
            "--index", idx_arg,
            "--out-dir", str(run_dir),
            *flags
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        except Exception as e:
            return {
                "preset": preset.name,
                "reason": reason,
                "result": {"ok": False, "error": str(e)},
            }
        
        if not pack_path.exists():
            return {
                "preset": preset.name,
                "reason": reason,
                "result": {"ok": False, "error": "v7 script did not create pack file"},
            }
        
        # Postfilter
        if post_script and pack_path.exists():
            try:
                subprocess.run([
                    "python3", post_script,
                    "--pack", str(pack_path)
                ], check=True, capture_output=True, text=True, timeout=10)
                
                filtered_txt = run_dir / "last_pack.filtered.txt"
                filtered_json = run_dir / "last_pack.filtered.json"
                
                if filtered_txt.exists():
                    filtered_path = filtered_txt
                elif filtered_json.exists():
                    filtered_path = filtered_json
                else:
                    filtered_path = pack_path
            except Exception as e:
                print(f"[ARIA] Postfilter failed: {e}", file=sys.stderr)
                filtered_path = pack_path
        else:
            filtered_path = pack_path
        
        elapsed = time.time() - start
        
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
        
        # Calculate reward
        reward = 0.0
        if self.exemplar_scorer and raw_chunks:
            try:
                reward, _ = self.exemplar_scorer.calculate_fit(retrieved_text[:5000], cov_score)
            except Exception:
                pass
        
        # Build metrics
        run_metrics = self.RunMetrics(
            retrieval=dataclasses.asdict(r),
            generation=dataclasses.asdict(gen),
            timing=dataclasses.asdict(timing),
            issues=[],
            reward=reward,
            query=query_text,
            preset=preset.name,
            flags=flags_dict,
        )
        
        # Detect issues
        run_metrics.issues = self.detect_issues(run_metrics, {
            "slow_seconds": 8.0,
            "dup_ratio": 0.6,
            "min_unique_sources": 3,
            "min_coverage": 0.2,
        })
        
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
            )
        except Exception as e:
            print(f"[ARIA] Bandit update failed: {e}", file=sys.stderr)
        
        # Save metadata
        (run_dir / "run.meta.json").write_text(
            json.dumps({
                "query": query_text,
                "preset": preset.name,
                "flags": flags_dict,
                "reward": reward,
                "phase": phase,
            }, indent=2),
            encoding="utf-8",
        )
        (run_dir / "pack.stats.txt").write_text(
            self.telemetry_line(run_metrics) + "\n", encoding="utf-8"
        )
        
        return {
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

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ARIA Core - Adaptive Resonant Intelligent Architecture")
    ap.add_argument("query", nargs="+", help="Query text")
    ap.add_argument("--index", default=os.getenv("RAG_INDEX_ROOT", "./data"))
    ap.add_argument("--out-dir", default="./rag_runs/aria")
    ap.add_argument("--state", default="~/.rag_bandit_state.json")
    ap.add_argument("--exemplars", default="data/exemplars.txt")
    ap.add_argument("--no-session", action="store_true", help="Disable session enforcement")
    args = ap.parse_args()
    
    aria = ARIA(
        index_roots=[p.strip() for p in args.index.replace(",", ":").split(":") if p.strip()],
        out_root=args.out_dir,
        state_path=args.state,
        exemplars_path=args.exemplars,
        enforce_session=not args.no_session,
    )
    res = aria.query(" ".join(args.query))
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    raise SystemExit(main())
