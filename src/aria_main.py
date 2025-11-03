#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, time, subprocess, sys, shutil, inspect, dataclasses
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Mapping, cast

# --- paths -----------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT if (MODULE_ROOT / "src").exists() else MODULE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# --------------------------------------------------------------------

from query_features import QueryFeatureExtractor
from presets import preset_to_cli_args, Preset
from metrics_utils import (
    compute_retrieval_stats,
    coverage_score,
    telemetry_line,
    RunMetrics,
    TimingStats,
    detect_issues,
)
from exemplar_fit import ExemplarFitScorer
from aria_terminal import get_terminal
from aria_retrieval import ARIARetrieval
from aria_postfilter import ARIAPostfilter
from aria_config import get_config
from anchor_selector import AnchorSelector  # NEW: Import anchor selector

# EXPLORATION: Quaternion State + PCA Rotation
try:
    from aria_exploration import ExplorationManager

    EXPLORATION_AVAILABLE = True
except ImportError as e:
    print(f"[ARIA] WARNING: Exploration components unavailable: {e}", file=sys.stderr)
    EXPLORATION_AVAILABLE = False
    ExplorationManager = None  # type: ignore

# PHASE 1: Curiosity Engine
from aria_curiosity import ARIACuriosity
import asyncio

# PHASE 6: Resonance Detection & Self-Tuning
from aria_resonance import ResonanceDetector, SelfTuningEngine, MetaPatternExtractor

# PHASE 3: Multi-Turn Memory
try:
    from aria_session import SessionManager

    SESSION_AVAILABLE = True
except ImportError:
    SESSION_AVAILABLE = False
    SessionManager = None  # type: ignore
    print(
        "[ARIA] WARNING: aria_session.py not found, Phase 3 disabled", file=sys.stderr
    )

# WAVE 1: Meta-Learning Components
try:
    from auto_tuner import AnchorAutoTuner
    from source_reliability_scorer import SourceReliabilityScorer

    WAVE1_AVAILABLE = True
except ImportError as e:
    print(f"[ARIA] WARNING: Wave 1 components unavailable: {e}", file=sys.stderr)
    WAVE1_AVAILABLE = False
    AnchorAutoTuner = None  # type: ignore
    SourceReliabilityScorer = None  # type: ignore

# WAVE 2: Advanced Reasoning Components
try:
    from temporal_tracker import TemporalTracker
    from conversation_state_graph import ConversationStateGraph
    from uncertainty_quantifier import UncertaintyQuantifier
    from multi_hop_reasoner import MultiHopReasoner
    from contradiction_detector import ContradictionDetector

    WAVE2_AVAILABLE = True
except ImportError as e:
    print(f"[ARIA] WARNING: Wave 2 components unavailable: {e}", file=sys.stderr)
    WAVE2_AVAILABLE = False
    TemporalTracker = None  # type: ignore
    ConversationStateGraph = None  # type: ignore
    UncertaintyQuantifier = None  # type: ignore
    MultiHopReasoner = None  # type: ignore
    ContradictionDetector = None  # type: ignore

# Import bandit functions
try:
    from bandit_context import (
        select_preset as bandit_select,
        give_reward as bandit_update,
    )
except Exception:
    import bandit_context as _bandit

    bandit_select = getattr(_bandit, "select_preset")
    bandit_update = getattr(_bandit, "give_reward")


def _slug(s: str) -> str:
    import hashlib

    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = "-".join(s.split())[:60]
    h = hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()[:6]
    return f"{s or 'query'}-{h}"


def _enforce_session(config) -> None:
    """Ensure a live session using the repo's session_cli.py (absolute path)."""
    if not config.session_enforce:
        return

    if os.environ.get("RAG_ENFORCE_SESSION", "1") not in (
        "1",
        "true",
        "TRUE",
        "Yes",
        "yes",
    ):
        return
    cli = os.environ.get("RAG_SESSION_CLI") or str(
        (REPO_ROOT / "session_cli.py").resolve()
    )
    if not Path(cli).exists():
        return
    py = os.environ.get("PYTHON", "python3")
    ttl = str(config.session_ttl_hours)
    env = os.environ.copy()
    try:
        subprocess.run(
            [py, cli, "ensure", "--ttl-hours", ttl],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"[SECURITY] Session enforcement failed: {msg}")


def _feats_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {}


def _load_exemplars(path: str) -> list[str]:
    """Load exemplar lines from file"""
    p = Path(path)
    if not p.exists():
        return []
    return [
        line.strip()
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def _load_anchor(anchor_dir: Path, mode: str) -> Optional[str]:
    """Load anchor markdown file for the given mode"""
    anchor_file = anchor_dir / f"{mode}.md"
    if not anchor_file.exists():
        print(f"[ARIA] Warning: Anchor file not found: {anchor_file}", file=sys.stderr)
        return None
    try:
        return anchor_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ARIA] Error loading anchor {mode}: {e}", file=sys.stderr)
        return None


class ARIA:
    def __init__(
        self,
        config_path: Optional[str] = None,
        with_anchor: bool = True,  # âœ… FIXED: Enable multi-anchor by default (v44)
    ) -> None:
        """
        Initialize ARIA with YAML config

        Args:
            config_path: Path to aria_config.yaml (auto-detects if None)
            with_anchor: Enable multi-anchor reasoning system
        """
        # Load config from YAML
        self.config = get_config(config_path)
        self.with_anchor = with_anchor

        # Setup paths from config
        self.index_roots = self.config.index_roots
        self.out_root = Path(self.config.output_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.config.bandit_state
        self.exemplars_path = self.config.exemplars

        # NEW: Setup anchor system
        if self.with_anchor:
            # Anchor directory (default: /media/notapplicable/ARIA-knowledge/anchors/)
            self.anchor_dir = Path("/media/notapplicable/ARIA-knowledge/anchors")
            if not self.anchor_dir.exists():
                self.anchor_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[ARIA] Created anchor directory: {self.anchor_dir}",
                    file=sys.stderr,
                )
                self.with_anchor = False
            else:
                # FIXED: Pass exemplar_path to AnchorSelector for pattern matching
                exemplar_path = (
                    Path(self.exemplars_path) if self.exemplars_path else None
                )
                self.anchor_selector = AnchorSelector(exemplar_path=exemplar_path)
                print(f"[ARIA] Multi-Anchor System enabled", file=sys.stderr)
                if exemplar_path and exemplar_path.exists():
                    print(
                        f"[ARIA] ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Loaded exemplars: {exemplar_path}",
                        file=sys.stderr,
                    )

        # PHASE 1: Initialize Curiosity Engine
        self.curiosity = None
        if self.config.curiosity_enabled:
            exemplar_file = Path(self.exemplars_path) if self.exemplars_path else None
            self.curiosity = ARIACuriosity(
                exemplar_path=(
                    exemplar_file
                    if (exemplar_file and exemplar_file.exists())
                    else None
                ),
                personality=self.config.curiosity_personality,
                gap_threshold=self.config.curiosity_gap_threshold,
            )
            print(
                f"[ARIA] ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Phase 1: Curiosity Engine enabled (personality={self.config.curiosity_personality})",
                file=sys.stderr,
            )

        # PHASE 6: Initialize Resonance Detector & Self-Tuning
        resonance_state_dir = self.out_root.parent / "var" / "resonance"
        resonance_state_dir.mkdir(parents=True, exist_ok=True)

        self.resonance_detector = ResonanceDetector(
            history_size=10, stable_threshold=5, resonance_floor=0.85
        )
        self.self_tuner = SelfTuningEngine(self.resonance_detector)
        self.meta_extractor = MetaPatternExtractor()

        # Load previous resonance patterns if they exist
        resonance_patterns_file = resonance_state_dir / "patterns.json"
        if resonance_patterns_file.exists():
            self.resonance_detector.load_patterns(resonance_patterns_file)
            if self.config.logging_debug_mode:
                num_patterns = len(self.resonance_detector.resonance_patterns)
                print(
                    f"[ARIA] ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Phase 6: Resonance Detection enabled ({num_patterns} patterns loaded)",
                    file=sys.stderr,
                )
        else:
            if self.config.logging_debug_mode:
                print(
                    f"[ARIA] ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ Phase 6: Resonance Detection enabled (learning mode)",
                    file=sys.stderr,
                )

        self.resonance_state_dir = resonance_state_dir
        self.query_sequence = []  # Track query sequence for meta-pattern extraction

        # PHASE 3: Initialize Session Manager for Multi-Turn Memory
        if SESSION_AVAILABLE and SessionManager is not None:
            session_dir = self.out_root.parent / "var" / "sessions"
            self.session_manager = SessionManager(state_dir=session_dir)
            if self.config.logging_debug_mode:
                print(
                    f"[ARIA] Ã¢Å“â€ Phase 3: Multi-Turn Memory enabled",
                    file=sys.stderr,
                )
        else:
            self.session_manager = None

        # WAVE 1: Initialize Meta-Learning Components
        if WAVE1_AVAILABLE and AnchorAutoTuner and SourceReliabilityScorer:
            # Auto-tuner for anchor weights
            tuner_state = self.out_root.parent / "var" / "anchor_tuner.json"
            self.auto_tuner = AnchorAutoTuner(
                state_path=tuner_state,
                learning_rate=0.1,
                success_threshold=0.6,
                adjustment_frequency=50,
            )

            # Source reliability scorer
            reliability_state = self.out_root.parent / "var" / "source_reliability.json"
            self.source_scorer = SourceReliabilityScorer(state_path=reliability_state)

            if self.config.logging_debug_mode:
                print(
                    "[ARIA] Ã¢Å“â€ Wave 1: Meta-Learning enabled (AutoTuner + Source Reliability)",
                    file=sys.stderr,
                )
        else:
            self.auto_tuner = None
            self.source_scorer = None

        # WAVE 2: Initialize Advanced Reasoning Components
        if WAVE2_AVAILABLE and TemporalTracker:
            # Temporal tracker for conversation drift
            temporal_state_dir = self.out_root.parent / "var" / "temporal"
            temporal_state_dir.mkdir(parents=True, exist_ok=True)
            temporal_state_file = temporal_state_dir / "conversation_arc.json"
            self.temporal_tracker = TemporalTracker(state_path=temporal_state_file)

            # Conversation state graph for entity tracking
            if ConversationStateGraph:
                graph_state = self.out_root.parent / "var" / "conversation_graph.json"
                self.conversation_graph = ConversationStateGraph(state_path=graph_state)
            else:
                self.conversation_graph = None

            # Uncertainty quantifier
            if UncertaintyQuantifier:
                self.uncertainty_quantifier = UncertaintyQuantifier()
            else:
                self.uncertainty_quantifier = None

            # Multi-hop reasoner
            if MultiHopReasoner:
                # Create retrieval function wrapper
                def retrieval_wrapper(query: str):
                    """Wrapper to make ARIARetrieval compatible with MultiHopReasoner"""
                    result = self.retrieval.retrieve(
                        query=query, top_k=20, per_file_limit=6000
                    )
                    return result.get("items", [])

                self.multi_hop_reasoner = MultiHopReasoner(
                    retrieval_fn=retrieval_wrapper, max_hops=3
                )
            else:
                self.multi_hop_reasoner = None

            # Contradiction detector
            if ContradictionDetector:
                self.contradiction_detector = ContradictionDetector()
            else:
                self.contradiction_detector = None

            if self.config.logging_debug_mode:
                components = []
                if self.temporal_tracker:
                    components.append("TemporalTracker")
                if self.conversation_graph:
                    components.append("StateGraph")
                if self.uncertainty_quantifier:
                    components.append("Uncertainty")
                if self.multi_hop_reasoner:
                    components.append("MultiHop")
                if self.contradiction_detector:
                    components.append("Contradiction")
                print(
                    f"[ARIA] Ã¢Å“â€ Wave 2: Advanced Reasoning enabled ({', '.join(components)})",
                    file=sys.stderr,
                )
        else:
            self.temporal_tracker = None
            self.conversation_graph = None
            self.uncertainty_quantifier = None
            self.multi_hop_reasoner = None
            self.contradiction_detector = None

        # EXPLORATION: Initialize Quaternion + PCA Exploration
        self.exploration_manager = None
        if EXPLORATION_AVAILABLE and ExplorationManager:
            exploration_state_dir = self.out_root.parent / "var" / "exploration"
            exploration_state_dir.mkdir(parents=True, exist_ok=True)

            try:
                self.exploration_manager = ExplorationManager(
                    state_dir=exploration_state_dir,
                    enable_quaternion=True,
                    enable_pca=True,  # âœ… FIXED: Enable PCA subspace rotations (v44)
                )

                # Fit semantic model on corpus (first 500 chars of 100 files)
                if self.index_roots:
                    corpus_texts = []
                    for root_path in self.index_roots[:3]:  # Sample from first 3 roots
                        root = Path(root_path)
                        if not root.exists():
                            continue

                        for fpath in root.rglob("*.txt"):
                            if len(corpus_texts) >= 100:
                                break
                            try:
                                text = fpath.read_text(
                                    encoding="utf-8", errors="ignore"
                                )[:500]
                                if len(text) > 50:  # Minimum viable text
                                    corpus_texts.append(text)
                            except:
                                pass

                        if len(corpus_texts) >= 100:
                            break

                    if len(corpus_texts) >= 3:  # Minimum for semantic model
                        self.exploration_manager.fit_semantic_model(corpus_texts)
                        if self.config.logging_debug_mode:
                            print(
                                f"[ARIA] 🌀 Exploration System enabled ({len(corpus_texts)} corpus samples)",
                                file=sys.stderr,
                            )
                    else:
                        if self.config.logging_debug_mode:
                            print(
                                f"[ARIA] Warning: Insufficient corpus for exploration fitting (<3 texts)",
                                file=sys.stderr,
                            )
                        self.exploration_manager = None

            except Exception as e:
                print(
                    f"[ARIA] Warning: Could not initialize exploration: {e}",
                    file=sys.stderr,
                )
                self.exploration_manager = None

        # DEBUG: Log config to stderr for visibility
        if self.config.logging_debug_mode:
            print(
                f"[ARIA] Config loaded from: {self.config._config_path}",
                file=sys.stderr,
            )
            print(f"[ARIA] Index roots: {self.index_roots}", file=sys.stderr)
            print(f"[ARIA] Output dir: {self.out_root}", file=sys.stderr)
            print(f"[ARIA] Anchor system: {self.with_anchor}", file=sys.stderr)
            print(
                f"[ARIA] Exploration system: {self.exploration_manager is not None}",
                file=sys.stderr,
            )

        # Initialize components
        self.feat_ex = QueryFeatureExtractor()

        # Load exemplars if provided
        exemplars_list = (
            _load_exemplars(self.exemplars_path) if self.exemplars_path else None
        )
        self.exemplar_scorer = (
            ExemplarFitScorer(exemplars_list) if exemplars_list else None
        )

        # Initialize ARIARetrieval with config values
        index_paths = [Path(p) for p in self.index_roots]
        self.retrieval = ARIARetrieval(
            index_roots=index_paths,
            cache_dir=None,
            prefer_dirs=[],
            max_per_file=self.config.retrieval_max_per_file,
        )

        # Initialize ARIAPostfilter with config values
        self.postfilter = ARIAPostfilter(
            max_per_source=self.config.postfilter_max_per_source,
            min_keep=self.config.postfilter_min_keep,
        )

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query through ARIA pipeline

        Hot-reloads config if it changed since last query
        """
        # Auto-reload config if it changed
        if self.config.needs_reload():
            print("[ARIA] Config file changed, reloading...", file=sys.stderr)
            self.config.reload()

        # Get terminal (respects config.terminal_enabled)
        term = get_terminal(enabled=self.config.terminal_enabled)
        pipeline_start = time.time()

        if self.config.terminal_enabled and self.config.terminal_show_chunks:
            term.show_query(query_text)

        # PHASE 3: Check if this is a follow-up query and get conversation context
        is_followup = False
        conversation_context = ""
        if self.session_manager:
            is_followup = self.session_manager.is_follow_up(query_text)
            if is_followup:
                conversation_context = self.session_manager.get_recent_context(
                    max_chars=2000
                )
                if self.config.logging_debug_mode:
                    print(
                        f"[ARIA] ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ Follow-up detected, using conversation context ({len(conversation_context)} chars)",
                        file=sys.stderr,
                    )

            # Add user message to session
            self.session_manager.add_message("user", query_text)

        _enforce_session(self.config)

        # NEW: Detect reasoning mode if anchor system is enabled
        reasoning_mode = None
        anchor_content = None
        if self.with_anchor:
            reasoning_mode = self.anchor_selector.select_mode(query_text)
            anchor_content = _load_anchor(self.anchor_dir, reasoning_mode)
            if reasoning_mode:
                print(
                    f"[ARIA] ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ Reasoning Mode: {reasoning_mode}",
                    file=sys.stderr,
                )

        slug = _slug(query_text)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.out_root / ts / slug
        run_dir.mkdir(parents=True, exist_ok=True)

        feats = _feats_to_dict(self.feat_ex.extract(query_text))

        # Bandit selection
        if self.config.bandit_enabled:
            preset, reason, meta = bandit_select(feats, state_path=self.state_path)
            phase = meta.get("phase", "unknown")
        else:
            # If bandit disabled, use balanced preset from config or create default
            balanced_config = self.config.presets.get(
                "balanced",
                {"top_k": 32, "sem_limit": 128, "rotations": 2, "rotation_subspace": 4},
            )
            preset = Preset(name="balanced", args=balanced_config)
            reason = "bandit_disabled"
            phase = "no_bandit"

        if self.config.terminal_enabled:
            term.show_bandit_selection(preset.name, reason, phase, feats)

        # Get top_k from config (overrides preset)
        top_k = self.config.retrieval_top_k

        flags = preset_to_cli_args(preset)
        pack_path = run_dir / "last_pack.json"
        filtered_path = run_dir / "last_pack.filtered.json"

        if self.config.terminal_enabled:
            term.show_retrieval_start(self.index_roots, flags)

        start = time.time()

        # Call ARIA retrieval directly
        try:
            result = self.retrieval.retrieve(
                query=query_text,
                top_k=top_k,
                per_file_limit=self.config.retrieval_per_file_limit,
            )

            # Save pack to JSON
            pack_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        except Exception as e:
            if self.config.terminal_enabled:
                term.show_error("Retrieval", str(e))
            return {
                "preset": preset.name,
                "reason": reason,
                "result": {"ok": False, "error": str(e)},
                "reasoning_mode": reasoning_mode,  # NEW
                "anchor": anchor_content,  # NEW
            }

        retrieval_time = time.time() - start

        # Extract chunks from result
        raw_chunks = result.get("items", [])
        if self.config.terminal_enabled and self.config.terminal_show_chunks:
            term.show_chunks_summary(raw_chunks, show_all=False)

        # Apply postfilter if enabled
        if self.config.postfilter_enabled:
            if self.config.terminal_enabled:
                term.show_postfilter_start("Quality + Diversity")
            postfilter_start = time.time()

            try:
                filter_result = self.postfilter.filter(
                    items=raw_chunks,
                    enable_quality=self.config.postfilter_quality,
                    enable_topic=self.config.postfilter_topic,
                    enable_diversity=self.config.postfilter_diversity,
                )

                filtered_chunks = filter_result["items"]
                postfilter_time = time.time() - postfilter_start

                # Save filtered pack
                filtered_path = run_dir / "last_pack.filtered.json"
                filtered_path.write_text(
                    json.dumps(
                        {"items": filtered_chunks}, ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )

                if self.config.terminal_enabled:
                    term.show_postfilter_results(len(raw_chunks), len(filtered_chunks))

            except Exception as e:
                print(f"[ARIA] Postfilter failed: {e}", file=sys.stderr)
                filtered_chunks = raw_chunks
                filtered_path = pack_path
                postfilter_time = 0.0
        else:
            # Postfilter disabled, use raw chunks
            filtered_chunks = raw_chunks
            filtered_path = pack_path
            postfilter_time = 0.0

        # 🌀 EXPLORATION: Apply quaternion state exploration (golden ratio spiral)
        exploration_output = None
        exploration_time = 0.0

        if self.exploration_manager and filtered_chunks:
            exploration_start = time.time()

            try:
                if self.config.logging_debug_mode:
                    print(
                        f"[ARIA] 🌀 Applying exploration (quaternion+PCA+φ spiral) to {len(filtered_chunks)} chunks",
                        file=sys.stderr,
                    )

                # Prepare results for exploration
                exploration_input = {"items": filtered_chunks, "query": query_text}

                # Apply golden ratio spiral exploration
                exploration_result = self.exploration_manager.explore(
                    query=query_text,
                    retrieval_results=exploration_input,
                    strategy="golden_ratio_spiral",
                    momentum_weight=0.25,
                    recall_similar=True,
                    save_state=True,
                )

                # Update filtered chunks with exploration reranking
                filtered_chunks = exploration_result.get("items", filtered_chunks)
                exploration_output = exploration_result.get("exploration", {})

                exploration_time = time.time() - exploration_start

                if self.config.logging_debug_mode and exploration_output.get(
                    "applied", False
                ):
                    print(
                        f"[ARIA] ✅ Exploration complete ({exploration_time*1000:.1f}ms, strategy: {exploration_output.get('strategy', 'none')})",
                        file=sys.stderr,
                    )

            except Exception as e:
                print(f"[ARIA] Warning: Exploration failed: {e}", file=sys.stderr)
                exploration_output = {
                    "strategy": "golden_ratio_spiral",
                    "applied": False,
                    "error": str(e),
                }
                exploration_time = 0.0

        # CRITICAL FIX: Apply exemplar scoring to rerank chunks
        if self.exemplar_scorer and filtered_chunks:
            if self.config.logging_debug_mode:
                print(
                    f"[ARIA] ðŸ“Š Applying exemplar fit scoring to {len(filtered_chunks)} chunks",
                    file=sys.stderr,
                )

            exemplar_start = time.time()

            # Score each chunk with exemplar fit
            for chunk in filtered_chunks:
                text = chunk.get("text", "") or chunk.get("content", "")
                if text:
                    # Calculate exemplar fit for this chunk
                    fit_score, _ = self.exemplar_scorer.calculate_fit(
                        text, coverage_score=1.0
                    )
                    # Add exemplar_fit to chunk
                    chunk["exemplar_fit"] = fit_score
                    # Boost the chunk's score based on exemplar fit
                    original_score = chunk.get("score", 0.5)
                    chunk["score"] = original_score * 0.6 + fit_score * 0.4

            # Rerank by combined score
            filtered_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

            exemplar_time = time.time() - exemplar_start

            if self.config.logging_debug_mode:
                avg_fit = sum(c.get("exemplar_fit", 0) for c in filtered_chunks) / len(
                    filtered_chunks
                )
                print(
                    f"[ARIA] âœ… Exemplar scoring complete ({exemplar_time*1000:.1f}ms, avg fit: {avg_fit:.3f})",
                    file=sys.stderr,
                )
        else:
            exemplar_time = 0.0

        elapsed = retrieval_time + postfilter_time

        # Use filtered chunks for stats
        chunks = filtered_chunks
        chunks_mapped = cast(List[Mapping[str, Any]], chunks)
        r = compute_retrieval_stats(chunks_mapped)

        # Extract text for coverage score
        chunk_texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
        retrieved_text = "\n\n".join(chunk_texts)

        cov_score = coverage_score(query_text, retrieved_text)

        # Calculate reward using exemplar fit
        reward = 0.0
        if self.exemplar_scorer and chunks:
            try:
                reward, _ = self.exemplar_scorer.calculate_fit(
                    retrieved_text[:5000], cov_score
                )
            except Exception:
                pass

        # PHASE 1: Process with Curiosity Engine
        curiosity_output = None
        if self.curiosity and self.config.curiosity_enabled:
            try:
                # Convert chunks to format expected by curiosity engine
                curiosity_chunks = [
                    {
                        "text": chunk.get("text", ""),
                        "source": chunk.get("source", "unknown"),
                        "score": chunk.get("score", 0.0),
                    }
                    for chunk in chunks
                ]

                # Run curiosity analysis (async)
                curiosity_result = asyncio.run(
                    self.curiosity.process(
                        query=query_text,
                        retrieved_chunks=curiosity_chunks,
                        confidence=float(cov_score),
                        mode="adaptive",  # Could be config-driven
                        enable_socratic=self.config.curiosity_enable_socratic,
                    )
                )

                # Format curiosity output for display and injection
                curiosity_output = {
                    "gap_score": curiosity_result.get("gaps", {}).get("gap_score", 0.0),
                    "missing_aspects": curiosity_result.get("gaps", {}).get(
                        "missing_aspects", []
                    ),
                    "socratic_questions": curiosity_result.get(
                        "socratic_questions", {}
                    ),
                    "needs_followup": curiosity_result.get("needs_followup", False),
                    "curiosity_level": curiosity_result.get("curiosity_level", 0.0),
                }

                # Display curiosity insights in terminal
                if self.config.terminal_enabled and self.config.logging_debug_mode:
                    gap_pct = curiosity_output["gap_score"] * 100
                    print(
                        f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€šÃ‚Â Gap Detection: {gap_pct:.1f}% ({len(curiosity_output['missing_aspects'])} aspects missing)",
                        file=sys.stderr,
                    )

                    if curiosity_output["socratic_questions"].get("internal"):
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢Ãƒâ€šÃ‚Â¡ Internal Questions: {len(curiosity_output['socratic_questions']['internal'])}",
                            file=sys.stderr,
                        )

                    if curiosity_output["socratic_questions"].get("external"):
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ External Questions: {len(curiosity_output['socratic_questions']['external'])}",
                            file=sys.stderr,
                        )
                        for q in curiosity_output["socratic_questions"]["external"][:2]:
                            print(
                                f"[ARIA]    ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {q}",
                                file=sys.stderr,
                            )

            except Exception as e:
                print(f"[ARIA] Warning: Curiosity engine failed: {e}", file=sys.stderr)
                curiosity_output = None

        # WAVE 2: Advanced Reasoning Processing
        uncertainty_breakdown = None
        contradictions = None
        temporal_analysis = None

        # Uncertainty Quantification
        if self.uncertainty_quantifier:
            try:
                # Quantify epistemic and aleatoric uncertainty
                uncertainty_breakdown = self.uncertainty_quantifier.quantify(
                    query=query_text,
                    chunks=[
                        {
                            "text": c.get("text", ""),
                            "score": c.get("score", 0.0),
                            "source": c.get("source", "unknown"),
                        }
                        for c in chunks
                    ],
                    contradictions=None,  # Will be filled after contradiction detection
                    retrieval_metrics={"coverage": cov_score},
                )

                if self.config.logging_debug_mode:
                    total_unc = uncertainty_breakdown.total_uncertainty
                    print(
                        f"[ARIA] Ã°Å¸â€Â® Uncertainty: {total_unc:.1%} (epistemic={uncertainty_breakdown.epistemic:.1%}, aleatoric={uncertainty_breakdown.aleatoric:.1%})",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(
                    f"[ARIA] Warning: Uncertainty quantification failed: {e}",
                    file=sys.stderr,
                )

        # Contradiction Detection
        if self.contradiction_detector and len(chunks) > 1:
            try:
                contradictions = self.contradiction_detector.detect(
                    chunks=[
                        {
                            "text": c.get("text", ""),
                            "source": c.get("source", "unknown"),
                        }
                        for c in chunks
                    ]
                )

                if contradictions and self.config.logging_debug_mode:
                    print(
                        f"[ARIA] Ã¢Å¡Â Ã¯Â¸Â  Contradictions detected: {len(contradictions)} conflicts",
                        file=sys.stderr,
                    )
                    for i, cont in enumerate(contradictions[:2], 1):
                        # Contradiction is a dataclass, access attributes directly
                        print(
                            f"[ARIA]    {i}. {cont.source1} Ã¢â€ â€ {cont.source2}",
                            file=sys.stderr,
                        )
            except Exception as e:
                print(
                    f"[ARIA] Warning: Contradiction detection failed: {e}",
                    file=sys.stderr,
                )

        # Temporal Tracking (for conversation drift)
        if self.temporal_tracker and self.session_manager:
            try:
                # Update temporal tracker with current query
                turn_index = len(getattr(self.session_manager, "turns", []))
                self.temporal_tracker.add_turn(
                    turn_index=turn_index,
                    user_message=query_text,
                    assistant_response="",  # Will be filled by LM Studio
                    entities=[],  # Could extract from conversation_graph if available
                )

                # Get temporal analysis
                temporal_analysis = {
                    "phase": (
                        self.temporal_tracker.arc.phase.value
                        if self.temporal_tracker.arc
                        else "opening"
                    ),
                    "drift_score": (
                        getattr(self.temporal_tracker.arc, "drift_score", 0.0)
                        if self.temporal_tracker.arc
                        else 0.0
                    ),
                    "coherence": (
                        getattr(self.temporal_tracker.arc, "coherence", 1.0)
                        if self.temporal_tracker.arc
                        else 1.0
                    ),
                }

                if temporal_analysis and self.config.logging_debug_mode:
                    phase = temporal_analysis.get("phase", "unknown")
                    drift_score = temporal_analysis.get("drift_score", 0.0)
                    print(
                        f"[ARIA] Ã°Å¸Å’Å  Conversation: {phase} (drift={drift_score:.2f})",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"[ARIA] Warning: Temporal tracking failed: {e}", file=sys.stderr)

        # Update the existing stats object with coverage and exemplar fit
        r.coverage_score = cov_score
        r.exemplar_fit = reward
        gen = r  # Use the complete stats object

        timing = TimingStats(
            retrieval_sec=retrieval_time,
            postfilter_sec=postfilter_time,
            total_sec=elapsed + exploration_time,  # Include exploration in total
        )

        # Extract flags safely from preset
        if hasattr(preset, "args") and isinstance(preset.args, dict):
            flags_dict = {
                "top_k": preset.args.get("top_k", top_k),
                "sem_limit": preset.args.get("sem_limit", 128),
                "rotations": preset.args.get("rotations", 1),
                "rotation_subspace": preset.args.get("rotation_subspace", 4),
            }
        else:
            flags_dict = {
                "top_k": getattr(preset, "top_k", top_k),
                "sem_limit": getattr(preset, "sem_limit", 128),
                "rotations": getattr(preset, "rotations", 1),
                "rotation_subspace": getattr(preset, "rotation_subspace", 4),
            }

        # PHASE 6: Measure Resonance (AFTER flags_dict is created)
        resonance_result = None
        if hasattr(self, "resonance_detector"):
            try:
                # Extract query features for resonance
                query_feats = _feats_to_dict(self.feat_ex.extract(query_text))

                # Get gap score from curiosity or default to 0
                gap_score = 0.0
                if curiosity_output:
                    gap_score = curiosity_output.get("gap_score", 0.0)

                # Create config snapshot
                config_snapshot = {
                    "preset": preset.name,
                    "reasoning_mode": reasoning_mode,
                    "flags": flags_dict,
                    "curiosity_enabled": self.config.curiosity_enabled,
                    "anchor_enabled": self.with_anchor,
                }

                # Measure resonance
                resonance_result = self.resonance_detector.measure_resonance(
                    coverage_score=cov_score,
                    gap_score=gap_score,
                    exemplar_fit=reward,
                    total_chunks=len(chunks),
                    unique_sources=r.unique_sources,
                    query_features=query_feats,
                    config_snapshot=config_snapshot,
                )

                # Display resonance in terminal
                if self.config.terminal_enabled and self.config.logging_debug_mode:
                    res_score = resonance_result["resonance"]
                    res_pct = res_score * 100
                    stability_icon = (
                        "ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢"
                        if resonance_result["is_stable"]
                        else "ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã‚Â¡Ãƒâ€šÃ‚Â¡"
                    )
                    confidence = resonance_result["confidence"]

                    print(
                        f"[ARIA] {stability_icon} Resonance: {res_pct:.1f}% (confidence: {confidence:.2f})",
                        file=sys.stderr,
                    )

                    if resonance_result["pattern_detected"]:
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€¦Ã‚Â½Ãƒâ€šÃ‚Â¯ New resonance pattern detected and saved!",
                            file=sys.stderr,
                        )

                    if resonance_result["is_stable"]:
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ Stable resonance achieved - optimal configuration learned",
                            file=sys.stderr,
                        )

                # Add to query sequence for meta-pattern extraction
                self.query_sequence.append(
                    {
                        "query": query_text,
                        "reasoning_mode": reasoning_mode,
                        "is_followup": is_followup,
                        "resonance": resonance_result["resonance"],
                        "timestamp": time.time(),
                    }
                )

                # Keep only recent queries (last 20)
                if len(self.query_sequence) > 20:
                    self.query_sequence = self.query_sequence[-20:]

                # Extract meta-patterns periodically (every 10 queries)
                if (
                    len(self.query_sequence) >= 10
                    and len(self.query_sequence) % 10 == 0
                ):
                    self.meta_extractor.add_query_sequence(self.query_sequence[-10:])
                    meta_patterns = self.meta_extractor.extract_patterns()

                    if meta_patterns and self.config.logging_debug_mode:
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸Ãƒâ€šÃ‚Â§Ãƒâ€šÃ‚Â  Meta-patterns extracted: {len(meta_patterns)}",
                            file=sys.stderr,
                        )

                # Check if self-tuning recommended
                avg_resonance = sum(
                    r["resonance"] for r in self.resonance_detector.resonance_history
                ) / max(len(self.resonance_detector.resonance_history), 1)

                if self.self_tuner.should_tune(
                    resonance_result["resonance"], avg_resonance
                ):
                    recommendations = self.self_tuner.extract_tuning_recommendations(
                        config_snapshot, query_feats
                    )

                    if recommendations and self.config.logging_debug_mode:
                        print(
                            f"[ARIA] ÃƒÆ’Ã‚Â°Ãƒâ€¦Ã‚Â¸ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€šÃ‚Â§ Self-tuning recommendations available:",
                            file=sys.stderr,
                        )
                        for key, value in recommendations.items():
                            print(
                                f"[ARIA]    ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {key}: {value}",
                                file=sys.stderr,
                            )

                # Save resonance patterns periodically (every 5 queries)
                if len(self.resonance_detector.resonance_history) % 5 == 0:
                    patterns_file = self.resonance_state_dir / "patterns.json"
                    self.resonance_detector.save_patterns(patterns_file)

            except Exception as e:
                print(
                    f"[ARIA] Warning: Resonance detection failed: {e}", file=sys.stderr
                )
                resonance_result = None

        # Build complete RunMetrics with all required fields
        run_metrics = RunMetrics(
            retrieval=dataclasses.asdict(r),
            generation=dataclasses.asdict(gen),
            timing=dataclasses.asdict(timing),
            issues=[],
            reward=reward,
            query=query_text,
            preset=preset.name,
            flags=flags_dict,
            exemplar_fit=reward,  # Include top-level exemplar fit
            coverage_score=cov_score,  # Include top-level coverage score
        )

        # Detect issues
        run_metrics.issues = detect_issues(
            run_metrics,
            {
                "slow_seconds": 8.0,
                "dup_ratio": 0.6,
                "min_unique_sources": 3,
                "min_coverage": 0.2,
            },
        )

        # Update bandit if enabled
        if self.config.bandit_enabled:
            try:
                bandit_update(
                    preset_name=preset.name,
                    reward=reward,
                    state_path=self.state_path,
                )
            except Exception as e:
                print(f"[ARIA] Bandit update failed: {e}", file=sys.stderr)

        # WAVE 1: Auto-Tuner Update (track anchor selection performance)
        if self.auto_tuner and self.with_anchor and hasattr(self, "anchor_selector"):
            try:
                # Get detected anchor mode (if multi-anchor is enabled)
                detected_mode = getattr(self, "_last_anchor_mode", "casual")

                # Track anchor selection with reward
                self.auto_tuner.record_selection(
                    query=query_text,
                    selected_mode=detected_mode,  # FIXED: Use selected_mode not anchor_mode
                    reward=reward,
                    patterns_matched={},  # Could extract from anchor_selector if needed
                )

                # Check if it's time to adjust weights
                if (
                    self.auto_tuner.selections_since_adjustment
                    >= self.auto_tuner.adjustment_frequency
                ):
                    adjustments = (
                        self.auto_tuner.export_weight_adjustments()
                    )  # FIXED: Use export_weight_adjustments()
                    if adjustments and self.config.logging_debug_mode:
                        print(
                            f"[ARIA] Ã°Å¸â€Â§ Auto-tuner: Computed {len(adjustments)} weight adjustments",
                            file=sys.stderr,
                        )
                    self.auto_tuner._save_state()  # FIXED: Use _save_state() private method
            except Exception as e:
                print(f"[ARIA] Warning: Auto-tuner update failed: {e}", file=sys.stderr)

        # WAVE 1: Source Reliability Scoring
        if self.source_scorer:
            try:
                # Update reliability scores for sources used
                self.source_scorer.record_retrieval(  # FIXED: Use record_retrieval() not update_source()
                    query=query_text,
                    chunks=chunks,
                    reward=reward,
                    chunk_scores=None,  # Could pass exemplar fit scores per chunk
                )

                # Save updated reliability scores
                self.source_scorer._save_state()  # FIXED: Use _save_state() private method
            except Exception as e:
                print(
                    f"[ARIA] Warning: Source reliability update failed: {e}",
                    file=sys.stderr,
                )

        # Save run metadata if telemetry enabled
        if self.config.logging_save_telemetry:
            (run_dir / "run.meta.json").write_text(
                json.dumps(
                    {
                        "query": query_text,
                        "preset": preset.name,
                        "flags": flags_dict,
                        "reward": reward,
                        "phase": phase,
                        "reasoning_mode": reasoning_mode,  # NEW
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "pack.stats.txt").write_text(
                telemetry_line(run_metrics) + "\n", encoding="utf-8"
            )

        # Display comprehensive metrics if terminal enabled
        if self.config.terminal_enabled:
            metrics_dict = {
                "retrieval": dataclasses.asdict(r),
                "generation": dataclasses.asdict(gen),
                "timing": dataclasses.asdict(timing),
                "issues": run_metrics.issues,
                "reward": reward,
            }
            if self.config.terminal_show_metrics:
                term.show_metrics(metrics_dict)
            if self.config.terminal_show_resources:
                term.show_resources()
            term.show_output_paths(str(pack_path), str(filtered_path))

            # Final summary
            total_pipeline_time = time.time() - pipeline_start
            term.show_final_summary(reward, total_pipeline_time)

        # PHASE 3: Save assistant response placeholder (LM Studio will fill in actual response)
        if self.session_manager:
            self.session_manager.add_message(
                "assistant", "[Response will be generated by LM Studio]"
            )

        return {
            "preset": preset.name,
            "reason": reason,
            "flags": flags_dict,
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
            "reasoning_mode": reasoning_mode,  # NEW: Add to JSON output
            "anchor": anchor_content,  # NEW: Add to JSON output
            # EXPLORATION: Quaternion + PCA exploration output
            "exploration": exploration_output,
            # PHASE 1: Curiosity Engine output
            "curiosity": curiosity_output,
            # PHASE 6: Resonance Detection output
            "resonance": resonance_result,
            # PHASE 3: Multi-Turn Memory info
            "is_followup": is_followup if self.session_manager else False,
            "conversation_context": (
                conversation_context if self.session_manager else ""
            ),
            # WAVE 2: Advanced Reasoning outputs
            "uncertainty": (
                {
                    "total": uncertainty_breakdown.total_uncertainty,
                    "confidence": uncertainty_breakdown.confidence,
                    "epistemic": uncertainty_breakdown.epistemic,
                    "aleatoric": uncertainty_breakdown.aleatoric,
                    "coverage": uncertainty_breakdown.coverage,
                    "reliability": uncertainty_breakdown.reliability,
                    "temporal": uncertainty_breakdown.temporal,
                    "factors": uncertainty_breakdown.factors,
                }
                if uncertainty_breakdown
                else None
            ),
            "contradictions": (
                [dataclasses.asdict(c) for c in contradictions]
                if contradictions
                else []
            ),
            "temporal": temporal_analysis if temporal_analysis else None,
        }


def main():
    import argparse

    # DEBUG: Log raw command line arguments BEFORE parsing
    if os.getenv("RAG_DEBUG"):
        print(f"[ARIA] Raw sys.argv: {sys.argv}", file=sys.stderr)

    ap = argparse.ArgumentParser(description="ARIA Orchestrator with YAML Config")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--config", default=None, help="Path to aria_config.yaml")
    ap.add_argument(
        "--with-anchor",
        action="store_true",
        help="Enable multi-anchor reasoning system",
    )  # NEW
    args = ap.parse_args()

    aria = ARIA(
        config_path=args.config, with_anchor=args.with_anchor
    )  # NEW: Pass anchor flag
    res = aria.query(" ".join(args.query))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
