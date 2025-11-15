#!/usr/bin/env python3
"""
aria_training.py — Unified training and exemplar analysis

Consolidates:
- exemplar_fit.py (exemplar fit scoring with TF-IDF/cosine similarity)
- exemplar_report.py (exemplar coverage reporting and topic analysis)

Features:
- Exemplar fit scoring (style, citation, confidence)
- TF-IDF/cosine similarity (with sklearn fallback to Jaccard)
- Exemplar coverage reporting
- Topic-based breakdown
- Auto-detection of latest pack
- Support for .txt, .json, .jsonl formats
"""
from __future__ import annotations

import argparse, json, math, os, re, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Optional sklearn imports (graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────
HAVE_SK = False
TfidfVectorizer: Any = None
cosine_similarity: Any = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TFV
    from sklearn.metrics.pairwise import cosine_similarity as _COS
    TfidfVectorizer = _TFV
    cosine_similarity = _COS
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(s: str) -> List[str]:
    """Extract alphanumeric tokens from text."""
    return [t for t in re.sub(r"[^\w]+", " ", s.lower()).split() if t]

def _simple_cosine(a: str, b: str) -> float:
    """
    Fallback cosine similarity using token counts (no sklearn).
    
    Args:
        a: First text
        b: Second text
    
    Returns:
        Cosine similarity between 0.0 and 1.0
    """
    A = Counter(_tokenize(a))
    B = Counter(_tokenize(b))
    terms = set(A) | set(B)
    dot = sum(A[t] * B[t] for t in terms)
    na = math.sqrt(sum(v * v for v in A.values()))
    nb = math.sqrt(sum(v * v for v in B.values()))
    return (dot / (na * nb)) if na * nb else 0.0

def _jaccard(a: str, b: str) -> float:
    """
    Jaccard similarity between token sets.
    
    Args:
        a: First text
        b: Second text
    
    Returns:
        Jaccard similarity between 0.0 and 1.0
    """
    tok = lambda s: set(re.findall(r"[a-zA-Z0-9]{3,}", s.lower()))
    A, B = tok(a), tok(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

# ─────────────────────────────────────────────────────────────────────────────
# Exemplar Fit Scorer (from exemplar_fit.py)
# ─────────────────────────────────────────────────────────────────────────────

class ExemplarFitScorer:
    """
    Score answer quality based on exemplar similarity.
    
    Combines three metrics:
    - Style (40%): TF-IDF/cosine similarity to exemplars
    - Citations (30%): Citation density around optimal rate
    - Confidence (30%): Hedge word usage calibrated to coverage
    
    Uses sklearn TF-IDF if available, falls back to simple cosine.
    """
    
    def __init__(self, exemplars: List[str] | None):
        """
        Initialize scorer with exemplars.
        
        Args:
            exemplars: List of exemplar texts for style matching
        """
        self.exemplars = exemplars or []
        
        if HAVE_SK and self.exemplars:
            try:
                self.vect = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 3),
                    stop_words='english'
                )
                self.ex_vecs = self.vect.fit_transform(self.exemplars)
            except Exception:
                self.vect = None
                self.ex_vecs = None
        else:
            self.vect = None
            self.ex_vecs = None
    
    def calculate_fit(self, answer: str, coverage_score: float) -> Tuple[float, Dict[str, float]]:
        """
        Calculate exemplar fit score for an answer.
        
        Args:
            answer: Generated answer text
            coverage_score: Coverage score (0-1) for confidence calibration
        
        Returns:
            Tuple of (overall_score, component_scores_dict)
        """
        if not self.exemplars:
            return 0.5, {}
        
        style = self._style(answer)
        cite = self._citation(answer)
        conf = self._confidence(answer, coverage_score)
        
        overall = 0.40 * style + 0.30 * cite + 0.30 * conf
        return overall, {"style": style, "citations": cite, "confidence": conf}
    
    def _style(self, ans: str) -> float:
        """
        Score style similarity to exemplars using TF-IDF/cosine or fallback.
        
        Args:
            ans: Answer text
        
        Returns:
            Style score (0-1), average of top-3 similarities
        """
        if HAVE_SK and self.vect is not None:
            try:
                v = self.vect.transform([ans])
                sims = cosine_similarity(v, self.ex_vecs)[0]
                top = sorted(sims, reverse=True)[:min(3, len(sims))]
                return float(sum(top) / len(top)) if top else 0.5
            except Exception:
                pass
        
        if not self.exemplars:
            return 0.5
        
        scores = sorted(
            (_simple_cosine(ans, ex) for ex in self.exemplars),
            reverse=True
        )
        top = scores[:min(3, len(scores))]
        return float(sum(top) / len(top)) if top else 0.5
    
    def _citation(self, ans: str) -> float:
        """
        Score citation density (optimal ~0.2 citations per sentence).
        
        Args:
            ans: Answer text
        
        Returns:
            Citation score (0-1), penalized for deviation from optimal
        """
        pats = [r'\[\d+\]', r'\[[\w\-\.]+\]', r'\([\w\-\.]+\)']
        n_sent = max(1, len(ans.split('.')))
        count = sum(len(re.findall(p, ans)) for p in pats)
        dens = count / n_sent
        diff = abs(dens - 0.2)
        return max(0.0, 1.0 - 2.0 * diff)
    
    def _confidence(self, ans: str, coverage: float) -> float:
        """
        Score confidence calibration based on hedge word usage.
        
        Lower coverage should have more hedges. Optimal hedge rate
        varies from 2% (high coverage) to 3% (low coverage).
        
        Args:
            ans: Answer text
            coverage: Coverage score (0-1)
        
        Returns:
            Confidence score (0-1), penalized for miscalibration
        """
        hedges = [
            'might', 'may', 'could', 'possibly', 'perhaps', 'likely',
            'appears', 'seems', 'suggests', 'indicates', 'probably',
            'generally', 'typically', 'often', 'usually', 'sometimes'
        ]
        words = max(1, len(ans.split()))
        rate = sum(1 for h in hedges if h in ans.lower()) / words
        target = 0.02 * (1.5 - max(0.0, min(1.0, coverage)))
        diff = abs(rate - target)
        return max(0.0, 1.0 - 10.0 * diff)

# ─────────────────────────────────────────────────────────────────────────────
# Exemplar Coverage Reporter (from exemplar_report.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_exemplars(path: Path) -> Tuple[List[str], List[str]]:
    """
    Load exemplars from file (supports .txt, .json, .jsonl).
    
    Format options:
    - .txt: "topic:: text" (topic optional) or plain text lines
    - .json/.jsonl: {"text": "...", "topic": "..."} objects
    
    Args:
        path: Path to exemplars file
    
    Returns:
        Tuple of (topics, texts) lists
    """
    topics: List[str] = []
    texts: List[str] = []
    
    if path.suffix.lower() in (".jsonl", ".json"):
        try:
            for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                t = (obj.get("text") or obj.get("body") or "").strip()
                topic = (obj.get("topic") or obj.get("tag") or "general").strip()
                if t:
                    texts.append(t)
                    topics.append(topic or "general")
            if texts:
                return topics, texts
        except Exception:
            pass
    
    # Fallback: txt format
    if not path.exists():
        return [], []
    
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "::" in ln:
            topic, text = ln.split("::", 1)
            topics.append(topic.strip() or "general")
            texts.append(text.strip())
        else:
            topics.append("general")
            texts.append(ln)
    
    return topics, texts

def parse_pack(path: Path) -> List[str]:
    """
    Parse pack file to extract text chunks.
    
    Supports .json and .jsonl formats from postfilter.
    
    Args:
        path: Path to pack file
    
    Returns:
        List of text chunks
    """
    texts: List[str] = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    
    if path.suffix.lower() == ".jsonl":
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    if "text" in obj or "body" in obj:
                        t = (obj.get("text") or obj.get("body") or "")
                        if t:
                            texts.append(t)
                        continue
                    # Handle nested items
                    seq = obj.get("items") or obj.get("selected_filtered") or obj.get("sources")
                    if isinstance(seq, list):
                        for it in seq:
                            t = (it.get("text") or it.get("body") or "")
                            if t:
                                texts.append(t)
                        continue
            except Exception:
                # Fallback regex extraction
                m = re.search(r'"text"\s*:\s*"([^"]+)"', ln)
                if m:
                    texts.append(m.group(1))
        return texts
    
    # JSON format
    try:
        obj = json.loads(raw)
        for key in ("items", "selected_filtered", "sources"):
            seq = obj.get(key)
            if isinstance(seq, list):
                for it in seq:
                    t = (it.get("text") or it.get("body") or "")
                    if t:
                        texts.append(t)
                break
        return texts
    except Exception:
        return texts

def best_sims(exemplars: List[str], ctxs: List[str]) -> List[float]:
    """
    Calculate best similarity score for each exemplar vs all contexts.
    
    Uses TF-IDF/cosine if sklearn available, otherwise Jaccard.
    
    Args:
        exemplars: List of exemplar texts
        ctxs: List of context texts to compare against
    
    Returns:
        List of max similarity scores (one per exemplar)
    """
    if HAVE_SK and exemplars and ctxs:
        try:
            vect = TfidfVectorizer(min_df=1, max_df=0.9)
            X = vect.fit_transform(exemplars + ctxs)
            n_ex = len(exemplars)
            ex = X[0:n_ex, :]
            cx = X[n_ex:, :]
            sims = cosine_similarity(ex, cx)
            return [
                float(sims[i, :].max()) if sims.shape[1] else 0.0
                for i in range(sims.shape[0])
            ]
        except Exception:
            pass
    
    # Fallback to Jaccard
    return [max((_jaccard(e, c) for c in ctxs), default=0.0) for e in exemplars]

def auto_find_latest_pack(root: Path) -> Path | None:
    """
    Auto-detect latest pack file in rag_runs/aria.
    
    Prefers filtered packs over raw packs.
    
    Args:
        root: Repository root path
    
    Returns:
        Path to latest pack, or None if not found
    """
    aria_root = root / "rag_runs" / "aria"
    if not aria_root.exists():
        return None
    
    runs = sorted(
        (p for p in aria_root.glob("*/*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for run in runs:
        for name in ("pack.filtered.jsonl", "pack.filtered.json", "pack.raw.json"):
            p = run / name
            if p.exists():
                return p
    
    return None

class ExemplarReporter:
    """Generate exemplar coverage reports with topic breakdown."""
    
    def __init__(self, exemplars_path: Path):
        """
        Initialize reporter with exemplars.
        
        Args:
            exemplars_path: Path to exemplars file
        """
        self.exemplars_path = exemplars_path
        self.topics, self.exemplars = load_exemplars(exemplars_path)
    
    def report(self, pack_path: Path) -> Dict[str, Any]:
        """
        Generate coverage report for a pack.
        
        Args:
            pack_path: Path to pack file
        
        Returns:
            Report dictionary with coverage metrics
        """
        ctxs = parse_pack(pack_path)
        sims = best_sims(self.exemplars, ctxs)
        
        # Coverage threshold (lower for Jaccard)
        thr = 0.25 if HAVE_SK else 0.15
        covered = sum(1 for s in sims if s >= thr)
        coverage = covered / max(1, len(sims))
        
        # Topic breakdown
        by_topic: Dict[str, List[float]] = {}
        for t, s in zip(self.topics, sims):
            by_topic.setdefault(t, []).append(s)
        
        topic_rows = [
            {
                "topic": t,
                "n": len(v),
                "covered": sum(1 for x in v if x >= thr),
                "avg_sim": (sum(v) / len(v)) if v else 0.0
            }
            for t, v in sorted(
                by_topic.items(),
                key=lambda kv: (-len(kv[1]), kv[0])
            )
        ]
        
        return {
            "exemplars_count": len(self.exemplars),
            "ctx_chunks_count": len(ctxs),
            "method": "tfidf/cosine" if HAVE_SK else "jaccard",
            "threshold": thr,
            "global_coverage": coverage,
            "covered_count": covered,
            "total_count": len(sims),
            "topics": topic_rows,
            "pack_path": str(pack_path)
        }
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """
        Print formatted coverage report.
        
        Args:
            report: Report dictionary from report()
        """
        print("== EXEMPLAR COVERAGE REPORT ==")
        print(f"exemplars: {report['exemplars_count']}  |  "
              f"ctx-chunks: {report['ctx_chunks_count']}  |  "
              f"method: {report['method']}")
        print(f"global_coverage@{report['threshold']:.2f}: "
              f"{report['global_coverage']:.3f}  "
              f"({report['covered_count']}/{report['total_count']})")
        print()
        print("topic                          n  covered  avg_sim")
        print("--------------------------------  -------  -------")
        
        for row in report['topics']:
            print(f"{row['topic'][:30]:<30}  "
                  f"{row['n']:>3}  "
                  f"{row['covered']:>7}  "
                  f"{row['avg_sim']:>7.3f}")
        
        print()
        print(f"[pack] {report['pack_path']}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ARIA Training - exemplar fit scoring and coverage reporting")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    # Report command
    report_ap = sub.add_parser("report", help="Generate exemplar coverage report")
    report_ap.add_argument("--exemplars", default="data/exemplars.txt")
    report_ap.add_argument("--pack", default="")
    report_ap.add_argument("--json", action="store_true", help="Output JSON instead of text")
    
    # Score command (for testing)
    score_ap = sub.add_parser("score", help="Score a single answer")
    score_ap.add_argument("--exemplars", default="data/exemplars.txt")
    score_ap.add_argument("--answer", required=True)
    score_ap.add_argument("--coverage", type=float, default=0.7)
    
    args = ap.parse_args()
    
    if args.cmd == "report":
        repo = Path.cwd()
        ex_path = (repo / args.exemplars).resolve()
        pk_path = Path(args.pack).resolve() if args.pack else (auto_find_latest_pack(repo) or Path(""))
        
        if not ex_path.exists():
            print(f"[error] exemplars not found: {ex_path}", file=sys.stderr)
            sys.exit(2)
        if not pk_path or not pk_path.exists():
            print(f"[error] pack not found. Pass --pack or run a query first.", file=sys.stderr)
            sys.exit(2)
        
        reporter = ExemplarReporter(ex_path)
        report = reporter.report(pk_path)
        
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            reporter.print_report(report)
    
    elif args.cmd == "score":
        ex_path = Path(args.exemplars).resolve()
        if not ex_path.exists():
            print(f"[error] exemplars not found: {ex_path}", file=sys.stderr)
            sys.exit(2)
        
        _, exemplars = load_exemplars(ex_path)
        scorer = ExemplarFitScorer(exemplars)
        overall, components = scorer.calculate_fit(args.answer, args.coverage)
        
        print(f"Overall fit: {overall:.3f}")
        print(f"Components:")
        for k, v in components.items():
            print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
