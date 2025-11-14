#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from collections import defaultdict

README = """v7 retriever (offline-friendly) - DIVERSITY FIXED
- Scans multiple roots (comma/colon separated) recursively
- Extracts text from many formats without external deps
- Optional slow parsers if libs are present (PyPDF2, python-docx)
- Lexical scoring against query (token overlap) + source boosting
- NEW: include/exclude GLOBs (path-level) and must/must-regex (content-level) filters
- FIXED: Enforces per-file chunk limits to prevent single-source dominance
- Writes pack JSON with top-K items for downstream filters
"""

def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(description="Lightweight v7 retriever (multi-root, multi-format).")
    ap.add_argument("--query", required=False, default=None)
    ap.add_argument("--index", required=True, help="One or more index roots, colon/comma-separated")
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--out-dir", required=False, default=None)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--semantic-limit", type=int, default=256, dest="semantic_limit")
    ap.add_argument("--max-per-file", type=int, default=6, dest="max_per_file", 
                    help="Maximum chunks per file (enforces diversity)")
    ap.add_argument("--rotations", type=int, default=0)
    ap.add_argument("--rotation-subspace", type=int, default=0, dest="rotation_subspace")
    ap.add_argument("--rotation-angle", type=float, default=0.0, dest="rotation_angle")
    ap.add_argument("--rotation-angle-deg", type=float, default=None, dest="rotation_angle_deg")
    ap.add_argument("--anchor-lexical", action="store_true")
    ap.add_argument("--round-combine", choices=["max", "mean", "sum"], default="max")
    ap.add_argument("--prefer", default="datasets,_ingested,lmstudio-chat-archive", help="comma list of path substrings to boost")
    ap.add_argument("--per-file-limit", type=int, default=6000, help="max chars to read per file")
    ap.add_argument("--include-glob", default="", help="comma-separated path patterns to include")
    ap.add_argument("--exclude-glob", default="", help="comma-separated path patterns to exclude")
    ap.add_argument("--must", default="", help="comma-separated substrings that must appear in content")
    ap.add_argument("--must-regex", default="", help="regex that must match content")
    return ap.parse_known_args()

def coerce_legacy(argv: List[str], args: argparse.Namespace) -> argparse.Namespace:
    if args.query is None:
        tokens: List[str] = []
        for tok in argv:
            if tok.startswith("-"):
                break
            tokens.append(tok)
        if tokens:
            args.query = " ".join(tokens).strip() or None
    if args.out is None and args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
        args.out = str((out_dir / "last_pack.json").resolve())
    return args

TEXT_EXTS = {
    ".txt",".md",".rst",".json",".jsonl",".yaml",".yml",".ini",".cfg",".toml",
    ".csv",".tsv",".py",".js",".ts",".tsx",".java",".go",".rs",".c",".cpp",".h",".hpp",
    ".html",".htm",".css"
}
SLOW_EXTS = {".pdf",".docx",".rtf"}

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)

def _strip_html(s: str) -> str:
    return _HTML_TAG.sub(" ", s)

def _read_small_text(path: Path, limit: int) -> str:
    try:
        t = path.read_text(encoding="utf-8", errors="ignore")
        return t[:limit]
    except Exception:
        return ""

def _read_json(path: Path, limit: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
        s = json.dumps(obj, ensure_ascii=False, indent=2)
        return s[:limit]
    except Exception:
        return ""

def _read_jsonl(path: Path, limit: int) -> str:
    try:
        out, n = [], 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                out.append(line)
                n += len(line) + 1
                if n > limit:
                    break
        return "\n".join(out)[:limit]
    except Exception:
        return ""

def _read_csv(path: Path, limit: int) -> str:
    try:
        out, n = [], 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                out.append(line.rstrip("\n"))
                n += len(line)
                if n > limit or i > 2000:
                    break
        return "\n".join(out)[:limit]
    except Exception:
        return ""

def _read_html(path: Path, limit: int) -> str:
    raw = _read_small_text(path, limit*2)
    return _strip_html(raw)[:limit]

def _read_pdf(path: Path, limit: int) -> str:
    try:
        import PyPDF2  # type: ignore
        txt_parts = []
        with path.open("rb") as f:
            pdf = PyPDF2.PdfReader(f)
            for i, page in enumerate(pdf.pages):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                if t:
                    txt_parts.append(t)
                if sum(len(x) for x in txt_parts) > limit:
                    break
        return "\n".join(txt_parts)[:limit]
    except Exception:
        return ""

def _read_docx(path: Path, limit: int) -> str:
    try:
        import docx  # type: ignore
        # Fix for Pylance: use getattr to handle optional module
        Document = getattr(docx, 'Document', None)
        if Document is None:
            return ""
        d = Document(str(path))
        parts = [p.text for p in d.paragraphs if p.text]
        return "\n".join(parts)[:limit]
    except Exception:
        return ""

def _read_rtf(path: Path, limit: int) -> str:
    raw = _read_small_text(path, limit*2)
    raw = re.sub(r"\\[a-zA-Z]+-?\d*", " ", raw)
    raw = raw.replace("{", " ").replace("}", " ")
    return _WS.sub(" ", raw)[:limit]

def extract_text(path: Path, limit: int) -> str:
    ext = path.suffix.lower()
    if ext in {".json"}:
        return _read_json(path, limit)
    if ext in {".jsonl"}:
        return _read_jsonl(path, limit)
    if ext in {".csv",".tsv"}:
        return _read_csv(path, limit)
    if ext in {".html",".htm"}:
        return _read_html(path, limit)
    if ext in {".pdf"}:
        t = _read_pdf(path, limit)
        if t: return t
        return ""
    if ext in {".docx"}:
        t = _read_docx(path, limit)
        if t: return t
        return ""
    if ext in {".rtf"}:
        return _read_rtf(path, limit)
    return _read_small_text(path, limit)

_STOP = {
    "the","a","an","and","or","but","if","when","of","to","in","on","for","with","as","by",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","you",
    "we","they","he","she","them","his","her","their","your","my","our","at","from","into",
    "about","over","after","before","than","so","not","no","yes"
}

def _tokens(s: str) -> List[str]:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    return [t for t in s.split() if t and t not in _STOP]

def lexical_score(query: str, text: str) -> float:
    """BM25-inspired scoring with term frequency and length normalization"""
    if not query or not text:
        return 0.0
    q = _tokens(query)
    d = _tokens(text)
    if not q or not d:
        return 0.0

    # BM25 parameters
    k1 = 1.5  # Term frequency saturation
    b = 0.75  # Length normalization
    avgdl = 100  # Average document length estimate

    # Calculate term frequencies
    d_freq: Dict[str, int] = {}
    for tok in d:
        d_freq[tok] = d_freq.get(tok, 0) + 1

    # BM25 scoring
    score = 0.0
    doc_len = len(d)
    for q_term in set(q):  # Unique query terms
        if q_term in d_freq:
            tf = d_freq[q_term]
            # BM25 formula (simplified IDF=1.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += numerator / denominator

    return score

def apply_boost(path: str, base: float, prefer: List[str]) -> float:
    p = path.lower()
    w = base
    for key in prefer:
        key = key.strip().lower()
        if key and key in p:
            w *= 1.25
    return w

def _roots_from_arg(index_arg: str) -> List[Path]:
    parts = [s.strip() for s in index_arg.replace(",", ":").split(":") if s.strip()]
    out: List[Path] = []
    for s in parts:
        p = Path(s).expanduser().resolve()
        if p.exists():
            out.append(p)
    return out

def _iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    for base in roots:
        for root, _dirs, files in os.walk(base):
            for fn in files:
                yield Path(root) / fn

def _path_ok(p: Path, includes: List[str], excludes: List[str]) -> bool:
    s = str(p)
    if includes:
        hit = any(fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(s.lower(), pat) for pat in includes)
        if not hit:
            return False
    if excludes:
        if any(fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(s.lower(), pat) for pat in excludes):
            return False
    return True

def _content_ok(text: str, must: List[str], must_regex: Optional[re.Pattern]) -> bool:
    if must:
        lo = text.lower()
        for m in must:
            if m.lower() not in lo:
                return False
    if must_regex:
        if not must_regex.search(text):
            return False
    return True

def _content_hash(text: str) -> str:
    """Generate short hash for deduplication"""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:12]

def _is_substantial(text: str, min_words: int = 20, min_unique: int = 10) -> bool:
    """Filter out trivial/boilerplate content like __init__.py"""
    words = text.split()
    if len(words) < min_words:
        return False
    if len(set(words)) < min_unique:
        return False

    # Reject if mostly import statements (>70%)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False

    import_lines = sum(1 for l in lines if l.startswith(('import ', 'from ')))
    import_ratio = import_lines / len(lines)
    if import_ratio > 0.7:
        return False

    return True

def main() -> None:
    args, rest = parse_args()
    args = coerce_legacy(rest, args)
    if args.rotation_angle_deg is not None:
        args.rotation_angle = float(args.rotation_angle_deg)
    if not args.query:
        raise SystemExit("missing --query (and no positional query discovered)")
    if not args.out:
        raise SystemExit("missing --out (and no --out-dir provided)")

    roots = _roots_from_arg(args.index)
    if not roots:
        raise SystemExit("no index roots found")

    prefer = [s for s in (args.prefer or "").split(",") if s.strip()]
    includes = [s for s in (args.include_glob or "").split(",") if s.strip()]
    excludes = [s for s in (args.exclude_glob or "").split(",") if s.strip()]
    must = [s for s in (args.must or "").split(",") if s.strip()]
    must_rx = re.compile(args.must_regex, re.IGNORECASE) if (args.must_regex or "").strip() else None

    # Scan all files and score them
    items: List[Dict[str, object]] = []
    seen_hashes: set = set()  # For deduplication
    n_scan = 0
    n_trivial = 0
    n_duplicate = 0

    for p in _iter_files(roots):
        ext = p.suffix.lower()
        if ext not in TEXT_EXTS and ext not in SLOW_EXTS:
            continue
        if not _path_ok(p, includes, excludes):
            continue
        text = extract_text(p, int(args.per_file_limit))
        if not text:
            continue
        if not _content_ok(text, must, must_rx):
            continue

        # Filter trivial/boilerplate content
        if not _is_substantial(text):
            n_trivial += 1
            continue

        # Deduplicate by content hash
        content_hash = _content_hash(text)
        if content_hash in seen_hashes:
            n_duplicate += 1
            continue
        seen_hashes.add(content_hash)

        n_scan += 1
        score = lexical_score(args.query, text)
        score = apply_boost(str(p), score, prefer)
        items.append({"path": str(p), "text": text, "score": round(score, 6)})
    
    # Sort by score (highest first)
    items.sort(key=lambda x: (x.get("score", 0.0), len(str(x.get("text","")))), reverse=True)
    
    # ======= DIVERSITY FIX: Limit chunks per file =======
    max_per_file = max(1, int(args.max_per_file))
    topn = max(1, int(args.top_k))
    
    # Group items by file
    items_by_file: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        items_by_file[str(item["path"])].append(item)
    
    # Sort files by their best score
    sorted_files = sorted(
        items_by_file.keys(),
        key=lambda path: max(it["score"] for it in items_by_file[path]),  # type: ignore
        reverse=True
    )
    
    # Collect chunks: max N per file, stop when we hit top_k total
    kept: List[Dict[str, object]] = []
    for file_path in sorted_files:
        file_items = items_by_file[file_path][:max_per_file]  # Take at most N chunks from this file
        kept.extend(file_items)
        if len(kept) >= topn:
            break
    
    # Trim to exact top_k
    kept = kept[:topn]
    # =====================================================

    pack = {
        "items": [{"path": it["path"], "text": it["text"], "score": it.get("score", 0.0)} for it in kept],
        "meta": {
            "roots": [str(r) for r in roots],
            "query": args.query,
            "scanned": n_scan,
            "kept": len(kept),
            "unique_sources": len(set(str(it["path"]) for it in kept)),
            "max_per_file": max_per_file,
            "prefer": prefer,
            "include_glob": includes,
            "exclude_glob": excludes,
            "must": must,
            "must_regex": args.must_regex or "",
            "readme": README.strip()
        }
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
