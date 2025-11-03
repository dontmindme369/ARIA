#!/usr/bin/env python3
"""
ARIA Unified Retrieval - Phase 2
# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv('ARIA_DATA_DIR', './data')
CACHE_DIR = os.getenv('ARIA_CACHE_DIR', './cache')
OUTPUT_DIR = os.getenv('ARIA_OUTPUT_DIR', './output')



Consolidates:
- local_rag_context_v7_guided_exploration.py - main retrieval
- rag_embed_cache.py - embedding caching
- query_features.py - query feature extraction
- rag_chunking.py - text chunking
- spherical_cap_sampler.py - golden ratio sampling

Features:
- Multi-root recursive scanning (data: directories)
- Multi-format text extraction (txt, md, json, csv, html, pdf, docx)
- Lexical scoring with source boosting
- Per-file diversity limits
- Query feature extraction
- Embedding cache for speed
- PCA rotation exploration (optional)
- Spherical cap sampling for search space expansion
"""

import fnmatch
import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


# Export public API
__all__ = [
    'extract_text', 'chunk_text', 'QueryAnalyzer', 'EmbeddingCache',
    'lexical_score', 'apply_boost', 'ARIARetrieval',
    'generate_spherical_cap_samples', 'tokenize'
]


# ============================================================================
# TEXT EXTRACTION - Multi-Format Support
# ============================================================================

TEXT_EXTS = {
    ".txt", ".md", ".rst", ".json", ".jsonl", ".yaml", ".yml",
    ".ini", ".cfg", ".toml", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
    ".html", ".htm", ".css", ".xml"
}

SLOW_EXTS = {".pdf", ".docx", ".rtf"}

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def strip_html(s: str) -> str:
    """Remove HTML tags"""
    return _HTML_TAG.sub(" ", s)


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace"""
    return _WS.sub(" ", s).strip()


def read_text_file(path: Path, limit: int = 6000) -> str:
    """Read text file with encoding fallback"""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def read_json_file(path: Path, limit: int = 6000) -> str:
    """Read JSON file and pretty-print"""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False, indent=2)[:limit]
    except Exception:
        return ""


def read_jsonl_file(path: Path, limit: int = 6000) -> str:
    """Read JSONL file (line-delimited JSON)"""
    try:
        lines = []
        total = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                total += len(line) + 1
                if total >= limit:
                    break
        return "\n".join(lines)
    except Exception:
        return ""


def read_csv_file(path: Path, limit: int = 6000) -> str:
    """Read CSV file"""
    try:
        lines = []
        total = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.rstrip())
                total += len(line)
                if total >= limit:
                    break
        return "\n".join(lines)
    except Exception:
        return ""


def read_html_file(path: Path, limit: int = 6000) -> str:
    """Read HTML and strip tags"""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = strip_html(raw)
        return normalize_whitespace(text)[:limit]
    except Exception:
        return ""


def read_pdf_file(path: Path, limit: int = 6000) -> str:
    """Read PDF with PyPDF2 (optional)"""
    try:
        import PyPDF2  # type: ignore[import-not-found]
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            chunks = []
            total = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                chunks.append(text)
                total += len(text)
                if total >= limit:
                    break
            return " ".join(chunks)[:limit]
    except ImportError:
        # Fallback: raw binary read (garbage but better than nothing)
        return read_text_file(path, limit)
    except Exception:
        return ""


def read_docx_file(path: Path, limit: int = 6000) -> str:
    """Read DOCX with python-docx (optional)"""
    try:
        from docx import Document  # type: ignore[import-not-found]
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n".join(paragraphs)
        return text[:limit]
    except ImportError:
        # No fallback for DOCX
        return ""
    except Exception:
        return ""


def extract_text(path: Path, limit: int = 6000) -> str:
    """
    Extract text from file based on extension
    
    Supports: txt, md, json, jsonl, csv, html, pdf, docx
    """
    ext = path.suffix.lower()
    
    if ext in {".json"}:
        return read_json_file(path, limit)
    elif ext in {".jsonl"}:
        return read_jsonl_file(path, limit)
    elif ext in {".csv", ".tsv"}:
        return read_csv_file(path, limit)
    elif ext in {".html", ".htm", ".xml"}:
        return read_html_file(path, limit)
    elif ext == ".pdf":
        return read_pdf_file(path, limit)
    elif ext == ".docx":
        return read_docx_file(path, limit)
    elif ext in TEXT_EXTS:
        return read_text_file(path, limit)
    else:
        return ""


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        max_chunk_size: Max characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence break in last 200 chars
            chunk_text = text[start:end]
            last_period = chunk_text.rfind(". ")
            last_newline = chunk_text.rfind("\n")
            break_point = max(last_period, last_newline)
            
            if break_point > max_chunk_size - 200:
                end = start + break_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


# ============================================================================
# QUERY FEATURE EXTRACTION
# ============================================================================

class QueryAnalyzer:
    """Extract features from queries for better retrieval"""
    
    QUESTION_WORDS = {"what", "how", "why", "when", "where", "who", "which"}
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "about", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did"
    }
    
    def __init__(self):
        pass
    
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract features from query (primary method)
        
        Returns:
            {
                'type': str,             # 'definition', 'how-to', 'comparison', etc
                'key_terms': List[str],  # Important terms
                'intent': str,           # 'factual', 'navigational', 'transactional'
                'complexity': float,     # 0-1 score
                'specificity': float,    # 0-1 score
            }
        """
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        
        # Detect query type
        query_type = self._detect_type(query_lower, words)
        
        # Extract key terms (non-stop words)
        key_terms = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        
        # Assess complexity
        complexity = min(1.0, len(words) / 20)
        
        # Assess specificity (more unique terms = more specific)
        specificity = len(set(key_terms)) / max(len(key_terms), 1)
        
        # Detect intent
        intent = self._detect_intent(query_lower, query_type)
        
        return {
            'type': query_type,
            'key_terms': key_terms[:10],  # Top 10 terms
            'intent': intent,
            'complexity': complexity,
            'specificity': specificity,
            'word_count': len(words)
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Alias for extract() for backwards compatibility"""
        return self.extract(query)
    
    def _detect_type(self, query_lower: str, words: List[str]) -> str:
        """Detect query type"""
        if any(w in words for w in ["what", "define", "definition"]):
            return "definition"
        elif any(w in words for w in ["how", "explain"]):
            return "how-to"
        elif any(w in words for w in ["why"]):
            return "explanation"
        elif any(w in words for w in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        elif any(w in words for w in ["best", "recommend", "should"]):
            return "recommendation"
        elif "?" in query_lower:
            return "question"
        else:
            return "general"
    
    def _detect_intent(self, query_lower: str, query_type: str) -> str:
        """Detect search intent"""
        if query_type in ["definition", "explanation", "how-to"]:
            return "informational"
        elif query_type == "recommendation":
            return "transactional"
        elif any(w in query_lower for w in ["where", "find", "location"]):
            return "navigational"
        else:
            return "informational"


# Backwards compatibility alias
QueryFeatureExtractor = QueryAnalyzer


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

class EmbeddingCache:
    """Cache embeddings to disk for speed"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = self._hash(text)
        
        # Check memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.memory_cache[key] = embedding
                return embedding
            except Exception:
                return None
        
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Put embedding in cache"""
        key = self._hash(text)
        
        # Memory cache
        self.memory_cache[key] = embedding
        
        # Disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception:
            pass
    
    def _hash(self, text: str) -> str:
        """Hash text for cache key"""
        return hashlib.md5(text.encode()).hexdigest()


# ============================================================================
# SPHERICAL CAP SAMPLING
# ============================================================================

def generate_spherical_cap_samples(
    center: np.ndarray,
    n_samples: int,
    cap_angle: float = 0.1
) -> List[np.ndarray]:
    """
    Generate samples on a spherical cap using golden ratio spiral
    
    Args:
        center: Center direction (unit vector)
        n_samples: Number of samples to generate
        cap_angle: Angular radius of cap (radians)
    
    Returns:
        List of unit vectors sampled from cap
    """
    if n_samples == 0:
        return []
    
    # Normalize center
    center = center / (np.linalg.norm(center) + 1e-10)
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    samples = []
    for i in range(n_samples):
        # Golden angle
        theta = 2 * np.pi * i / phi
        
        # Uniform distribution in cap
        t = (i + 0.5) / n_samples
        cos_angle = 1 - (1 - np.cos(cap_angle)) * t
        sin_angle = np.sqrt(1 - cos_angle**2)
        
        # Generate point
        x = sin_angle * np.cos(theta)
        y = sin_angle * np.sin(theta)
        z = cos_angle
        
        # Rotate to align with center
        sample = _rotate_to_direction(np.array([x, y, z]), center)
        samples.append(sample)
    
    return samples


def _rotate_to_direction(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Rotate point to align z-axis with direction"""
    # Find rotation axis (cross product with z-axis)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # Already aligned or opposite
        if direction[2] > 0:
            return point
        else:
            return -point
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
    
    # Rodrigues rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    return (
        point * cos_a +
        np.cross(axis, point) * sin_a +
        axis * np.dot(axis, point) * (1 - cos_a)
    )


# ============================================================================
# LEXICAL SCORING
# ============================================================================

def tokenize(text: str) -> Set[str]:
    """Tokenize text into words"""
    return set(_PUNCT.sub(" ", text.lower()).split())


def lexical_score(query: str, document: str) -> float:
    """
    Score document relevance using lexical overlap
    
    Args:
        query: Query text
        document: Document text
    
    Returns:
        Score (0-1) based on term overlap
    """
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)
    
    if not q_tokens or not d_tokens:
        return 0.0
    
    overlap = len(q_tokens & d_tokens)
    return overlap / len(q_tokens)


def apply_boost(path: str, base_score: float, prefer: List[str]) -> float:
    """
    Boost scores for preferred sources
    
    Args:
        path: File path
        base_score: Base relevance score
        prefer: List of path substrings to boost
    
    Returns:
        Boosted score
    """
    path_lower = path.lower()
    score = base_score
    
    for keyword in prefer:
        if keyword.lower() in path_lower:
            score *= 1.25
    
    return score


# ============================================================================
# MAIN RETRIEVER
# ============================================================================

class ARIARetrieval:
    """
    Unified retrieval system
    
    Features:
    - Multi-root scanning
    - Multi-format extraction
    - Lexical scoring + boosting
    - Per-file diversity limits
    - Query analysis
    - Embedding cache (optional)
    """
    
    def __init__(
        self,
        index_roots: List[Path],
        cache_dir: Optional[Path] = None,
        prefer_dirs: Optional[List[str]] = None,
        max_per_file: int = 6
    ):
        # Ensure all paths are expanded and resolved (defensive programming)
        self.index_roots = [Path(str(r)).expanduser().resolve() for r in index_roots]
        self.prefer_dirs = prefer_dirs or []
        self.max_per_file = max_per_file
        
        # Query analyzer
        self.query_analyzer = QueryAnalyzer()
        
        # Embedding cache (optional)
        if cache_dir:
            cache_dir = Path(str(cache_dir)).expanduser().resolve()
        self.embedding_cache = EmbeddingCache(cache_dir) if cache_dir else None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 64,  # Increased for 100k token model
        per_file_limit: int = 6000,  # Longer chunks for richer context
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        must_contain: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: Query text
            top_k: Number of chunks to return
            per_file_limit: Max chars to read per file
            include_globs: Path patterns to include
            exclude_globs: Path patterns to exclude
            must_contain: Terms that must appear in content
        
        Returns:
            {
                'items': List[Dict],  # Retrieved chunks
                'meta': Dict,         # Metadata
                'query_analysis': Dict  # Query features
            }
        """
        # Analyze query
        query_features = self.query_analyzer.analyze(query)
        
        # Scan and score files
        items: List[Dict[str, Any]] = []
        n_scanned = 0
        n_skipped_empty = 0
        n_skipped_filter = 0
        
        for path in self._iter_files():
            # Check path filters
            if not self._path_ok(path, include_globs, exclude_globs):
                n_skipped_filter += 1
                continue
            
            # Extract text
            text = extract_text(path, per_file_limit)
            if not text or len(text.strip()) < 50:  # Skip very short/empty files
                n_skipped_empty += 1
                continue
            
            # Check content filters
            if must_contain and not self._content_ok(text, must_contain):
                n_skipped_filter += 1
                continue
            
            n_scanned += 1
            
            # Score
            score = lexical_score(query, text)
            score = apply_boost(str(path), score, self.prefer_dirs)
            
            # Include source metadata for better visibility
            items.append({
                'path': str(path),
                'source': str(path),  # Explicit source field
                'filename': path.name,  # Just filename for easy reference
                'text': text,
                'score': round(score, 6),
                'length': len(text)
            })
        
        # Sort by score (desc) - but keep ALL items, even with score 0
        items.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply diversity limits (max N chunks per file)
        diverse_items = self._enforce_diversity(items, self.max_per_file, top_k)
        
        return {
            'items': diverse_items,
            'meta': {
                'query': query,
                'scanned': n_scanned,
                'skipped_empty': n_skipped_empty,
                'skipped_filter': n_skipped_filter,
                'kept': len(diverse_items),
                'unique_sources': len(set(it['path'] for it in diverse_items)),
                'max_per_file': self.max_per_file,
                'index_roots': [str(r) for r in self.index_roots]
            },
            'query_analysis': query_features
        }
    
    def _iter_files(self) -> Iterable[Path]:
        """Iterate all files in index roots"""
        import sys
        for root in self.index_roots:
            if not root.exists():
                print(f"[ARIA] WARNING: Index root does not exist: {root}", file=sys.stderr)
                continue
            print(f"[ARIA] Scanning index root: {root}", file=sys.stderr)
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    yield Path(dirpath) / filename
    
    def _path_ok(
        self,
        path: Path,
        includes: Optional[List[str]],
        excludes: Optional[List[str]]
    ) -> bool:
        """Check if path passes glob filters"""
        path_str = str(path)
        
        # Check extension
        if path.suffix.lower() not in TEXT_EXTS and path.suffix.lower() not in SLOW_EXTS:
            return False
        
        # Include filter
        if includes:
            if not any(fnmatch.fnmatch(path_str, pat) for pat in includes):
                return False
        
        # Exclude filter
        if excludes:
            if any(fnmatch.fnmatch(path_str, pat) for pat in excludes):
                return False
        
        return True
    
    def _content_ok(self, text: str, must_contain: List[str]) -> bool:
        """Check if text contains required terms"""
        text_lower = text.lower()
        return all(term.lower() in text_lower for term in must_contain)
    
    def _enforce_diversity(
        self,
        items: List[Dict[str, Any]],
        max_per_file: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Enforce per-file chunk limits for diversity
        
        Returns top_k items with at most max_per_file from any single source
        """
        # Group by file
        by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_file[item['path']].append(item)
        
        # Sort files by best score
        sorted_files = sorted(
            by_file.keys(),
            key=lambda p: max(it['score'] for it in by_file[p]),
            reverse=True
        )
        
        # Collect chunks
        result = []
        for file_path in sorted_files:
            file_items = by_file[file_path][:max_per_file]
            result.extend(file_items)
            if len(result) >= top_k:
                break
        
        return result[:top_k]


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for testing retrieval"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIA Unified Retrieval")
    parser.add_argument('query', help="Query text")
    parser.add_argument('--index', required=True, help="Index roots (colon-separated)")
    parser.add_argument('--top-k', type=int, default=32)
    parser.add_argument('--max-per-file', type=int, default=6)
    parser.add_argument('--prefer', default='datasets,_ingested', help="Boost these dirs")
    parser.add_argument('--out', help="Output JSON file")
    args = parser.parse_args()
    
    # Parse index roots
    roots = [Path(r.strip()).expanduser() for r in args.index.replace(',', ':').split(':') if r.strip()]
    roots = [r for r in roots if r.exists()]
    
    if not roots:
        print("No valid index roots found")
        return
    
    # Initialize retriever
    retriever = ARIARetrieval(
        index_roots=roots,
        prefer_dirs=args.prefer.split(','),
        max_per_file=args.max_per_file
    )
    
    # Retrieve
    result = retriever.retrieve(args.query, top_k=args.top_k)
    
    # Display
    print(f"\n{'='*70}")
    print(f"QUERY: {args.query}")
    print(f"{'='*70}")
    print(f"\nQuery Analysis:")
    for key, val in result['query_analysis'].items():
        print(f"  {key}: {val}")
    
    print(f"\nRetrieval Stats:")
    for key, val in result['meta'].items():
        if key != 'index_roots':
            print(f"  {key}: {val}")
    
    print(f"\nTop {len(result['items'])} chunks:")
    for i, item in enumerate(result['items'][:5], 1):
        print(f"\n{i}. {Path(item['path']).name} (score: {item['score']})")
        print(f"   {item['text'][:100]}...")
    
    # Save if requested
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Saved to {args.out}")


if __name__ == '__main__':
    main()
