import os
import re
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Configuration
# Set these paths in config.yaml or as environment variables
DATA_DIR = os.getenv("ARIA_DATA_DIR", "./data")
CACHE_DIR = os.getenv("ARIA_CACHE_DIR", "./cache")
OUTPUT_DIR = os.getenv("ARIA_OUTPUT_DIR", "./output")


@dataclass
class QueryFeatures:
    length: int
    word_count: int
    avg_word_length: float
    domain: str
    domain_confidence: float
    complexity: str
    has_multiple_questions: bool
    technical_density: float
    is_follow_up: bool
    conversation_depth: int
    vector: np.ndarray | None = None


class QueryFeatureExtractor:
    DOMAIN_PATTERNS = {
        "code": {
            "keywords": [
                "function",
                "class",
                "def",
                "import",
                "error",
                "debug",
                "variable",
                "method",
                "api",
                "syntax",
                "compile",
                "stacktrace",
            ],
            "patterns": [
                r"\\b[a-z_]+\\([^)]*\\)",
                r"\\b[A-Z][a-zA-Z]*Error\\b",
                r"`[^`]+`",
            ],
        },
        "paper": {
            "keywords": [
                "paper",
                "study",
                "research",
                "analysis",
                "theory",
                "hypothesis",
                "experiment",
                "methodology",
            ],
            "patterns": [r"\\bet al\\.", r"\\b\\d{4}\\b"],
        },
        "conversation": {
            "keywords": ["said", "discussed", "talked", "mentioned", "we", "our"],
            "patterns": [r"\\bwe\\b|\\bour\\b|\\bus\\b"],
        },
        "concept": {
            "keywords": ["explain", "how", "why", "what", "understand", "concept"],
            "patterns": [r"\\b(how|why|what)\\s+(does|is|are)", r"\\bexplain\\b"],
        },
        "factual": {
            "keywords": ["when", "where", "who", "which", "date", "time"],
            "patterns": [r"\\b(when|where|who)\\s+(did|was|is)"],
        },
    }

    def extract(
        self, query: str, conversation_context: List[str] | None = None
    ) -> QueryFeatures:
        q = query or ""
        words = q.split()
        length = len(q)
        word_count = len(words)
        avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

        domain, domain_conf = self._detect_domain(q)
        complexity = self._estimate_complexity(q)
        has_multiple = "?" in q and q.count("?") > 1
        tech_density = self._technical_density(q)

        is_follow_up = self._is_follow_up(q, conversation_context or [])
        conv_depth = len(conversation_context or [])

        vec = self._vectorize(
            length,
            word_count,
            avg_word_length,
            domain,
            domain_conf,
            complexity,
            tech_density,
            is_follow_up,
            conv_depth,
        )

        return QueryFeatures(
            length,
            word_count,
            avg_word_length,
            domain,
            domain_conf,
            complexity,
            has_multiple,
            tech_density,
            is_follow_up,
            conv_depth,
            vec,
        )

    def _detect_domain(self, query: str) -> Tuple[str, float]:
        ql = query.lower()
        scores = {}
        for d, patt in self.DOMAIN_PATTERNS.items():
            s = 0.0
            for k in patt["keywords"]:
                if k in ql:
                    s += 1.0
            for rgx in patt["patterns"]:
                if re.search(rgx, query, flags=re.IGNORECASE):
                    s += 2.0
            scores[d] = s
        if not scores:
            return "default", 0.0
        best, best_s = max(scores.items(), key=lambda kv: kv[1])
        max_possible = len(self.DOMAIN_PATTERNS[best]["keywords"]) + 2 * len(
            self.DOMAIN_PATTERNS[best]["patterns"]
        )
        conf = min(best_s / max(1, max_possible), 1.0)
        if best_s >= 1.0:
            return best, conf
        return "default", 0.0

    def _estimate_complexity(self, q: str) -> str:
        simple = ["what is", "who is", "when did", "where is", "define"]
        complex_i = [
            "compare",
            "analyze",
            "explain why",
            "how does",
            "relationship between",
        ]
        ql = q.lower()
        if any(x in ql for x in complex_i):
            return "complex"
        if any(x in ql for x in simple):
            return "simple"
        return "medium"

    def _technical_density(self, text: str) -> float:
        technical_terms = [
            "system",
            "algorithm",
            "function",
            "process",
            "architecture",
            "implementation",
            "protocol",
            "interface",
            "mechanism",
            "framework",
            "model",
            "vector",
            "embedding",
            "retrieval",
            "optimization",
            "parameter",
            "metric",
            "performance",
            "evaluation",
        ]
        words = text.lower().split()
        if not words:
            return 0.0
        tech = sum(1 for w in words if any(t in w for t in technical_terms))
        return tech / len(words)

    def _is_follow_up(self, q: str, ctx: List[str]) -> bool:
        if not ctx:
            return False
        indicators = [
            "also",
            "and",
            "what about",
            "how about",
            "similarly",
            "in addition",
            "furthermore",
            "that",
            "this",
            "it",
        ]
        ql = q.lower()
        if any(ql.startswith(ind) for ind in indicators):
            return True
        if any(w in ql for w in ["that", "this", "it", "they"]):
            return True
        return False

    def _vectorize(
        self,
        length,
        word_count,
        avg_word_length,
        domain,
        domain_conf,
        complexity,
        tech_density,
        is_follow_up,
        conv_depth,
    ) -> np.ndarray:
        vec: list[float] = []
        vec.append(min(length / 500, 1.0))
        vec.append(min(word_count / 100, 1.0))
        vec.append(min(avg_word_length / 10, 1.0))
        domains = ["code", "paper", "conversation", "concept", "factual", "default"]
        vec.extend([1.0 if d == domain else 0.0 for d in domains])
        complexities = ["simple", "medium", "complex"]
        vec.extend([1.0 if c == complexity else 0.0 for c in complexities])
        vec.append(float(domain_conf))
        vec.append(float(tech_density))
        vec.append(min(float(conv_depth) / 10.0, 1.0))
        return np.array(vec, dtype=np.float32)


# Backwards-compat: legacy simple extract()
def extract(q: str) -> dict:
    feats = QueryFeatureExtractor().extract(q)
    return {
        "len": feats.length,
        "domain": feats.domain,
        "complexity": feats.complexity,
        "is_q": "?" in (q or ""),
        "word_count": feats.word_count,
        "technical_density": feats.technical_density,
    }
