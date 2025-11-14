#!/usr/bin/env python3
"""
exemplar_fit.py - Exemplar fit scoring for ARIA
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re
import math

try:
    from typing import Any
    TfidfVectorizer: Any = None
    cosine_similarity: Any = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
        from sklearn.metrics.pairwise import cosine_similarity as _CS
        TfidfVectorizer = _TV
        cosine_similarity = _CS
        _HAVE_SK = True
    except Exception:
        _HAVE_SK = False
except Exception:
    _HAVE_SK = False

def _simple_cosine(a: str, b: str) -> float:
    from collections import Counter
    def toks(s):
        import re
        return [t for t in re.sub(r"[^\w]+", " ", s.lower()).split() if t]
    A = Counter(toks(a)); B = Counter(toks(b))
    terms = set(A)|set(B)
    import math
    dot = sum(A[t]*B[t] for t in terms)
    na = math.sqrt(sum(v*v for v in A.values()))
    nb = math.sqrt(sum(v*v for v in B.values()))
    return (dot/(na*nb)) if na*nb else 0.0

class ExemplarFitScorer:
    def __init__(self, exemplars: List[str] | None):
        self.exemplars = exemplars or []
        if _HAVE_SK and self.exemplars:
            self.vect = TfidfVectorizer(max_features=500, ngram_range=(1,3), stop_words='english')
            self.ex_vecs = self.vect.fit_transform(self.exemplars)
        else:
            self.vect = None; self.ex_vecs = None

    def calculate_fit(self, answer: str, coverage_score: float) -> Tuple[float, Dict[str,float]]:
        if not self.exemplars:
            return 0.5, {}
        style = self._style(answer)
        cite = self._citation(answer)
        conf = self._confidence(answer, coverage_score)
        overall = 0.40*style + 0.30*cite + 0.30*conf
        return overall, {"style": style, "citations": cite, "confidence": conf}

    def _style(self, ans: str) -> float:
        if _HAVE_SK and self.vect is not None:
            try:
                v = self.vect.transform([ans])
                sims = cosine_similarity(v, self.ex_vecs)[0]
                top = sorted(sims, reverse=True)[:min(3, len(sims))]
                return float(sum(top)/len(top)) if top else 0.5
            except Exception:
                pass
        if not self.exemplars: return 0.5
        scores = sorted((_simple_cosine(ans, ex) for ex in self.exemplars), reverse=True)
        top = scores[:min(3, len(scores))]
        return float(sum(top)/len(top)) if top else 0.5

    def _citation(self, ans: str) -> float:
        pats = [r'\[\d+\]', r'\[[\w\-\.]+\]', r'\([\w\-\.]+\)']
        n_sent = max(1, len(ans.split('.')))
        count = sum(len(re.findall(p, ans)) for p in pats)
        dens = count / n_sent
        diff = abs(dens - 0.2)
        return max(0.0, 1.0 - 2.0*diff)

    def _confidence(self, ans: str, coverage: float) -> float:
        hedges = ['might','may','could','possibly','perhaps','likely','appears','seems','suggests','indicates','probably','generally','typically','often','usually','sometimes']
        words = max(1, len(ans.split()))
        rate = sum(1 for h in hedges if h in ans.lower()) / words
        target = 0.02 * (1.5 - max(0.0, min(1.0, coverage)))
        diff = abs(rate - target)
        return max(0.0, 1.0 - 10.0*diff)
