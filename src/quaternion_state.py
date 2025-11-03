#!/usr/bin/env python3
"""
Quaternion State Manager for ARIA
==================================

Manages system state as a quaternion on the unit 3-sphere (S³), enabling:
- Smooth semantic space exploration via SLERP interpolation
- Momentum-based state evolution (not chaotic/random)
- Associative memory (save/recall states by label)
- 4D semantic vectors from TF-IDF + SVD

Based on quaternion_chaotic_rag.py but integrated with ARIA v44 architecture.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import re

# ============================================================================
# TEXT TOKENIZATION
# ============================================================================

WORD_RX = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    """Simple word tokenization"""
    return [w.lower() for w in WORD_RX.findall(text)]


# ============================================================================
# QUATERNION OPERATIONS ON S³
# ============================================================================

def s3_normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit 3-sphere (quaternion space)
    
    Args:
        v: 4D vector
    
    Returns:
        Unit vector on S³
    """
    norm = np.linalg.norm(v) + 1e-12
    return v / norm


def s3_nudge(q: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
    """
    Smooth interpolation on S³ using SLERP (Spherical Linear Interpolation)
    
    This is the KEY quaternion operation - smoothly rotates the current state
    toward a target on the 3-sphere. Unlike discrete jumps, this creates
    continuous semantic space exploration.
    
    Args:
        q: Current quaternion state (4D unit vector)
        target: Target quaternion (4D unit vector)
        step: Interpolation amount (0.0 to 1.0)
    
    Returns:
        New quaternion state after smooth rotation
    """
    # Ensure we're moving toward closest representation on S³
    dot = float(np.clip(np.dot(q, target), -1.0, 1.0))
    t = -target if dot < 0 else target
    
    # Already at target
    if abs(dot) > 0.9999:
        return q
    
    # SLERP interpolation
    theta = math.acos(abs(dot))
    u = min(1.0, step / (theta + 1e-12))
    
    new = (math.sin((1 - u) * theta) * q + math.sin(u * theta) * t) / max(1e-12, math.sin(theta))
    
    return s3_normalize(new)


# ============================================================================
# 4D SEMANTIC EMBEDDINGS (TF-IDF + SVD)
# ============================================================================

class TfidfSVD4:
    """
    Convert text to 4D semantic vectors using TF-IDF + SVD
    
    This creates semantically meaningful 4D vectors (not random hashes):
    - Texts with similar content → similar 4D vectors
    - 4D vectors live on unit 3-sphere (S³)
    - Can be used as quaternions for smooth interpolation
    """
    
    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.Vt = None  # Top 4 singular vectors
        self.doc_vecs = None  # N x 4 document vectors
        self.N = 0
    
    def fit(self, texts: List[str]):
        """
        Fit TF-IDF model and extract top 4 components via SVD
        
        Args:
            texts: List of document texts
        """
        # Tokenize all texts
        toks_list = [tokenize(t) for t in texts]
        
        # Build vocabulary
        vocab = {}
        for toks in toks_list:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        
        self.vocab = vocab
        V = len(vocab)
        N = len(texts)
        
        if V == 0 or N == 0:
            self.idf = np.zeros(0)
            self.Vt = np.zeros((4, 0))
            self.doc_vecs = np.zeros((0, 4))
            self.N = 0
            return
        
        # Build TF matrix
        rows = []
        for toks in toks_list:
            vec = np.zeros(V)
            for t in toks:
                vec[vocab[t]] += 1.0
            rows.append(vec)
        
        X = np.vstack(rows)  # N x V
        
        # Compute IDF
        df = (X > 0).sum(axis=0)
        idf = np.log((N + 1) / (df + 1)) + 1.0
        self.idf = idf
        self.N = N
        
        # TF-IDF
        tf = X / np.maximum(1.0, X.sum(axis=1, keepdims=True))
        X_tfidf = tf * idf
        
        # SVD - take top 4 components
        U, S, Vt = np.linalg.svd(X_tfidf, full_matrices=False)
        k = min(4, len(S))
        self.Vt = Vt[:k]
        
        # Project documents to 4D space
        doc_vecs = U[:, :k] * S[:k]
        
        # Normalize to unit vectors (S³)
        doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
        
        # Pad to 4D if needed
        if k < 4:
            doc_vecs = np.hstack([doc_vecs, np.zeros((N, 4 - k))])
        
        self.doc_vecs = doc_vecs
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to 4D semantic vectors
        
        Args:
            texts: List of texts to transform
        
        Returns:
            N x 4 array of unit vectors on S³
        """
        if self.Vt is None or self.N == 0:
            return np.zeros((len(texts), 4))
        
        V = len(self.vocab)
        out = []
        
        for text in texts:
            vec = np.zeros(V)
            toks = tokenize(text)
            
            for t in toks:
                j = self.vocab.get(t, -1)
                if j >= 0:
                    vec[j] += 1.0
            
            # TF-IDF
            tf = vec / vec.sum() if vec.sum() > 0 else vec
            if self.idf is not None:
                tfidf = tf * self.idf
            else:
                tfidf = tf
            
            # Project to 4D
            proj = tfidf @ self.Vt.T
            
            # Pad if needed
            if proj.shape[0] < 4:
                proj = np.pad(proj, (0, 4 - proj.shape[0]))
            
            # Normalize
            proj = proj / (np.linalg.norm(proj) + 1e-12)
            out.append(proj)
        
        return np.vstack(out) if len(out) > 1 else np.array(out)


# ============================================================================
# QUATERNION STATE MANAGER
# ============================================================================

class QuaternionStateManager:
    """
    Manages ARIA's quaternion state on S³
    
    Key Features:
    - State evolves smoothly via momentum (not chaotic)
    - SLERP interpolation for smooth semantic exploration
    - Associative memory (save/recall states)
    - Bias toward relevant past states
    """
    
    def __init__(self, state_dir: Path):
        """
        Initialize quaternion state manager
        
        Args:
            state_dir: Directory for saving state history
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Current quaternion state
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Momentum for smooth evolution
        self.momentum = np.zeros(4)
        
        # 4D semantic model
        self.svd4 = None
        self._fitted = False  # Track if semantic model is fitted
        
        # Associative memory path
        self.assoc_path = self.state_dir / "quaternion_states.jsonl"
        
        # Load last state if exists
        self._load_last_state()
    
    def fit_semantic_model(self, texts: List[str]):
        """
        Fit 4D semantic model on corpus
        
        Args:
            texts: List of document texts
        """
        self.svd4 = TfidfSVD4()
        self.svd4.fit(texts)
        
        # Initialize state to corpus centroid
        if self.svd4.doc_vecs is not None and len(self.svd4.doc_vecs) > 0:
            self.q = s3_normalize(self.svd4.doc_vecs.mean(axis=0))
        
        self._fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if semantic model has been fitted"""
        return self._fitted and self.svd4 is not None
    
    def query_to_4d(self, query: str) -> np.ndarray:
        """
        Convert query to 4D semantic vector
        
        Args:
            query: Query text
        
        Returns:
            4D unit vector
        """
        if self.svd4 is None:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        return self.svd4.transform([query])[0]
    
    def update_state(self, retrieved_vecs: List[np.ndarray], momentum_weight: float = 0.25):
        """
        Update quaternion state based on retrieved documents
        
        Args:
            retrieved_vecs: List of 4D vectors from retrieved docs
            momentum_weight: How much momentum to apply (0.0 to 1.0)
        """
        if not retrieved_vecs:
            return
        
        # Compute centroid of retrieved docs
        centroid = s3_normalize(np.mean(retrieved_vecs, axis=0))
        
        # Update momentum (85% old, 15% new direction)
        self.momentum = 0.85 * self.momentum + 0.15 * (centroid - self.q)
        
        # Move state with momentum
        self.q = s3_normalize(self.q + momentum_weight * self.momentum)
    
    def nudge_toward(self, target: np.ndarray, step: float = 0.15):
        """
        Gently nudge state toward target using SLERP
        
        Args:
            target: Target 4D vector
            step: Interpolation amount
        """
        self.q = s3_nudge(self.q, target, step)
    
    def save_state(self, label: str, metadata: Optional[Dict] = None):
        """
        Save current state to associative memory
        
        Args:
            label: Human-readable label for this state
            metadata: Additional metadata to store
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "q": self.q.tolist(),
            "momentum": self.momentum.tolist(),
            **(metadata or {})
        }
        
        with self.assoc_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    
    def recall_similar_states(self, query: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Recall similar states from associative memory
        
        Args:
            query: Optional query to match labels
            top_k: Number of states to return
        
        Returns:
            List of state records sorted by similarity
        """
        if not self.assoc_path.exists():
            return []
        
        states = []
        for line in self.assoc_path.read_text().splitlines():
            try:
                state = json.loads(line)
                states.append(state)
            except:
                pass
        
        if not states:
            return []
        
        # Filter by label if query provided
        if query:
            query_lower = query.lower()
            states = [s for s in states if query_lower in s.get("label", "").lower()]
        
        # If no matches, use recent states
        if not states:
            states = states[-20:]
        
        # Score by quaternion similarity
        scored = []
        for state in states:
            q_stored = np.array(state.get("q", [1, 0, 0, 0]), dtype=float)
            similarity = float(np.dot(self.q, q_stored))
            scored.append((similarity, state))
        
        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [state for _, state in scored[:top_k]]
    
    def _load_last_state(self):
        """Load last state from associative memory"""
        if not self.assoc_path.exists():
            return
        
        try:
            lines = self.assoc_path.read_text().splitlines()
            if lines:
                last_record = json.loads(lines[-1])
                last_q = np.array(last_record.get("q", [1, 0, 0, 0]), dtype=float)
                
                if np.linalg.norm(last_q) > 0:
                    self.q = s3_normalize(last_q)
                
                # Restore momentum if available
                if "momentum" in last_record:
                    self.momentum = np.array(last_record["momentum"], dtype=float)
        except:
            pass


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QuaternionStateManager',
    'TfidfSVD4',
    's3_normalize',
    's3_nudge',
    'tokenize'
]
