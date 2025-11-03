#!/usr/bin/env python3
"""
ARIA Exploration Strategies Integration
========================================

Integrates quaternion state management and PCA exploration into ARIA retrieval:

Strategies:
1. golden_ratio_spiral - Uses quaternion state on S³ with momentum evolution
2. pca_exploration - Rotates queries in PCA subspace for multi-perspective retrieval
3. hybrid_exploration - Combines both approaches

Usage in aria_main.py:
    from aria_exploration import ExplorationManager
    
    explorer = ExplorationManager(state_dir=var_dir / "exploration")
    results = explorer.explore(query, retrieval_results, strategy="golden_ratio_spiral")
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from quaternion_state import QuaternionStateManager, s3_normalize
from pca_exploration import PCAExplorer


# ============================================================================
# EXPLORATION MANAGER
# ============================================================================

class ExplorationManager:
    """
    Manages exploration strategies for ARIA retrieval
    
    Coordinates:
    - Quaternion state evolution on S³
    - PCA rotation exploration
    - Strategy selection and execution
    """
    
    def __init__(
        self,
        state_dir: Path,
        enable_quaternion: bool = True,
        enable_pca: bool = True
    ):
        """
        Initialize exploration manager
        
        Args:
            state_dir: Directory for saving state
            enable_quaternion: Enable quaternion state management
            enable_pca: Enable PCA exploration
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Quaternion state manager
        self.quaternion_mgr = None
        if enable_quaternion:
            try:
                self.quaternion_mgr = QuaternionStateManager(state_dir / "quaternion")
            except Exception as e:
                print(f"[ARIA] Warning: Could not initialize quaternion manager: {e}")
        
        # PCA explorer
        self.pca_explorer = None
        if enable_pca:
            try:
                self.pca_explorer = PCAExplorer(
                    n_rotations=8,
                    subspace_dim=16,
                    max_angle_deg=12.0,
                    combine_mode="max"
                )
            except Exception as e:
                print(f"[ARIA] Warning: Could not initialize PCA explorer: {e}")
        
        self.stats = {
            "quaternion_updates": 0,
            "pca_explorations": 0,
            "total_queries": 0
        }
    
    def fit_semantic_model(self, texts: List[str]):
        """
        Fit semantic models on corpus
        
        Args:
            texts: List of document texts
        """
        if self.quaternion_mgr:
            try:
                print("[ARIA] Fitting 4D semantic model...")
                self.quaternion_mgr.fit_semantic_model(texts)
                print(f"[ARIA] ✓ Quaternion state initialized from {len(texts)} documents")
            except Exception as e:
                print(f"[ARIA] Warning: Could not fit semantic model: {e}")
    
    def explore(
        self,
        query: str,
        retrieval_results: Dict[str, Any],
        strategy: str = "adaptive",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply exploration strategy to retrieval results
        
        Args:
            query: User query
            retrieval_results: Results from ARIARetrieval
            strategy: Exploration strategy name
            **kwargs: Strategy-specific parameters
        
        Returns:
            Enhanced retrieval results with exploration metadata
        """
        self.stats["total_queries"] += 1
        
        if strategy == "golden_ratio_spiral":
            return self._explore_quaternion(query, retrieval_results, **kwargs)
        
        elif strategy == "pca_exploration":
            return self._explore_pca(query, retrieval_results, **kwargs)
        
        elif strategy == "hybrid_exploration":
            return self._explore_hybrid(query, retrieval_results, **kwargs)
        
        else:
            # No exploration - return as-is
            retrieval_results["exploration"] = {
                "strategy": "none",
                "message": "No exploration applied"
            }
            return retrieval_results
    
    def _explore_quaternion(
        self,
        query: str,
        results: Dict[str, Any],
        momentum_weight: float = 0.25,
        recall_similar: bool = True,
        save_state: bool = True
    ) -> Dict[str, Any]:
        """
        Explore using quaternion state on S³
        
        Args:
            query: User query
            results: Retrieval results
            momentum_weight: Momentum influence (0.0 to 1.0)
            recall_similar: Whether to bias toward similar past states
            save_state: Whether to save current state
        
        Returns:
            Enhanced results with quaternion reranking
        """
        if not self.quaternion_mgr:
            results["exploration"] = {
                "strategy": "golden_ratio_spiral",
                "message": "Quaternion manager not available",
                "applied": False
            }
            return results
        
        try:
            # Get query 4D vector
            query_4d = self.quaternion_mgr.query_to_4d(query)
            
            # Recall similar past states if enabled
            if recall_similar:
                similar_states = self.quaternion_mgr.recall_similar_states(query, top_k=3)
                if similar_states:
                    # Nudge state toward historical context
                    avg_past_q = s3_normalize(np.mean(
                        [np.array(s["q"]) for s in similar_states],
                        axis=0
                    ))
                    self.quaternion_mgr.nudge_toward(avg_past_q, step=0.15)
            
            # Extract retrieved document texts for 4D projection
            items = results.get("items", [])
            if not items:
                results["exploration"] = {
                    "strategy": "golden_ratio_spiral",
                    "message": "No items to rerank",
                    "applied": False
                }
                return results
            
            # Get 4D vectors for retrieved docs
            doc_texts = [item.get("text", "") for item in items]
            doc_4ds = self.quaternion_mgr.svd4.transform(doc_texts) if self.quaternion_mgr.svd4 else None
            
            if doc_4ds is None:
                results["exploration"] = {
                    "strategy": "golden_ratio_spiral",
                    "message": "Semantic model not fitted",
                    "applied": False
                }
                return results
            
            # Convert to list of vectors for update_state
            doc_4ds_list = [doc_4ds[i] for i in range(len(doc_4ds))]
            
            # Compute quaternion-based scores
            q_state = self.quaternion_mgr.q
            q_scores = np.dot(doc_4ds, q_state) * 0.5 + 0.5  # Map to [0, 1]
            
            # Combine with original scores (if available)
            for i, item in enumerate(items):
                original_score = item.get("score", 0.5)
                q_score = float(q_scores[i])
                
                # Weighted combination (70% quaternion, 30% original)
                combined_score = 0.7 * q_score + 0.3 * original_score
                
                item["quaternion_score"] = q_score
                item["original_score"] = original_score
                item["score"] = combined_score
            
            # Re-sort by combined score
            items.sort(key=lambda x: x["score"], reverse=True)
            results["items"] = items
            
            # Update quaternion state based on retrieved docs
            self.quaternion_mgr.update_state(doc_4ds_list, momentum_weight)
            self.stats["quaternion_updates"] += 1
            
            # Save state if requested
            if save_state:
                self.quaternion_mgr.save_state(
                    label=query[:100],
                    metadata={"num_results": len(items)}
                )
            
            results["exploration"] = {
                "strategy": "golden_ratio_spiral",
                "applied": True,
                "quaternion_state": q_state.tolist(),
                "momentum": self.quaternion_mgr.momentum.tolist(),
                "message": f"Reranked {len(items)} items using quaternion state"
            }
            
            return results
            
        except Exception as e:
            print(f"[ARIA] Quaternion exploration failed: {e}")
            results["exploration"] = {
                "strategy": "golden_ratio_spiral",
                "applied": False,
                "error": str(e)
            }
            return results
    
    def _explore_pca(
        self,
        query: str,
        results: Dict[str, Any],
        n_rotations: int = 8,
        max_angle: float = 12.0
    ) -> Dict[str, Any]:
        """
        Explore using PCA rotation
        
        NOTE: This requires embeddings from a model like sentence-transformers.
        If not available, falls back to lexical scores.
        
        Args:
            query: User query
            results: Retrieval results
            n_rotations: Number of rotations
            max_angle: Maximum angle (degrees)
        
        Returns:
            Enhanced results (currently a placeholder for full embedding integration)
        """
        if not self.pca_explorer:
            results["exploration"] = {
                "strategy": "pca_exploration",
                "message": "PCA explorer not available",
                "applied": False
            }
            return results
        
        # TODO: Integrate with sentence-transformers or other embedding model
        # For now, this is a placeholder that would need:
        # 1. Query embedding
        # 2. Document embeddings
        # 3. PCA exploration on those embeddings
        
        results["exploration"] = {
            "strategy": "pca_exploration",
            "message": "PCA exploration requires embedding model integration",
            "applied": False,
            "note": "Use quaternion strategy for semantic exploration without external models"
        }
        
        self.stats["pca_explorations"] += 1
        return results
    
    def _explore_hybrid(
        self,
        query: str,
        results: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hybrid exploration using both quaternion and PCA
        
        Args:
            query: User query
            results: Retrieval results
        
        Returns:
            Enhanced results
        """
        # Apply quaternion first
        results = self._explore_quaternion(query, results, **kwargs)
        
        # Then PCA (if available)
        if self.pca_explorer and results["exploration"].get("applied", False):
            results = self._explore_pca(query, results, **kwargs)
        
        results["exploration"]["strategy"] = "hybrid_exploration"
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        return {
            **self.stats,
            "quaternion_enabled": self.quaternion_mgr is not None,
            "pca_enabled": self.pca_explorer is not None
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_exploration_manager(
    state_dir: Path,
    corpus_texts: Optional[List[str]] = None
) -> ExplorationManager:
    """
    Create and initialize exploration manager
    
    Args:
        state_dir: Directory for state files
        corpus_texts: Optional corpus for fitting semantic model
    
    Returns:
        Initialized ExplorationManager
    """
    manager = ExplorationManager(state_dir)
    
    if corpus_texts:
        manager.fit_semantic_model(corpus_texts)
    
    return manager


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ExplorationManager',
    'create_exploration_manager'
]
