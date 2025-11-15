#!/usr/bin/env python3
"""
Contextual Multi-Armed Bandit - Wave 1 Foundation

Upgrades Thompson Sampling to consider query context/features when selecting strategies.

Key Improvements over basic bandits:
1. Uses query features (length, complexity, domain, etc.) to inform decisions
2. Learns which strategies work best for which types of queries
3. Builds feature-strategy performance models
4. Enables personalization and domain-specific optimization

Integration:
- Replaces simple Thompson Sampling in bandit_context.py
- Feeds from query_features.py for feature extraction
- Updates from aria_telemetry.py reward signals
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


# ============================================================================
# PRESET MAPPING CONSTANTS
# ============================================================================

# Map preset names to arm names (for ARIA integration)
PRESET_TO_ARM = {
    "fast": "fast",
    "balanced": "balanced",
    "deep": "deep",
    "diverse": "diverse"
}

# Map arm names to preset configurations
ARM_TO_PRESET_ARGS = {
    "fast": {"top_k": 40, "sem_limit": 64, "rotations": 1, "max_per_file": 8},
    "balanced": {"top_k": 64, "sem_limit": 128, "rotations": 2, "max_per_file": 6},
    "deep": {"top_k": 96, "sem_limit": 256, "rotations": 3, "max_per_file": 5},
    "diverse": {"top_k": 80, "sem_limit": 128, "rotations": 2, "max_per_file": 4}
}

# Default arms for ARIA (retrieval presets)
DEFAULT_ARMS = ["fast", "balanced", "deep", "diverse"]


@dataclass
class ContextualArm:
    """
    An arm (strategy) with context-aware performance tracking
    """

    name: str

    # Feature-specific performance (feature_hash -> {successes, failures})
    feature_performance: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"successes": 0, "failures": 0})
    )

    # Overall statistics
    total_pulls: int = 0
    total_successes: int = 0
    total_failures: int = 0

    # Linear model: reward = feature_weights · features
    feature_weights: Dict[str, float] = field(default_factory=dict)

    # Confidence bounds
    uncertainty: float = 1.0

    def __post_init__(self):
        """Initialize default weights"""
        if not self.feature_weights:
            self.feature_weights = {
                "query_length": 0.0,
                "complexity": 0.0,
                "domain_technical": 0.0,
                "domain_creative": 0.0,
                "has_question": 0.0,
                "num_entities": 0.0,
            }


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit using LinUCB algorithm

    LinUCB (Linear Upper Confidence Bound):
    - Maintains linear model per arm: reward = θ · x (features)
    - Selects arm maximizing: θ · x + α * sqrt(x^T * A^-1 * x)
    - Updates model with ridge regression after each pull

    Benefits:
    - Generalizes across similar contexts
    - Efficient learning with limited data
    - Provably optimal regret bounds
    """

    def __init__(
        self,
        arms: List[str],
        feature_dim: int = 10,
        alpha: float = 1.0,
        state_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            arms: List of arm names (strategies)
            feature_dim: Dimensionality of feature vectors
            alpha: Exploration parameter (higher = more exploration)
            state_path: Path to save/load state (str or Path)
        """
        self.arms = {name: ContextualArm(name) for name in arms}
        self.feature_dim = feature_dim
        self.alpha = alpha
        # BUG FIX: Convert string to Path if needed
        if state_path is None:
            self.state_path = Path.home() / ".aria_contextual_bandit.json"
        elif isinstance(state_path, str):
            self.state_path = Path(state_path)
        else:
            self.state_path = state_path

        # LinUCB matrices per arm
        # A: feature covariance matrix (d x d)
        # b: reward-weighted feature sum (d x 1)
        self.A = {name: np.identity(feature_dim) for name in arms}
        self.b = {name: np.zeros(feature_dim) for name in arms}

        # Feature normalizer (running mean/std)
        self.feature_mean = np.zeros(feature_dim)
        self.feature_std = np.ones(feature_dim)
        self.n_observations = 0

        # Performance tracking
        self.history: List[Dict[str, Any]] = []

        # Load existing state
        self._load_state()

    def extract_features(self, query_context: Dict[str, Any]) -> np.ndarray:
        """
        Extract and normalize feature vector from query context

        Compatible with QueryFeatureExtractor output from ARIA.

        Args:
            query_context: Dict with query metadata
                - length: int (character count)
                - word_count: int
                - complexity: str ('simple', 'moderate', 'complex')
                - domain: str (or 'default')
                - is_code_query: bool
                - is_conceptual: bool
                - is_diagnostic: bool
                - technical_density: float
                - has_multiple_questions: bool
                ... etc

        Returns:
            Normalized feature vector (feature_dim,)
        """
        features = np.zeros(self.feature_dim)

        # Feature 0: Query length (normalized to 0-1, cap at 200 chars)
        length = query_context.get("length", query_context.get("query_length", 50))
        features[0] = min(length / 200.0, 1.0)

        # Feature 1: Complexity score (convert string to numeric if needed)
        complexity = query_context.get("complexity", "moderate")
        if isinstance(complexity, str):
            complexity_map = {"simple": 0.2, "moderate": 0.5, "complex": 0.8, "expert": 1.0}
            features[1] = complexity_map.get(complexity, 0.5)
        else:
            features[1] = float(complexity)

        # Feature 2: Domain - technical
        domain = query_context.get("domain", "default")
        features[2] = 1.0 if domain in ["technical", "code"] or query_context.get("is_code_query", False) else 0.0

        # Feature 3: Domain - creative
        features[3] = 1.0 if domain == "creative" else 0.0

        # Feature 4: Domain - analytical
        features[4] = 1.0 if domain == "analytical" or query_context.get("technical_density", 0.0) > 0.3 else 0.0

        # Feature 5: Domain - conceptual/philosophical
        features[5] = 1.0 if domain in ["concept", "philosophical"] or query_context.get("is_conceptual", False) else 0.0

        # Feature 6: Has question indicator
        has_question = query_context.get("has_question", False)
        if not isinstance(has_question, bool):
            # Might be a count
            has_question = bool(has_question)
        features[6] = 1.0 if has_question or query_context.get("has_multiple_questions", False) else 0.0

        # Feature 7: Entity/term count (normalized)
        entities = query_context.get("entities", [])
        term_count = query_context.get("term_count", query_context.get("word_count", 0))
        num_items = len(entities) if entities else max(term_count / 5, 0)
        features[7] = min(num_items / 10.0, 1.0)

        # Feature 8: Diagnostic/problem-solving indicator
        is_diagnostic = query_context.get("is_diagnostic", False)
        features[8] = 1.0 if is_diagnostic else 0.0

        # Feature 9: Bias term (always 1)
        features[9] = 1.0

        # Normalize using running statistics (only after warmup)
        if self.n_observations > 10:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return features

    def select_arm(
        self, query_context: Dict[str, Any], mode: str = "ucb"
    ) -> Tuple[str, float, np.ndarray]:
        """
        Select best arm using LinUCB

        Args:
            query_context: Query metadata dict
            mode: 'ucb' (exploration), 'greedy' (exploitation), 'thompson' (sampling), 'random' (pure exploration)

        Returns:
            (arm_name, expected_reward, features)
        """
        features = self.extract_features(query_context)

        # Random exploration mode
        if mode == "random":
            import random
            best_arm = random.choice(list(self.arms.keys()))
            return best_arm, 0.0, features

        scores = {}
        for name, arm in self.arms.items():
            # Compute θ = A^-1 * b (ridge regression solution)
            A_inv = np.linalg.inv(self.A[name])
            theta = A_inv @ self.b[name]

            # Expected reward: θ^T * x
            expected_reward = theta @ features

            # Confidence bound: α * sqrt(x^T * A^-1 * x)
            confidence = self.alpha * np.sqrt(features @ A_inv @ features)

            if mode == "ucb":
                # Upper Confidence Bound
                scores[name] = expected_reward + confidence
            elif mode == "greedy":
                # Pure exploitation
                scores[name] = expected_reward
            elif mode == "thompson":
                # Thompson sampling with linear posterior
                sampled_theta = np.random.multivariate_normal(
                    theta, self.alpha**2 * A_inv
                )
                scores[name] = sampled_theta @ features

        # Select best arm
        best_arm = max(scores.keys(), key=lambda k: scores[k])

        return best_arm, scores[best_arm], features

    def select_arm_epsilon_greedy(
        self, query_context: Dict[str, Any], epsilon: float = 0.10
    ) -> Tuple[str, float, np.ndarray, str]:
        """
        Select arm with epsilon-greedy exploration

        With probability epsilon: select random arm (exploration)
        With probability (1-epsilon): select best UCB arm (exploitation)

        Args:
            query_context: Query metadata dict
            epsilon: Probability of random exploration (default 0.10 = 10%)

        Returns:
            (arm_name, expected_reward, features, selection_method)
        """
        import random

        if random.random() < epsilon:
            # Exploration: random selection
            arm_name, _, features = self.select_arm(query_context, mode="random")
            return arm_name, 0.0, features, "epsilon_random"
        else:
            # Exploitation: UCB selection
            arm_name, reward, features = self.select_arm(query_context, mode="ucb")
            return arm_name, reward, features, "epsilon_ucb"

    def update(
        self,
        arm_name: str,
        features: np.ndarray,
        reward: float,
        query_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Update arm statistics with observed reward

        Args:
            arm_name: Arm that was pulled
            features: Feature vector used for selection
            reward: Observed reward (0-1, or can be continuous)
            query_context: Optional context for detailed tracking
        """
        if arm_name not in self.arms:
            return

        arm = self.arms[arm_name]

        # Update LinUCB matrices
        # A = A + x * x^T
        self.A[arm_name] += np.outer(features, features)

        # b = b + r * x
        self.b[arm_name] += reward * features

        # Update arm statistics
        arm.total_pulls += 1
        if reward > 0.5:  # Binary classification threshold
            arm.total_successes += 1
        else:
            arm.total_failures += 1

        # Update feature normalizer
        self.n_observations += 1
        delta = features - self.feature_mean
        self.feature_mean += delta / self.n_observations
        self.feature_std = np.sqrt(
            (self.n_observations - 1) * self.feature_std**2 / self.n_observations
            + delta**2 / self.n_observations
        )

        # Track history
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "arm": arm_name,
                "reward": reward,
                "features": features.tolist(),
                "context": query_context or {},
            }
        )

        # Periodic save
        if len(self.history) % 10 == 0:
            self._save_state()

    def get_arm_stats(self, arm_name: str) -> Dict[str, Any]:
        """Get detailed statistics for an arm"""
        if arm_name not in self.arms:
            return {}

        arm = self.arms[arm_name]

        # Compute current theta (weights)
        A_inv = np.linalg.inv(self.A[arm_name])
        theta = A_inv @ self.b[arm_name]

        return {
            "name": arm_name,
            "total_pulls": arm.total_pulls,
            "success_rate": arm.total_successes / max(arm.total_pulls, 1),
            "feature_weights": theta.tolist(),
            "uncertainty": np.trace(A_inv),  # Trace of covariance = total uncertainty
            "last_10_rewards": [
                h["reward"] for h in self.history[-10:] if h["arm"] == arm_name
            ],
        }

    def get_rankings(self, query_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get arm rankings for a given context

        Returns:
            List of (arm_name, ucb_score) sorted by score descending
        """
        features = self.extract_features(query_context)

        rankings = []
        for name, arm in self.arms.items():
            A_inv = np.linalg.inv(self.A[name])
            theta = A_inv @ self.b[name]

            expected_reward = theta @ features
            confidence = self.alpha * np.sqrt(features @ A_inv @ features)
            ucb_score = expected_reward + confidence

            rankings.append((name, ucb_score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def _save_state(self):
        """Save bandit state to disk"""
        try:
            state = {
                "version": "1.0",
                "arms": {
                    name: {
                        "total_pulls": arm.total_pulls,
                        "total_successes": arm.total_successes,
                        "total_failures": arm.total_failures,
                    }
                    for name, arm in self.arms.items()
                },
                "A": {name: A.tolist() for name, A in self.A.items()},
                "b": {name: b.tolist() for name, b in self.b.items()},
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
                "n_observations": self.n_observations,
                "history": self.history[-1000:],  # Keep last 1000
            }

            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save contextual bandit state: {e}")

    def save_state(self):
        """Public method to manually save bandit state"""
        self._save_state()

    def _load_state(self):
        """Load bandit state from disk"""
        if not self.state_path.exists():
            return

        try:
            state = json.loads(self.state_path.read_text())

            # Restore arm statistics
            for name, arm_data in state.get("arms", {}).items():
                if name in self.arms:
                    self.arms[name].total_pulls = arm_data["total_pulls"]
                    self.arms[name].total_successes = arm_data["total_successes"]
                    self.arms[name].total_failures = arm_data["total_failures"]

            # Restore LinUCB matrices
            for name in self.arms:
                if name in state.get("A", {}):
                    self.A[name] = np.array(state["A"][name], dtype=np.float64)
                if name in state.get("b", {}):
                    self.b[name] = np.array(state["b"][name], dtype=np.float64)  # type: ignore[assignment]

            # Restore feature normalizer
            if "feature_mean" in state:
                self.feature_mean = np.array(state["feature_mean"])  # type: ignore[assignment]
            if "feature_std" in state:
                self.feature_std = np.array(state["feature_std"])  # type: ignore[assignment]
            if "n_observations" in state:
                self.n_observations = state["n_observations"]

            # Restore history
            self.history = state.get("history", [])

            print(
                f"✓ Loaded contextual bandit state: {len(self.arms)} arms, {self.n_observations} observations"
            )

        except Exception as e:
            print(f"Warning: Could not load contextual bandit state: {e}")


# ============================================================================
# Integration Example
# ============================================================================


def integrate_with_aria():
    """
    Example: How to integrate contextual bandit into ARIA

    Replace calls to bandit_context.py Thompson sampling with contextual selection
    """

    # Initialize contextual bandit with ARIA's retrieval strategies
    strategies = [
        "bm25_only",
        "hybrid_base",
        "hybrid_aggressive",
        "semantic_only",
        "cross_encoder_heavy",
        "pca_exploration",
        "golden_ratio_spiral",
        "adaptive_threshold",
    ]

    bandit = ContextualBandit(
        arms=strategies,
        feature_dim=10,
        alpha=1.5,  # Moderate exploration
        state_path=Path.home() / ".aria_contextual_bandit.json",
    )

    # Example query context (extracted from query_features.py)
    query_context = {
        "query": "How does quantum entanglement work?",
        "query_length": 35,
        "complexity_score": 0.7,
        "domain": "technical",
        "entities": ["quantum", "entanglement"],
        "has_question": True,
    }

    # Select strategy using contextual features
    strategy, expected_reward, features = bandit.select_arm(query_context, mode="ucb")

    print(f"Selected strategy: {strategy} (expected reward: {expected_reward:.3f})")

    # ... run retrieval with selected strategy ...
    # retrieval_result = run_retrieval(query, strategy)

    # After retrieval, compute reward from metrics
    # reward = compute_reward(retrieval_result)

    # Example reward (0-1 scale)
    reward = 0.85

    # Update bandit
    bandit.update(strategy, features, reward, query_context)

    # Get current rankings for this context
    rankings = bandit.get_rankings(query_context)
    print("\nStrategy rankings for this context:")
    for name, score in rankings[:5]:
        print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    integrate_with_aria()
