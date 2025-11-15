#!/usr/bin/env python3
"""
Neural Contextual Bandit - Wave 1 Foundation

Uses neural networks to learn complex, non-linear relationships between
query features and strategy performance.

Integration with Current ARIA:
- Uses query_features.py QueryFeatureExtractor for 15-dim feature vectors
- Upgrades from Thompson Sampling in bandit_context.py
- Feeds from aria_telemetry.py for reward signals
- Persists state like current bandit_context.py

Key Advantages over Linear Contextual Bandits:
1. Learns non-linear feature interactions
2. Handles high-dimensional features better
3. Adapts to changing query distributions
4. Can discover complex patterns (e.g., "technical + long + follow-up" queries work best with strategy X)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

# Import current ARIA feature extraction
try:
    from query_features import QueryFeatureExtractor, QueryFeatures
except ImportError:
    print("Warning: Could not import query_features. Using fallback.")
    QueryFeatureExtractor = None


@dataclass
class NeuralArm:
    """
    An arm (strategy) with neural network prediction model
    """
    name: str
    
    # Network weights (simple 2-layer network)
    # input_dim -> hidden_dim -> 1 (predicted reward)
    W1: Optional[np.ndarray] = None  # (input_dim, hidden_dim)
    b1: Optional[np.ndarray] = None  # (hidden_dim,)
    W2: Optional[np.ndarray] = None  # (hidden_dim, 1)
    b2: Optional[np.ndarray] = None  # (1,)
    
    # Training hyperparameters
    learning_rate: float = 0.01
    hidden_dim: int = 32
    
    # Experience replay buffer (for stable training)
    experience_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Statistics
    total_pulls: int = 0
    total_reward: float = 0.0
    
    # Uncertainty tracking (using dropout at test time)
    uncertainty_samples: int = 10
    
    def __post_init__(self):
        """Initialize network if not loaded"""
        if self.W1 is None:
            # He initialization for ReLU networks
            input_dim = 15  # From query_features.py vectorization
            self.W1 = np.random.randn(input_dim, self.hidden_dim) * np.sqrt(2.0 / input_dim)
            self.b1 = np.zeros(self.hidden_dim)
            self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
            self.b2 = np.zeros(1)


class NeuralBandit:
    """
    Neural Contextual Bandit using Thompson Sampling with Neural Networks
    
    Architecture:
    - Each arm has a 2-layer neural network
    - Networks predict expected reward given query features
    - Thompson Sampling: sample from posterior using dropout as Bayesian approximation
    - Experience replay for stable training
    
    Benefits:
    - Learns complex non-linear patterns
    - Generalizes across similar contexts
    - Uncertainty quantification via dropout
    - Online learning with experience replay
    """
    
    def __init__(
        self,
        arms: List[str],
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        dropout_rate: float = 0.1,
        state_path: Optional[Path] = None
    ):
        """
        Args:
            arms: List of arm names (retrieval strategies)
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for SGD
            dropout_rate: Dropout rate for uncertainty estimation
            state_path: Path to save/load state
        """
        self.arms = {
            name: NeuralArm(name, hidden_dim=hidden_dim, learning_rate=learning_rate)
            for name in arms
        }
        self.dropout_rate = dropout_rate
        self.state_path = state_path or Path.home() / '.aria_neural_bandit.json'
        
        # Feature extractor
        if QueryFeatureExtractor is not None:
            self.feature_extractor = QueryFeatureExtractor()
        else:
            self.feature_extractor = None
        
        # Training config
        self.batch_size = 32
        self.train_frequency = 10  # Train every N selections
        self.selections_since_train = 0
        
        # History
        self.history: List[Dict[str, Any]] = []
        
        # Load existing state
        self._load_state()
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _dropout_mask(self, shape: Tuple[int, ...], rate: float) -> np.ndarray:
        """Generate dropout mask"""
        return (np.random.rand(*shape) > rate).astype(np.float32)
    
    def _forward(
        self,
        arm: NeuralArm,
        features: np.ndarray,
        training: bool = False
    ) -> float:
        """
        Forward pass through network
        
        Args:
            arm: Arm with network weights
            features: Input feature vector (15,)
            training: If True, apply dropout
        
        Returns:
            Predicted reward (scalar)
        """
        # Ensure weights are initialized
        assert arm.W1 is not None and arm.b1 is not None and arm.W2 is not None and arm.b2 is not None
        
        x = features
        
        # Layer 1: input -> hidden
        h = self._relu(x @ arm.W1 + arm.b1)
        
        # Dropout (if training or for uncertainty)
        if training:
            mask = self._dropout_mask(h.shape, self.dropout_rate)
            h = h * mask / (1 - self.dropout_rate)
        
        # Layer 2: hidden -> output
        out = h @ arm.W2 + arm.b2
        
        return float(out[0])
    
    def _backward(
        self,
        arm: NeuralArm,
        features: np.ndarray,
        target: float
    ):
        """
        Backward pass - update network weights using SGD
        
        Args:
            arm: Arm to update
            features: Input features
            target: Target reward
        """
        # Ensure weights are initialized
        assert arm.W1 is not None and arm.b1 is not None and arm.W2 is not None and arm.b2 is not None
        
        # Forward pass (save activations)
        x = features
        h = self._relu(x @ arm.W1 + arm.b1)
        pred = (h @ arm.W2 + arm.b2)[0]
        
        # Compute loss gradient
        loss_grad = 2 * (pred - target)  # MSE gradient
        
        # Backprop through layer 2
        dW2 = np.outer(h, loss_grad)
        db2 = np.array([loss_grad])
        
        # Backprop through ReLU and layer 1
        dh = loss_grad * arm.W2.flatten()
        dh[h <= 0] = 0  # ReLU gradient
        
        dW1 = np.outer(x, dh)
        db1 = dh
        
        # SGD update with gradient clipping
        clip_value = 1.0
        dW1 = np.clip(dW1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        
        arm.W1 -= arm.learning_rate * dW1
        arm.b1 -= arm.learning_rate * db1
        arm.W2 -= arm.learning_rate * dW2
        arm.b2 -= arm.learning_rate * db2
    
    def extract_features(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract feature vector from query using current ARIA feature extraction
        
        Args:
            query: Query string
            query_context: Optional context dict
        
        Returns:
            Feature vector (15,) from query_features.py
        """
        if self.feature_extractor is not None:
            conversation_context = query_context.get('conversation_context', []) if query_context else None
            query_features = self.feature_extractor.extract(query, conversation_context)
            if query_features.vector is not None:
                return query_features.vector
        
        # Fallback: simple feature vector if extractor unavailable or returns None
        return np.random.rand(15).astype(np.float32)
    
    def select_arm(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]] = None,
        mode: str = 'thompson'
    ) -> Tuple[str, float, np.ndarray]:
        """
        Select best arm using Thompson Sampling with neural networks
        
        Args:
            query: Query string
            query_context: Optional metadata
            mode: 'thompson' (sampling), 'greedy' (exploitation), 'ucb' (upper confidence)
        
        Returns:
            (arm_name, score, features)
        """
        features = self.extract_features(query, query_context)
        
        scores = {}
        for name, arm in self.arms.items():
            if mode == 'thompson':
                # Thompson Sampling: sample multiple predictions with dropout
                samples = [
                    self._forward(arm, features, training=True)
                    for _ in range(arm.uncertainty_samples)
                ]
                # Sample from distribution
                scores[name] = np.random.choice(samples)
                
            elif mode == 'greedy':
                # Pure exploitation: use mean prediction
                scores[name] = self._forward(arm, features, training=False)
                
            elif mode == 'ucb':
                # Upper Confidence Bound: mean + std
                samples = [
                    self._forward(arm, features, training=True)
                    for _ in range(arm.uncertainty_samples)
                ]
                mean = np.mean(samples)
                std = np.std(samples)
                scores[name] = mean + 2.0 * std  # UCB with α=2
        
        # Select best arm
        best_arm = max(scores.keys(), key=lambda k: scores[k])
        
        return best_arm, scores[best_arm], features
    
    def update(
        self,
        arm_name: str,
        features: np.ndarray,
        reward: float,
        query_context: Optional[Dict[str, Any]] = None
    ):
        """
        Update arm with observed reward
        
        Args:
            arm_name: Arm that was pulled
            features: Feature vector used
            reward: Observed reward (0-1)
            query_context: Optional context
        """
        if arm_name not in self.arms:
            return
        
        arm = self.arms[arm_name]
        
        # Add to experience buffer
        arm.experience_buffer.append({
            'features': features.copy(),
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update statistics
        arm.total_pulls += 1
        arm.total_reward += reward
        
        # Track history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'arm': arm_name,
            'reward': reward,
            'context': query_context or {}
        })
        
        # Train networks periodically
        self.selections_since_train += 1
        if self.selections_since_train >= self.train_frequency:
            self._train_all_arms()
            self.selections_since_train = 0
        
        # Periodic save
        if len(self.history) % 50 == 0:
            self._save_state()
    
    def _train_all_arms(self):
        """
        Train all arms using experience replay
        """
        for arm in self.arms.values():
            if len(arm.experience_buffer) < self.batch_size:
                continue
            
            # Sample mini-batch from experience buffer
            indices = np.random.choice(
                len(arm.experience_buffer),
                size=min(self.batch_size, len(arm.experience_buffer)),
                replace=False
            )
            
            batch = [arm.experience_buffer[i] for i in indices]
            
            # Train on mini-batch
            for exp in batch:
                self._backward(arm, exp['features'], exp['reward'])
    
    def get_arm_stats(self, arm_name: str) -> Dict[str, Any]:
        """Get detailed statistics for an arm"""
        if arm_name not in self.arms:
            return {}
        
        arm = self.arms[arm_name]
        
        # Compute average reward
        avg_reward = arm.total_reward / max(arm.total_pulls, 1)
        
        # Recent performance
        recent_history = [
            h for h in self.history[-100:]
            if h['arm'] == arm_name
        ]
        recent_rewards = [h['reward'] for h in recent_history]
        
        return {
            'name': arm_name,
            'total_pulls': arm.total_pulls,
            'avg_reward': avg_reward,
            'experience_buffer_size': len(arm.experience_buffer),
            'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'recent_std_reward': np.std(recent_rewards) if recent_rewards else 0.0,
            'last_10_rewards': recent_rewards[-10:] if recent_rewards else []
        }
    
    def get_rankings(
        self,
        query: str,
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Get arm rankings with uncertainty for a given query
        
        Returns:
            List of (arm_name, mean_score, std_score) sorted by mean descending
        """
        features = self.extract_features(query, query_context)
        
        rankings = []
        for name, arm in self.arms.items():
            # Sample predictions
            samples = [
                self._forward(arm, features, training=True)
                for _ in range(arm.uncertainty_samples)
            ]
            mean_score = np.mean(samples)
            std_score = np.std(samples)
            
            rankings.append((name, mean_score, std_score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def _save_state(self):
        """Save bandit state to disk"""
        try:
            state = {
                'version': '1.0',
                'arms': {
                    name: {
                        'W1': arm.W1.tolist() if arm.W1 is not None else [],
                        'b1': arm.b1.tolist() if arm.b1 is not None else [],
                        'W2': arm.W2.tolist() if arm.W2 is not None else [],
                        'b2': arm.b2.tolist() if arm.b2 is not None else [],
                        'total_pulls': arm.total_pulls,
                        'total_reward': arm.total_reward,
                        'experience_buffer': list(arm.experience_buffer)[-100:]  # Save last 100
                    }
                    for name, arm in self.arms.items()
                },
                'history': self.history[-1000:]  # Keep last 1000
            }
            
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save neural bandit state: {e}")
    
    def _load_state(self):
        """Load bandit state from disk"""
        if not self.state_path.exists():
            return
        
        try:
            state = json.loads(self.state_path.read_text())
            
            for name, arm_data in state.get('arms', {}).items():
                if name in self.arms:
                    arm = self.arms[name]
                    
                    # Restore network weights
                    arm.W1 = np.array(arm_data['W1'])
                    arm.b1 = np.array(arm_data['b1'])
                    arm.W2 = np.array(arm_data['W2'])
                    arm.b2 = np.array(arm_data['b2'])
                    
                    # Restore statistics
                    arm.total_pulls = arm_data['total_pulls']
                    arm.total_reward = arm_data['total_reward']
                    
                    # Restore experience buffer
                    for exp in arm_data.get('experience_buffer', []):
                        exp['features'] = np.array(exp['features'])
                        arm.experience_buffer.append(exp)
            
            # Restore history
            self.history = state.get('history', [])
            
            print(f"✓ Loaded neural bandit state: {len(self.arms)} arms, {sum(a.total_pulls for a in self.arms.values())} total pulls")
        
        except Exception as e:
            print(f"Warning: Could not load neural bandit state: {e}")


# ============================================================================
# Integration with Current ARIA
# ============================================================================

def integrate_with_current_aria():
    """
    Example: Replace bandit_context.py with neural bandit
    
    Drop-in replacement for BanditState in current bandit_context.py
    """
    
    # Initialize with current ARIA strategies (from presets.py / bandit_context.py)
    strategies = [
        'fast',      # top_k: 40, sem_limit: 64, rotations: 1
        'balanced',  # top_k: 64, sem_limit: 128, rotations: 2
        'deep',      # top_k: 96, sem_limit: 256, rotations: 3
        'diverse',   # top_k: 80, sem_limit: 128, rotations: 2, max_per_file: 6
    ]
    
    bandit = NeuralBandit(
        arms=strategies,
        hidden_dim=32,
        learning_rate=0.01,
        dropout_rate=0.1,
        state_path=Path.home() / '.aria_neural_bandit.json'
    )
    
    # Example: Select strategy for a query
    query = "How does quantum entanglement enable quantum computing?"
    
    strategy, score, features = bandit.select_arm(
        query,
        query_context={'conversation_context': []},
        mode='thompson'
    )
    
    print(f"Selected strategy: {strategy} (score: {score:.3f})")
    print(f"Features: {features[:5]}...")  # First 5 features
    
    # ... run retrieval with strategy ...
    # retrieval_result = run_v7_retrieval(query, strategy_preset)
    
    # Compute reward from metrics (e.g., from aria_telemetry.py)
    reward = 0.82  # Example: coverage_score * 0.5 + exemplar_fit * 0.5
    
    # Update bandit
    bandit.update(strategy, features, reward, {'query': query})
    
    # Get current rankings
    rankings = bandit.get_rankings(query)
    print("\nStrategy rankings:")
    for name, mean, std in rankings:
        print(f"  {name}: {mean:.3f} ± {std:.3f}")


if __name__ == '__main__':
    integrate_with_current_aria()
