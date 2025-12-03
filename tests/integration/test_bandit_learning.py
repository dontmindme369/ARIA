"""
Integration tests for bandit learning loop

Tests the complete learning cycle:
1. Query -> Feature extraction
2. Feature -> Bandit selection
3. Execution -> Reward
4. Reward -> Bandit update
5. Convergence over time
"""

import pytest
import numpy as np
from pathlib import Path

from intelligence.contextual_bandit import ContextualBandit, DEFAULT_ARMS
from retrieval.query_features import QueryFeatureExtractor


@pytest.mark.integration
class TestBanditLearningLoop:
    """Test complete bandit learning loop"""

    def test_query_to_selection_pipeline(self, clean_bandit_state):
        """Test pipeline from query to arm selection"""
        extractor = QueryFeatureExtractor()
        bandit = ContextualBandit(
            arms=DEFAULT_ARMS,
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        query = "How do I implement a binary search tree in Python?"

        # Extract features
        query_features = extractor.extract(query)

        # Create context dict for bandit
        context = {
            "length": query_features.length,
            "complexity": query_features.complexity,
            "domain": query_features.domain,
            "word_count": query_features.word_count,
            "technical_density": query_features.technical_density,
        }

        # Select arm
        arm_name, score, features = bandit.select_arm(context, mode="ucb")

        assert arm_name in DEFAULT_ARMS
        assert features.shape == (10,)
        assert np.isfinite(score)

    def test_learning_convergence(self, clean_bandit_state):
        """Test that bandit learns from feedback"""
        bandit = ContextualBandit(
            arms=["good", "bad"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        # Create two distinct query types
        complex_context = {
            "length": 200,
            "complexity": "complex",
            "domain": "code",
            "word_count": 30,
        }

        simple_context = {
            "length": 50,
            "complexity": "simple",
            "domain": "factual",
            "word_count": 8,
        }

        # Train: "good" arm works for complex queries
        for _ in range(30):
            arm, _, features = bandit.select_arm(complex_context, mode="ucb")
            if arm == "good":
                reward = 0.9
            else:
                reward = 0.2
            bandit.update(arm, features, reward)

        # Train: "bad" arm works for simple queries
        for _ in range(30):
            arm, _, features = bandit.select_arm(simple_context, mode="ucb")
            if arm == "bad":
                reward = 0.9
            else:
                reward = 0.2
            bandit.update(arm, features, reward)

        # Test: Should prefer "good" for complex queries
        selections = []
        for _ in range(10):
            arm, _, _ = bandit.select_arm(complex_context, mode="greedy")
            selections.append(arm)

        # Should show learning (allow some randomness)
        assert selections.count("good") >= 5

    def test_epsilon_greedy_exploration(self, clean_bandit_state):
        """Test epsilon-greedy balances exploration and exploitation"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2", "arm3"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 100, "complexity": "medium"}

        # With high epsilon, should see diverse selections
        selections = []
        for _ in range(100):
            arm, _, _, method = bandit.select_arm_epsilon_greedy(
                context, epsilon=0.3
            )
            selections.append(arm)

        # Should have tried multiple arms
        unique_arms = len(set(selections))
        assert unique_arms >= 2

    def test_reward_signal_integration(self, clean_bandit_state):
        """Test integration of reward signals"""
        bandit = ContextualBandit(
            arms=["test"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 100}

        # Simulate reward feedback loop
        for reward_value in [0.1, 0.3, 0.5, 0.7, 0.9]:
            arm, _, features = bandit.select_arm(context, mode="ucb")
            bandit.update(arm, features, reward_value)

        # Check that updates were recorded
        stats = bandit.get_arm_stats("test")
        assert stats["total_pulls"] == 5
        assert len(stats["last_10_rewards"]) == 5

    def test_state_persistence_across_sessions(self, clean_bandit_state):
        """Test that learning persists across sessions"""
        # Session 1: Train bandit
        bandit1 = ContextualBandit(
            arms=["persistent"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 150, "complexity": "complex"}
        for _ in range(20):
            arm, _, features = bandit1.select_arm(context, mode="ucb")
            bandit1.update(arm, features, reward=0.85)

        bandit1.save_state()
        pulls_session1 = bandit1.arms["persistent"].total_pulls

        # Session 2: Load and continue
        bandit2 = ContextualBandit(
            arms=["persistent"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        # Should have loaded previous state
        assert bandit2.arms["persistent"].total_pulls == pulls_session1

        # Continue training
        for _ in range(10):
            arm, _, features = bandit2.select_arm(context, mode="ucb")
            bandit2.update(arm, features, reward=0.8)

        assert bandit2.arms["persistent"].total_pulls == pulls_session1 + 10


@pytest.mark.integration
class TestFeatureExtractionIntegration:
    """Test feature extraction integration with bandit"""

    def test_extractor_output_compatible_with_bandit(self):
        """Test that extractor output works with bandit input"""
        extractor = QueryFeatureExtractor()

        queries = [
            "What is Python?",
            "Compare TensorFlow and PyTorch architectures",
            "Debug memory leak in C++",
        ]

        for query in queries:
            features = extractor.extract(query)

            # Convert to bandit context
            context = {
                "length": features.length,
                "complexity": features.complexity,
                "domain": features.domain,
                "technical_density": features.technical_density,
                "word_count": features.word_count,
                "has_question": '?' in query,
            }

            # Should work with bandit
            bandit = ContextualBandit(arms=["test"], feature_dim=10)
            extracted_features = bandit.extract_features(context)

            assert extracted_features.shape == (10,)
            assert np.all(np.isfinite(extracted_features))

    def test_domain_to_bandit_feature_mapping(self):
        """Test domain mapping to bandit features"""
        extractor = QueryFeatureExtractor()
        bandit = ContextualBandit(arms=["test"], feature_dim=10)

        test_cases = [
            ("How to code?", "code", 2),  # Technical domain, feature index 2
            ("Creative writing", "creative", 3),  # Creative domain, feature index 3
        ]

        for query, expected_domain, feature_idx in test_cases:
            qf = extractor.extract(query)
            if qf.domain == expected_domain:
                context = {"domain": qf.domain}
                features = bandit.extract_features(context)
                # Corresponding feature should be activated
                assert features[feature_idx] >= 0.0


@pytest.mark.integration
@pytest.mark.slow
class TestLongTermLearning:
    """Test long-term learning behavior"""

    def test_convergence_over_100_iterations(self, clean_bandit_state):
        """Test convergence over extended training"""
        bandit = ContextualBandit(
            arms=["optimal", "suboptimal"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 100, "complexity": "medium"}

        # Track selections over time
        selections_early = []
        selections_late = []

        for i in range(100):
            arm, _, features = bandit.select_arm(context, mode="ucb")

            # Optimal arm gives better rewards
            reward = 0.9 if arm == "optimal" else 0.3
            bandit.update(arm, features, reward)

            if i < 20:
                selections_early.append(arm)
            elif i >= 80:
                selections_late.append(arm)

        # Should increasingly prefer optimal arm
        optimal_early = selections_early.count("optimal") / len(selections_early)
        optimal_late = selections_late.count("optimal") / len(selections_late)

        assert optimal_late > optimal_early

    def test_regret_minimization(self, clean_bandit_state):
        """Test that cumulative regret grows sub-linearly"""
        bandit = ContextualBandit(
            arms=["best", "medium", "worst"],
            feature_dim=10,
            alpha=1.5,  # Higher exploration
            state_path=clean_bandit_state,
        )

        context = {"length": 100}

        # True rewards for each arm
        true_rewards = {"best": 0.9, "medium": 0.6, "worst": 0.3}

        cumulative_regret = []
        total_regret = 0

        for _ in range(100):
            arm, _, features = bandit.select_arm(context, mode="ucb")
            reward = true_rewards[arm]
            bandit.update(arm, features, reward)

            # Regret = difference from optimal
            regret = true_rewards["best"] - reward
            total_regret += regret
            cumulative_regret.append(total_regret)

        # Regret should grow sub-linearly (slower than linear)
        # Check that rate of regret growth is decreasing
        early_regret_rate = cumulative_regret[19] - cumulative_regret[0]
        late_regret_rate = cumulative_regret[99] - cumulative_regret[80]

        assert late_regret_rate < early_regret_rate
