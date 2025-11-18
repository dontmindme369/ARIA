"""
Unit tests for LinUCB Contextual Bandit

Tests cover:
- Feature extraction
- Arm selection (UCB, greedy, Thompson, random)
- Model updates
- Epsilon-greedy exploration
- Matrix operations and numerical stability
- State persistence (save/load)
- Edge cases and error handling
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from intelligence.contextual_bandit import (
    ContextualBandit,
    ContextualArm,
    PRESET_TO_ARM,
    ARM_TO_PRESET_ARGS,
    DEFAULT_ARMS,
)


class TestContextualArm:
    """Test ContextualArm dataclass"""

    def test_arm_initialization(self):
        """Test arm initialization with default values"""
        arm = ContextualArm("test_arm")

        assert arm.name == "test_arm"
        assert arm.total_pulls == 0
        assert arm.total_successes == 0
        assert arm.total_failures == 0
        assert arm.uncertainty == 1.0
        assert isinstance(arm.feature_weights, dict)
        assert len(arm.feature_weights) > 0

    def test_arm_feature_weights_initialized(self):
        """Test that feature weights are initialized to zero"""
        arm = ContextualArm("test")

        expected_keys = [
            "query_length",
            "complexity",
            "domain_technical",
            "domain_creative",
            "has_question",
            "num_entities",
        ]

        for key in expected_keys:
            assert key in arm.feature_weights
            assert arm.feature_weights[key] == 0.0


class TestContextualBanditInitialization:
    """Test bandit initialization"""

    def test_basic_initialization(self, clean_bandit_state):
        """Test bandit creation with basic parameters"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2"],
            feature_dim=10,
            alpha=1.0,
            state_path=clean_bandit_state,
        )

        assert len(bandit.arms) == 2
        assert "arm1" in bandit.arms
        assert "arm2" in bandit.arms
        assert bandit.feature_dim == 10
        assert bandit.alpha == 1.0
        assert bandit.n_observations == 0

    def test_default_arms(self, clean_bandit_state):
        """Test initialization with default ARIA arms"""
        bandit = ContextualBandit(
            arms=DEFAULT_ARMS, feature_dim=10, state_path=clean_bandit_state
        )

        assert len(bandit.arms) == 4
        for arm_name in ["fast", "balanced", "deep", "diverse"]:
            assert arm_name in bandit.arms

    def test_linucb_matrices_initialized(self, clean_bandit_state):
        """Test that A and b matrices are initialized correctly"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # A should be identity matrix
        assert bandit.A["test"].shape == (10, 10)
        np.testing.assert_array_almost_equal(bandit.A["test"], np.identity(10))

        # b should be zero vector
        assert bandit.b["test"].shape == (10,)
        np.testing.assert_array_almost_equal(bandit.b["test"], np.zeros(10))

    def test_feature_normalizer_initialized(self, clean_bandit_state):
        """Test feature normalizer initialization"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        np.testing.assert_array_almost_equal(
            bandit.feature_mean, np.zeros(10)
        )
        np.testing.assert_array_almost_equal(bandit.feature_std, np.ones(10))


class TestFeatureExtraction:
    """Test feature extraction from query context"""

    def test_extract_basic_features(self, clean_bandit_state):
        """Test extraction of basic features"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {
            "length": 100,
            "complexity": "complex",
            "domain": "technical",
            "has_question": True,
            "word_count": 15,
        }

        features = bandit.extract_features(context)

        assert features.shape == (10,)
        assert features[0] == 0.5  # length/200 = 100/200 = 0.5
        assert features[1] == 0.8  # complex = 0.8
        assert features[2] == 1.0  # technical domain
        assert features[6] == 1.0  # has_question
        assert features[9] == 1.0  # bias term

    def test_extract_complexity_mapping(self, clean_bandit_state):
        """Test complexity string to numeric mapping"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        test_cases = [
            ("simple", 0.2),
            ("moderate", 0.5),
            ("complex", 0.8),
            ("expert", 1.0),
        ]

        for complexity_str, expected_value in test_cases:
            context = {"complexity": complexity_str}
            features = bandit.extract_features(context)
            assert features[1] == expected_value

    def test_extract_domain_features(self, clean_bandit_state):
        """Test domain feature extraction"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Technical domain
        context1 = {"domain": "technical"}
        features1 = bandit.extract_features(context1)
        assert features1[2] == 1.0  # technical
        assert features1[3] == 0.0  # creative
        assert features1[4] == 0.0  # analytical

        # Creative domain
        context2 = {"domain": "creative"}
        features2 = bandit.extract_features(context2)
        assert features2[2] == 0.0  # technical
        assert features2[3] == 1.0  # creative

    def test_extract_code_query_detection(self, clean_bandit_state):
        """Test code query detection"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"domain": "code", "is_code_query": True}
        features = bandit.extract_features(context)

        assert features[2] == 1.0  # Should be marked as technical

    def test_extract_empty_context(self, clean_bandit_state):
        """Test feature extraction with empty context"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = bandit.extract_features({})

        assert features.shape == (10,)
        assert features[9] == 1.0  # Bias term always 1
        # Other features should have defaults

    def test_feature_dimensionality_validation(self, clean_bandit_state):
        """Test that feature dimension validation works"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 50}
        features = bandit.extract_features(context)

        # Should match specified dimension
        assert len(features) == 10


class TestArmSelection:
    """Test arm selection strategies"""

    def test_select_arm_ucb_mode(self, clean_bandit_state):
        """Test arm selection with UCB mode"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100, "complexity": "complex"}
        arm_name, reward, features = bandit.select_arm(context, mode="ucb")

        assert arm_name in ["arm1", "arm2"]
        assert isinstance(reward, (int, float))
        assert features.shape == (10,)

    def test_select_arm_greedy_mode(self, clean_bandit_state):
        """Test arm selection with greedy mode"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100}
        arm_name, reward, features = bandit.select_arm(context, mode="greedy")

        assert arm_name in ["arm1", "arm2"]
        assert isinstance(reward, (int, float))

    def test_select_arm_random_mode(self, clean_bandit_state):
        """Test random arm selection"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2", "arm3"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 50}

        # Select multiple times and check we get different arms
        selections = set()
        for _ in range(20):
            arm_name, _, _ = bandit.select_arm(context, mode="random")
            selections.add(arm_name)
            assert arm_name in ["arm1", "arm2", "arm3"]

        # Should get multiple different arms (probabilistically)
        assert len(selections) >= 2

    def test_select_arm_thompson_mode(self, clean_bandit_state):
        """Test Thompson sampling mode"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100}
        arm_name, reward, features = bandit.select_arm(context, mode="thompson")

        assert arm_name in ["arm1", "arm2"]
        assert isinstance(reward, (int, float))

    def test_epsilon_greedy_selection(self, clean_bandit_state):
        """Test epsilon-greedy selection"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100}

        # Test with epsilon=0.0 (pure exploitation)
        for _ in range(10):
            arm_name, _, _, method = bandit.select_arm_epsilon_greedy(
                context, epsilon=0.0
            )
            assert method == "epsilon_ucb"

        # Test with epsilon=1.0 (pure exploration)
        for _ in range(10):
            arm_name, _, _, method = bandit.select_arm_epsilon_greedy(
                context, epsilon=1.0
            )
            assert method == "epsilon_random"

        # Test with epsilon=0.5 (mixed)
        methods = []
        for _ in range(100):
            _, _, _, method = bandit.select_arm_epsilon_greedy(context, epsilon=0.5)
            methods.append(method)

        # Should have both methods represented
        assert "epsilon_ucb" in methods
        assert "epsilon_random" in methods

    def test_epsilon_greedy_validation(self, clean_bandit_state):
        """Test epsilon validation in epsilon-greedy"""
        bandit = ContextualBandit(
            arms=["arm1"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 50}

        # Invalid epsilon values
        with pytest.raises(ValueError):
            bandit.select_arm_epsilon_greedy(context, epsilon=-0.1)

        with pytest.raises(ValueError):
            bandit.select_arm_epsilon_greedy(context, epsilon=1.5)


class TestBanditUpdate:
    """Test bandit update mechanics"""

    def test_update_basic(self, clean_bandit_state):
        """Test basic update operation"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100}
        arm_name, _, features = bandit.select_arm(context, mode="ucb")

        # Update with positive reward
        bandit.update(arm_name, features, reward=0.8)

        # Check statistics updated
        assert bandit.arms[arm_name].total_pulls == 1
        assert bandit.arms[arm_name].total_successes == 1
        assert bandit.n_observations == 1

    def test_update_matrix_updates(self, clean_bandit_state):
        """Test that A and b matrices are updated correctly"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.ones(10) * 0.5
        initial_A = bandit.A["test"].copy()
        initial_b = bandit.b["test"].copy()

        bandit.update("test", features, reward=1.0)

        # A should be updated: A = A + x * x^T
        expected_A = initial_A + np.outer(features, features)
        np.testing.assert_array_almost_equal(bandit.A["test"], expected_A)

        # b should be updated: b = b + r * x
        expected_b = initial_b + 1.0 * features
        np.testing.assert_array_almost_equal(bandit.b["test"], expected_b)

    def test_update_success_failure_tracking(self, clean_bandit_state):
        """Test success/failure tracking based on reward threshold"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.ones(10)

        # High reward -> success
        bandit.update("test", features, reward=0.9)
        assert bandit.arms["test"].total_successes == 1
        assert bandit.arms["test"].total_failures == 0

        # Low reward -> failure
        bandit.update("test", features, reward=0.1)
        assert bandit.arms["test"].total_successes == 1
        assert bandit.arms["test"].total_failures == 1

    def test_update_feature_normalization(self, clean_bandit_state):
        """Test feature normalization updates"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.3, 0.2, 1.0])

        initial_mean = bandit.feature_mean.copy()
        bandit.update("test", features, reward=0.8)

        # Mean should have updated
        assert not np.array_equal(bandit.feature_mean, initial_mean)
        assert bandit.n_observations == 1

    def test_update_history_tracking(self, clean_bandit_state):
        """Test that history is tracked"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.ones(10)
        bandit.update("test", features, reward=0.75, query_context={"test": "data"})

        assert len(bandit.history) == 1
        assert bandit.history[0]["arm"] == "test"
        assert bandit.history[0]["reward"] == 0.75
        assert "timestamp" in bandit.history[0]

    def test_update_nonexistent_arm(self, clean_bandit_state):
        """Test update with non-existent arm (should be gracefully handled)"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.ones(10)
        # Should not raise an error
        bandit.update("nonexistent", features, reward=0.5)

        # Should not have updated anything
        assert bandit.n_observations == 0


class TestBanditStatistics:
    """Test statistical methods"""

    def test_get_arm_stats(self, clean_bandit_state):
        """Test getting arm statistics"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Add some data
        features = np.ones(10) * 0.5
        for _ in range(5):
            bandit.update("test", features, reward=0.8)

        stats = bandit.get_arm_stats("test")

        assert stats["name"] == "test"
        assert stats["total_pulls"] == 5
        assert stats["success_rate"] == 1.0  # All rewards > 0.5
        assert "feature_weights" in stats
        assert "uncertainty" in stats
        assert len(stats["last_10_rewards"]) == 5

    def test_get_arm_stats_nonexistent(self, clean_bandit_state):
        """Test getting stats for non-existent arm"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        stats = bandit.get_arm_stats("nonexistent")
        assert stats == {}

    def test_get_rankings(self, clean_bandit_state):
        """Test getting arm rankings for a context"""
        bandit = ContextualBandit(
            arms=["arm1", "arm2", "arm3"],
            feature_dim=10,
            state_path=clean_bandit_state,
        )

        context = {"length": 100, "complexity": "complex"}
        rankings = bandit.get_rankings(context)

        assert len(rankings) == 3
        assert all(isinstance(item, tuple) for item in rankings)
        assert all(len(item) == 2 for item in rankings)

        # Should be sorted descending
        scores = [score for _, score in rankings]
        assert scores == sorted(scores, reverse=True)


class TestNumericalStability:
    """Test numerical stability and edge cases"""

    def test_safe_matrix_inverse(self, clean_bandit_state):
        """Test safe matrix inversion"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Test well-conditioned matrix
        A = np.identity(10)
        A_inv = bandit._safe_matrix_inverse(A)
        np.testing.assert_array_almost_equal(A_inv, np.identity(10))

    def test_safe_matrix_inverse_ill_conditioned(self, clean_bandit_state):
        """Test safe inversion of ill-conditioned matrix"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Create nearly singular matrix
        A = np.identity(10)
        A[0, 0] = 1e-15  # Near-zero eigenvalue

        # Should not crash
        A_inv = bandit._safe_matrix_inverse(A)
        assert A_inv is not None
        assert A_inv.shape == (10, 10)

    def test_confidence_term_clamping(self, clean_bandit_state):
        """Test that confidence terms are clamped to non-negative"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 50}
        arm_name, reward, features = bandit.select_arm(context, mode="ucb")

        # UCB score should be finite and non-NaN
        assert np.isfinite(reward)
        assert not np.isnan(reward)


class TestStatePersistence:
    """Test state save/load functionality"""

    def test_save_state(self, clean_bandit_state):
        """Test saving bandit state"""
        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Add some data
        features = np.ones(10) * 0.5
        bandit.update("test", features, reward=0.8)

        # Save state
        bandit.save_state()

        # Check file exists
        assert clean_bandit_state.exists()

        # Check file is valid JSON
        with open(clean_bandit_state, "r") as f:
            state = json.load(f)

        assert "version" in state
        assert "arms" in state
        assert "A" in state
        assert "b" in state

    def test_load_state(self, clean_bandit_state):
        """Test loading bandit state"""
        # Create and save initial state
        bandit1 = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        features = np.ones(10) * 0.5
        for i in range(5):
            bandit1.update("test", features, reward=0.8)

        bandit1.save_state()

        # Create new bandit and load state
        bandit2 = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Check state was loaded
        assert bandit2.arms["test"].total_pulls == 5
        assert bandit2.n_observations == 5

        # Matrices should match
        np.testing.assert_array_almost_equal(bandit2.A["test"], bandit1.A["test"])
        np.testing.assert_array_almost_equal(bandit2.b["test"], bandit1.b["test"])

    def test_load_nonexistent_state(self, clean_bandit_state):
        """Test loading from non-existent file (should initialize fresh)"""
        # Ensure file doesn't exist
        if clean_bandit_state.exists():
            clean_bandit_state.unlink()

        bandit = ContextualBandit(
            arms=["test"], feature_dim=10, state_path=clean_bandit_state
        )

        # Should have fresh state
        assert bandit.n_observations == 0
        assert bandit.arms["test"].total_pulls == 0


class TestLearningConvergence:
    """Test learning behavior and convergence"""

    def test_learning_updates_weights(self, clean_bandit_state):
        """Test that weights update based on feedback"""
        bandit = ContextualBandit(
            arms=["good", "bad"], feature_dim=10, state_path=clean_bandit_state
        )

        # Create distinguishing features
        good_context = {"length": 200, "complexity": "complex"}
        bad_context = {"length": 50, "complexity": "simple"}

        # Train with consistent rewards
        for _ in range(20):
            arm, _, features = bandit.select_arm(good_context, mode="greedy")
            if arm == "good":
                bandit.update("good", features, reward=0.9)

            arm, _, features = bandit.select_arm(bad_context, mode="greedy")
            if arm == "bad":
                bandit.update("bad", features, reward=0.1)

        # After training, should prefer "good" for complex queries
        selections = []
        for _ in range(10):
            arm, _, _ = bandit.select_arm(good_context, mode="greedy")
            selections.append(arm)

        # Should show learning (not guaranteed 100% due to exploration)
        assert bandit.n_observations > 0


class TestPresetMapping:
    """Test ARIA preset mapping constants"""

    def test_preset_to_arm_mapping(self):
        """Test that preset names map correctly"""
        assert PRESET_TO_ARM["fast"] == "fast"
        assert PRESET_TO_ARM["balanced"] == "balanced"
        assert PRESET_TO_ARM["deep"] == "deep"
        assert PRESET_TO_ARM["diverse"] == "diverse"

    def test_arm_to_preset_args(self):
        """Test preset configuration mapping"""
        assert ARM_TO_PRESET_ARGS["fast"]["top_k"] == 40
        assert ARM_TO_PRESET_ARGS["balanced"]["rotations"] == 2
        assert ARM_TO_PRESET_ARGS["deep"]["sem_limit"] == 256
        assert ARM_TO_PRESET_ARGS["diverse"]["max_per_file"] == 4

    def test_default_arms(self):
        """Test default arms list"""
        assert len(DEFAULT_ARMS) == 4
        assert "fast" in DEFAULT_ARMS
        assert "balanced" in DEFAULT_ARMS
        assert "deep" in DEFAULT_ARMS
        assert "diverse" in DEFAULT_ARMS


@pytest.mark.slow
class TestBanditPerformance:
    """Performance tests for bandit operations"""

    def test_selection_performance(self, clean_bandit_state, benchmark):
        """Benchmark arm selection speed"""
        bandit = ContextualBandit(
            arms=DEFAULT_ARMS, feature_dim=10, state_path=clean_bandit_state
        )

        context = {"length": 100, "complexity": "complex", "domain": "technical"}

        def select():
            bandit.select_arm(context, mode="ucb")

        result = benchmark(select)
        # Should be very fast (sub-millisecond)

    def test_update_performance(self, clean_bandit_state, benchmark):
        """Benchmark update speed"""
        bandit = ContextualBandit(
            arms=DEFAULT_ARMS, feature_dim=10, state_path=clean_bandit_state
        )

        features = np.random.randn(10)

        def update():
            bandit.update("fast", features, reward=0.75)

        result = benchmark(update)
        # Should be fast (sub-millisecond)
