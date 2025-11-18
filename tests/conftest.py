"""
Pytest configuration and shared fixtures for ARIA test suite
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any


# ============================================================================
# Session-level fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Root directory for test data"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs"""
    tmpdir = tempfile.mkdtemp(prefix="aria_test_")
    yield Path(tmpdir)
    # Cleanup after all tests
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Sample data fixtures
# ============================================================================

@pytest.fixture
def sample_queries() -> List[str]:
    """Sample queries covering different domains and complexities"""
    return [
        "What is a binary search tree?",
        "How do I debug a memory leak in Python?",
        "Explain quantum entanglement",
        "def foo():",
        "What is the meaning of justice?",
        "Compare TensorFlow and PyTorch architectures",
        "When was the Declaration of Independence signed?",
        "Analyze the relationship between entropy and information theory",
        "",  # Edge case: empty query
        "a" * 1000,  # Edge case: very long query
    ]


@pytest.fixture
def sample_query_features() -> List[Dict[str, Any]]:
    """Known query-feature mappings for regression testing"""
    return [
        {
            "query": "What is a binary search tree?",
            "expected_domain": "code",
            "expected_complexity": "simple",
            "expected_has_question": True,
        },
        {
            "query": "Compare gradient descent optimization algorithms",
            "expected_domain": "concept",
            "expected_complexity": "complex",
            "expected_has_question": False,
        },
        {
            "query": "Debug ValueError in line 42",
            "expected_domain": "code",
            "expected_complexity": "medium",
            "expected_has_question": False,
        },
    ]


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for retrieval testing"""
    return [
        "Binary search trees are hierarchical data structures where each node has at most two children.",
        "Python memory leaks often occur due to circular references or unclosed resources.",
        "Quantum entanglement is a phenomenon where quantum states of particles become correlated.",
        "Justice is the concept of moral rightness based on ethics, law, and fairness.",
        "TensorFlow uses static computational graphs while PyTorch uses dynamic computation.",
    ]


# ============================================================================
# Bandit fixtures
# ============================================================================

@pytest.fixture
def clean_bandit_state(temp_dir):
    """Provide a clean bandit state file path"""
    state_file = temp_dir / "test_bandit_state.json"
    yield state_file
    # Cleanup
    if state_file.exists():
        state_file.unlink()


@pytest.fixture
def sample_feature_vector() -> np.ndarray:
    """Sample 10D feature vector for bandit testing"""
    return np.array([
        0.5,   # query_length (normalized)
        1.0,   # complexity (complex)
        1.0,   # domain_technical
        0.0,   # domain_creative
        0.0,   # domain_analytical
        0.0,   # domain_philosophical
        1.0,   # has_question
        0.3,   # entity_count
        0.5,   # time_of_day
        1.0,   # bias_term
    ])


@pytest.fixture
def multiple_feature_vectors() -> List[np.ndarray]:
    """Multiple feature vectors for batch testing"""
    return [
        np.array([0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.1, 0.3, 1.0]),  # Simple technical
        np.array([0.8, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.7, 1.0]),  # Complex creative
        np.array([0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.3, 0.5, 1.0]),  # Medium analytical
        np.array([0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.2, 1.0]),  # Simple philosophical
    ]


# ============================================================================
# Quaternion fixtures
# ============================================================================

@pytest.fixture
def identity_quaternion():
    """Identity quaternion (no rotation)"""
    from intelligence.quaternion import Quaternion
    return Quaternion(1.0, 0.0, 0.0, 0.0)


@pytest.fixture
def sample_quaternions():
    """Sample quaternions for testing"""
    from intelligence.quaternion import Quaternion
    return {
        "identity": Quaternion(1.0, 0.0, 0.0, 0.0),
        "90_deg_z": Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2),
        "180_deg_x": Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi),
        "45_deg_y": Quaternion.from_axis_angle(np.array([0, 1, 0]), np.pi/4),
    }


@pytest.fixture
def sample_vectors() -> Dict[str, np.ndarray]:
    """Sample 3D vectors for rotation testing"""
    return {
        "x_axis": np.array([1.0, 0.0, 0.0]),
        "y_axis": np.array([0.0, 1.0, 0.0]),
        "z_axis": np.array([0.0, 0.0, 1.0]),
        "diagonal": np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        "arbitrary": np.array([0.6, 0.8, 0.0]),
    }


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def minimal_config(temp_dir) -> Dict[str, Any]:
    """Minimal ARIA configuration for testing"""
    return {
        "knowledge_base": str(temp_dir / "test_kb"),
        "embeddings": str(temp_dir / "test_embeddings"),
        "output_dir": str(temp_dir / "test_output"),
        "bandit": {
            "epsilon": 0.10,
            "alpha": 1.0,
            "feature_dim": 10,
        },
        "presets": {
            "fast": {"top_k": 40, "sem_limit": 64, "rotations": 1},
            "balanced": {"top_k": 64, "sem_limit": 128, "rotations": 2},
            "deep": {"top_k": 96, "sem_limit": 256, "rotations": 3},
            "diverse": {"top_k": 80, "sem_limit": 128, "rotations": 2},
        },
    }


@pytest.fixture
def config_file(temp_dir, minimal_config):
    """Create a temporary config file"""
    config_path = temp_dir / "test_config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(minimal_config, f)
    return config_path


# ============================================================================
# Mock data fixtures
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing without loading actual models"""
    def _get_embedding(text: str) -> np.ndarray:
        # Deterministic pseudo-embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).astype(np.float32)
    return _get_embedding


@pytest.fixture
def sample_perspective_signatures(test_data_dir):
    """Sample perspective signatures for testing"""
    signatures = {
        "educational": {
            "keywords": ["explain", "understand", "learn", "tutorial"],
            "weight": 1.0
        },
        "technical": {
            "keywords": ["implement", "debug", "optimize", "error"],
            "weight": 1.0
        },
        "philosophical": {
            "keywords": ["meaning", "purpose", "essence", "truth"],
            "weight": 1.0
        },
    }

    sig_file = test_data_dir / "test_perspective_signatures.json"
    sig_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sig_file, "w") as f:
        json.dump(signatures, f, indent=2)

    return sig_file


# ============================================================================
# Performance testing fixtures
# ============================================================================

@pytest.fixture
def benchmark_queries() -> List[str]:
    """Queries for performance benchmarking"""
    base_queries = [
        "How do I implement a hash table?",
        "Explain neural network backpropagation",
        "What is the difference between mutex and semaphore?",
        "Debug segmentation fault in C++",
    ]
    # Replicate for bulk testing
    return base_queries * 250  # 1000 queries total


@pytest.fixture
def performance_threshold():
    """Performance thresholds from README claims"""
    return {
        "bandit_selection_qps": 22000,  # 22,658 ops/sec claimed
        "bandit_update_qps": 10000,     # 10,347 ops/sec claimed
        "query_throughput_qps": 1500,   # 1,527 qps claimed
        "concurrent_qps": 2000,          # 2,148 qps claimed
    }


# ============================================================================
# Utility functions for tests
# ============================================================================

def assert_normalized_vector(vec: np.ndarray, tolerance: float = 1e-6):
    """Assert that a vector is normalized (unit length)"""
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < tolerance, f"Vector not normalized: ||v|| = {norm}"


def assert_rotation_preserves_length(original: np.ndarray, rotated: np.ndarray, tolerance: float = 1e-6):
    """Assert that rotation preserves vector length"""
    original_norm = np.linalg.norm(original)
    rotated_norm = np.linalg.norm(rotated)
    assert abs(original_norm - rotated_norm) < tolerance, \
        f"Rotation changed vector length: {original_norm} -> {rotated_norm}"


def assert_valid_probability_distribution(probs: np.ndarray, tolerance: float = 1e-6):
    """Assert that array is a valid probability distribution"""
    assert np.all(probs >= 0), "Probabilities must be non-negative"
    assert abs(np.sum(probs) - 1.0) < tolerance, f"Probabilities must sum to 1: sum={np.sum(probs)}"


# Export utility functions
pytest.assert_normalized_vector = assert_normalized_vector
pytest.assert_rotation_preserves_length = assert_rotation_preserves_length
pytest.assert_valid_probability_distribution = assert_valid_probability_distribution
