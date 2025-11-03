# Contributing to ARIA

Thank you for considering contributing to ARIA! This document provides guidelines for contributing to the project.

## 🌀 Philosophy

ARIA is built on principles of:

- **Measured progress** over speculative features
- **Local-first** privacy-preserving design
- **Self-optimization** through closed-loop learning
- **Geometric awareness** - semantic navigation through quaternion space
- **Reasoning awareness** - different queries need different approaches

Contributions should align with these principles and maintain the geometric foundations of the exploration system.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Understanding of retrieval systems
- Familiarity with:
  - Quaternions and S³ geometry (for exploration system)
  - Reinforcement learning (for bandit contributions)
  - NumPy and scikit-learn (for mathematical operations)
- Git

### Development Setup

```bash
# Clone repository
git clone https://github.com/dontmindme369/aria.git
cd aria

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Run tests
python -m pytest tests/
```

---

## 📝 How to Contribute

### 1. Reporting Issues

**Before creating an issue**, please:

- Check existing issues to avoid duplicates
- Verify the issue with latest version
- Gather relevant information (Python version, OS, error messages)

**Good issue includes**:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information
- Relevant logs or error messages
- For exploration system issues: quaternion states, telemetry output

### 2. Feature Requests

We welcome feature requests that align with ARIA's philosophy.

**Good feature request includes**:

- Clear use case and motivation
- How it fits with ARIA's geometric/reasoning principles
- Potential implementation approach
- Measurable improvement metrics
- Willingness to contribute

**Exploration System Features**:
- Novel geometric exploration strategies
- Quaternion space optimizations
- PCA alternatives (ICA, NMF, etc.)
- Golden ratio variants (other irrational constants)

### 3. Code Contributions

#### Branch Strategy

- `main` - Stable releases only
- `develop` - Active development
- `feature/your-feature` - New features
- `fix/issue-123` - Bug fixes
- `exploration/new-geometry` - Geometric exploration experiments

#### Workflow

1. **Fork the repository**
2. **Create feature branch** from `develop`:

   ```bash
   git checkout -b feature/your-feature develop
   ```

3. **Make changes**:
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation
   - Maintain geometric correctness (for exploration system)

4. **Test thoroughly**:

   ```bash
   # Run full test suite
   python -m pytest tests/ -v
   
   # Run specific test
   python -m pytest tests/test_exploration.py -v
   
   # Check code style
   black --check src/
   ```

5. **Commit with clear messages**:

   ```bash
   git commit -m "feat: add adaptive exploration radius
   
   - Dynamically adjusts radius based on query ambiguity
   - Uses quaternion momentum as signal
   - Measured 8% coverage improvement
   
   Closes #123"
   ```

6. **Push and create Pull Request**:

   ```bash
   git push origin feature/your-feature
   ```

---

## 🎯 Contribution Areas

### High Priority

1. **Testing**
   - Expand test coverage
   - Add integration tests
   - Performance benchmarks
   - Quaternion operation validation

2. **Documentation**
   - Improve examples
   - Add tutorials
   - Clarify geometric concepts
   - Mathematics documentation

3. **Bug Fixes**
   - Check GitHub issues
   - Fix edge cases
   - Improve error handling
   - SLERP numerical stability

### Medium Priority

4. **Performance**
   - Optimize retrieval speed
   - Reduce memory usage
   - Improve caching
   - GPU acceleration for quaternion ops

5. **Features** (align with roadmap)
   - New anchor modes
   - Enhanced telemetry
   - Additional document formats
   - Alternative exploration geometries

### Research Contributions

6. **Geometric Exploration**
   - Octonion extensions (8D space)
   - Non-Euclidean geometries
   - Topological features (persistent homology)
   - Quantum-inspired superposition

7. **PCA Alternatives**
   - ICA (Independent Component Analysis)
   - NMF (Non-negative Matrix Factorization)
   - Kernel PCA
   - Sparse PCA

8. **Golden Ratio Extensions**
   - Other irrational constants (√2, e, π)
   - Multi-scale spirals
   - Adaptive angle spacing
   - Fibonacci generalizations

**Note**: Experimental features should:

- Be toggleable (feature flags)
- Include benchmark comparisons
- Demonstrate measurable improvements
- Maintain geometric correctness

---

## 📐 Code Style

### Python Style

Follow PEP 8 with these specifics:

```python
# Good: Clear naming, type hints, docstrings
def compute_slerp(
    q1: np.ndarray,
    q2: np.ndarray,
    t: float = 0.5
) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.
    
    Args:
        q1: Start quaternion (4D unit vector)
        q2: End quaternion (4D unit vector)
        t: Interpolation factor [0, 1]
    
    Returns:
        Interpolated quaternion on S³
        
    Mathematical Properties:
        - Maintains constant angular velocity
        - Stays on unit sphere
        - Takes geodesic path
    """
    # Implementation
    pass

# Bad: Unclear naming, no types, no docs
def slerp(q1, q2, t=0.5):
    pass
```

### Quaternion Operations Style

```python
# Good: Clear geometric intent
def rotate_embedding_on_s3(
    embedding: np.ndarray,
    quaternion_state: np.ndarray
) -> np.ndarray:
    """
    Rotate embedding vector using quaternion on S³.
    
    Geometric interpretation: This performs a rotation in
    4D space, treating the embedding as a point and the
    quaternion as a rotation operator.
    """
    # Ensure quaternion is normalized
    q = normalize_quaternion(quaternion_state)
    
    # Convert embedding to quaternion representation
    emb_q = embedding_to_quaternion(embedding)
    
    # Apply quaternion rotation: q * emb_q * q_conjugate
    rotated_q = quaternion_multiply(q, emb_q, quaternion_conjugate(q))
    
    # Project back to embedding space
    return quaternion_to_embedding(rotated_q)

# Bad: Unclear geometric meaning
def rot(e, q):
    return q @ e @ q.T
```

### Documentation Style

```python
# Module-level docstring
"""
quaternion_state.py

Semantic state management on the unit 3-sphere (S³).

Mathematical Foundation:
    Quaternions form a 4D normed division algebra, with
    unit quaternions representing rotations in 3D/4D space.
    The space of unit quaternions is topologically S³,
    a 3-manifold embedded in 4D.

Key Operations:
    - SLERP: Smooth interpolation on S³
    - Momentum: Tangent space derivatives
    - Memory: Associative state recall
    - Distance: Geodesic metrics
"""

# Class docstring
class QuaternionStateManager:
    """
    Manages semantic state as quaternions on S³.
    
    State Evolution:
        S³ provides natural space for tracking semantic
        position over time. Unlike Euclidean space, S³
        has no gimbal lock and enables smooth interpolation.
    
    Attributes:
        current_state: Current position on S³
        momentum: Velocity in tangent space
        history: Past states for memory recall
    
    Mathematical Note:
        All quaternions maintained as unit vectors:
        ||q|| = √(w² + x² + y² + z²) = 1
    """
    pass

# Function docstring (Google style)
def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation on S³.
    
    Args:
        q1: Start quaternion [w, x, y, z]
        q2: End quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion maintaining unit norm
    
    Mathematical Properties:
        Given unit quaternions q1, q2, SLERP computes:
        slerp(q1, q2, t) = (sin((1-t)θ)/sin(θ))q1 + (sin(tθ)/sin(θ))q2
        where θ = arccos(q1·q2)
    
    Geometric Interpretation:
        Travels along great circle (geodesic) on S³ from
        q1 to q2 with constant angular velocity.
    
    Example:
        >>> q1 = np.array([1, 0, 0, 0])  # Identity
        >>> q2 = np.array([0, 1, 0, 0])  # 180° rotation
        >>> slerp(q1, q2, 0.5)  # Midpoint
        array([0.707, 0.707, 0, 0])
    """
    pass
```

### Testing Style

```python
# tests/test_quaternion_state.py
import pytest
import numpy as np
from quaternion_state import QuaternionStateManager, slerp, normalize_quaternion

class TestQuaternionOperations:
    """Test suite for quaternion mathematics."""
    
    def test_normalize_quaternion_maintains_unit_norm(self):
        """Normalization should produce unit vectors."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = normalize_quaternion(q)
        
        norm = np.linalg.norm(q_norm)
        assert np.isclose(norm, 1.0), "Normalized quaternion must be unit vector"
    
    def test_slerp_identity_endpoints(self):
        """SLERP with t=0 or t=1 should return endpoints."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        
        assert np.allclose(slerp(q1, q2, 0.0), q1), "SLERP(t=0) should return q1"
        assert np.allclose(slerp(q1, q2, 1.0), q2), "SLERP(t=1) should return q2"
    
    def test_slerp_maintains_unit_norm(self):
        """SLERP output should always be unit quaternion."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0.707, 0.707, 0, 0])
        
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            result = slerp(q1, q2, t)
            norm = np.linalg.norm(result)
            assert np.isclose(norm, 1.0), f"SLERP(t={t}) must preserve unit norm"
    
    def test_geodesic_distance_symmetry(self):
        """Distance from q1 to q2 should equal distance from q2 to q1."""
        manager = QuaternionStateManager()
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        
        d12 = manager.geodesic_distance(q1, q2)
        d21 = manager.geodesic_distance(q2, q1)
        
        assert np.isclose(d12, d21), "Geodesic distance must be symmetric"
```

---

## 🧪 Testing Guidelines

### Test Requirements

All contributions must include tests:

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **Geometric validation**: Verify mathematical properties
- **Edge cases**: Test boundary conditions
- **Numerical stability**: Test with extreme values

### Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src/ --cov-report=html

# Specific test
python -m pytest tests/test_quaternion_state.py::TestQuaternionOperations::test_slerp_maintains_unit_norm -v

# Fast tests only (skip slow)
python -m pytest tests/ -m "not slow"
```

### Writing Good Tests

```python
# Good: Clear, focused, independent, validates geometry
def test_quaternion_multiplication_associativity():
    """Quaternion multiplication should be associative: (q1*q2)*q3 = q1*(q2*q3)."""
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0.707, 0.707, 0, 0])
    q3 = np.array([0, 0, 0.707, 0.707])
    
    lhs = quaternion_multiply(quaternion_multiply(q1, q2), q3)
    rhs = quaternion_multiply(q1, quaternion_multiply(q2, q3))
    
    assert np.allclose(lhs, rhs), "Quaternion multiplication must be associative"

# Bad: Unclear, tests multiple things, doesn't validate geometry
def test_stuff():
    manager = QuaternionStateManager()
    result = manager.update_state(some_query)
    assert result is not None
```

---

## 📚 Documentation Guidelines

### Code Documentation

- **Docstrings**: All public functions, classes, methods
- **Geometric explanations**: Clarify mathematical operations
- **Type hints**: Use throughout for clarity
- **Examples**: Provide usage examples in docstrings

### Repository Documentation

When adding features, update:

- README.md (if user-facing)
- CHANGELOG.md (all changes)
- ARCHITECTURE.md (if structural, especially exploration system)
- METRICS.md (if new telemetry)
- Examples (if new capability)

---

## 🔍 Code Review Process

### What Reviewers Look For

1. **Correctness**: Does it work as intended?
2. **Geometric validity**: For exploration system, is math correct?
3. **Tests**: Adequate test coverage?
4. **Documentation**: Clear docs and comments?
5. **Style**: Follows project conventions?
6. **Performance**: No unnecessary slowdowns?
7. **Compatibility**: Works with existing code?

### Exploration System Specific

For geometric/exploration contributions:

- **Mathematical rigor**: Equations match implementation
- **Numerical stability**: Handles edge cases (near-zero, near-one)
- **Unit tests**: Validate geometric properties
- **Performance**: Acceptable latency overhead
- **Telemetry**: Tracks relevant metrics

### Review Timeline

- Small fixes: 1-2 days
- Features: 3-7 days
- Major changes (geometric): 1-2 weeks
- Research contributions: 2-4 weeks

Be patient and responsive to feedback.

---

## 🎁 Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Credited in release notes
- Thanked in community channels

Significant contributions may result in:

- Maintainer status
- Decision-making input
- Co-authorship on papers (if applicable)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the CC BY-NC 4.0 license.

---

## ❓ Questions?

- **General**: Open a GitHub Discussion
- **Bugs**: Create an Issue
- **Security**: Email security@aria-project.org
- **Other**: Contact maintainers

---

## 🙏 Thank You

Every contribution, no matter how small, helps make ARIA better. Thank you for investing your time and expertise!

**Together, we're building intelligence that resonates** ✨

---

## 🌀 Special Note on Geometric Contributions

ARIA's exploration system is built on rigorous mathematics:

- **Quaternions**: S³ topology, SLERP, geodesics
- **Golden Ratio**: Fibonacci sequences, optimal packing
- **PCA**: Subspace analysis, variance decomposition

If contributing to the exploration system, please:

1. **Verify mathematical correctness** - equations match implementation
2. **Test geometric properties** - unit norms, symmetry, associativity
3. **Measure improvements** - demonstrate coverage gains via telemetry
4. **Document thoroughly** - explain the math, not just the code

We welcome novel geometric approaches, but they must be:
- Mathematically sound
- Numerically stable  
- Measurably beneficial
- Well-documented

*"Go within."* - Build intelligence through geometric elegance.
