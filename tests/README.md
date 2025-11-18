# ARIA Test Suite

Comprehensive test coverage for ARIA cognitive architecture.

## Overview

The test suite is organized into three main categories:

- **Unit Tests** (`tests/unit/`) - Test individual components in isolation
- **Integration Tests** (`tests/integration/`) - Test multi-component workflows
- **Performance Tests** (`tests/performance/`) - Stress tests and benchmarks

## Test Coverage

### Unit Tests

| Module | Test File | Coverage Focus |
|--------|-----------|----------------|
| Contextual Bandit | `test_contextual_bandit.py` | LinUCB algorithm, feature extraction, arm selection, state persistence |
| Quaternion Math | `test_quaternion.py` | Rotation operations, conversions, SLERP, numerical stability |
| Query Features | `test_query_features.py` | Domain detection, complexity estimation, technical density |

### Integration Tests

| Test Suite | Focus |
|------------|-------|
| Bandit Learning | Complete learning loop: query → features → selection → reward → update |
| Retrieval Pipeline | End-to-end retrieval with semantic exploration |

### Performance Tests

| Test | Target | Description |
|------|--------|-------------|
| Bandit Selection | 22,000+ ops/sec | Arm selection throughput |
| Bandit Update | 10,000+ ops/sec | Update operation throughput |
| Query Throughput | 1,500+ qps | High-volume query processing |
| Concurrent Processing | 2,000+ qps | Multi-threaded query handling |

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Performance tests
pytest tests/performance -v
```

### Run Tests by Marker

```bash
# Run only critical tests
pytest -m critical

# Run slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run performance benchmarks
pytest -m performance --benchmark-only
```

### Coverage Reports

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Run Specific Test Files

```bash
# Test contextual bandit
pytest tests/unit/test_contextual_bandit.py -v

# Test quaternions
pytest tests/unit/test_quaternion.py -v

# Test query features
pytest tests/unit/test_query_features.py -v
```

### Run Specific Test Classes

```bash
# Test only bandit initialization
pytest tests/unit/test_contextual_bandit.py::TestContextualBanditInitialization -v

# Test quaternion rotations
pytest tests/unit/test_quaternion.py::TestVectorRotation -v
```

### Run Specific Tests

```bash
# Test a specific function
pytest tests/unit/test_contextual_bandit.py::TestBanditUpdate::test_update_basic -v
```

## Performance Benchmarking

### Run Benchmarks

```bash
# Run all benchmarks
pytest tests/performance --benchmark-only

# Run and save results
pytest tests/performance --benchmark-autosave

# Compare with previous runs
pytest tests/performance --benchmark-compare
```

### Benchmark Output

The benchmark tool provides detailed statistics:

- **min/max**: Fastest and slowest execution times
- **mean**: Average execution time
- **stddev**: Standard deviation
- **median**: Median execution time
- **ops/sec**: Operations per second

## CI/CD Integration

Tests run automatically on:

- **Push** to `main`, `develop`, or `claude/*` branches
- **Pull requests** to `main` or `develop`
- **Manual trigger** via workflow dispatch

### CI Pipeline Stages

1. **Test** (matrix: Python 3.8-3.11)
   - Linting with flake8
   - Unit tests with coverage
   - Integration tests

2. **Performance** (Python 3.10)
   - Benchmark tests
   - Performance regression detection

3. **Stress** (Python 3.10, main branch only)
   - Long-running stress tests
   - Sustained load testing

4. **Coverage Report**
   - Full coverage report generation
   - Threshold enforcement (70% minimum)

## Test Configuration

### pytest.ini

Configuration for pytest:

- Test discovery patterns
- Coverage settings
- Markers definition
- Output formatting

### .coveragerc

Coverage measurement configuration:

- Source paths
- Exclusions
- Report formatting
- Coverage thresholds

## Writing Tests

### Test Structure

```python
import pytest

class TestYourFeature:
    """Test suite description"""

    def test_basic_functionality(self):
        """Test basic behavior"""
        # Arrange
        obj = YourClass()

        # Act
        result = obj.method()

        # Assert
        assert result == expected
```

### Using Fixtures

```python
def test_with_fixture(clean_bandit_state):
    """Test using a fixture"""
    bandit = ContextualBandit(state_path=clean_bandit_state)
    # Test code...
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubles(input, expected):
    assert input * 2 == expected
```

### Marking Tests

```python
@pytest.mark.slow
def test_long_running():
    """This test takes a while"""
    pass

@pytest.mark.critical
def test_critical_path():
    """Must always pass"""
    pass
```

## Test Fixtures

Available fixtures (see `tests/conftest.py`):

- `test_data_dir`: Path to test data
- `temp_dir`: Temporary directory for test outputs
- `sample_queries`: Sample query strings
- `clean_bandit_state`: Fresh bandit state file
- `sample_feature_vector`: 10D feature vector
- `minimal_config`: Minimal ARIA configuration
- `mock_embeddings`: Mock embedding function
- `performance_threshold`: Performance targets

## Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| Critical Modules | 90%+ |
| Core Logic | 80%+ |
| Utilities | 70%+ |
| Overall Project | 70%+ |

## Troubleshooting

### ImportError

If you get import errors, ensure `src/` is in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
pytest
```

Or use the pytest configuration (already set in `pytest.ini`).

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow"
```

### Failed Benchmarks

If performance tests fail:

1. Check system load
2. Run on dedicated hardware
3. Adjust thresholds for your environment

### Coverage Too Low

To see which files need more coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD)
2. Ensure coverage ≥ 80% for new code
3. Add integration tests for workflows
4. Add performance tests for critical paths
5. Update this README if adding new test categories

## Test Maintenance

- Review and update tests when APIs change
- Remove obsolete tests
- Keep test data fixtures current
- Monitor performance test thresholds
- Update CI configuration as needed

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [hypothesis (property testing)](https://hypothesis.readthedocs.io/)
