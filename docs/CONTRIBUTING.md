# Contributing to ARIA

Thank you for your interest in contributing to ARIA! This guide will help you get started.

---

## Ways to Contribute

### 1. Report Bugs
Found a bug? Help us fix it:
- Check [existing issues](https://github.com/dontmindme369/ARIA/issues) first
- Create detailed bug report with reproduction steps
- Include system info, error messages, and logs

### 2. Suggest Features
Have an idea for improvement?
- Start a [discussion](https://github.com/dontmindme369/ARIA/discussions)
- Explain use case and benefits
- Consider implementation complexity

### 3. Improve Documentation
Documentation is always welcome:
- Fix typos and errors
- Add missing examples
- Clarify confusing sections
- Write tutorials

### 4. Write Code
Contribute new features or fixes:
- Follow code style guidelines
- Add tests for new features
- Update documentation
- Submit pull request

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork repository on GitHub, then:
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Main dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest black isort mypy pylance
```

### 4. Run Tests

```bash
# Comprehensive test suite
python3 tests/comprehensive_test_suite.py

# Should see: 14/14 tests passing
```

### 5. Create Branch

```bash
git checkout -b feature/my-new-feature
# OR
git checkout -b fix/bug-description
```

---

## Code Style

### Python Style Guide

ARIA follows **PEP 8** with some modifications:

**Line Length**: 100 characters (not 80)

**Imports**:
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local
from core.aria_core import ARIA
from utils.config_loader import load_config
```

**Type Hints**:
```python
from typing import Dict, List, Optional, Tuple, Any

def my_function(
    param1: str,
    param2: int,
    param3: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Clear docstring with type information.

    Args:
        param1: Description
        param2: Description
        param3: Optional description

    Returns:
        Description of return value
    """
    pass
```

**Naming Conventions**:
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Formatting Tools

```bash
# Auto-format code
black aria/ --line-length 100

# Sort imports
isort aria/

# Type checking
mypy aria/ --ignore-missing-imports
```

---

## File Structure

### Where to Add Code

```
aria/
├── src/
│   ├── core/              # Core orchestration
│   │   └── aria_core.py
│   ├── retrieval/         # Search & retrieval
│   │   ├── aria_v7_hybrid_semantic.py
│   │   ├── aria_postfilter.py
│   │   └── query_features.py
│   ├── intelligence/      # Bandit, quaternions
│   │   ├── bandit_context.py
│   │   ├── quaternion.py
│   │   └── aria_exploration.py
│   ├── perspective/       # Perspective detection
│   │   ├── detector.py
│   │   └── rotator.py
│   ├── anchors/          # Anchor reasoning
│   │   └── exemplar_fit.py
│   ├── monitoring/        # Telemetry & logs
│   │   ├── aria_telemetry.py
│   │   └── aria_terminal.py
│   └── utils/            # Utilities
│       ├── config_loader.py
│       ├── paths.py
│       └── presets.py
├── tests/                # Test suite
│   └── comprehensive_test_suite.py
├── data/                 # Domain dictionaries
│   └── domain_dictionaries/
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── ...
└── aria_control_center.py  # Main interface
```

### Adding New Features

**New Retrieval Strategy**:
1. Add to `src/retrieval/`
2. Update `aria_v7_hybrid_semantic.py` or create new file
3. Add tests
4. Update documentation

**New Perspective**:
1. Edit `src/perspective/detector.py`
2. Add to `PERSPECTIVES` list
3. Add base angle to `BASE_ANGLES`
4. Update `perspective_signatures_v2.json`
5. Add tests

**New Preset**:
1. Edit `src/intelligence/bandit_context.py`
2. Add to `DEFAULT_PRESETS`
3. Test with various queries
4. Document use case

**New Postfilter**:
1. Add function to `src/retrieval/aria_postfilter.py`
2. Integrate in `apply_postfilter()`
3. Add command-line argument
4. Update presets if needed

---

## Testing

### Running Tests

```bash
# Full test suite
python3 tests/comprehensive_test_suite.py

# Specific test (modify test suite)
# Comment out other tests, run individual ones
```

### Writing Tests

Add tests to `tests/comprehensive_test_suite.py`:

```python
def test_my_feature():
    """Test description"""
    # Setup
    input_data = setup_test_data()

    # Execute
    result = my_function(input_data)

    # Assert
    assert result is not None
    assert len(result) > 0
    assert result['key'] == expected_value

    return {
        'status': 'pass',
        'description': 'My feature works correctly',
        'metric1': value1,
        'metric2': value2
    }

# Add to TESTS list
TESTS = [
    # ... existing tests
    ("My Feature Test", test_my_feature),
]
```

### Test Guidelines

- **One test per function**: Test one thing at a time
- **Clear assertions**: Use descriptive assert messages
- **Independent tests**: Tests should not depend on each other
- **Clean up**: Reset state after tests
- **Fast tests**: Tests should run quickly (< 5s each)

---

## Pull Request Process

### 1. Before Submitting

**Checklist**:
- [ ] Code follows style guidelines
- [ ] Tests pass (14/14)
- [ ] New code has tests
- [ ] Documentation updated
- [ ] No hardcoded paths
- [ ] Type hints added
- [ ] Commit messages clear

### 2. Create Pull Request

1. Push branch to your fork
2. Open PR on GitHub
3. Fill out PR template:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this needed?
   - **Testing**: How was it tested?
   - **Related Issues**: Link to issues

### 3. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manually tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 4. Review Process

- Maintainers will review within 1-2 weeks
- Address feedback and push updates
- Once approved, PR will be merged

---

## Feature Development Examples

### Example 1: Add New Perspective

```python
# 1. Edit src/perspective/detector.py

PERSPECTIVES = [
    # ... existing
    "scientific"  # New perspective
]

BASE_ANGLES = {
    # ... existing
    "scientific": 65.0  # New base angle
}

# 2. Add signatures to data/domain_dictionaries/perspective_signatures_v2.json

{
  "scientific": {
    "signal_words": ["hypothesis", "experiment", "data", "analysis"],
    "question_patterns": ["does X cause Y", "correlation between"],
    "action_words": ["test", "measure", "observe", "validate"]
  }
}

# 3. Test
python3 -c "
from perspective.detector import PerspectiveOrientationDetector
detector = PerspectiveOrientationDetector('data/domain_dictionaries/perspective_signatures_v2.json')
result = detector.detect_perspective('Does temperature affect reaction rate?')
print(result)
"

# 4. Add test case
def test_scientific_perspective():
    detector = PerspectiveOrientationDetector(...)
    result = detector.detect_perspective("What is the correlation between X and Y?")
    assert result['perspective'] == 'scientific'
    return {'status': 'pass', 'description': 'Scientific perspective detected'}
```

### Example 2: Add New Preset

```python
# 1. Edit src/intelligence/bandit_context.py

DEFAULT_PRESETS = [
    # ... existing presets
    {
        "name": "exhaustive",
        "args": {
            "top_k": 128,
            "sem_limit": 512,
            "rotations": 5,
            "max_per_file": 3
        }
    }
]

# 2. Test manually
python3 aria_main.py "Test query" --preset exhaustive

# 3. Document in docs/USAGE.md

| Preset | Chunks | Rotations | Per-File | Best For |
|--------|--------|-----------|----------|----------|
| exhaustive | 128 | 5 | 3 | Maximum coverage research |

# 4. Let bandit learn when to use it
# Run 20+ queries, check bandit_state.json
```

### Example 3: Improve Postfilter

```python
# 1. Add new filter to src/retrieval/aria_postfilter.py

def semantic_clustering_filter(chunks, max_clusters=10):
    """
    Group chunks by semantic similarity, keep diverse representatives.
    """
    embeddings = get_embeddings([c['content'] for c in chunks])
    clusters = kmeans_cluster(embeddings, n_clusters=max_clusters)

    # Keep highest scoring chunk from each cluster
    filtered = []
    for cluster_id in range(max_clusters):
        cluster_chunks = [c for i, c in enumerate(chunks) if clusters[i] == cluster_id]
        if cluster_chunks:
            best = max(cluster_chunks, key=lambda x: x['score'])
            filtered.append(best)

    return filtered

# 2. Integrate in apply_postfilter()

def apply_postfilter(chunks, ..., semantic_clustering=False, ...):
    # ... existing filters

    if semantic_clustering:
        chunks = semantic_clustering_filter(chunks)

    return chunks

# 3. Add CLI argument in aria_postfilter.py main()

parser.add_argument('--semantic-clustering', action='store_true')

# 4. Update preset_to_postfilter_args() in utils/presets.py

if args.get('semantic_clustering'):
    flags.append('--semantic-clustering')

# 5. Test
python3 -c "
from retrieval.aria_postfilter import semantic_clustering_filter
chunks = [...]  # Test data
filtered = semantic_clustering_filter(chunks)
assert len(filtered) <= 10
"
```

---

## Code Review Criteria

### What We Look For

**Code Quality**:
- Clear, readable code
- Proper error handling
- No code duplication
- Efficient algorithms

**Type Safety**:
- Type hints on all functions
- No type: ignore comments (unless necessary)
- Handles None cases

**Testing**:
- Tests for new features
- Edge cases covered
- Tests actually run (not commented out)

**Documentation**:
- Clear docstrings
- Updated README/docs
- Code comments for complex logic

**Portability**:
- No hardcoded paths
- Cross-platform compatible
- Handles missing dependencies gracefully

---

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'core'`

**Solution**:
```python
# Add at top of file
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

### Type Errors

**Problem**: Pylance complains about types

**Solution**:
```python
# Use Optional for None values
from typing import Optional

def func(param: Optional[str] = None) -> Optional[int]:
    if param is None:
        return None
    return int(param)

# Cast when needed
from typing import cast
result = cast(str, some_value)
```

### Path Issues

**Problem**: Hardcoded paths

**Solution**:
```python
# Use relative paths
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
data_path = PROJECT_ROOT / "data" / "file.json"

# Or config-based
from utils.config_loader import load_config
config = load_config()
path = config['paths']['index_roots'][0]
```

---

## Getting Help

### Questions During Development

- **Unclear how something works?**
  → Read [ARCHITECTURE.md](ARCHITECTURE.md) or [API_REFERENCE.md](API_REFERENCE.md)

- **Not sure where to add code?**
  → See [File Structure](#file-structure) above

- **Need design guidance?**
  → Start a [discussion](https://github.com/dontmindme369/ARIA/discussions)

- **Stuck on implementation?**
  → Ask in discussion, maintainers will help

---

## Recognition

Contributors will be recognized in:
- README.md contributors section
- CHANGELOG.md for releases
- GitHub contributors page

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to ARIA!** 🌀
