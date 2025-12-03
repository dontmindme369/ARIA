# CLAUDE.md - AI Assistant Guide for ARIA

## Project Overview

**ARIA** (Adaptive Resonant Intelligence Architecture) is a self-learning cognitive architecture for knowledge retrieval. It combines LinUCB contextual bandits, quaternion semantic exploration, and anchor-based perspective detection to optimize retrieval quality.

**Repository**: https://github.com/dontmindme369/ARIA
**License**: MIT
**Python Version**: 3.9+

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a query
python aria_main.py "Your question here"

# Run with anchor mode
python aria_main.py "What is justice?" --with-anchor

# Run with specific preset
python aria_main.py "Debug memory leak" --preset deep

# Run tests
python aria_systems_test_and_analysis/stress_tests/test_stress.py
python aria_systems_test_and_analysis/bandit_intelligence/test_bandit_intelligence.py
```

## Project Structure

```
ARIA/
├── aria_main.py              # CLI entry point
├── aria_control_center.py    # Control center interface
├── aria_config.yaml          # Main configuration file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
│
├── src/                      # Source code
│   ├── core/
│   │   └── aria_core.py      # Main ARIA orchestrator class
│   │
│   ├── retrieval/            # Search and retrieval
│   │   ├── aria_v7_hybrid_semantic.py    # Hybrid BM25 + semantic search
│   │   ├── aria_postfilter.py            # Quality/diversity filtering
│   │   ├── query_features.py             # Query feature extraction
│   │   └── local_rag_context_v7_guided_exploration.py
│   │
│   ├── intelligence/         # Learning algorithms
│   │   ├── contextual_bandit.py   # LinUCB bandit implementation
│   │   ├── bandit_context.py      # Preset selection logic
│   │   ├── quaternion.py          # 4D semantic rotations
│   │   ├── aria_exploration.py    # Golden ratio spiral exploration
│   │   └── presets.py             # Retrieval preset definitions
│   │
│   ├── perspective/          # Query context understanding
│   │   ├── detector.py       # 8-perspective classification
│   │   ├── rotator.py        # Perspective-aware rotation params
│   │   └── user_profile.py   # User pattern learning
│   │
│   ├── anchors/              # Response mode system
│   │   ├── anchor_selector.py    # 16-anchor mode selection
│   │   ├── exemplar_fit.py       # Quality scoring
│   │   └── *.md                  # Anchor templates (casual, code, etc.)
│   │
│   ├── analysis/             # Feedback and scoring
│   │   ├── conversation_scorer.py
│   │   └── pattern_miner.py
│   │
│   ├── monitoring/           # Telemetry and logging
│   │   ├── aria_telemetry.py
│   │   ├── aria_terminal.py
│   │   └── metrics_utils.py
│   │
│   └── utils/                # Utilities
│       ├── config_loader.py  # YAML config parsing
│       ├── paths.py          # Path resolution
│       └── presets.py        # Preset-to-CLI conversion
│
├── data/                     # Static data files
│   ├── domain_dictionaries/  # Vocabulary files (JSON)
│   ├── perspective_signatures/   # Perspective detection signatures
│   ├── topology_maps/        # Concept network graphs
│   └── exemplars.txt         # Training exemplars
│
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md       # System architecture
│   ├── API_REFERENCE.md      # Programmatic API
│   ├── CONTRIBUTING.md       # Contribution guide
│   ├── QUATERNIONS.md        # Math deep dive
│   └── USAGE.md              # Usage examples
│
└── aria_systems_test_and_analysis/  # Test suites
    ├── stress_tests/
    ├── bandit_intelligence/
    └── integration/
```

## Key Architecture Concepts

### 1. LinUCB Contextual Bandit (`src/intelligence/contextual_bandit.py`)

Selects optimal retrieval presets based on query features:
- **4 Presets**: `fast`, `balanced`, `deep`, `diverse`
- **10D Feature Vector**: query length, complexity, domain, etc.
- **UCB Selection**: `score = expected_reward + alpha * sqrt(uncertainty)`
- **State File**: `.aria_contextual_bandit.json`

### 2. Quaternion Semantic Exploration (`src/intelligence/quaternion.py`)

4D rotations through embedding space:
- **Golden Ratio Spiral**: phi (1.618) based uniform sphere sampling
- **Multi-rotation**: 1-3 iterations for progressive depth
- **PCA Alignment**: Follows semantic space structure

### 3. Perspective Detection (`src/perspective/detector.py`)

8 query perspectives with rotation angles:
- educational (30°), diagnostic (90°), security (45°), implementation (60°)
- research (120°), theoretical (75°), practical (50°), reference (15°)

### 4. Retrieval Pipeline

```
Query → Feature Extraction → Bandit Preset Selection
  → BM25 Lexical Search → Semantic Embedding
  → Quaternion Rotation → Hybrid Scoring (0.3 BM25 + 0.7 Semantic)
  → Postfilter (diversity + quality) → Pack Generation
  → Reward Update (bandit learning)
```

## Configuration (`aria_config.yaml`)

Key sections:
- `paths`: File locations (index_roots, output_dir, bandit_state)
- `retrieval`: Search params (top_k, semantic_model, semantic_weight)
- `postfilter`: Quality filters (diversity, quality, topic)
- `bandit`: LinUCB settings (exploration_pulls)
- `perspective`: 8-perspective detection toggle
- `anchors`: 16-anchor modes (core + technical)

## Code Conventions

### Import Style
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local imports
from core.aria_core import ARIA
from utils.config_loader import load_config
```

### Type Hints
All functions should have type hints:
```python
from typing import Dict, List, Optional, Any

def my_function(
    param1: str,
    param2: int,
    param3: Optional[Dict[str, Any]] = None
) -> List[str]:
    ...
```

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Line Length
100 characters (not 80)

## Key Classes and Functions

### ARIA Class (`src/core/aria_core.py`)
```python
aria = ARIA(
    index_roots=["/path/to/knowledge"],
    out_root="./aria_packs",
    enforce_session=False,
)

result = aria.query(
    "Your question",
    with_anchor=True,      # Enable 16-anchor mode
    preset_override=None,  # Let bandit choose
)
```

### ContextualBandit (`src/intelligence/contextual_bandit.py`)
```python
bandit = ContextualBandit(
    arms=["fast", "balanced", "deep", "diverse"],
    feature_dim=10,
    alpha=1.0,  # Exploration parameter
)

arm, reward, features = bandit.select_arm(query_context, mode="ucb")
bandit.update(arm, features, reward)
```

### QueryFeatureExtractor (`src/retrieval/query_features.py`)
```python
extractor = QueryFeatureExtractor()
features = extractor.extract("How does gradient descent work?")
# Returns: QueryFeatures dataclass with length, domain, complexity, etc.
```

## Testing

Run test suites:
```bash
# Stress tests (performance)
python aria_systems_test_and_analysis/stress_tests/test_stress.py

# Bandit intelligence tests
python aria_systems_test_and_analysis/bandit_intelligence/test_bandit_intelligence.py

# Integration tests
python aria_systems_test_and_analysis/integration/test_integration.py
```

## Common Tasks

### Adding a New Preset
1. Edit `src/intelligence/contextual_bandit.py`:
   ```python
   ARM_TO_PRESET_ARGS = {
       # ... existing
       "exhaustive": {"top_k": 128, "sem_limit": 512, "rotations": 5, "max_per_file": 3}
   }
   DEFAULT_ARMS = ["fast", "balanced", "deep", "diverse", "exhaustive"]
   ```
2. Update `src/utils/presets.py` if needed
3. Test with: `python aria_main.py "Test query" --preset exhaustive`

### Adding a New Perspective
1. Edit `src/perspective/detector.py`:
   ```python
   PERSPECTIVES = [..., "scientific"]
   BASE_ANGLES = {..., "scientific": 65.0}
   ```
2. Add signatures to `data/domain_dictionaries/perspective_signatures_v2.json`

### Adding a Postfilter
1. Add function to `src/retrieval/aria_postfilter.py`
2. Integrate in `apply_postfilter()`
3. Add CLI argument if needed

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARIA_DATA_DIR` | Data directory | `./data` |
| `ARIA_CACHE_DIR` | Cache directory | `./cache` |
| `ARIA_OUTPUT_DIR` | Output directory | `./output` |
| `RAG_INDEX_ROOT` | Knowledge base path | Config value |
| `ARIA_DEBUG_REWARD` | Enable reward debugging | Unset |

## Files to Avoid Editing

- `.aria_contextual_bandit.json` - Auto-generated bandit state
- `var/` - Runtime state and logs
- `aria_packs/` - Output packs (user data)

## Debugging Tips

1. **Enable verbose mode**: Set `terminal.verbose: true` in config
2. **Debug rewards**: `export ARIA_DEBUG_REWARD=1`
3. **Check bandit state**: `cat .aria_contextual_bandit.json | python -m json.tool`
4. **View retrieval stats**: Check `pack.stats.txt` in run directory

## Dependencies

Core:
- `numpy>=1.21.0` - Numerical operations
- `scikit-learn>=1.0.0` - ML utilities
- `PyYAML>=6.0` - Config parsing

Optional:
- `sentence-transformers>=2.2.0` - Semantic search (recommended)
- `PyPDF2>=3.0.0` - PDF support
- `python-docx>=0.8.11` - Word document support

Dev:
- `pytest>=7.0.0`, `black>=22.0.0`, `mypy>=0.990`

## Git Workflow

Branch naming:
- Features: `feature/description`
- Fixes: `fix/description`
- Claude branches: `claude/claude-md-*`

Commit messages: Be concise and descriptive.

## Performance Characteristics

- **Retrieval latency**: 500-2000ms per query
- **Bandit selection**: ~22,000 ops/sec
- **Memory usage**: ~200-500MB typical
- **Scalability**: Works well up to 100k documents

## Further Reading

- [README.md](README.md) - Project overview
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed architecture
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - API documentation
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Contribution guide
- [docs/QUATERNIONS.md](docs/QUATERNIONS.md) - Math deep dive
