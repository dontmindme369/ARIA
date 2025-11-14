# ARIA API Reference

Complete API documentation for programmatic usage of ARIA.

---

## Core API

### ARIA Class

**Location**: `src/core/aria_core.py`

```python
from core.aria_core import ARIA

aria = ARIA(
    index_roots: List[str],
    out_root: str,
    state_path: str = "~/.aria/bandit_state.json",
    enforce_session: bool = False
)
```

#### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `index_roots` | `List[str]` | Yes | - | Paths to knowledge base directories |
| `out_root` | `str` | Yes | - | Output directory for packs |
| `state_path` | `str` | No | `~/.aria/bandit_state.json` | Bandit state file path |
| `enforce_session` | `bool` | No | `False` | Require session management |

#### Methods

##### `query()`

Execute a query and retrieve relevant chunks.

```python
result = aria.query(
    query_text: str,
    with_anchor: bool = False,
    preset_override: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters**:
- `query_text` (str): The query string
- `with_anchor` (bool): Enable 16-anchor reasoning (optional)
- `preset_override` (str): Manual preset selection: "fast", "balanced", "deep", "diverse" (optional)

**Returns**:
```python
{
    "query": str,              # Original query
    "preset": str,             # Selected preset name
    "run_dir": str,            # Output directory path
    "pack": str,               # Pack JSON file path
    "chunks_retrieved": int,   # Number of chunks
    "files_used": int,         # Number of source files
    "perspective": str,        # Detected perspective
    "rotation_angle": float,   # Rotation angle used
    "bandit_reason": str,      # Why this preset was chosen
    "metadata": Dict[str, Any] # Additional metadata
}
```

**Example**:
```python
from core.aria_core import ARIA

aria = ARIA(
    index_roots=["./datasets", "./docs"],
    out_root="./aria_packs"
)

result = aria.query("How does gradient descent work?")

print(f"Retrieved {result['chunks_retrieved']} chunks")
print(f"Pack saved to: {result['pack']}")
```

---

## Retrieval API

### Hybrid Semantic Search

**Location**: `src/retrieval/aria_v7_hybrid_semantic.py`

```python
from retrieval.aria_v7_hybrid_semantic import retrieve_hybrid_semantic

results = retrieve_hybrid_semantic(
    query: str,
    index_root: str,
    top_k: int = 64,
    semantic_limit: int = 256,
    semantic_weight: float = 0.7,
    rotations: int = 2,
    rotation_angle: float = 45.0,
    max_per_file: int = 6
) -> List[Dict[str, Any]]
```

**Parameters**:
- `query`: Search query
- `index_root`: Path to knowledge base
- `top_k`: Number of results to return
- `semantic_limit`: Top N docs to score semantically (performance optimization)
- `semantic_weight`: Weight of semantic vs BM25 (0.0 = all BM25, 1.0 = all semantic)
- `rotations`: Number of quaternion rotation iterations
- `rotation_angle`: Rotation angle in degrees
- `max_per_file`: Maximum chunks from single file (diversity)

**Returns**:
```python
[
    {
        "path": str,           # Source file path
        "content": str,        # Chunk content
        "score": float,        # Hybrid relevance score
        "start_line": int,     # Starting line number
        "end_line": int,       # Ending line number
        "source_type": str     # File type
    },
    # ...
]
```

### Postfilter

**Location**: `src/retrieval/aria_postfilter.py`

```python
from retrieval.aria_postfilter import apply_postfilter

filtered = apply_postfilter(
    chunks: List[Dict[str, Any]],
    max_per_source: int = 15,
    min_keep: int = 20,
    quality_filter: bool = False,
    diversity_filter: bool = True,
    topic_filter: bool = False,
    min_alpha_ratio: float = 0.2,
    min_score: float = 0.001
) -> List[Dict[str, Any]]
```

**Parameters**:
- `chunks`: Retrieved chunks from semantic search
- `max_per_source`: Maximum chunks per source file
- `min_keep`: Minimum chunks to keep (safety threshold)
- `quality_filter`: Enable quality filtering
- `diversity_filter`: Enable diversity filtering
- `topic_filter`: Enable topic coherence filtering
- `min_alpha_ratio`: Minimum alphabetic character ratio
- `min_score`: Minimum relevance score

---

## Intelligence API

### Bandit (Thompson Sampling)

**Location**: `src/intelligence/bandit_context.py`

```python
from intelligence.bandit_context import BanditState

# Initialize
state = BanditState(state_path="~/.aria/bandit_state.json")

# Get available presets
presets = state.presets
# [Preset(name='fast', args={...}), ...]

# Select preset
preset, reason, meta = state.select_preset(
    features: Dict[str, Any]
)

# Update with reward
state.update(
    preset_name: str,
    reward: float,
    features: Dict[str, Any]
)

# Save state
state.save()
```

#### Preset Structure

```python
class Preset:
    name: str                # "fast", "balanced", "deep", "diverse"
    args: Dict[str, Any]     # Retrieval parameters
    alpha: float             # Successes (Beta distribution)
    beta: float              # Failures (Beta distribution)
    pulls: int               # Times selected
    wins: int                # Successful queries
    avg_reward: float        # Average reward
```

### Quaternion Math

**Location**: `src/intelligence/quaternion.py`

```python
from intelligence.quaternion import Quaternion
import numpy as np

# Create quaternion
q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

# From axis-angle
axis = np.array([0, 0, 1])  # Z-axis
angle = np.pi / 4           # 45 degrees
q = Quaternion.from_axis_angle(axis, angle)

# From Euler angles
q = Quaternion.from_euler(roll=0, pitch=0, yaw=np.pi/2)

# Operations
q_norm = q.normalize()
q_conj = q.conjugate()
q_inv = q.inverse()

# Composition
q_result = q1 * q2

# Rotate vector
vector = np.array([1, 0, 0])
rotated = q.rotate_vector(vector)

# Interpolation (slerp)
q_mid = Quaternion.slerp(q1, q2, t=0.5)

# Convert to rotation matrix
R = q.to_rotation_matrix()  # 3x3 numpy array
```

#### Quaternion Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `normalize()` | Normalize to unit quaternion | `Quaternion` |
| `conjugate()` | Compute conjugate (inverse rotation) | `Quaternion` |
| `inverse()` | Compute inverse | `Quaternion` |
| `magnitude()` | Compute magnitude | `float` |
| `rotate_vector(v)` | Rotate 3D vector | `np.ndarray` |
| `to_rotation_matrix()` | Convert to 3x3 rotation matrix | `np.ndarray` |
| `to_axis_angle()` | Convert to axis-angle representation | `(np.ndarray, float)` |
| `to_euler()` | Convert to Euler angles (roll, pitch, yaw) | `(float, float, float)` |

### Quaternion Exploration

**Location**: `src/intelligence/aria_exploration.py`

```python
from intelligence.aria_exploration import QuaternionExplorer

explorer = QuaternionExplorer(
    embedding_dim: int = 384,
    num_rotations: int = 100
)

# Explore semantic space
results = explorer.explore_rotations(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    num_iterations: int = 3,
    angle_degrees: float = 45.0
) -> List[Tuple[int, float]]

# Returns: [(doc_idx, score), ...]
```

#### Helper Functions

```python
from intelligence.aria_exploration import (
    golden_ratio_spiral,
    compute_rotation_params_from_perspective
)

# Generate uniform sphere points
points = golden_ratio_spiral(n=100)  # np.ndarray shape (100, 3)

# Compute rotation parameters from perspective
angle, rotations = compute_rotation_params_from_perspective(
    perspective="educational",
    confidence=0.85,
    user_adjustment=1.0
)
```

---

## Perspective API

### Perspective Detection

**Location**: `src/perspective/detector.py`

```python
from perspective.detector import PerspectiveOrientationDetector

detector = PerspectiveOrientationDetector(
    signatures_path="data/domain_dictionaries/perspective_signatures_v2.json"
)

# Detect perspective
result = detector.detect_perspective(query="How to debug Python errors?")

{
    "perspective": "diagnostic",      # Detected perspective
    "confidence": 0.87,               # Confidence score
    "signals": [...],                 # Matching signal words
    "all_scores": {                   # All perspective scores
        "educational": 0.23,
        "diagnostic": 0.87,
        # ...
    }
}
```

#### 8 Perspectives

| Perspective | Base Angle | Typical Queries |
|-------------|------------|-----------------|
| educational | 30° | "explain", "what is", "how does" |
| diagnostic | 90° | "debug", "error", "fix", "troubleshoot" |
| security | 45° | "vulnerability", "attack", "secure" |
| implementation | 60° | "build", "create", "implement" |
| research | 120° | "investigate", "explore", "analyze" |
| theoretical | 75° | "theory", "concept", "principle" |
| practical | 50° | "tutorial", "guide", "how-to" |
| reference | 15° | "definition", "what is" (factual) |

### Rotation Parameters

**Location**: `src/perspective/rotator.py`

```python
from perspective.rotator import compute_rotation_params

angle, iterations = compute_rotation_params(
    perspective: str,
    confidence: float,
    base_preset: str = "balanced"
) -> Tuple[float, int]
```

---

## Configuration API

### Config Loader

**Location**: `src/utils/config_loader.py`

```python
from utils.config_loader import load_config, get_config_value

# Load config file
config = load_config(config_path="aria_config.yaml")
# Returns: Dict[str, Any]

# Get specific value with default
value = get_config_value(
    key_path="retrieval.top_k",
    config=config,
    default=64
)
```

### Path Utilities

**Location**: `src/utils/paths.py`

```python
from utils.paths import (
    get_project_root,
    expand_path,
    ensure_dir
)

# Get project root
root = get_project_root()  # Path object

# Expand path (handles ~/ and ./)
expanded = expand_path("~/Documents/knowledge")  # Path object

# Ensure directory exists
dir_path = ensure_dir("./aria_packs")  # Path object
```

### Preset Utilities

**Location**: `src/utils/presets.py`

```python
from utils.presets import preset_to_postfilter_args

# Convert preset to postfilter arguments
args = preset_to_postfilter_args(preset)
# Returns: ["--max-per-source", "6", "--min-keep", "32"]
```

---

## Monitoring API

### Telemetry

**Location**: `src/monitoring/aria_telemetry.py`

```python
from monitoring.aria_telemetry import log_event, log_metric

# Log event
log_event(
    category="query",
    event="executed",
    data={"query": "...", "preset": "balanced"}
)

# Log metric
log_metric(
    metric_name="query_latency",
    value=1.23,
    unit="seconds"
)
```

### Terminal Output

**Location**: `src/monitoring/aria_terminal.py`

```python
from monitoring.aria_terminal import (
    print_status,
    print_error,
    print_success,
    print_info
)

print_status("Processing query...")
print_success("Query completed!")
print_error("Failed to load config")
print_info("Using preset: balanced")
```

---

## Data Structures

### Pack Format

```python
{
    "query": str,
    "timestamp": str,              # ISO format
    "preset": str,
    "perspective": str,
    "items": [
        {
            "path": str,           # Absolute path to source
            "content": str,        # Chunk content
            "score": float,        # Relevance score
            "start_line": int,
            "end_line": int,
            "source_type": str,
            "metadata": Dict[str, Any]
        }
    ],
    "metadata": {
        "chunks_retrieved": int,
        "files_used": int,
        "rotation_angle": float,
        "rotations": int,
        "query_features": Dict[str, Any],
        "bandit_meta": Dict[str, Any]
    }
}
```

### Query Features

```python
{
    "length": int,                 # Character count
    "word_count": int,             # Word count
    "avg_word_length": float,      # Average word length
    "question_words": List[str],   # Question words found
    "domain": str,                 # Detected domain
    "technical_density": float,    # Technical term ratio
    "complexity": str              # "simple", "moderate", "complex"
}
```

---

## Complete Example

### Full Pipeline

```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from core.aria_core import ARIA
from intelligence.bandit_context import BanditState
from perspective.detector import PerspectiveOrientationDetector
import json

# Initialize components
aria = ARIA(
    index_roots=["./datasets"],
    out_root="./aria_packs",
    state_path="~/.aria/bandit_state.json"
)

detector = PerspectiveOrientationDetector(
    "data/domain_dictionaries/perspective_signatures_v2.json"
)

# Execute query
query = "How does gradient descent optimize neural networks?"

# Detect perspective (optional - ARIA does this internally)
perspective = detector.detect_perspective(query)
print(f"Perspective: {perspective['perspective']} ({perspective['confidence']:.2f})")

# Run query
result = aria.query(query)

# Load pack
with open(result["pack"]) as f:
    pack = json.load(f)

# Process results
print(f"\nRetrieved {len(pack['items'])} chunks from {result['files_used']} files")
print(f"Preset used: {result['preset']}")

for i, item in enumerate(pack['items'][:3], 1):
    print(f"\n{i}. {Path(item['path']).name} (score: {item['score']:.3f})")
    print(f"   {item['content'][:200]}...")

# Check bandit state
state = BanditState(state_path="~/.aria/bandit_state.json")
for preset in state.presets:
    print(f"\n{preset.name}: α={preset.alpha:.1f}, β={preset.beta:.1f}, "
          f"pulls={preset.pulls}, avg_reward={preset.avg_reward:.3f}")
```

---

## Error Handling

```python
from core.aria_core import ARIA

try:
    aria = ARIA(
        index_roots=["./nonexistent"],
        out_root="./aria_packs"
    )
except FileNotFoundError as e:
    print(f"Knowledge base not found: {e}")
except Exception as e:
    print(f"Initialization failed: {e}")

try:
    result = aria.query("test query")
except ValueError as e:
    print(f"Invalid query: {e}")
except RuntimeError as e:
    print(f"Query execution failed: {e}")
```

---

## Type Hints

```python
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Function signature example
def retrieve_documents(
    query: str,
    index_root: Path,
    top_k: int = 64,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve documents matching query.

    Args:
        query: Search query string
        index_root: Knowledge base root path
        top_k: Number of results
        filters: Optional filters

    Returns:
        List of document dictionaries
    """
    pass
```

---

## Testing Your Integration

```python
import unittest
from core.aria_core import ARIA

class TestARIAIntegration(unittest.TestCase):
    def setUp(self):
        self.aria = ARIA(
            index_roots=["./test_data"],
            out_root="./test_packs"
        )

    def test_query(self):
        result = self.aria.query("test query")
        self.assertIsNotNone(result)
        self.assertIn("pack", result)
        self.assertGreater(result["chunks_retrieved"], 0)

    def test_preset_override(self):
        result = self.aria.query("test", preset_override="fast")
        self.assertEqual(result["preset"], "fast")

if __name__ == "__main__":
    unittest.main()
```

---

## Next Steps

- See [USAGE.md](USAGE.md) for practical examples
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system internals
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for extending the API
