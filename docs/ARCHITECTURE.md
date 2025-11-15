# ARIA System Architecture

## Overview

ARIA (Adaptive Resonant Intelligent Architecture) is a self-learning retrieval system that combines:
- **Teacher ARIA**: Query-driven knowledge retrieval
- **Student ARIA**: Conversation corpus learning
- **Quaternion Mathematics**: 4D semantic space exploration
- **LinUCB**: Adaptive preset selection
- **Perspective Detection**: Query context understanding

---

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA Control Center                       │
│  ┌──────────────────────┐    ┌────────────────────────┐     │
│  │   Teacher ARIA       │    │   Student ARIA         │     │
│  │   (Query/Retrieval)  │    │   (Corpus Learning)    │     │
│  └──────────────────────┘    └────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    ┌───▼────┐     ┌──────▼──────┐   ┌────▼─────┐
    │ Query  │     │  Retrieval  │   │ Learning │
    │ Layer  │     │   Engine    │   │  System  │
    └───┬────┘     └──────┬──────┘   └────┬─────┘
        │                 │               │
        │                 │               │
    ┌───▼─────────────────▼───────────────▼─────┐
    │           Intelligence Layer               │
    │  • Bandit (LinUCB)              │
    │  • Quaternion Exploration                  │
    │  • Perspective Detection                   │
    │  • Anchor Reasoning                        │
    └────────────────────────────────────────────┘
```

---

## Core Modules

### 1. Query Layer

**Location**: `src/core/aria_core.py`

**Responsibilities**:
- Query orchestration
- Session management
- Pack generation
- Result formatting

**Key Classes**:
- `ARIA` - Main orchestrator

**Flow**:
```
User Query → Feature Extraction → Preset Selection → Retrieval → Postfilter → Pack
```

### 2. Retrieval Engine

**Location**: `src/retrieval/`

**Components**:
- `local_rag_context_v7_guided_exploration.py` - Main retrieval with quaternion guidance
- `aria_v7_hybrid_semantic.py` - Hybrid BM25 + semantic search
- `aria_postfilter.py` - Quality and diversity filtering
- `query_features.py` - Query feature extraction

**Retrieval Pipeline**:
```
Query
  ↓
BM25 Lexical Search (top 10,000 docs)
  ↓
Semantic Embedding (query + docs)
  ↓
Quaternion Rotation (multi-perspective exploration)
  ↓
PCA Alignment (semantic space structure)
  ↓
Hybrid Scoring (0.3 * BM25 + 0.7 * Semantic)
  ↓
Top-K Selection (64-96 chunks)
  ↓
Postfilter (diversity + quality)
  ↓
Final Pack
```

### 3. Intelligence Layer

**Location**: `src/intelligence/`

**Components**:

#### Bandit Context (`bandit_context.py`)
- **LinUCB** - Bayesian multi-armed bandit
- **Preset Selection** - Dynamic strategy selection
- **Reward Learning** - Multi-objective optimization

**Algorithm**:
```python
for each preset:
    sample = Beta(α, β)  # α = successes, β = failures

selected_preset = argmax(samples)

# After query:
reward = 0.4 * exemplar_fit + 0.3 * coverage + 0.3 * diversity - 0.2 * issues
α += reward
β += (1 - reward)
```

#### Quaternion Math (`quaternion.py`)
- **Hypercomplex Numbers** - 4D rotations: `q = w + xi + yj + zk`
- **Semantic Rotations** - Explore embedding space
- **Slerp Interpolation** - Smooth transitions
- **Composition** - Chain multiple rotations

**Key Operations**:
```python
# Rotation of vector v by quaternion q
v_rotated = q * v * q.conjugate()

# Composition
q_total = q1 * q2 * q3

# Interpolation
q_interpolated = slerp(q1, q2, t=0.5)
```

#### ARIA Exploration (`aria_exploration.py`)
- **Golden Ratio Spiral** - Uniform sphere sampling (φ = 1.618...)
- **Multi-Rotation** - Iterative refinement
- **Perspective-Aware** - Rotation angles from query context

**Exploration Strategy**:
```python
# Generate N rotation points on sphere
points = golden_ratio_spiral(N)

for each rotation_point:
    q = quaternion_from_point(point, angle)
    rotated_embeddings = apply_rotation(q, embeddings)
    scores = compute_similarity(query, rotated_embeddings)

aggregate_scores(all_rotations)
```

### 4. Perspective Layer

**Location**: `src/perspective/`

**Components**:
- `detector.py` - 8-perspective classification
- `rotator.py` - Perspective-aware rotation parameters
- `user_profile.py` - User pattern learning
- `signature_learner.py` - Domain signature extraction

**8 Perspectives**:
1. **Educational** (30°) - Teaching, explaining, learning
2. **Diagnostic** (90°) - Debugging, troubleshooting, error analysis
3. **Security** (45°) - Threat analysis, vulnerabilities
4. **Implementation** (60°) - Building, coding, creating
5. **Research** (120°) - Investigation, exploration, discovery
6. **Theoretical** (75°) - Concepts, principles, abstractions
7. **Practical** (50°) - Applied knowledge, how-to
8. **Reference** (15°) - Factual lookup, quick answers

**Detection Algorithm**:
```python
# Extract query features
features = {
    'question_words': ['what', 'how', 'why'],
    'action_words': ['debug', 'fix', 'explain'],
    'technical_density': count_technical_terms(),
    'domain_signals': detect_domain_keywords()
}

# Score each perspective
scores = perspective_matcher(features)
selected = argmax(scores)

# Compute rotation angle
angle = base_angle * confidence * user_adjustment
```

### 5. Anchor Layer

**Location**: `src/anchors/`

**Components**:
- `exemplar_fit.py` - Response quality scoring

**16 Anchors** (Response Modes):
- **Core**: formal, casual, philosophical, analytical, factual, creative, feedback_correction, educational
- **Technical**: code, technical, engineering, medical, science, mathematics, business, law

### 6. Monitoring Layer

**Location**: `src/monitoring/`

**Components**:
- `aria_telemetry.py` - Event logging and metrics
- `aria_terminal.py` - Colorized console output
- `metrics_utils.py` - Performance tracking

---

## Data Flow

### Query Execution

```
1. User Query
   ├─> Query Features Extraction
   └─> Perspective Detection

2. Bandit Preset Selection
   ├─> LinUCB
   ├─> Query Features → Preset Mapping
   └─> Selected Preset (fast/balanced/deep/diverse)

3. Retrieval
   ├─> BM25 Lexical Search
   ├─> Semantic Embedding
   ├─> Quaternion Rotation (N iterations)
   └─> Hybrid Scoring

4. Postfilter
   ├─> Diversity Filter (max per file)
   ├─> Quality Filter (min score)
   └─> Topic Filter (relevance)

5. Pack Generation
   ├─> Format Chunks
   ├─> Add Metadata
   └─> Save JSON

6. Reward Update
   ├─> Exemplar Fit (40%)
   ├─> Coverage (30%)
   ├─> Diversity (30%)
   ├─> Issues Penalty (-20%)
   └─> Update Bandit α, β
```

### Student Learning

```
1. Conversation Monitor
   └─> Watch ~/.lmstudio/conversations/

2. New Conversation Detected
   ├─> Parse JSON
   ├─> Extract Messages
   └─> Filter Quality

3. Pattern Extraction
   ├─> Turn-taking patterns
   ├─> Reasoning styles
   ├─> Domain transitions
   └─> Response structures

4. Corpus Building
   ├─> Save to training_data/conversation_corpus/
   ├─> Update Statistics
   └─> Log Metrics
```

---

## Mathematical Foundation

### Quaternion Semantic Space

**Representation**:
```
q = w + xi + yj + zk

where:
  w = real component
  (x, y, z) = imaginary components (rotation axis)
```

**Rotation Matrix** (3D vector rotation):
```
R(q) = [
  [1-2(y²+z²),   2(xy-wz),   2(xz+wy)  ]
  [2(xy+wz),   1-2(x²+z²),   2(yz-wx)  ]
  [2(xz-wy),   2(yz+wx),   1-2(x²+y²) ]
]

v_rotated = R(q) * v
```

**Why Quaternions?**
- No gimbal lock (unlike Euler angles)
- Efficient composition (multiply quaternions)
- Smooth interpolation (slerp)
- Natural for 4D+ spaces (embeddings are high-dimensional)

### Golden Ratio Spiral

**Formula**:
```python
φ = (1 + √5) / 2 ≈ 1.618  # Golden ratio

for i in range(N):
    θ = 2π * i / φ          # Azimuthal angle
    h = -1 + 2*i/(N-1)      # Height on sphere
    r = √(1 - h²)           # Radius at height h

    x = r * cos(θ)
    y = r * sin(θ)
    z = h
```

**Properties**:
- Uniform distribution on sphere
- No clustering or gaps
- φ is irrational → no resonance

### LinUCB

**Beta Distribution**:
```
P(θ | α, β) = θ^(α-1) * (1-θ)^(β-1) / B(α, β)

where:
  α = successes (wins)
  β = failures (losses)
  θ = success probability
```

**Selection**:
```python
for each preset:
    θ_sample = Beta(α, β).sample()

selected = argmax(θ_samples)
```

**Update**:
```python
reward = compound_reward(query_result)
α += reward
β += (1 - reward)
```

---

## Configuration System

**Location**: `aria_config.yaml`

**Structure**:
```yaml
paths:              # File locations
retrieval:          # Search parameters
postfilter:         # Quality/diversity filters
bandit:             # LinUCB settings
context:            # LLM context limits
perspective:        # 8-perspective detection
anchors:            # 16-anchor reasoning
student:            # Corpus learning
monitoring:         # Telemetry
```

**Config Loader**: `src/utils/config_loader.py`
- YAML parsing
- Path expansion (`~/`, `./`)
- Type validation
- Default values

---

## Preset System

**Location**: `src/intelligence/bandit_context.py`

**4 Presets**:

| Preset     | top_k | rotations | max_per_file | Use Case           |
|------------|-------|-----------|--------------|-------------------|
| fast       | 40    | 1         | 8            | Quick lookups     |
| balanced   | 64    | 2         | 6            | General queries   |
| deep       | 96    | 3         | 5            | Complex research  |
| diverse    | 80    | 2         | 4            | Broad exploration |

**Preset Selection**:
- First 20 queries: Random exploration
- After 20: LinUCB exploitation

---

## State Management

**Bandit State**: `.aria_contextual_bandit.json` (project root)
```json
{
  "total_pulls": 45,
  "presets": [
    {
      "name": "balanced",
      "alpha": 12.5,
      "beta": 8.2,
      "pulls": 15,
      "wins": 12,
      "avg_reward": 0.78
    }
  ]
}
```

**Watcher State**: `../var/watcher_state.json`
```json
{
  "seen_conversations": ["conv_123.json", "conv_456.json"],
  "last_check": "2025-11-14T12:00:00",
  "stats": {
    "total_captured": 42,
    "total_messages": 1337
  }
}
```

---

## Performance Characteristics

**Retrieval Latency**:
- BM25 search: ~100-500ms (10k documents)
- Embedding: ~200-800ms (depends on model/GPU)
- Quaternion rotation: ~50-200ms (3 iterations)
- Postfilter: ~10-50ms
- **Total**: ~500-2000ms per query

**Memory Usage**:
- Sentence transformer model: ~100MB
- Document embeddings (cached): ~10MB per 1000 docs
- Quaternion operations: negligible
- **Total**: ~200-500MB for typical use

**Scalability**:
- Works well up to 100k documents
- For larger corpora, consider document pre-filtering
- Embeddings can be pre-computed and cached

---

## Extension Points

### Adding New Presets

Edit `src/intelligence/bandit_context.py`:
```python
DEFAULT_PRESETS = [
    # ... existing presets
    {"name": "my_preset", "args": {
        "top_k": 128,
        "sem_limit": 256,
        "rotations": 4,
        "max_per_file": 3
    }}
]
```

### Adding New Perspectives

Edit `src/perspective/detector.py`:
```python
PERSPECTIVES = [
    # ... existing perspectives
    "my_perspective"
]

BASE_ANGLES = {
    # ... existing angles
    "my_perspective": 42.0
}
```

### Custom Postfilters

Implement in `src/retrieval/aria_postfilter.py`:
```python
def my_custom_filter(chunks, **kwargs):
    # Filter logic
    return filtered_chunks
```

---

## Testing

**Test Suite**: `tests/comprehensive_test_suite.py`

**14 Tests**:
1. Bandit initialization
2. Preset configuration
3. Quaternion math
4. Rotation operations
5. Normalization
6. Conjugate
7. Inverse
8. Composition
9. Slerp interpolation
10. Axis-angle conversion
11. Vector rotation
12. Golden ratio spiral
13. Perspective rotation parameters
14. Multi-rotation exploration

**Run Tests**:
```bash
python3 tests/comprehensive_test_suite.py
```

---

## Further Reading

- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- [USAGE.md](USAGE.md) - How to use ARIA
- [API_REFERENCE.md](API_REFERENCE.md) - Programmatic API
- [QUATERNIONS.md](QUATERNIONS.md) - Mathematical deep dive
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
