# ARIA Metrics Guide

**Understanding and Interpreting ARIA Telemetry**

## Overview

ARIA collects comprehensive metrics to enable self-optimization through closed-loop learning. This guide explains each metric category, computation methods, and interpretation guidelines.

---

## Metric Categories

### 1. Retrieval Metrics
### 2. Postfilter Metrics  
### 3. Exploration Metrics ⭐ NEW
### 4. Bandit Metrics
### 5. Curiosity Metrics
### 6. Quality Metrics
### 7. Conversation Metrics

---

## 1. Retrieval Metrics

### Retrieved Chunk Count
**What**: Number of chunks returned from initial retrieval  
**Range**: 0 to configured `k` (typically 20-50)  
**Interpretation**:
- ✅ Meeting target `k` = Good corpus coverage
- ⚠️ Significantly fewer than `k` = Sparse corpus or overly specific query
- ❌ Zero results = No matching content

### Retrieval Latency
**What**: Time to execute retrieval (milliseconds)  
**Typical**: 150-700ms depending on corpus size and strategy  
**Factors**:
- Corpus size (larger = slower BM25)
- GPU availability (faster embeddings)
- Cache hits (faster if embeddings cached)
- Strategy (hybrid slower than single-method)

### Source Diversity
**What**: Number of unique sources in results  
**Range**: 1 to `k` (max = all chunks from different sources)  
**Formula**: `unique_sources / total_chunks`  
**Interpretation**:
- ✅ >0.7 = High diversity (good)
- ⚠️ 0.4-0.7 = Moderate diversity
- ❌ <0.4 = Too concentrated (single-source bias)

### Average Chunk Score
**What**: Mean retrieval relevance score  
**Range**: 0.0 to 1.0  
**Interpretation**:
- ✅ >0.8 = High relevance
- ⚠️ 0.5-0.8 = Moderate relevance
- ❌ <0.5 = Weak relevance (query mismatch)

---

## 2. Postfilter Metrics

### Quality-Filtered Count
**What**: Chunks passing quality threshold  
**Range**: 0 to retrieved count  
**Formula**: Chunks with `score >= min_quality_score`  
**Interpretation**:
- ✅ High retention (>80%) = Quality corpus
- ⚠️ Moderate retention (50-80%) = Mixed quality
- ❌ Excessive filtering (<50%) = Threshold too high or poor corpus

### Diversity Score
**What**: Semantic diversity of final pack  
**Range**: 0.0 (all duplicates) to 1.0 (all unique)  
**Method**: Average pairwise cosine distance  
**Target**: >0.6

### Pack Coverage
**What**: Proportion of query aspects covered  
**Range**: 0.0 to 1.0  
**Method**: Query term presence in chunks  
**Interpretation**:
- ✅ >0.7 = Comprehensive coverage
- ⚠️ 0.4-0.7 = Partial coverage
- ❌ <0.4 = Incomplete (missing key aspects)

### Source Distribution Entropy
**What**: Entropy of source distribution  
**Range**: 0 (single source) to log(N) (uniform)  
**Formula**: `-Σ p_i * log(p_i)` where p_i = fraction from source i  
**Interpretation**: Higher = more diverse sources

---

## 3. Exploration Metrics ⭐ NEW

### Quaternion State Vector

**What**: Current semantic position on S³  
**Format**: `[w, x, y, z]` where w² + x² + y² + z² = 1  
**Example**: `[0.707, 0.707, 0.0, 0.0]`

**Interpretation**:
- **Identity state** [1, 0, 0, 0]: Initial state, no prior queries
- **Smooth transitions**: SLERP creates gradual evolution
- **Large rotations**: Different topic = large quaternion change
- **Momentum visible**: Non-zero off-diagonal components

**Tracking Over Time**:
```python
# Plot quaternion evolution
states = [run['exploration']['quaternion_state'] for run in runs]
plt.plot(states)  # See semantic trajectory
```

### Quaternion Momentum

**What**: Velocity in tangent space of S³  
**Format**: `[dw, dx, dy, dz]`  
**Typical**: Small values (~0.01-0.1)

**Interpretation**:
- **Zero momentum**: First query or unrelated to previous
- **Non-zero momentum**: Semantic inertia from past queries
- **Large momentum**: Rapid semantic shifts
- **Decaying momentum**: Gradual slowdown over time

### Geodesic Distance

**What**: Rotation angle on S³ between queries  
**Range**: 0 to π radians (0° to 180°)  
**Formula**: `arccos(|dot(q1, q2)|)`

**Interpretation**:
- ✅ < π/6 (30°) = Very similar queries
- ⚠️ π/6 to π/3 (30° to 60°) = Related topics
- ⚠️ π/3 to 2π/3 (60° to 120°) = Different but not opposite
- ❌ > 2π/3 (120°) = Opposite semantic regions

### Cross-Query Memory Hits

**What**: Number of times similar past states were recalled  
**Range**: 0 to size of history  
**Interpretation**:
- ✅ Frequent hits = Consistent exploration area
- ⚠️ Occasional hits = Varied topics
- ❌ No hits = Always exploring new territory

### PCA Exploration Stats

**PCA Components Used**:  
**What**: Number of principal components  
**Typical**: 32 (configured)  
**Interpretation**: Higher = more semantic dimensions captured

**Variance Explained**:  
**What**: Cumulative variance captured by PCA  
**Typical**: 0.85-0.95 (85-95%)  
**Interpretation**: Higher = better corpus representation

**Rotations Generated**:  
**What**: Number of rotated query versions  
**Typical**: 8  
**Interpretation**: More rotations = more perspectives (but slower)

**PCA Retrieval Count**:  
**What**: Chunks retrieved from PCA-rotated queries  
**Range**: 0 to (rotations × per-rotation k)  
**Interpretation**: Higher = more multi-perspective content

### Golden Ratio Spiral Stats

**Samples Generated**:  
**What**: Number of φ-spaced samples  
**Typical**: 13 (Fibonacci number)  
**Why 13**: Optimal for sphere packing via φ

**Spiral Retrieval Count**:  
**What**: Chunks retrieved from spiral samples  
**Range**: 0 to (samples × per-sample k)  
**Interpretation**: Higher = more comprehensive coverage

**Unique Spiral Chunks**:  
**What**: Non-duplicate chunks from spiral  
**Range**: 0 to spiral retrieval count  
**Interpretation**:
- ✅ High ratio (>0.7) = Good exploration (finding new content)
- ⚠️ Medium ratio (0.4-0.7) = Some overlap with initial retrieval
- ❌ Low ratio (<0.4) = Redundant (not exploring effectively)

### Exploration Strategy

**What**: Which exploration mode was used  
**Values**:
- `golden_ratio_spiral`: φ-based coverage
- `pca_exploration`: Subspace rotations
- `quaternion_only`: State-based reranking
- `combined`: All methods

**Interpretation**: Combined typically gives best results

### Exploration Latency

**What**: Time spent in exploration phase  
**Typical**: 75-220ms  
**Breakdown**:
- Quaternion state update: 5-15ms
- PCA rotations: 30-80ms
- Golden ratio spiral: 40-120ms

**Overhead**: ~25% of total pipeline time  
**Benefit**: 15-25% better semantic coverage

### Reranking Impact

**What**: Change in chunk ordering after quaternion reranking  
**Metric**: Kendall's τ distance  
**Range**: 0 (no change) to 1 (complete reversal)  
**Interpretation**:
- ✅ 0.2-0.5 = Meaningful reordering
- ⚠️ <0.2 = Little impact (quaternion state close to query)
- ⚠️ >0.7 = Dramatic reordering (state very different from query)

---

## 4. Bandit Metrics

### Selected Strategy
**What**: Which retrieval strategy was chosen  
**Values**: `lexical_only`, `semantic_only`, `hybrid_balanced`, etc.  
**Tracked**: Per-query selection frequency

### Exploration Rate (Epsilon)
**What**: Probability of random strategy selection  
**Default**: 0.1 (10% exploration)  
**Adaptive**: Can decay over time

### Strategy Reward History
**What**: Cumulative reward per strategy  
**Formula**: Σ rewards over all uses  
**Used for**: Thompson Sampling updates

### Alpha/Beta Parameters
**What**: Beta distribution parameters for each strategy  
**Initial**: (1, 1) - uniform prior  
**Updates**:
- α += reward (successes)
- β += (1 - reward) (failures)
**Interpretation**: α/(α+β) = estimated win rate

---

## 5. Curiosity Metrics

### Gap Counts

**Semantic Gaps**:  
**What**: Missing topics via embedding analysis  
**Method**: Query entities not in retrieved chunks  
**Range**: 0 to # query entities

**Factual Gaps**:  
**What**: Missing named entities  
**Method**: Entity extraction + coverage check  
**Range**: 0 to # required entities

**Logical Gaps**:  
**What**: Incomplete reasoning chains  
**Method**: Dependency graph analysis  
**Range**: 0 to # missing links

**Total Gaps**: Sum of all gap types

### Confidence Score
**What**: System confidence in response  
**Range**: 0.0 (uncertain) to 1.0 (certain)  
**Factors**:
- Retrieval score quality
- Source agreement
- Coverage completeness
- Exploration success

### Socratic Questions Generated
**What**: Number of follow-up questions  
**Range**: 0 to configured max  
**Interpretation**: More questions = more uncertainty

### Synthesis Strategy Used
**Values**: `speed`, `depth`, `adaptive`  
**Selection logic**:
- High confidence → speed
- Low confidence + gaps → depth
- Otherwise → adaptive

---

## 6. Quality Metrics

### Fluency Score
**What**: Language quality assessment  
**Method**: Perplexity-based (lower = more fluent)  
**Range**: Typically 10-100  
**Interpretation**:
- ✅ <30 = Fluent
- ⚠️ 30-80 = Acceptable
- ❌ >80 = Incoherent

### Factual Correctness
**What**: Alignment with retrieved sources  
**Method**: Entailment scoring  
**Range**: 0.0 to 1.0  
**Target**: >0.7

### Relevance Score
**What**: Response alignment to query  
**Method**: Semantic similarity  
**Range**: 0.0 to 1.0  
**Target**: >0.6

### Source Quality
**What**: Reliability of sources used  
**Factors**:
- Source authority (.edu, .gov, .org)
- Recency
- Citation count
- Peer review status
**Range**: 0.0 to 1.0

### Composite Reward
**Formula**:
```python
reward = (
    0.3 * fluency +
    0.3 * factual +
    0.2 * relevance +
    0.2 * source_quality
)
```
**Range**: 0.0 to 1.0  
**Used for**: Bandit updates

---

## 7. Conversation Metrics

### Turn Count
**What**: Number of exchanges in conversation  
**Interpretation**: Longer = more engagement

### Topic Drift
**What**: Semantic distance from initial query  
**Method**: Embedding similarity to turn 1  
**Range**: 0.0 (on-topic) to 1.0 (drifted)

### User Satisfaction (Implicit)
**Signals**:
- Follow-up questions = engagement
- Reformulations = dissatisfaction
- Termination = completion or frustration

---

## Telemetry Output Format

### Complete JSON Example

```json
{
  "timestamp": "2025-11-03T12:34:56",
  "query": "How does Thompson Sampling work?",
  "anchor_mode": "technical",
  "session_id": "abc123",
  
  "retrieval": {
    "strategy": "hybrid_balanced",
    "latency_ms": 245,
    "chunks_retrieved": 20,
    "avg_score": 0.82,
    "source_diversity": 0.75
  },
  
  "postfilter": {
    "chunks_after_quality": 18,
    "chunks_after_diversity": 15,
    "diversity_score": 0.68,
    "pack_coverage": 0.85
  },
  
  "exploration": {
    "applied": true,
    "strategy": "combined",
    "latency_ms": 123,
    
    "quaternion_state": {
      "vector": [0.712, 0.702, 0.000, 0.001],
      "momentum": [0.012, -0.003, 0.008, 0.001],
      "geodesic_distance": 0.523,
      "memory_hits": 2
    },
    
    "pca": {
      "enabled": true,
      "components": 32,
      "variance_explained": 0.91,
      "rotations": 8,
      "chunks_retrieved": 24,
      "unique_chunks": 18
    },
    
    "golden_ratio": {
      "enabled": true,
      "samples": 13,
      "chunks_retrieved": 35,
      "unique_chunks": 27,
      "coverage_improvement": 0.22
    },
    
    "reranking": {
      "applied": true,
      "kendall_tau": 0.34,
      "final_count": 32
    }
  },
  
  "bandit": {
    "selected_strategy": "hybrid_balanced",
    "exploration": false,
    "alpha": 12.5,
    "beta": 3.8,
    "estimated_win_rate": 0.767
  },
  
  "curiosity": {
    "enabled": true,
    "semantic_gaps": 1,
    "factual_gaps": 0,
    "logical_gaps": 0,
    "total_gaps": 1,
    "confidence": 0.87,
    "questions_generated": 1,
    "synthesis_strategy": "speed"
  },
  
  "quality": {
    "fluency": 0.91,
    "factual": 0.88,
    "relevance": 0.92,
    "source_quality": 0.85,
    "composite_reward": 0.89
  }
}
```

---

## Interpreting Metrics

### High-Quality Exploration

✅ **Good Signs:**
- Quaternion state shows smooth evolution (small geodesic distances)
- Golden ratio spiral discovers new chunks (>0.6 unique ratio)
- PCA rotations find complementary content
- Exploration overhead <150ms
- Coverage improvement >0.15

❌ **Warning Signs:**
- Quaternion state jumps erratically (large geodesic distances)
- Spiral mostly duplicates initial retrieval (<0.4 unique)
- PCA disabled (corpus too small)
- Exploration latency >250ms
- No coverage improvement

### Effective Bandit Learning

✅ **Good Signs:**
- Converging strategy preferences
- Rising average rewards over time
- Appropriate exploration rate
- α/(α+β) ratios diverging (strategies differentiate)

❌ **Warning Signs:**
- Flat reward curves (not learning)
- Excessive exploration (not exploiting)
- Single strategy dominance (premature convergence)
- All strategies have similar α/β ratios

### Healthy Curiosity

✅ **Good Signs:**
- Gaps detected when appropriate
- Questions relevant to gaps
- Adaptive synthesis based on confidence
- Confidence correlates with coverage

❌ **Warning Signs:**
- No gaps ever detected (too confident)
- Irrelevant questions
- Fixed synthesis strategy
- Confidence doesn't match actual quality

---

## Performance Optimization

### If Exploration is Slow:

1. **Reduce golden ratio samples**: 13 → 8
2. **Disable PCA**: Set `enabled: false` if corpus small
3. **Lower exploration radius**: 0.3 → 0.2
4. **Enable GPU**: Faster embeddings
5. **Cache embeddings**: Reuse across queries

### If Exploration Not Helping:

1. **Check unique chunk ratio**: Should be >0.5
2. **Verify quaternion state persists**: Check state file
3. **Increase exploration radius**: 0.2 → 0.4
4. **Enable all methods**: Ensure PCA + spiral both active
5. **Review telemetry**: Is coverage actually improving?

### If Quality Declining:

1. **Check postfilter**: May be too aggressive
2. **Review exploration chunks**: Ensure relevance
3. **Adjust quaternion decay**: Slower state evolution
4. **Verify PCA variance**: Should explain >0.85
5. **Increase quality threshold**: Filter low-quality spiral chunks

---

## Advanced Analysis

### Quaternion State Trajectories

Plot semantic paths through S³:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract quaternion states over time
states = [run['exploration']['quaternion_state']['vector'] 
          for run in runs]

# Project to 3D for visualization (drop w component)
trajectories = [(s[1], s[2], s[3]) for s in states]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*zip(*trajectories))
ax.set_title('Semantic Trajectory on S³')
plt.show()
```

### Golden Ratio Coverage Analysis

Measure actual coverage improvement:

```python
# Calculate overlap between retrieval and exploration
def coverage_analysis(runs):
    for run in runs:
        initial = set(c['id'] for c in run['retrieval']['chunks'])
        after_spiral = set(c['id'] for c in run['exploration']['final_chunks'])
        
        new_chunks = after_spiral - initial
        coverage_gain = len(new_chunks) / len(after_spiral)
        
        print(f"Coverage gain: {coverage_gain:.1%}")
```

### Strategy Performance by Anchor Mode

Compare bandit strategies across reasoning modes:

```python
from collections import defaultdict

strategy_by_mode = defaultdict(lambda: defaultdict(list))

for run in runs:
    mode = run['anchor_mode']
    strategy = run['bandit']['selected_strategy']
    reward = run['quality']['composite_reward']
    
    strategy_by_mode[mode][strategy].append(reward)

# Identify best strategy per mode
for mode, strategies in strategy_by_mode.items():
    best_strategy = max(strategies.items(), 
                       key=lambda x: np.mean(x[1]))
    print(f"{mode}: {best_strategy[0]} (μ={np.mean(best_strategy[1]):.3f})")
```

---

## Telemetry Best Practices

### Data Collection

- ✅ Record every query with full context
- ✅ Include all exploration metadata
- ✅ Timestamp with millisecond precision
- ✅ Version telemetry schema

### Data Analysis

- ✅ Aggregate weekly/monthly
- ✅ Track trends over time (learning curves)
- ✅ Identify outliers (anomalous queries)
- ✅ Correlate metrics (e.g., coverage vs latency)

### Privacy

- ✅ Store locally only (no external telemetry)
- ✅ Optionally sanitize queries (remove PII)
- ✅ Rotate old logs (configurable retention)
- ✅ Secure storage (encrypted if sensitive)

---

## Metric Relationships

**Exploration → Coverage**:  
More exploration samples → Better coverage (but slower)

**Coverage → Quality**:  
Better coverage → Higher quality (more context)

**Bandit → Latency**:  
Optimal strategy selection → Lower latency

**Quaternion → Consistency**:  
State continuity → More consistent results

**Curiosity → Confidence**:  
Gap detection → Accurate confidence

---

**Metrics enable measurement. Measurement enables optimization. Optimization enables intelligence.** ✨
