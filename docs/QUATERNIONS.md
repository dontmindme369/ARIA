# Quaternions in ARIA

## Introduction

ARIA uses **quaternions** for semantic space exploration. This document explains why, how, and what makes quaternions uniquely suited for high-dimensional retrieval.

---

## What Are Quaternions?

Quaternions are **hypercomplex numbers** extending complex numbers to 4 dimensions.

### Mathematical Definition

```
q = w + xi + yj + zk

where:
  w, x, y, z ∈ ℝ (real numbers)
  i² = j² = k² = ijk = -1
```

### Components

- **w**: Real (scalar) component
- **x, y, z**: Imaginary (vector) components
- **i, j, k**: Imaginary units

### Hamilton's Rules

```
i² = j² = k² = -1
ij = k,  jk = i,  ki = j
ji = -k, kj = -i, ik = -j
```

---

## Why Quaternions for Retrieval?

### 1. No Gimbal Lock

**Euler Angles Problem**:
```python
# Euler angles can lose a degree of freedom
pitch = 90°  # Gimbal lock!
# Roll and yaw now rotate same axis
```

**Quaternion Solution**:
```python
# Always 4 independent parameters
q = Quaternion(w, x, y, z)
# No singularities, no gimbal lock
```

### 2. Efficient Composition

**Multiple Rotations**:
```python
# Euler angles: Convert → Rotate → Convert (expensive)
R_total = euler_to_matrix(angles3) @ euler_to_matrix(angles2) @ euler_to_matrix(angles1)

# Quaternions: Just multiply (cheap)
q_total = q3 * q2 * q1
```

### 3. Smooth Interpolation

**Slerp** (Spherical Linear Interpolation):
```python
# Smooth rotation between two orientations
q_mid = slerp(q_start, q_end, t=0.5)

# Euler angles: Non-smooth, axis-dependent
# Quaternions: Shortest path on 4D sphere
```

### 4. Natural for High Dimensions

**Semantic Embeddings**:
- ARIA uses 384D embeddings (MiniLM-L6-v2)
- Quaternions rotate 3D subspaces within 384D
- Apply rotation matrix to embedding vectors
- Explore semantic space systematically

---

## Quaternion Operations

### Normalization

**Unit Quaternion** (magnitude = 1):
```python
q_norm = q / |q|

where |q| = √(w² + x² + y² + z²)
```

**Why**: Only unit quaternions represent rotations.

### Conjugate

```python
q* = w - xi - yj - zk
```

**Property**: `q * q* = |q|²`

**Use**: Reverse rotation (if q is unit)

### Inverse

```python
q⁻¹ = q* / |q|²

For unit quaternion: q⁻¹ = q*
```

### Multiplication

```python
(w₁ + x₁i + y₁j + z₁k) * (w₂ + x₂i + y₂j + z₂k) =

w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂ +
(w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i +
(w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j +
(w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k
```

**Non-commutative**: `q₁ * q₂ ≠ q₂ * q₁` (order matters!)

---

## Representing Rotations

### Axis-Angle Representation

**Given**:
- Rotation axis: `n = (nₓ, nᵧ, n_z)` (unit vector)
- Rotation angle: `θ`

**Quaternion**:
```python
q = cos(θ/2) + sin(θ/2) * (nₓi + nᵧj + n_zk)

# Example: 90° around Z-axis
θ = π/2
n = [0, 0, 1]

q = cos(π/4) + sin(π/4) * k
  = 0.707 + 0.707k
  = (0.707, 0, 0, 0.707)
```

### Rotation Matrix

**3x3 Matrix** from quaternion:
```python
R = [
  [1-2(y²+z²),   2(xy-wz),   2(xz+wy)  ]
  [2(xy+wz),   1-2(x²+z²),   2(yz-wx)  ]
  [2(xz-wy),   2(yz+wx),   1-2(x²+y²) ]
]
```

**Rotate vector**:
```python
v' = R * v
```

### Direct Vector Rotation

**Formula**:
```python
v' = q * v * q*

where v is treated as pure quaternion: v = 0 + vₓi + vᵧj + v_zk
```

---

## ARIA's Quaternion Strategy

### 1. Golden Ratio Spiral

**Why Golden Ratio?**

φ = (1 + √5) / 2 ≈ 1.618...

**Property**: φ is **maximally irrational**
- No simple fraction approximation
- No resonance patterns
- Uniform sphere coverage

**Algorithm**:
```python
def golden_ratio_spiral(n):
    phi = (1 + np.sqrt(5)) / 2
    points = []

    for i in range(n):
        theta = 2 * np.pi * i / phi    # Azimuthal angle
        h = -1 + 2 * i / (n - 1)       # Height: -1 to 1
        r = np.sqrt(1 - h**2)          # Radius at height h

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h

        points.append([x, y, z])

    return np.array(points)
```

**Visualization**:
```
    z
    │
    ○  ← Evenly distributed points
   ○ ○  on unit sphere surface
  ○   ○
 ○     ○ ← No clustering
○───○───○─── y
  ○   ○
   ○ ○
    ○
   /
  x
```

### 2. Multi-Rotation Exploration

**Single Rotation**: Limited exploration
**Multi-Rotation**: Comprehensive coverage

**Algorithm**:
```python
# Iteration 1: Initial rotation
rotations_1 = [generate_quaternion(point) for point in spiral_points]
scores_1 = [compute_similarity(rotate(q, embeddings)) for q in rotations_1]

# Iteration 2: Rotate around best from iteration 1
best_q1 = rotations_1[argmax(scores_1)]
rotations_2 = [best_q1 * q for q in generate_new_spiral()]
scores_2 = [compute_similarity(...) for q in rotations_2]

# Iteration 3: Further refinement
best_q2 = rotations_2[argmax(scores_2)]
rotations_3 = [best_q2 * q for q in generate_new_spiral()]
scores_3 = [compute_similarity(...) for q in rotations_3]

# Aggregate scores
final_scores = aggregate([scores_1, scores_2, scores_3])
```

**Effect**: Progressively refine search in semantic space

### 3. PCA Alignment

**Problem**: Embeddings have intrinsic structure
**Solution**: Align quaternion rotations with principal components

**Algorithm**:
```python
# 1. Compute PCA of document embeddings
pca = PCA(n_components=3)
pca.fit(document_embeddings[:, :3])  # Use first 3 dims

# 2. Get principal axes
pc1, pc2, pc3 = pca.components_

# 3. Align quaternion rotation axes with PCs
rotation_axis = pc1  # Rotate around first principal component
q = Quaternion.from_axis_angle(rotation_axis, angle)

# 4. Apply rotation
rotated_embeddings = apply_quaternion_rotation(q, embeddings)
```

**Effect**: Rotations follow natural structure of semantic space

---

## Perspective-Based Rotation Angles

ARIA adjusts rotation angle based on query perspective.

### Angle Mapping

| Perspective | Base Angle | Exploration Style |
|-------------|------------|-------------------|
| Reference | 15° | Minimal (direct lookup) |
| Educational | 30° | Gentle (broad concepts) |
| Security | 45° | Moderate (focused scan) |
| Practical | 50° | Moderate-Strong |
| Implementation | 60° | Strong (code/building) |
| Theoretical | 75° | Strong (abstractions) |
| Diagnostic | 90° | Aggressive (debugging) |
| Research | 120° | Very Aggressive (exploration) |

### Why Different Angles?

**Small Angles** (15°-30°):
- Stay close to original query
- Find very similar content
- Good for factual lookup

**Medium Angles** (45°-75°):
- Explore related concepts
- Balance precision and recall
- Good for learning, implementation

**Large Angles** (90°-120°):
- Aggressive exploration
- Find tangentially related content
- Good for debugging, research

### Computation

```python
def compute_rotation_angle(perspective, confidence, user_adjustment=1.0):
    """
    Compute rotation angle from perspective and confidence.

    Args:
        perspective: Detected perspective (e.g., "educational")
        confidence: Detection confidence (0.0 - 1.0)
        user_adjustment: User preference multiplier

    Returns:
        Rotation angle in degrees
    """
    base_angles = {
        "reference": 15.0,
        "educational": 30.0,
        "security": 45.0,
        "practical": 50.0,
        "implementation": 60.0,
        "theoretical": 75.0,
        "diagnostic": 90.0,
        "research": 120.0
    }

    base = base_angles.get(perspective, 60.0)
    angle = base * confidence * user_adjustment

    return np.clip(angle, 10.0, 150.0)  # Safety bounds
```

---

## Practical Example

### Complete Rotation Pipeline

```python
import numpy as np
from intelligence.quaternion import Quaternion
from intelligence.aria_exploration import golden_ratio_spiral

# 1. Generate rotation points
n_rotations = 100
rotation_points = golden_ratio_spiral(n_rotations)

# 2. Create quaternions
angle = 45.0  # degrees
angle_rad = np.radians(angle)

quaternions = []
for point in rotation_points:
    # point is rotation axis (unit vector)
    q = Quaternion.from_axis_angle(point, angle_rad)
    quaternions.append(q)

# 3. Apply rotations to embeddings
query_embedding = get_embedding("How does gradient descent work?")
document_embeddings = get_embeddings(documents)

all_scores = []
for q in quaternions:
    # Rotate embeddings (apply to 3D subspace)
    R = q.to_rotation_matrix()
    rotated_docs = document_embeddings.copy()
    rotated_docs[:, :3] = (R @ document_embeddings[:, :3].T).T

    # Compute similarity
    scores = cosine_similarity(query_embedding, rotated_docs)
    all_scores.append(scores)

# 4. Aggregate scores (max across rotations)
final_scores = np.max(all_scores, axis=0)

# 5. Select top-k
top_k_indices = np.argsort(final_scores)[-64:][::-1]
```

---

## Mathematical Properties

### Quaternion Norm Preservation

**Property**: Quaternion multiplication preserves norm
```
|q₁ * q₂| = |q₁| * |q₂|
```

**For unit quaternions**:
```
|q₁| = |q₂| = 1  ⟹  |q₁ * q₂| = 1
```

**Implication**: Rotation composition stays on unit sphere

### Rotation Angle Addition

**Property**: Rotating by θ₁ then θ₂ around same axis = rotating by (θ₁ + θ₂)

**Quaternion**:
```python
q₁ = from_axis_angle(n, θ₁)
q₂ = from_axis_angle(n, θ₂)

q_total = q₂ * q₁
        = from_axis_angle(n, θ₁ + θ₂)
```

### Double Coverage

**Property**: q and -q represent the same rotation

```python
q = (w, x, y, z)
-q = (-w, -x, -y, -z)

# Both rotate by same amount around same axis
```

**Why?**:
```
θ and (θ + 2π) are equivalent rotations
cos(θ/2) vs cos((θ + 2π)/2) = -cos(θ/2)
```

---

## Slerp: Spherical Linear Interpolation

### Formula

```python
def slerp(q₁, q₂, t):
    """
    Spherical linear interpolation between quaternions.

    Args:
        q₁, q₂: Quaternions
        t: Interpolation parameter (0 ≤ t ≤ 1)

    Returns:
        Interpolated quaternion
    """
    # Compute angle between quaternions
    dot = q₁.w*q₂.w + q₁.x*q₂.x + q₁.y*q₂.y + q₁.z*q₂.z

    # If negative, negate one quaternion (shorter path)
    if dot < 0:
        q₂ = -q₂
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        return lerp(q₁, q₂, t).normalize()

    # Slerp formula
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta

    return a * q₁ + b * q₂
```

### Properties

- **Constant angular velocity**: Smooth rotation
- **Shortest path**: Always takes geodesic on sphere
- **Normalized**: Result is unit quaternion if inputs are

---

## Comparison: Quaternions vs Euler Angles

| Aspect | Euler Angles | Quaternions |
|--------|--------------|-------------|
| **Parameters** | 3 (roll, pitch, yaw) | 4 (w, x, y, z) |
| **Gimbal Lock** | Yes (at ±90°) | No |
| **Composition** | Matrix multiply | Quaternion multiply |
| **Interpolation** | Non-smooth | Smooth (slerp) |
| **Memory** | 3 floats | 4 floats |
| **Efficiency** | Moderate | High |
| **Intuition** | High | Low |

---

## Implementation in ARIA

### File Structure

```
src/intelligence/
├── quaternion.py           # Core quaternion math
├── aria_exploration.py     # Exploration strategy
└── presets.py             # Preset configurations
```

### Key Classes

**Quaternion** (`quaternion.py`):
- Basic quaternion operations
- Rotation representations
- Conversions (axis-angle, Euler, matrix)

**QuaternionExplorer** (`aria_exploration.py`):
- Golden ratio spiral generation
- Multi-rotation exploration
- PCA alignment
- Score aggregation

### Usage in Retrieval

```python
# In local_rag_context_v7_guided_exploration.py

from intelligence.aria_exploration import QuaternionExplorer

# Initialize explorer
explorer = QuaternionExplorer(
    embedding_dim=384,
    num_rotations=100
)

# Explore with multiple iterations
results = explorer.explore_rotations(
    query_embedding=query_emb,
    document_embeddings=doc_embs,
    num_iterations=3,      # 3 rotations (from preset)
    angle_degrees=45.0     # From perspective
)

# results = [(doc_idx, aggregated_score), ...]
```

---

## Further Reading

### Books
- "Quaternions and Rotation Sequences" by Kuipers
- "Visualizing Quaternions" by Hanson

### Papers
- Hamilton, W.R. (1844). "On Quaternions"
- Shoemake, K. (1985). "Animating Rotation with Quaternion Curves" (SLERP)

### Online Resources
- 3Blue1Brown: Quaternions visualization
- Wikipedia: Quaternion mathematics
- Wolfram MathWorld: Quaternion algebra

---

## Appendix: Quaternion Identities

### Basic Identities

```
q + 0 = q
q * 1 = q
q * q⁻¹ = 1

(q₁ * q₂) * q₃ = q₁ * (q₂ * q₃)  (associative)
q₁ * q₂ ≠ q₂ * q₁                (non-commutative)

(q₁ * q₂)* = q₂* * q₁*           (anti-distributive)
(q₁ * q₂)⁻¹ = q₂⁻¹ * q₁⁻¹        (reverse order)
```

### Rotation Identities

```
q * v * q* = R(q) * v    (vector rotation)

q₂ * (q₁ * v * q₁*) * q₂* = (q₂ * q₁) * v * (q₂ * q₁)*

R(q₂) * R(q₁) = R(q₂ * q₁)
```

---

## Questions & Answers

**Q: Why not just use rotation matrices?**

A: Quaternions are more efficient (4 vs 9 parameters), no gimbal lock, and better for interpolation.

**Q: Can quaternions rotate in 4D space?**

A: Quaternions represent rotations in 3D. For 4D+, we apply the 3D rotation to subspaces.

**Q: Why golden ratio specifically?**

A: φ is the "most irrational" number, giving optimal sphere coverage without patterns.

**Q: How many rotations are needed?**

A: ARIA uses 100 rotation points × 2-3 iterations ≈ 200-300 total rotations per query.

**Q: Does this work for any embedding dimension?**

A: Yes! We rotate 3D subspaces within the full embedding space.

---

*For implementation details, see `src/intelligence/quaternion.py` and `src/intelligence/aria_exploration.py`*
