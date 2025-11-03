# ARIA Integration Testing Mock Data

This directory contains all the mock data files used by the ARIA integration test suite.

## Directory Structure

```
data/
├── ml/                          # Machine Learning documents
│   ├── supervised_learning.txt  # 350 words - ML paradigm overview
│   ├── neural_networks.txt      # 280 words - Neural network architecture
│   └── deep_learning.txt        # 320 words - Deep learning innovations
├── physics/                     # Physics documents
│   ├── quantum_mechanics.txt    # 290 words - Quantum theory fundamentals
│   ├── relativity.txt           # 270 words - Einstein's theories
│   └── thermodynamics.txt       # 300 words - Laws of thermodynamics
├── philosophy/                  # Philosophy documents
│   ├── consciousness.txt        # 310 words - Philosophy of mind
│   ├── epistemology.txt         # 260 words - Theory of knowledge
│   └── ethics.txt               # 290 words - Moral philosophy
└── examples/                    # Test exemplars
    └── exemplars.txt            # 16 patterns + 6 meta-patterns
```

## File Contents

### Machine Learning (ml/)

**supervised_learning.txt**
- Training paradigm with labeled data
- Key concepts: features, labels, loss functions
- Common algorithms and applications

**neural_networks.txt**
- Computational models inspired by biology
- Architecture: input, hidden, output layers
- Training via backpropagation and gradient descent

**deep_learning.txt**
- Multi-layer hierarchical feature learning
- CNNs, RNNs, Transformers, transfer learning
- State-of-the-art performance across domains

### Physics (physics/)

**quantum_mechanics.txt**
- Fundamental theory of atomic/subatomic scales
- Wave-particle duality, uncertainty, superposition, entanglement
- Experimental verification and modern applications

**relativity.txt**
- Special relativity: time dilation, E=mc²
- General relativity: spacetime curvature, gravity
- Black holes, gravitational waves

**thermodynamics.txt**
- Four laws governing energy and entropy
- Conservation, entropy increase, absolute zero
- Statistical mechanics foundations

### Philosophy (philosophy/)

**consciousness.txt**
- Hard problem of consciousness and qualia
- Dualism, physicalism, panpsychism, functionalism
- Philosophy-neuroscience-cognitive science intersection

**epistemology.txt**
- Study of knowledge, belief, justification
- Sources: perception, reason, testimony
- Empiricism, rationalism, pragmatism

**ethics.txt**
- Moral philosophy frameworks
- Consequentialism, deontology, virtue ethics, care ethics
- Applied ethics domains

### Examples (examples/)

**exemplars.txt**
- 8 anchor mode examples (technical, formal, educational, etc.)
- 6 meta-pattern categories for query detection
- Format: `topic:: query -> response`

## Usage in Tests

The integration test suite automatically creates this mock data structure at:
```
./data/
```

Each test phase uses these documents:
1. **Retrieval tests** - Verify document scanning and chunking
2. **Postfilter tests** - Check diversity enforcement
3. **Anchor tests** - Validate mode detection accuracy
4. **Curiosity tests** - Test gap detection
5. **End-to-end tests** - Complete pipeline validation

## Total Statistics

- **9 documents** across 3 domains
- **~2,700 words** total content
- **22 exemplar patterns** for training
- **100% self-contained** - no external dependencies

## File Links

### Machine Learning
- [supervised_learning.txt](computer:///mnt/user-data/outputs/data/ml/supervised_learning.txt)
- [neural_networks.txt](computer:///mnt/user-data/outputs/data/ml/neural_networks.txt)
- [deep_learning.txt](computer:///mnt/user-data/outputs/data/ml/deep_learning.txt)

### Physics
- [quantum_mechanics.txt](computer:///mnt/user-data/outputs/data/physics/quantum_mechanics.txt)
- [relativity.txt](computer:///mnt/user-data/outputs/data/physics/relativity.txt)
- [thermodynamics.txt](computer:///mnt/user-data/outputs/data/physics/thermodynamics.txt)

### Philosophy
- [consciousness.txt](computer:///mnt/user-data/outputs/data/philosophy/consciousness.txt)
- [epistemology.txt](computer:///mnt/user-data/outputs/data/philosophy/epistemology.txt)
- [ethics.txt](computer:///mnt/user-data/outputs/data/philosophy/ethics.txt)

### Examples
- [exemplars.txt](computer:///mnt/user-data/outputs/data/examples/exemplars.txt)

---

**Note**: The test suite creates these files programmatically. These standalone files are provided for reference and manual testing.
