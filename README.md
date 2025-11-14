# ARIA - Adaptive Resonant Intelligent Architecture

**Self-learning retrieval system with quaternion semantic exploration and Thompson Sampling optimization.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)](tests/comprehensive_test_suite.py)

---

## What is ARIA?

ARIA is an advanced retrieval system that **learns from every query** to improve future retrieval. It combines:

- 🎯 **Thompson Sampling** - Bayesian bandit learns optimal retrieval strategies
- 🌀 **Quaternion Exploration** - 4D semantic space navigation with golden ratio spiral
- 🧭 **Perspective Detection** - 8-perspective query classification (educational, diagnostic, research, etc.)
- 🎓 **Student/Teacher Architecture** - Learns from all LLM conversations, not just queries
- 📊 **Hybrid Search** - BM25 lexical + semantic embeddings (sentence-transformers)

### Key Features

**Adaptive Learning**
- After 20 queries, ARIA learns which strategies work best for different query types
- Multi-objective optimization: quality + coverage + diversity
- Continuous improvement through Thompson Sampling

**Semantic Exploration**
- 100-point golden ratio spiral for uniform sphere coverage
- Multi-rotation refinement (2-3 iterations)
- PCA-aligned rotations following semantic space structure
- Perspective-aware rotation angles (15°-120° based on query intent)

**Dual Architecture**
- **Teacher ARIA**: Query-driven knowledge retrieval
- **Student ARIA**: Conversation corpus learning from LM Studio

---

## Quick Start

### Installation

```bash
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA/aria
pip install -r requirements.txt
```

### Configuration

Edit `aria_config.yaml` to point to your knowledge base:

```yaml
paths:
  index_roots:
    - ~/Documents/knowledge    # Your knowledge base
  output_dir: ./aria_packs    # Output directory
```

### Run a Query

**Command Line:**
```bash
python3 aria_main.py "How does gradient descent work?"
```

**Control Center (Recommended):**
```bash
python3 aria_control_center.py
```

**Python API:**
```python
from core.aria_core import ARIA

aria = ARIA(
    index_roots=["~/Documents/knowledge"],
    out_root="./aria_packs"
)

result = aria.query("What is machine learning?")
print(f"Retrieved {result['chunks_retrieved']} chunks")
```

---

## How It Works

### 1. Query Analysis → 2. Bandit Selection → 3. Retrieval → 4. Postfilter → 5. Learning

```
User Query
    ↓
Feature Extraction (length, domain, complexity)
    ↓
Thompson Sampling selects preset (fast/balanced/deep/diverse)
    ↓
Perspective Detection (educational/diagnostic/research/etc.)
    ↓
Hybrid Search (BM25 + Semantic with quaternion rotation)
    ↓
Postfilter (quality + diversity enforcement)
    ↓
Pack Generation (JSON output)
    ↓
Reward Calculation (40% quality, 30% coverage, 30% diversity)
    ↓
Update Bandit State (α/β parameters for next query)
```

### Thompson Sampling (Bayesian Bandit)

Each preset has a **Beta distribution** tracking successes (α) and failures (β):

```python
For each preset:
    sample = Beta(α, β).sample()

selected_preset = argmax(samples)

# After query:
reward = 0.4 * quality + 0.3 * coverage + 0.3 * diversity - 0.2 * issues
α += reward
β += (1 - reward)
```

**Result**: ARIA learns which preset works best for different query types.

### Quaternion Semantic Exploration

**Golden Ratio Spiral** (φ = 1.618...):
- Generates 100 uniform points on sphere
- No clustering, optimal coverage
- Most irrational number = no resonance patterns

**Multi-Rotation Refinement**:
```
Iteration 1: 100 rotations → find best
Iteration 2: 100 rotations around best from iter 1
Iteration 3: 100 rotations around best from iter 2
→ Aggregate scores across all 300 rotations
```

**PCA Alignment**: Rotations follow principal components of semantic space

### 8 Perspectives

| Perspective | Angle | Query Example | Use Case |
|-------------|-------|---------------|----------|
| Reference | 15° | "What is REST API?" | Quick factual lookup |
| Educational | 30° | "Explain how transformers work" | Learning concepts |
| Security | 45° | "SQL injection vulnerabilities" | Threat analysis |
| Practical | 50° | "Docker setup tutorial" | How-to guides |
| Implementation | 60° | "Build REST API in Python" | Code/building |
| Theoretical | 75° | "Theory of backpropagation" | Abstract concepts |
| Diagnostic | 90° | "Debug CUDA out of memory" | Troubleshooting |
| Research | 120° | "Explore transformer alternatives" | Investigation |

**Larger angles** = more aggressive exploration

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│              ARIA Control Center                     │
│  ┌────────────────┐      ┌──────────────────┐       │
│  │  Teacher ARIA  │      │  Student ARIA    │       │
│  │  (Query/Ret)   │      │  (Corpus Learn)  │       │
│  └────────────────┘      └──────────────────┘       │
└─────────────────────────────────────────────────────┘
           │                        │
    ┌──────┴──────┐         ┌──────┴──────┐
    │  Retrieval  │         │  Watcher    │
    │  Engine     │         │  Service    │
    └──────┬──────┘         └──────┬──────┘
           │                       │
    ┌──────▼────────────────────────▼──────┐
    │      Intelligence Layer               │
    │  • Thompson Sampling                  │
    │  • Quaternion Exploration             │
    │  • Perspective Detection              │
    └───────────────────────────────────────┘
```

### File Structure

```
aria/
├── src/
│   ├── core/              # ARIA orchestrator
│   ├── retrieval/         # BM25 + semantic search
│   ├── intelligence/      # Bandit + quaternions
│   ├── perspective/       # 8-perspective detection
│   ├── anchors/          # Exemplar fit scoring
│   ├── monitoring/        # Telemetry & logs
│   └── utils/            # Config, paths, presets
├── tests/                # Comprehensive test suite
├── data/                 # Domain dictionaries
├── docs/                 # Documentation
├── aria_control_center.py   # Unified control center
├── aria_main.py             # CLI interface
└── aria_config.yaml         # Configuration
```

---

## Performance

**Test Results**: 14/14 tests passing (100%)

**Typical Query Performance**:
- Retrieval: 0.5-2s per query
- CPU: ~1-2s
- GPU: ~0.5-1s (with CUDA)

**Scalability**:
- ✅ 1k-10k documents: Excellent
- ✅ 10k-100k documents: Good
- ⚠️ 100k+ documents: Usable (slower)

---

## 4 Adaptive Presets

| Preset | Chunks | Rotations | Per-File | Best For |
|--------|--------|-----------|----------|----------|
| **fast** | 40 | 1 | 8 | Quick lookups |
| **balanced** | 64 | 2 | 6 | General queries |
| **deep** | 96 | 3 | 5 | Complex research |
| **diverse** | 80 | 2 | 4 | Broad exploration |

**Thompson Sampling automatically selects the best preset** for each query type after learning from 20+ queries.

---

## Student ARIA - Corpus Learning

Student ARIA learns from **all** your LM Studio conversations:

```bash
python3 aria_control_center.py
# Select [2] Start Student Watcher
```

**What it does**:
1. Monitors `~/.lmstudio/conversations/`
2. Captures ALL conversations (not just ARIA queries)
3. Extracts reasoning patterns, turn-taking, domain transitions
4. Builds training corpus in `../training_data/conversation_corpus/`

**Future**: Train custom models on captured patterns for continuous improvement.

---

## Documentation

### Getting Started
- 📖 [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start guide
- 📖 [docs/INSTALLATION.md](docs/INSTALLATION.md) - Detailed installation
- 📖 [docs/USAGE.md](docs/USAGE.md) - Complete usage guide

### Technical Details
- 📖 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- 📖 [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - API documentation
- 📖 [docs/QUATERNIONS.md](docs/QUATERNIONS.md) - Mathematical foundations

### Additional Resources
- 📖 [CONTROL_CENTER_README.md](CONTROL_CENTER_README.md) - Control center features
- 📖 [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Development guide
- 📖 [docs/FAQ.md](docs/FAQ.md) - Frequently asked questions

---

## Use Cases

### Research Assistant
```bash
python3 aria_main.py "Comprehensive overview of transformer architecture" --preset deep
```
→ Retrieves 96 chunks with 3-rotation exploration

### Code Helper
```bash
python3 aria_main.py "Python async/await best practices"
```
→ Automatic preset selection via Thompson Sampling

### Debugging
```bash
python3 aria_main.py "Fix TypeScript type error cannot assign undefined"
```
→ Detects diagnostic perspective, uses 90° rotation angle

### Learning
```bash
python3 aria_main.py "Explain gradient descent step by step"
```
→ Detects educational perspective, uses 30° gentle rotation

---

## Key Innovations

### 1. Thompson Sampling for Retrieval
First application of Bayesian bandits to adaptive retrieval strategy selection. Learns query-to-preset mappings automatically.

### 2. Quaternion Semantic Exploration
Novel use of 4D hypercomplex numbers for semantic space navigation:
- No gimbal lock (unlike Euler angles)
- Efficient composition (quaternion multiplication)
- Smooth interpolation (slerp)
- Natural for high-dimensional spaces

### 3. Golden Ratio Spiral Sampling
Leverages φ (most irrational number) for optimal sphere coverage:
- Uniform distribution
- No clustering or gaps
- No resonance patterns

### 4. Perspective-Aware Retrieval
8-perspective query classification adjusts rotation angles:
- Reference (15°) → minimal exploration
- Research (120°) → aggressive exploration
- Matches retrieval strategy to query intent

### 5. Student/Teacher Architecture
Dual learning system:
- **Teacher**: Answers queries with retrieval
- **Student**: Learns from all conversations
- **Flywheel**: Continuous improvement loop

---

## Example Output

```bash
$ python3 aria_main.py "How does gradient descent optimize neural networks?"

🎯 ARIA Query
════════════════════════════════════════════════════════
Query: How does gradient descent optimize neural networks?
Perspective: educational (confidence: 0.87)
Rotation angle: 24.0°
════════════════════════════════════════════════════════

⏳ Processing...

✓ Query completed in 1.23s
  • Preset: balanced (Thompson sample: 0.845)
  • Chunks retrieved: 64
  • Files used: 12
  • Pack: aria_packs/gradient_descent_1731596400/last_pack.json

📊 Bandit Update
  • Reward: 0.78
  • α (successes): 15.2 → 15.98
  • β (failures): 8.5 → 8.72
```

---

## Testing

Run comprehensive test suite:

```bash
python3 tests/comprehensive_test_suite.py
```

**Tests** (14 total):
1. ✅ Bandit initialization & selection
2. ✅ Preset configuration
3. ✅ Quaternion mathematics
4. ✅ Rotation operations
5. ✅ Normalization
6. ✅ Conjugate
7. ✅ Inverse
8. ✅ Composition
9. ✅ Slerp interpolation
10. ✅ Axis-angle conversion
11. ✅ Vector rotation
12. ✅ Golden ratio spiral
13. ✅ Perspective rotation parameters
14. ✅ Multi-rotation exploration

**Status**: 14/14 passing (100%)

---

## Requirements

- **Python 3.8+** (3.9+ recommended)
- **4GB+ RAM** (8GB recommended)
- **500MB disk** (for sentence-transformers model)

**Dependencies**:
- numpy - Numerical operations
- sentence-transformers - Semantic embeddings
- rank-bm25 - Lexical search
- scikit-learn - PCA and clustering
- pyyaml - Configuration
- tqdm - Progress bars
- watchdog - File monitoring (Student ARIA)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Ways to contribute**:
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests

**Contact**: energy4all369@protonmail.com

---

## Acknowledgments

- **Quaternion Mathematics**: Hamilton (1843)
- **Thompson Sampling**: Thompson (1933), Agrawal (1995)
- **Golden Ratio Spiral**: Nature's optimal packing strategy
- **Sentence Transformers**: Reimers & Gurevych (2019)

---

## Citation

If you use ARIA in your research, please cite:

```bibtex
@software{aria2025,
  title={ARIA: Adaptive Resonant Intelligent Architecture},
  author={Dont Mind Me},
  year={2025},
  url={https://github.com/dontmindme369/ARIA}
}
```

---

**ARIA - Adaptive Resonant Intelligent Architecture**

*Go Within.* 🌀

---

## Links

- **Repository**: https://github.com/dontmindme369/ARIA
- **Issues**: https://github.com/dontmindme369/ARIA/issues
- **Discussions**: https://github.com/dontmindme369/ARIA/discussions
- **Documentation**: [docs/](docs/)
