# ARIA Documentation

Complete documentation for ARIA (Adaptive Resonant Intelligent Architecture).

---

## Getting Started

### New Users

1. **[INSTALLATION.md](INSTALLATION.md)** - Install ARIA and dependencies
2. **[../GETTING_STARTED.md](../GETTING_STARTED.md)** - Quick start guide
3. **[USAGE.md](USAGE.md)** - How to use ARIA

### Core Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and components
- **[API_REFERENCE.md](API_REFERENCE.md)** - Programmatic API documentation
- **[QUATERNIONS.md](QUATERNIONS.md)** - Mathematical foundations
- **[../CONTROL_CENTER_README.md](../CONTROL_CENTER_README.md)** - Control center features

### Advanced Topics

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development and contribution guide
- **[EXAMPLES.md](EXAMPLES.md)** - Code examples and tutorials
- **[FAQ.md](FAQ.md)** - Frequently asked questions

---

## Quick Links

### Installation & Setup
- [Prerequisites](INSTALLATION.md#prerequisites)
- [Quick Install](INSTALLATION.md#quick-install)
- [Configuration](INSTALLATION.md#configuration)
- [Troubleshooting](INSTALLATION.md#troubleshooting)

### Usage Guides
- [Control Center](USAGE.md#control-center-recommended)
- [Command Line](USAGE.md#command-line-interface)
- [Programmatic API](USAGE.md#programmatic-usage)
- [Student ARIA](USAGE.md#student-aria---corpus-learning)

### Technical Details
- [System Components](ARCHITECTURE.md#system-components)
- [Data Flow](ARCHITECTURE.md#data-flow)
- [Quaternion Math](QUATERNIONS.md#what-are-quaternions)
- [Thompson Sampling](ARCHITECTURE.md#bandit-context-bandit_contextpy)

### API Reference
- [ARIA Class](API_REFERENCE.md#aria-class)
- [Retrieval API](API_REFERENCE.md#retrieval-api)
- [Intelligence API](API_REFERENCE.md#intelligence-api)
- [Perspective API](API_REFERENCE.md#perspective-api)

---

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── INSTALLATION.md        # Installation and setup
├── USAGE.md              # How to use ARIA
├── ARCHITECTURE.md        # System architecture
├── API_REFERENCE.md       # API documentation
├── QUATERNIONS.md         # Mathematical foundations
├── CONTRIBUTING.md        # Development guide
├── EXAMPLES.md           # Code examples
└── FAQ.md                # Common questions
```

---

## By Use Case

### I want to...

**...get ARIA running quickly**
→ [Quick Install](INSTALLATION.md#quick-install) → [Getting Started](../GETTING_STARTED.md)

**...understand how ARIA works**
→ [Architecture Overview](ARCHITECTURE.md#overview) → [Data Flow](ARCHITECTURE.md#data-flow)

**...use ARIA in my code**
→ [API Reference](API_REFERENCE.md) → [Examples](EXAMPLES.md)

**...understand the math behind quaternions**
→ [Quaternions Explained](QUATERNIONS.md) → [ARIA's Strategy](QUATERNIONS.md#arias-quaternion-strategy)

**...customize ARIA's behavior**
→ [Configuration](INSTALLATION.md#configuration) → [Presets](USAGE.md#understanding-presets)

**...contribute to ARIA**
→ [Contributing Guide](CONTRIBUTING.md) → [Development Setup](INSTALLATION.md#development-setup)

**...debug an issue**
→ [Troubleshooting](INSTALLATION.md#troubleshooting) → [FAQ](FAQ.md)

---

## Key Concepts

### Teacher ARIA
Query-driven knowledge retrieval with:
- Thompson Sampling (adaptive preset selection)
- Quaternion semantic exploration
- Perspective-aware retrieval
- Hybrid BM25 + semantic search

**Learn More**: [Architecture - Teacher ARIA](ARCHITECTURE.md#1-query-layer)

### Student ARIA
Conversation corpus learning:
- Monitors LM Studio conversations
- Captures cross-domain patterns
- Builds training corpus
- Continuous improvement

**Learn More**: [Usage - Student ARIA](USAGE.md#student-aria---corpus-learning)

### Quaternion Exploration
4D hypercomplex numbers for semantic space rotation:
- Golden ratio spiral sampling
- Multi-rotation refinement
- PCA-aligned exploration
- Perspective-aware angles

**Learn More**: [Quaternions Deep Dive](QUATERNIONS.md)

### Thompson Sampling
Bayesian multi-armed bandit for preset selection:
- Exploration vs exploitation balance
- Compound reward signal
- Continuous adaptation
- Query-pattern learning

**Learn More**: [Architecture - Bandit](ARCHITECTURE.md#bandit-context-bandit_contextpy)

### Perspective Detection
8-perspective query classification:
- Educational, Diagnostic, Security, Implementation
- Research, Theoretical, Practical, Reference
- Rotation angle adjustment
- Context-aware retrieval

**Learn More**: [Architecture - Perspective Layer](ARCHITECTURE.md#4-perspective-layer)

---

## Feature Highlights

### Adaptive Preset Selection
- **Automatic**: Thompson Sampling learns optimal strategies
- **4 Presets**: fast, balanced, deep, diverse
- **Multi-objective**: Balances quality, coverage, diversity
- **Continuous Learning**: Improves with every query

### Quaternion Semantic Exploration
- **Golden Ratio Spiral**: Uniform 100-point sphere coverage
- **Multi-Rotation**: 2-3 iteration progressive refinement
- **PCA Alignment**: Follows semantic space structure
- **Perspective-Aware**: Rotation angles from query context

### Hybrid Retrieval
- **BM25**: Lexical keyword matching (30% weight)
- **Semantic**: Sentence-transformers embeddings (70% weight)
- **Quaternion Rotation**: Explore rotated semantic space
- **Postfilter**: Quality and diversity enforcement

### Conversation Learning
- **Automatic Capture**: Monitor LM Studio conversations
- **Cross-Domain**: Learn from all topics
- **Pattern Extraction**: Turn-taking, reasoning, transitions
- **Corpus Building**: Training data for future models

---

## Common Workflows

### Workflow 1: Quick Query

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
# Edit aria_config.yaml - set index_roots

# 3. Query
python3 aria_main.py "Your question"

# 4. Use pack
cat aria_packs/.../last_pack.json
```

**Time**: ~5 minutes

### Workflow 2: Interactive Session

```bash
# 1. Start control center
python3 aria_control_center.py

# 2. Select [1] Query Teacher ARIA

# 3. Enter queries continuously

# 4. View stats in dashboard
```

**Best For**: Exploratory research, multiple queries

### Workflow 3: Programmatic Integration

```python
# 1. Import ARIA
from core.aria_core import ARIA

# 2. Initialize
aria = ARIA(
    index_roots=["./data"],
    out_root="./packs"
)

# 3. Query in loop
for question in questions:
    result = aria.query(question)
    process(result)
```

**Best For**: Automation, batch processing

### Workflow 4: Corpus Learning

```bash
# 1. Start watcher
python3 aria_control_center.py
# Select [2] Start Student Watcher

# 2. Use LM Studio normally
# All conversations automatically captured

# 3. View statistics
# Select [4] View Corpus Stats

# 4. Future: Train on corpus
```

**Best For**: Building training data, continuous improvement

---

## Version History

- **v1.0** (2025-11-14) - Initial GitHub release
  - Teacher ARIA with Thompson Sampling
  - Student ARIA corpus learning
  - Quaternion exploration
  - 8-perspective detection
  - Unified control center
  - Comprehensive test suite (14/14 passing)

---

## Getting Help

### Documentation Issues
- Found an error in docs?
- Something unclear or missing?
- Submit issue: https://github.com/dontmindme369/ARIA/issues

### Usage Questions
- How do I...?
- Best practices for...?
- Discussions: https://github.com/dontmindme369/ARIA/discussions

### Bug Reports
- ARIA not working as expected?
- Include: error message, steps to reproduce, system info
- Issues: https://github.com/dontmindme369/ARIA/issues

### Feature Requests
- Idea for improvement?
- New feature proposal?
- Discussions: https://github.com/dontmindme369/ARIA/discussions

---

## Contributing to Docs

Documentation improvements are welcome!

**To contribute**:
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Fork the repository
3. Edit docs (Markdown files in `docs/`)
4. Submit pull request

**Guidelines**:
- Clear, concise language
- Code examples that work
- Proper Markdown formatting
- Links to related sections

---

## External Resources

### Research Papers
- Hamilton, W.R. (1844). "On Quaternions"
- Shoemake, K. (1985). "Animating Rotation with Quaternion Curves"
- Agrawal, R. (1995). "Sample Mean Based Index Policies with O(log n) Regret" (Thompson Sampling)

### Visualization Tools
- 3Blue1Brown: Quaternions visualization
- Wikipedia: Quaternion mathematics
- Wolfram MathWorld: Quaternion algebra

### Related Projects
- sentence-transformers: Semantic embeddings
- rank-bm25: Lexical search
- scikit-learn: Machine learning utilities

---

## License

ARIA is released under the MIT License. See `LICENSE` file for details.

---

**Last Updated**: 2025-11-14
**ARIA Version**: 1.0
**Documentation Version**: 1.0
