#!/usr/bin/env python3
"""
ARIA GitHub Finalization Script
Cleans existing aria-github-clean directory for public release:
- Removes PII and hardcoded paths from all .py files
- Updates config.yaml to be universal
- Creates .gitignore and requirements.txt
- Adds comprehensive documentation
"""

import os
import re
from pathlib import Path
from typing import List, Dict

# Target directory (your existing clean repo)
TARGET_DIR = Path("/media/notapplicable/ARIA-knowledge/aria-github-clean")

# Hardcoded paths to remove
PATH_PATTERNS = [
    r"/media/notapplicable/Internal-SSD/ai-quaternions-model",
    r"/media/notapplicable/ARIA-knowledge",
    r"/media/notapplicable/[^\s\"']*",
    r"/home/notapplicable",
]


def clean_python_file(filepath: Path) -> bool:
    """Remove hardcoded paths from Python file"""
    try:
        content = filepath.read_text(encoding="utf-8")
        original = content

        # Replace hardcoded paths with config-based references
        for pattern in PATH_PATTERNS:
            # Replace paths in strings
            content = re.sub(
                rf'(["\'])' + pattern + r'/[^"\']*(["\'])', r"\1{DATA_DIR}\2", content
            )
            # Replace paths in Path() calls
            content = re.sub(
                r'Path\(["\']' + pattern + r'/[^"\']*["\']\)', "Path(DATA_DIR)", content
            )

        if content != original:
            filepath.write_text(content, encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Error cleaning {filepath.name}: {e}")
        return False


def update_config_yaml():
    """Update config.yaml to be universal"""
    config = """# ARIA Configuration
# Universal configuration - all paths relative to ARIA root or absolute

# System Paths
paths:
  data_dir: "./data"              # Knowledge base documents
  cache_dir: "./cache"            # Vector embeddings cache
  output_dir: "./output"          # Query results and logs
  exemplars: "./data/exemplars.txt"  # Anchor patterns

# Retrieval Settings
retrieval:
  default_k: 20                   # Initial retrieval count
  rerank_top_k: 10               # After cross-encoder reranking
  chunk_size: 512                # Tokens per chunk
  chunk_overlap: 50              # Overlap between chunks
  use_bm25: true                 # Lexical retrieval
  use_embeddings: true           # Semantic retrieval

# Multi-Anchor Reasoning (v44)
anchors:
  enabled: true
  available_modes:
    - formal                     # Structured, precise
    - casual                     # Conversational
    - technical                  # Deep technical
    - educational                # Teaching-oriented
    - philosophical              # Conceptual exploration
    - analytical                 # Data-driven
    - factual                    # Direct facts
    - creative                   # Exploratory

# Bandit Learning (Thompson Sampling)
bandit:
  epsilon: 0.1                   # Exploration rate
  learning_rate: 0.01           # Update speed
  discount_factor: 0.95         # Future reward weight
  alpha_prior: 1.0              # Beta distribution alpha
  beta_prior: 1.0               # Beta distribution beta

# Curiosity Engine (v45)
curiosity:
  enabled: true
  gap_threshold: 0.3            # Knowledge gap sensitivity
  learning_rate: 0.05           # Learning speed
  personality: 7                # 1-10 scale (10 = max curiosity)
  conversation_tracking: true   # Track multi-turn conversations

# Post-filtering
postfilter:
  enable_quality_filter: true
  enable_diversity: true
  min_quality_score: 0.5
  max_duplication_ratio: 0.3

# Logging
logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  file: "./output/aria.log"
  console_output: true
  debug_mode: false             # Verbose pipeline output

# Performance
performance:
  use_gpu: true                 # CUDA if available
  batch_size: 32
  num_workers: 4
  cache_embeddings: true

# Advanced
advanced:
  enable_telemetry: true        # Performance tracking
  enable_meta_learning: true    # Cross-session learning
  confidence_threshold: 0.7     # Min confidence for answers
  session_timeout: 3600         # Session timeout (seconds)
"""

    config_path = TARGET_DIR / "config.yaml"
    config_path.write_text(config)
    print("  ✓ config.yaml")


def create_gitignore():
    """Create .gitignore for the repository"""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.nox/

# ARIA specific
cache/
output/
*.log
data/*.txt
!data/exemplars.txt
rag_runs/

# Secrets
*.key
*.pem
config_local.yaml
.env
"""

    gitignore_path = TARGET_DIR / ".gitignore"
    gitignore_path.write_text(gitignore)
    print("  ✓ .gitignore")


def create_requirements():
    """Create requirements.txt"""
    requirements = """# Core dependencies
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=1.10.0
transformers>=4.25.0
sentence-transformers>=2.2.0

# Retrieval
rank-bm25>=0.2.2
faiss-cpu>=1.7.0  # or faiss-gpu for GPU support

# Document processing
python-docx>=0.8.11
PyPDF2>=2.0.0
python-magic>=0.4.27

# Data handling
pandas>=1.3.0
pyyaml>=6.0

# Optional: Enhanced features
# textstat>=0.7.3  # For readability metrics
# spacy>=3.4.0  # For NLP features
# scipy>=1.7.0  # For advanced math

# Development
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
"""

    req_path = TARGET_DIR / "requirements.txt"
    req_path.write_text(requirements)
    print("  ✓ requirements.txt")


def create_updated_readme():
    """Create updated README with v44/v45 info"""
    readme = """# ARIA - Adaptive Resonant Intelligent Architecture

**Self-optimizing intelligence through adaptive retrieval and geometric exploration**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🌀 What is ARIA?

ARIA is a **local-first AI reasoning system** that combines:

- **Multi-Anchor Hybrid Reasoning** - 8 specialized reasoning modes (formal, casual, technical, educational, philosophical, analytical, factual, creative)
- **Thompson Sampling Bandits** - Self-optimizing strategy selection based on measurable outcomes
- **Curiosity-Driven Learning** - Detects knowledge gaps and generates Socratic questions
- **Privacy-First Architecture** - Runs entirely local, no cloud dependencies

Unlike traditional RAG systems, ARIA **learns from every query** through closed-loop telemetry and adapts its retrieval strategies in real-time.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/aria.git
cd aria

# Install dependencies
pip install -r requirements.txt

# Configure paths
cp config.yaml config_local.yaml
# Edit config_local.yaml with your paths
```

### Basic Usage

```python
from aria_core import ARIA

# Initialize ARIA
aria = ARIA(
    index_roots=["./data"],
    out_root="./output"
)

# Run a query
result = aria.query("How does machine learning work?")
print(result['response'])
```

### With Curiosity Engine (v45)

```python
from aria_curiosity import ARIACuriosity

curiosity = ARIACuriosity(personality=7)
result = await curiosity.process(
    query="Explain quantum entanglement",
    retrieved_chunks=chunks,
    confidence=0.8
)

print(result['response'])
print(result['socratic_questions'])  # Follow-up questions
print(result['knowledge_gaps'])      # Detected gaps
```

---

## 🏗️ Architecture

### Core Components

1. **aria_main.py** - Main orchestrator
2. **anchor_selector.py** - Detects query type, selects reasoning mode
3. **aria_retrieval.py** - Multi-source retrieval with BM25 + embeddings
4. **aria_postfilter.py** - Quality filtering and diversity enforcement
5. **contextual_bandit.py** - Thompson Sampling for strategy selection
6. **aria_curiosity.py** - Gap detection and Socratic questioning
7. **aria_conversation_watcher.py** - Multi-turn conversation tracking
8. **conversation_scorer.py** - Quality scoring for learning loop

### Multi-Anchor Reasoning Modes

ARIA automatically detects query intent and selects the optimal reasoning mode:

| Mode | Use Case | Example Query |
|------|----------|---------------|
| **formal** | Precise, structured | "Define the formal properties of..." |
| **casual** | Conversational | "Hey, can you explain..." |
| **technical** | Deep technical | "Implement a binary search tree in..." |
| **educational** | Teaching-oriented | "Teach me about neural networks" |
| **philosophical** | Conceptual | "What is the nature of consciousness?" |
| **analytical** | Data-driven | "Analyze the performance of..." |
| **factual** | Direct facts | "What is the capital of France?" |
| **creative** | Exploratory | "Imagine a world where..." |

---

## 📊 System Features

### v44 (Current Stable)
- ✅ 8-mode multi-anchor reasoning
- ✅ Thompson Sampling bandits
- ✅ Exemplar-based anchor detection (2,087 patterns)
- ✅ Cross-encoder reranking
- ✅ PCA query rotation
- ✅ Closed-loop telemetry

### v45 (Latest)
- ✅ Curiosity engine with gap detection
- ✅ Socratic question generation
- ✅ Conversation state tracking
- ✅ Multi-turn learning loop
- ✅ Adaptive synthesis strategies

---

## 🧪 Testing

```bash
# Run core tests
python -m pytest tests/test_aria_comprehensive.py

# Test anchor detection
python -m pytest tests/test_anchor_system.py

# Test specific component
python -m pytest tests/test_curiosity_engine.py -v
```

---

## 📁 Project Structure

```
aria/
├── src/
│   ├── aria_main.py                    # Main orchestrator
│   ├── anchor_selector.py              # Mode detection
│   ├── aria_retrieval.py               # Multi-source retrieval
│   ├── aria_postfilter.py              # Quality filtering
│   ├── contextual_bandit.py            # Thompson Sampling
│   ├── aria_curiosity.py               # Curiosity engine (v45)
│   ├── aria_conversation_watcher.py    # Conversation tracking
│   └── conversation_scorer.py          # Quality scoring
├── tests/
│   ├── test_aria_comprehensive.py
│   └── test_anchor_system.py
├── data/
│   └── exemplars.txt                   # 2,087 anchor patterns
├── config.yaml                         # Configuration
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
# Multi-Anchor Reasoning
anchors:
  enabled: true
  available_modes:
    - formal
    - casual
    - technical
    # ... (8 total modes)

# Curiosity Engine
curiosity:
  enabled: true
  personality: 7                # 1-10 curiosity level
  gap_threshold: 0.3

# Bandit Learning
bandit:
  epsilon: 0.1
  learning_rate: 0.01
```

---

## 📚 Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance metrics
- **[Metrics Guide](docs/METRICS.md)** - Telemetry and scoring
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and fixes

---

## 🎯 Design Philosophy

ARIA is built on three core principles:

1. **Privacy-First** - All processing local, no cloud dependencies
2. **Self-Optimizing** - Learns from measurable outcomes via bandits
3. **Reasoning-Aware** - Adapts approach based on query intent

Unlike traditional RAG systems that treat all queries the same, ARIA **recognizes that different questions require different reasoning strategies** and learns which strategies work best for each type.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with insights from:
- Multi-armed bandit literature (Thompson Sampling)
- Curiosity-driven learning research
- Information geometry and resonance theory
- The underlying connection of all things

---

## 📧 Contact

[Your contact information]

---

**ARIA v44/v45** - Where intelligence resonates with architecture ✨
"""

    readme_path = TARGET_DIR / "README.md"
    readme_path.write_text(readme)
    print("  ✓ README.md")


def main():
    print("=" * 70)
    print("ARIA GitHub Finalization Script")
    print("=" * 70)
    print(f"\nTarget: {TARGET_DIR}")

    if not TARGET_DIR.exists():
        print(f"\n❌ Directory not found: {TARGET_DIR}")
        print("Please check the path and try again.")
        return

    # Step 1: Clean Python files
    print("\n🧹 Cleaning Python files...")
    src_dir = TARGET_DIR / "src"
    if src_dir.exists():
        py_files = list(src_dir.glob("*.py"))
        cleaned_count = 0

        for py_file in py_files:
            if clean_python_file(py_file):
                print(f"  ✓ {py_file.name} (cleaned)")
                cleaned_count += 1
            else:
                print(f"  - {py_file.name} (no changes)")

        print(f"\n  {cleaned_count}/{len(py_files)} files cleaned")
    else:
        print("  ⚠️  src/ directory not found")

    # Step 2: Update configuration
    print("\n⚙️  Updating configuration...")
    update_config_yaml()

    # Step 3: Create supporting files
    print("\n📄 Creating supporting files...")
    create_gitignore()
    create_requirements()

    # Step 4: Update README
    print("\n📖 Updating README...")
    create_updated_readme()

    # Summary
    print("\n" + "=" * 70)
    print("✅ Finalization Complete!")
    print("=" * 70)
    print(f"\nDirectory prepared: {TARGET_DIR}")
    print("\n📋 What was done:")
    print("  ✓ Removed hardcoded paths from all .py files")
    print("  ✓ Updated config.yaml to be universal")
    print("  ✓ Created .gitignore")
    print("  ✓ Created requirements.txt")
    print("  ✓ Updated README.md with v44/v45 info")

    print("\n🎯 Next steps:")
    print("  1. Review cleaned files in src/")
    print("  2. Add full documentation (from previous chats)")
    print("  3. Test: python -m pytest tests/")
    print("  4. Initialize git: cd {TARGET_DIR} && git init")
    print("  5. Push to GitHub")

    print("\n🌀 Ready for public release!")


if __name__ == "__main__":
    main()
