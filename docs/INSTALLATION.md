# ARIA Installation Guide

## Quick Install

### Prerequisites

- **Python 3.8+** (3.9+ recommended)
- **pip** package manager
- **Git** (for cloning repository)

### 1. Clone Repository

```bash
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA/aria
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical operations and quaternion math
- `sentence-transformers` - Semantic embeddings (all-MiniLM-L6-v2 model)
- `rank-bm25` - Lexical search
- `scikit-learn` - PCA and clustering
- `pyyaml` - Configuration files
- `tqdm` - Progress bars
- `watchdog` - File system monitoring (Student ARIA)

### 3. Configure ARIA

Edit `aria_config.yaml` to set your knowledge base location:

```yaml
paths:
  index_roots:
    - ~/Documents/knowledge        # Your knowledge base
    - ./sample_data               # Or use sample data
  output_dir: ./aria_packs
```

### 4. Verify Installation

```bash
# Run comprehensive test suite
python3 tests/comprehensive_test_suite.py

# Should see: 14/14 tests passing
```

---

## Advanced Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Acceleration (Optional)

For faster semantic search with GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then sentence-transformers will automatically use CUDA if available.

### Development Setup

For development with type checking:

```bash
pip install -r requirements.txt
pip install pylance mypy black isort pytest
```

---

## Configuration

### Minimal Config

Create `aria_config.yaml`:

```yaml
paths:
  index_roots:
    - ~/Documents/knowledge
  output_dir: ./aria_packs

retrieval:
  top_k: 64
  semantic_model: all-MiniLM-L6-v2
```

### Full Config

See `aria_config.yaml` for all configuration options including:
- Retrieval settings (BM25 + semantic)
- Postfilter parameters
- Perspective detection
- Bandit settings
- Monitoring and telemetry

---

## Student ARIA Setup (Optional)

For conversation corpus learning:

### 1. Install LM Studio

Download from: https://lmstudio.ai/

### 2. Configure Paths

Student ARIA automatically monitors:
- `~/.lmstudio/conversations/` - LM Studio conversations
- `../training_data/conversation_corpus/` - Captured corpus

### 3. Start Watcher

```bash
python3 aria_control_center.py
# Select option [2] - Start Student Watcher
```

---

## Troubleshooting

### Import Errors

If you see import errors:

```bash
# Make sure you're in the aria/ folder
cd ARIA/aria

# Verify Python path
python3 -c "import sys; print(sys.path)"
```

### Semantic Model Download

First run will download the sentence-transformers model (~90MB):

```bash
# Pre-download model
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Permission Issues

On Linux/Mac, you may need to make scripts executable:

```bash
chmod +x aria_control_center.py
chmod +x aria_main.py
```

### Missing Dependencies

If tests fail due to missing modules:

```bash
pip install --upgrade -r requirements.txt
```

---

## Platform-Specific Notes

### Windows

- Use `python` instead of `python3`
- Use backslashes in paths: `C:\Users\YourName\Documents\knowledge`
- Paths in config should use forward slashes or escaped backslashes

### macOS

- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```

### Linux

- Works out of the box on most distributions
- Tested on Ubuntu 20.04+, Debian 11+, Fedora 35+

---

## Verification

After installation, verify everything works:

```bash
# 1. Test imports
python3 -c "from core.aria_core import ARIA; print('✓ ARIA core loaded')"

# 2. Run tests
python3 tests/comprehensive_test_suite.py

# 3. Start control center
python3 aria_control_center.py

# 4. Try a query (Ctrl+C to exit control center first)
python3 aria_main.py "What is machine learning?"
```

---

## Next Steps

- Read [GETTING_STARTED.md](../GETTING_STARTED.md) for usage guide
- Read [CONTROL_CENTER_README.md](../CONTROL_CENTER_README.md) for control center features
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [API_REFERENCE.md](API_REFERENCE.md) for programmatic usage

---

## Getting Help

- **Issues**: https://github.com/dontmindme369/ARIA/issues
- **Discussions**: https://github.com/dontmindme369/ARIA/discussions
- **Documentation**: https://github.com/dontmindme369/ARIA/tree/main/aria/docs
