# Getting Started with ARIA

This guide will help you set up and run ARIA on your system.

---

## Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dontmindme369/aria.git
cd aria
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- numpy - Numerical computing
- scikit-learn - PCA and ML utilities
- sentence-transformers - Semantic embeddings (optional)
- PyYAML - Configuration parsing

### 3. Verify Installation

```bash
python3 aria_main.py --help
```

You should see the help text with available options.

---

## Configuration

### Default Configuration

ARIA uses `aria_config.yaml` for configuration. The default config works out of the box:

```yaml
paths:
  index_roots:
    - ~/Documents/knowledge  # Your knowledge base
  output_dir: ./aria_packs   # Where packs are saved
```

### Create Your Knowledge Base

Create a directory for your documents:

```bash
mkdir -p ~/Documents/knowledge
```

Add some test documents:

```bash
cat > ~/Documents/knowledge/test.md << 'TESTDOC'
# Test Document

This is a test document for ARIA.

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed.
TESTDOC
```

### Custom Configuration (Optional)

For custom settings, create a local config:

```bash
cp aria_config.yaml aria_config.local.yaml
```

Edit `aria_config.local.yaml` to customize:
- Knowledge base locations
- Output directories
- Retrieval parameters
- Postfilter settings
- Bandit behavior

---

## Your First Query

### Basic Query

```bash
python3 aria_main.py "What is machine learning?"
```

**Expected output:**
```
🎯 BANDIT SELECTION
  preset: balanced
  reason: Thompson sample: 0.85

📚 RETRIEVAL
  scanned: 5 files
  kept: 3 chunks

📊 METRICS
  coverage: 78.5%
  diversity: 0.95
  reward: 0.612

✓ COMPLETE (1.2s)
```

### With Custom Config

```bash
python3 aria_main.py "Your question" --config aria_config.local.yaml
```

### Force a Specific Preset

```bash
python3 aria_main.py "Your question" --preset deep
```

**Available presets:**
- `balanced` - Good all-around (default)
- `diverse` - Maximum source variety
- `focused` - Narrow, precise
- `creative` - Explore unusual connections
- `fast` - Quick, fewer chunks
- `thorough` - Comprehensive coverage
- `deep` - Maximum depth

### Custom Output Directory

```bash
python3 aria_main.py "Your question" --output ./my_custom_packs
```

---

## Understanding the Output

### CLI Output

ARIA provides rich, colorized output in the terminal:

```
🎯 BANDIT SELECTION
  preset: deep
  reason: epsilon-greedy random (ε=0.1)
  phase: exploitation

📚 RETRIEVAL
  Index Roots: ~/Documents/knowledge

🔬 POSTFILTER
  raw chunks: 28
  filtered chunks: 27
  removed: 1 (3.6%)

📊 METRICS
  • Retrieval
    total chunks: 27
    unique sources: 27
    diversity (MM): 1.00

  • Generation
    coverage: 85.71%
    exemplar fit: 0.689

✓ COMPLETE
  Reward: 0.689
  Total Time: 11.15s
```

### JSON Output

The final JSON output includes:

```json
{
  "preset": "deep",
  "pack": "path/to/last_pack.json",
  "filtered": "path/to/last_pack.filtered.json",
  "metrics": {
    "retrieval": {
      "total": 27,
      "unique_sources": 27,
      "diversity_mm": 1.0,
      "coverage_score": 0.857
    },
    "reward": 0.689
  },
  "quaternion_state": [...],
  "curiosity": { "gap_score": 0.65 },
  "uncertainty": { "epistemic": 0.37 }
}
```

### Output Files

ARIA creates timestamped directories for each query:

```
aria_packs/
└── 20251112-234120/
    └── what-is-machine-learning-abc123/
        ├── last_pack.json           # Raw retrieval results
        ├── last_pack.filtered.json  # Postfiltered results
        ├── run.meta.json            # Run metadata
        └── pack.stats.txt           # One-line telemetry
```

---

## Common Use Cases

### 1. Build a Personal Knowledge Base

```bash
# Organize your notes
mkdir -p ~/Documents/knowledge/{ml,programming,research}

# Add markdown notes
echo "# Python Basics" > ~/Documents/knowledge/programming/python.md

# Query across all notes
python3 aria_main.py "How do I use Python decorators?"
```

### 2. Index Code Repositories

```yaml
# aria_config.yaml
paths:
  index_roots:
    - ~/workspace/my-project/src
    - ~/workspace/my-project/docs
```

```bash
python3 aria_main.py "How does the authentication system work?"
```

### 3. Research Assistant

```yaml
# aria_config.yaml
paths:
  index_roots:
    - ~/Documents/papers
    - ~/Documents/notes
```

```bash
python3 aria_main.py "What are the latest approaches to transformer optimization?"
```

### 4. Multi-Domain Knowledge

```yaml
# aria_config.yaml
paths:
  index_roots:
    - ~/Documents/work
    - ~/Documents/personal
    - ~/workspace/projects
```

ARIA will search across all roots and combine results intelligently.

---

## Advanced Configuration

### Retrieval Settings

```yaml
retrieval:
  top_k: 128                      # Max chunks to retrieve
  per_file_limit: 10000           # Max chars per file
  max_per_file: 32                # Max chunks per file
  use_semantic: true              # Enable semantic search
  semantic_model: all-MiniLM-L6-v2
  semantic_weight: 0.7            # Semantic vs lexical balance
```

### Postfilter Settings

```yaml
postfilter:
  enabled: true
  quality_filter: false           # Disable aggressive filtering
  topic_filter: false
  diversity_filter: true          # Keep source variety
  max_per_source: 15              # Max chunks per file
  min_keep: 20                    # Always keep at least this many
  min_score: 0.001                # Min relevance (lenient)
```

### Bandit Settings

```yaml
bandit:
  enabled: true                   # Adaptive selection
  exploration_pulls: 20           # Explore first 20 queries
```

The bandit learns which presets work best for different query types.

---

## Python API

For programmatic use:

```python
from core.aria_core import ARIA

# Initialize
aria = ARIA(
    index_roots=["~/Documents/knowledge"],
    out_root="./my_packs"
)

# Run query
result = aria.query(
    "What is machine learning?",
    preset_override="deep"  # Optional
)

# Access metrics
print(f"Coverage: {result['metrics']['retrieval']['coverage_score']:.2%}")
print(f"Sources: {result['metrics']['retrieval']['unique_sources']}")
print(f"Reward: {result['metrics']['reward']:.3f}")

# Get pack path
pack_path = result['pack']
```

---

## Troubleshooting

### Empty Results

**Problem:** `"scanned": 0` in output

**Solution:**
1. Check paths exist: `ls -la ~/Documents/knowledge`
2. Add test documents (see "Create Your Knowledge Base" above)
3. Verify supported file types: .txt, .md, .py, .json, .yaml, etc.

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'retrieval'`

**Solution:**
```bash
# Run from project root
cd /path/to/aria
export PYTHONPATH=./src:$PYTHONPATH
python3 aria_main.py "test"
```

### Low Coverage Scores

**Problem:** Coverage consistently below 50%

**Solutions:**
- Add more relevant content to knowledge base
- Use `--preset thorough` for broader retrieval
- Lower `min_score` in postfilter config
- Check query matches content in knowledge base

### Slow Queries

**Problem:** Queries take >10 seconds

**Solutions:**
- Use `--preset fast` for quicker retrieval
- Reduce `top_k` in config (try 64 instead of 128)
- Disable semantic search if not needed: `use_semantic: false`
- Index smaller directories

---

## Next Steps

- **Read [README.md](README.md)** for feature overview
- **Check [PERFORMANCE.md](PERFORMANCE.md)** for benchmarks
- **Explore [SYSTEM_WORKING.md](SYSTEM_WORKING.md)** for test results
- **Review [aria_config.yaml](aria_config.yaml)** for all options

---

## Need Help?

- **Issues:** [GitHub Issues](https://github.com/dontmindme369/aria/issues)
- **Discussions:** [GitHub Discussions](https://github.com/dontmindme369/aria/discussions)

---

**Happy retrieving!** 🚀
