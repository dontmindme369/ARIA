# ARIA Installation Guide

**Complete setup instructions for local deployment**

---

## 📋 Prerequisites

### Required
- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended)
- 2GB free disk space for dependencies
- Git (for cloning repository)

### Optional
- CUDA-capable GPU (for faster embeddings)
- 10GB+ storage (for large knowledge bases)

---

## 🚀 Quick Start (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Paths

**CRITICAL**: Edit `config.yaml` to set your local paths.

```bash
cp config.yaml config.yaml.example  # Backup original
nano config.yaml  # or vim, code, etc.
```

**Required edits in config.yaml**:

```yaml
paths:
  # WHERE YOUR DOCUMENTS ARE STORED
  data_dir: "./data"  # ← CHANGE THIS to your docs location
  
  # WHERE TO CACHE EMBEDDINGS (SPEEDS UP QUERIES)
  cache_dir: "./cache"  # ← CHANGE THIS to your cache location
  
  # WHERE TO SAVE QUERY RESULTS
  output_dir: "./output"  # ← CHANGE THIS to your output location
  
  # WHERE EXPLORATION STATE IS SAVED
  quaternion_state_path: "./state/quaternion_states.jsonl"  # ← CHANGE THIS
  
  # ANCHOR PATTERN FILE (746 patterns)
  exemplars: "./data/exemplars.txt"  # ← Should exist after setup
```

### 5. Create Required Directories

```bash
mkdir -p data cache output state/exploration/quaternion
```

### 6. Add Your Documents

```bash
# Copy your documents to data directory
cp -r /path/to/your/documents/* ./data/

# Supported formats: .txt, .md, .pdf, .docx, .html
```

### 7. Test Installation

```bash
# Run test query
python src/aria_main.py "test query" --config config.yaml

# Should see:
# [ARIA] Multi-Anchor System enabled
# [ARIA] 🌀 Exploration System enabled
# [ARIA] ✅ Query complete
```

---

## 📁 Directory Structure After Setup

Your ARIA directory should look like:

```
ARIA/
├── src/                          # Core system (don't modify)
│   ├── aria_main.py
│   ├── quaternion_state.py
│   ├── pca_exploration.py
│   └── ... (20+ files)
│
├── anchors/                      # 8 reasoning mode instructions
│   ├── technical.md
│   ├── formal.md
│   └── ...
│
├── data/                         # ← YOUR DOCUMENTS GO HERE
│   ├── exemplars.txt            # (746 anchor patterns)
│   ├── your_file_1.txt
│   ├── your_file_2.pdf
│   └── subdirs_ok/
│
├── cache/                        # ← EMBEDDINGS CACHED HERE
│   └── (auto-generated)
│
├── output/                       # ← QUERY RESULTS SAVED HERE
│   └── rag_runs/
│       └── aria/
│
├── state/                        # ← QUATERNION STATE SAVED HERE
│   └── exploration/
│       └── quaternion/
│           └── quaternion_states.jsonl
│
├── venv/                         # Python virtual environment
├── config.yaml                   # ← YOUR CONFIGURATION
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration Guide

### config.yaml - Complete Reference

**Section 1: Paths** (MUST EDIT)

```yaml
paths:
  # Your knowledge base documents
  # Absolute: ~/documents
  # Relative: ./data (relative to ARIA directory)
  data_dir: "./data"
  
  # Embedding cache (speeds up repeat queries)
  # Can be large (100MB-5GB depending on corpus)
  cache_dir: "./cache"
  
  # Query results and telemetry
  output_dir: "./output"
  
  # Quaternion state for cross-query memory
  quaternion_state_path: "./state/quaternion_states.jsonl"
  
  # 746 anchor detection patterns
  exemplars: "./data/exemplars.txt"
```

**Section 2: Retrieval** (can customize)

```yaml
retrieval:
  default_k: 20                   # Initial chunks to retrieve
  rerank_top_k: 10               # After reranking
  chunk_size: 512                # Tokens per chunk
  chunk_overlap: 50              # Overlap between chunks
  use_bm25: true                 # Lexical retrieval
  use_embeddings: true           # Semantic retrieval
```

**Section 3: Exploration** (can customize)

```yaml
exploration:
  enabled: true                   # Enable quaternion+PCA+spiral
  quaternion_decay: 0.5          # Momentum decay rate
  pca_enabled: true              # Multi-perspective exploration
  pca_components: 32             # PCA dimensionality
  golden_ratio_samples: 13       # Spiral samples (Fibonacci #)
  exploration_radius: 0.3        # Search radius in semantic space
```

**Section 4: Anchors** (can customize modes)

```yaml
anchors:
  enabled: true
  available_modes:
    - technical
    - formal
    - educational
    - philosophical
    - analytical
    - factual
    - creative
    - casual
```

**Section 5: Performance** (adjust for your system)

```yaml
performance:
  use_gpu: true                   # Use GPU if available (faster)
  batch_size: 32                  # Batch size for embeddings
  num_workers: 4                  # Parallel workers
  cache_embeddings: true          # Cache for speed
```

---

## 🔧 Path Configuration Examples

### Example 1: Default (Relative Paths)

**Best for**: Most users, simple setup

```yaml
paths:
  data_dir: "./data"
  cache_dir: "./cache"
  output_dir: "./output"
  quaternion_state_path: "./state/quaternion_states.jsonl"
  exemplars: "./data/exemplars.txt"
```

**Advantages**:
- ✅ Works anywhere you move ARIA folder
- ✅ Easy to understand
- ✅ No absolute path dependencies

**Setup**:
```bash
mkdir -p data cache output state/exploration/quaternion
cp /your/docs/* ./data/
```

### Example 2: Absolute Paths

**Best for**: Fixed installation, multiple ARIA instances

```yaml
paths:
  data_dir: "~/documents/aria-knowledge"
  cache_dir: "~/.cache/aria"
  output_dir: "~/aria-output"
  quaternion_state_path: "~/.cache/aria/quaternion_states.jsonl"
  exemplars: "~/documents/aria-knowledge/exemplars.txt"
```

**Advantages**:
- ✅ Can run ARIA from any directory
- ✅ Centralized data location
- ✅ Separate ARIA instances can share data

**Setup**:
```bash
mkdir -p ~/documents/aria-knowledge
mkdir -p ~/.cache/aria/exploration/quaternion
mkdir -p ~/aria-output

# Use your actual documents
ln -s ~/actual-docs ~/documents/aria-knowledge
```

### Example 3: Mixed (Relative + Absolute)

**Best for**: Shared corpus, local processing

```yaml
paths:
  data_dir: "/shared/knowledge-base"        # Shared across users
  cache_dir: "./cache"                      # Local cache
  output_dir: "./output"                    # Local output
  quaternion_state_path: "./state/quaternion_states.jsonl"  # Local state
  exemplars: "/shared/knowledge-base/exemplars.txt"
```

**Advantages**:
- ✅ Share large corpus without duplication
- ✅ Separate caches prevent conflicts
- ✅ Each user has own exploration state

---

## 🐍 Python Path Configuration

### Running from Different Directories

If you run ARIA from outside its directory, set `PYTHONPATH`:

```bash
# Option 1: Export PYTHONPATH
export PYTHONPATH=/path/to/ARIA:$PYTHONPATH
python /path/to/ARIA/src/aria_main.py "query"

# Option 2: Use absolute path in command
cd /anywhere
python /path/to/ARIA/src/aria_main.py "query" --config /path/to/ARIA/config.yaml
```

### Fixing "ModuleNotFoundError"

If you see: `ModuleNotFoundError: No module named 'aria_...'`

**Solution**:
```bash
# Make sure you're in ARIA directory
cd /path/to/ARIA

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/ARIA/src:$PYTHONPATH
```

---

## 🗂️ Data Directory Setup

### Supported File Formats

ARIA can read:
- ✅ `.txt` - Plain text
- ✅ `.md` - Markdown
- ✅ `.pdf` - PDF documents
- ✅ `.docx` - Word documents
- ✅ `.html` - HTML files

### Organizing Your Documents

**Option 1: Flat Structure** (simple)
```
data/
├── doc1.txt
├── doc2.pdf
├── doc3.md
└── doc4.docx
```

**Option 2: Categorized** (organized)
```
data/
├── research/
│   ├── paper1.pdf
│   └── paper2.pdf
├── notes/
│   ├── note1.md
│   └── note2.md
└── books/
    └── book1.pdf
```

**Option 3: Large Corpus** (scalable)
```
data/
├── 2024/
│   ├── january/
│   ├── february/
│   └── ...
├── 2025/
└── archived/
```

**ARIA automatically searches all subdirectories** - organize however you want!

### Recommended: Separate Corpus from ARIA

```bash
# Keep corpus separate
/your/documents/knowledge-base/
  ├── research/
  ├── notes/
  └── books/

# Configure ARIA to point there
# In config.yaml:
data_dir: "/your/documents/knowledge-base"
```

**Why separate?**
- Can upgrade ARIA without moving documents
- Can use same corpus with multiple ARIA instances
- Cleaner git updates (no accidental commits of personal docs)

---

## 🎯 First Run Checklist

Before your first query, verify:

- [ ] Virtual environment activated (`source venv/bin/activate`)
- [ ] Dependencies installed (`pip list | grep sentence-transformers`)
- [ ] `config.yaml` paths configured correctly
- [ ] Directories created (`ls -la data cache output state`)
- [ ] Documents copied to data directory (`ls data/*.txt`)
- [ ] `exemplars.txt` exists (`wc -l data/exemplars.txt` should show ~746)

### Test Installation

```bash
# Comprehensive test
python -c "
from src.aria_main import ARIA
from src.anchor_selector import AnchorSelector
from src.aria_exploration import ExplorationManager
print('✅ All imports successful!')
"

# Test query
python src/aria_main.py "What is machine learning?" --config config.yaml

# Should see exploration system activate:
# [ARIA] 🌀 Exploration System enabled
# [ARIA] 🌀 Quaternion state: [1.0, 0.0, 0.0, 0.0]
# [ARIA] ✅ Exploration complete
```

---

## 🐛 Troubleshooting Installation

### "No such file or directory: config.yaml"

```bash
# Make sure you're in ARIA directory
pwd  # Should show /path/to/ARIA

# Check config exists
ls -la config.yaml

# If missing, it was deleted - restore from git
git checkout config.yaml
```

### "ModuleNotFoundError: No module named 'torch'"

```bash
# Virtual environment not activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or dependencies not installed
pip install -r requirements.txt
```

### "FileNotFoundError: './data'"

```bash
# Directory doesn't exist
mkdir -p data cache output state/exploration/quaternion

# Or config.yaml points to wrong location
nano config.yaml  # Fix data_dir path
```

### "No documents found in data directory"

```bash
# Check data directory
ls -la data/

# Copy documents
cp /your/docs/*.txt ./data/

# Or update config.yaml to point to your documents
```

### "CUDA out of memory"

```yaml
# In config.yaml, disable GPU:
performance:
  use_gpu: false  # Use CPU instead
  batch_size: 16  # Reduce batch size
```

---

## 🔄 Updating ARIA

When pulling updates from GitHub:

```bash
# Pull latest changes
git pull origin main

# Backup your config (important!)
cp config.yaml config.yaml.backup

# Update dependencies (if requirements.txt changed)
pip install -r requirements.txt --upgrade

# Restore your config
cp config.yaml.backup config.yaml

# Test
python src/aria_main.py "test" --config config.yaml
```

**Important**: Never commit `config.yaml` with your personal paths!

```bash
# Add to .gitignore
echo "config.yaml" >> .gitignore
git add .gitignore
git commit -m "Ignore personal config.yaml"
```

---

## 🌐 LM Studio Plugin (Optional)

If you want seamless integration with LM Studio:

### Plugin Paths to Configure

The plugin needs to know where ARIA backend is running.

**In `~/.lmstudio/extensions/plugins/lmstudio/aria-rag/src/index.ts`**:

```typescript
// ARIA backend URL
const ARIA_BACKEND_URL = 'http://localhost:5000';  // ← Default

// Or if running on different port:
const ARIA_BACKEND_URL = 'http://localhost:8080';  // ← CHANGE THIS
```

**Plugin installation location**:
- Linux/Mac: `~/.lmstudio/extensions/plugins/lmstudio/aria-rag/`
- Windows: `%USERPROFILE%\.lmstudio\extensions\plugins\lmstudio\aria-rag\`

See README.md "LM Studio Plugin" section for full setup.

---

## 📚 Additional Resources

- **README.md** - System overview and features
- **docs/ARCHITECTURE.md** - Detailed system design
- **docs/TROUBLESHOOTING.md** - Common issues and solutions
- **docs/METRICS.md** - Understanding telemetry
- **docs/CONTRIBUTING.md** - Development guidelines

---

## ✅ Installation Complete!

If everything works, you should see:

```bash
$ python src/aria_main.py "test query" --config config.yaml

[ARIA] Loading configuration from config.yaml
[ARIA] Multi-Anchor System enabled
[ARIA] 🌀 Exploration System enabled (corpus: 42 documents)
[ARIA] 🌀 Quaternion state loaded: [1.0, 0.0, 0.0, 0.0]
[ARIA] 📐 Reasoning Mode: analytical
[ARIA] 🔍 Retrieval: hybrid_balanced (20 chunks)
[ARIA] 🌀 Applying exploration (quaternion+PCA+φ spiral)
[ARIA] ✅ Exploration complete (123.4ms)
[ARIA] ✅ Query complete (678ms total)
```

**Welcome to ARIA!** 🎉

*"Go within." - Start exploring your knowledge base with geometric intelligence.*
