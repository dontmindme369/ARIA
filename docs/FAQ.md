# ARIA FAQ

Frequently Asked Questions about ARIA.

---

## General Questions

### What is ARIA?

ARIA (Adaptive Resonant Intelligent Architecture) is a self-learning retrieval system that combines:
- **Hybrid search** (BM25 + semantic embeddings)
- **Quaternion exploration** (4D geometric semantic space navigation)
- **LinUCB** (adaptive strategy selection)
- **Perspective detection** (8-way query classification)

It's designed to retrieve relevant context from your knowledge base for use with LLMs.

### What makes ARIA different from other cognitive architectures?

**Unique Features**:
1. **Quaternion Semantic Exploration** - Rotates embeddings in semantic space for broader coverage
2. **LinUCB** - Learns which retrieval strategies work best for different queries
3. **Perspective-Aware** - Adjusts retrieval based on query intent (educational vs diagnostic vs research, etc.)
4. **Student/Teacher Architecture** - Learns from all your LLM conversations, not just explicit queries

### Is ARIA production-ready?

**Yes**, for personal and small-team use:
- ✅ 14/14 tests passing
- ✅ Handles 10k-100k documents well
- ✅ Stable API
- ✅ Portable paths
- ⚠️  Large-scale deployment untested (>1M documents)
- ⚠️  Pre-alpha Student ARIA (learning features experimental)

### What are the system requirements?

**Minimum**:
- Python 3.8+
- 4GB RAM
- 500MB disk space (for model)

**Recommended**:
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU (for faster semantic search)
- SSD storage

---

## Installation & Setup

### How do I install ARIA?

```bash
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA/aria
pip install -r requirements.txt
```

See [INSTALLATION.md](INSTALLATION.md) for details.

### Do I need a GPU?

**No**, but it helps:
- **CPU**: Works fine, ~1-2s per query
- **GPU**: Faster semantic encoding, ~0.5-1s per query

Install PyTorch with CUDA for GPU support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Where should I put my knowledge base?

**Recommended structure**:
```
~/Documents/knowledge/
├── programming/
├── research/
├── docs/
└── notes/
```

Then configure in `aria_config.yaml`:
```yaml
paths:
  index_roots:
    - ~/Documents/knowledge
```

### What file formats are supported?

**Supported**:
- ✅ `.txt` - Plain text
- ✅ `.md` - Markdown
- ✅ `.json` - JSON (flat text extraction)
- ✅ `.csv` - CSV (text columns)
- ✅ `.py`, `.js`, `.java`, etc. - Source code

**Not Supported**:
- ❌ `.pdf` - Use text extraction tool first
- ❌ `.docx` - Convert to `.txt` or `.md`
- ❌ Binary files

---

## Usage Questions

### How do I run a simple query?

**Command Line**:
```bash
python3 aria_main.py "How does gradient descent work?"
```

**Control Center**:
```bash
python3 aria_control_center.py
# Select [1] Query Teacher ARIA
```

**Programmatic**:
```python
from core.aria_core import ARIA

aria = ARIA(index_roots=["./data"], out_root="./packs")
result = aria.query("How does gradient descent work?")
```

### How do I choose a preset?

**Don't choose - let the bandit learn!**

After 20+ queries, LinUCB will automatically select the best preset for each query type.

**Manual override** (testing only):
```bash
python3 aria_main.py "Complex query" --preset deep
```

**Presets**:
- `fast` - Quick lookups (40 chunks, 1 rotation)
- `balanced` - General use (64 chunks, 2 rotations)
- `deep` - Thorough research (96 chunks, 3 rotations)
- `diverse` - Broad exploration (80 chunks, max diversity)

### What is a "pack"?

A **pack** is the JSON output containing retrieved chunks:
```json
{
  "query": "Your question",
  "items": [
    {
      "path": "/path/to/source.txt",
      "content": "Retrieved text chunk",
      "score": 0.89,
      "start_line": 42,
      "end_line": 68
    }
    // ... more chunks
  ]
}
```

**Location**: `aria_packs/<query>_<timestamp>/last_pack.json`

### How do I use packs with an LLM?

**Example with any LLM**:
```python
import json

# Load pack
with open("aria_packs/.../last_pack.json") as f:
    pack = json.load(f)

# Build context
context = "\n\n---\n\n".join([item['content'] for item in pack['items']])

# Send to LLM
prompt = f"Context:\n{context}\n\nQuestion: {pack['query']}\n\nAnswer:"

# Use with OpenAI, Anthropic, LM Studio, etc.
response = llm.complete(prompt)
```

### How does Student ARIA work?

**Student ARIA** learns from your LM Studio conversations:

1. **Monitors** `~/.lmstudio/conversations/` directory
2. **Captures** ALL conversations (not just ARIA queries)
3. **Extracts** reasoning patterns, turn-taking, domain transitions
4. **Builds** training corpus in `../training_data/conversation_corpus/`

**Start watcher**:
```bash
python3 aria_control_center.py
# Select [2] Start Student Watcher
```

**Future**: Train custom models on captured corpus.

---

## Technical Questions

### What are quaternions and why use them?

**Quaternions** are 4D hypercomplex numbers that represent 3D rotations.

**Why for retrieval?**
- Rotate embeddings in semantic space
- Explore different "angles" of meaning
- Find content that's semantically related but not lexically similar

**Example**:
```
Query: "debug authentication error"
→ Rotate embeddings 90° (diagnostic perspective)
→ Find error-related content from different angles
→ Better coverage than direct similarity
```

See [QUATERNIONS.md](QUATERNIONS.md) for deep dive.

### What is LinUCB?

**LinUCB** is a Bayesian algorithm for the multi-armed bandit problem.

**In ARIA**:
- **Arms**: 4 presets (fast, balanced, deep, diverse)
- **Reward**: Multi-objective score (quality + coverage + diversity)
- **Learning**: Updates α (successes) and β (failures) after each query
- **Selection**: Samples from Beta(α, β) distributions, picks highest

**Result**: ARIA learns which preset works best for different query types.

### What are the 8 perspectives?

ARIA detects query intent and adjusts retrieval:

| Perspective | Example Query | Rotation Angle |
|-------------|---------------|----------------|
| Reference | "What is machine learning?" | 15° (minimal) |
| Educational | "Explain how neural networks work" | 30° (gentle) |
| Security | "SQL injection vulnerabilities" | 45° (moderate) |
| Practical | "Step-by-step Docker setup" | 50° (moderate) |
| Implementation | "Build REST API in Python" | 60° (strong) |
| Theoretical | "Theory behind backpropagation" | 75° (strong) |
| Diagnostic | "Debug CUDA out of memory error" | 90° (aggressive) |
| Research | "Explore alternatives to transformers" | 120° (very aggressive) |

**Larger angles** = more exploration = find related but not directly similar content.

### How does the hybrid search work?

**Formula**: `Score = 0.3 × BM25 + 0.7 × Semantic`

**BM25** (Lexical):
- Keyword matching
- TF-IDF based
- Fast, exact matches

**Semantic** (Embeddings):
- Meaning-based similarity
- Sentence-transformers (all-MiniLM-L6-v2)
- Cosine similarity in embedding space
- Quaternion-rotated for exploration

**Combined**: Best of both worlds - catches exact matches AND semantic meaning.

---

## Troubleshooting

### Why am I getting no results?

**Possible causes**:

1. **Query too specific**
   - Try broader query
   - Example: "Python async/await event loop internals" → "Python async basics"

2. **Knowledge base doesn't contain relevant info**
   - Check: `ls -R ~/Documents/knowledge/`
   - Add more documents

3. **Semantic model not downloaded**
   - First run downloads ~90MB model
   - Check internet connection
   - Pre-download: `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

4. **Postfilter too aggressive**
   - Edit `aria_config.yaml`:
     ```yaml
     postfilter:
       quality_filter: false
       min_score: 0.001  # Lower threshold
     ```

### Why are queries slow?

**Optimization steps**:

1. **Use fast preset**
   ```bash
   python3 aria_main.py "Quick query" --preset fast
   ```

2. **Reduce top_k** in config
   ```yaml
   retrieval:
     top_k: 40  # Lower from default 64
   ```

3. **Use GPU**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Reduce rotations** in config
   ```yaml
   retrieval:
     semantic_rotations: 1  # Lower from default 3
   ```

**Typical performance**:
- CPU: ~1-2s per query
- GPU: ~0.5-1s per query

### Student watcher not capturing conversations

**Check**:

1. **LM Studio path exists**
   ```bash
   ls ~/.lmstudio/conversations/
   ```

2. **Watcher is running**
   ```bash
   ps aux | grep conversation_watcher
   ```

3. **Permissions**
   ```bash
   ls -l ~/.lmstudio/conversations/
   # Should be readable
   ```

4. **Logs**
   ```bash
   tail -f ../var/telemetry/conversation_watcher.log
   ```

### Import errors

**Problem**: `ModuleNotFoundError: No module named 'core'`

**Solution**:
```bash
# Make sure you're in aria/ folder
cd ARIA/aria

# Check Python path
python3 -c "import sys; print(sys.path)"

# Should include /path/to/ARIA/aria/src
```

**Fix in code**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### Type errors from Pylance

**These are often false positives**:

1. **Verify code works**
   ```bash
   python3 -c "from core.aria_core import ARIA; print('Works!')"
   ```

2. **If it works, it's a Pylance issue** - safe to ignore

3. **Real type errors** will show as Python exceptions at runtime

---

## Performance & Scaling

### How many documents can ARIA handle?

**Tested scale**:
- ✅ 1k-10k documents: Excellent performance
- ✅ 10k-100k documents: Good performance
- ⚠️  100k-1M documents: Slower but usable
- ❌ >1M documents: Not tested, likely needs optimization

**Bottlenecks**:
- BM25 search: O(n) with documents
- Semantic encoding: O(n) with document count
- Quaternion rotation: O(k) with rotations (constant)

**For large scale**:
- Pre-compute embeddings (future feature)
- Use document filtering before retrieval
- Consider sharding knowledge base

### Can I run ARIA on a server?

**Yes!** ARIA is designed to be portable:

```bash
# On server
git clone https://github.com/dontmindme369/ARIA.git
cd ARIA/aria
pip install -r requirements.txt

# Run as service
nohup python3 aria_control_center.py &

# Or via API (future feature)
```

### How much disk space does ARIA use?

**ARIA itself**: ~50MB
**Sentence-transformers model**: ~90MB
**Your knowledge base**: Varies
**Generated packs**: ~1-5MB per 1000 queries

**Total**: ~150MB + knowledge base size

---

## Integration & Advanced Usage

### Can I use ARIA with my own LLM?

**Yes!** ARIA generates packs, you integrate them however you want:

```python
import json
from your_llm import YourLLM

# Get pack from ARIA
from core.aria_core import ARIA
aria = ARIA(...)
result = aria.query("Question")

# Load pack
with open(result['pack']) as f:
    pack = json.load(f)

# Build context
context = "\n\n".join([item['content'] for item in pack['items']])

# Use with your LLM
llm = YourLLM()
response = llm.generate(f"Context:\n{context}\n\nQuestion: {pack['query']}")
```

### Can I customize the retrieval algorithm?

**Yes!** Several extension points:

**1. Add custom preset**:
```python
# Edit src/intelligence/bandit_context.py
DEFAULT_PRESETS.append({
    "name": "my_preset",
    "args": {"top_k": 128, "rotations": 5, "max_per_file": 2}
})
```

**2. Add custom postfilter**:
```python
# Edit src/retrieval/aria_postfilter.py
def my_custom_filter(chunks):
    # Custom filtering logic
    return filtered_chunks
```

**3. Add custom perspective**:
```python
# Edit src/perspective/detector.py
PERSPECTIVES.append("my_perspective")
BASE_ANGLES["my_perspective"] = 55.0
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Can I use a different embedding model?

**Yes!** Edit `aria_config.yaml`:

```yaml
retrieval:
  semantic_model: sentence-transformers/all-mpnet-base-v2  # Larger, more accurate
  # OR
  semantic_model: sentence-transformers/paraphrase-MiniLM-L3-v2  # Smaller, faster
```

**Available models**: https://www.sbert.net/docs/pretrained_models.html

**Trade-offs**:
- Larger models: Better accuracy, slower, more memory
- Smaller models: Faster, less memory, lower accuracy

---

## Contributing & Development

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guide.

**Quick start**:
1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

**Ways to contribute**:
- Report bugs
- Suggest features
- Improve documentation
- Write code

### Where should I report bugs?

**GitHub Issues**: https://github.com/dontmindme369/ARIA/issues

**Include**:
- ARIA version
- Python version
- Operating system
- Error message (full traceback)
- Steps to reproduce
- Expected vs actual behavior

### How do I run tests?

```bash
cd ARIA/aria
python3 tests/comprehensive_test_suite.py
```

**Expected output**: `14/14 tests passing`

---

## Miscellaneous

### What does "ARIA" stand for?

**Adaptive Resonant Intelligent Architecture**

- **Adaptive**: LinUCB learns optimal strategies
- **Resonant**: Quaternion rotations explore semantic "resonances"
- **Intelligent**: Perspective detection, bandit learning
- **Architecture**: Unified Teacher/Student system design

### Is ARIA related to the opera "ARIA"?

No, but the metaphor works:
- **Opera aria**: Solo vocal piece expressing emotion
- **ARIA system**: Expresses knowledge through quaternion "songs" in semantic space

### What's the license?

**MIT License** - free to use, modify, distribute.

See `LICENSE` file for details.

### Who created ARIA?

ARIA was created as a research project exploring:
- Quaternion mathematics in information retrieval
- Multi-armed bandits for adaptive systems
- Perspective-aware semantic search
- Self-learning retrieval architectures

**Maintainer**: See GitHub repository

### Can I use ARIA commercially?

**Yes!** MIT License permits commercial use.

**Requirements**:
- Include license and copyright notice
- No warranty (use at your own risk)

**Recommended**:
- Test thoroughly for your use case
- Contribute improvements back (optional but appreciated)

---

## Still Have Questions?

### Documentation
- [INSTALLATION.md](INSTALLATION.md) - Setup and config
- [USAGE.md](USAGE.md) - How to use ARIA
- [ARCHITECTURE.md](ARCHITECTURE.md) - System internals
- [API_REFERENCE.md](API_REFERENCE.md) - Programmatic API

### Community
- **Issues**: https://github.com/dontmindme369/ARIA/issues
- **Discussions**: https://github.com/dontmindme369/ARIA/discussions

### Direct Contact
- See GitHub repository for maintainer contact info
- Prefer public discussions/issues for community benefit

---

**Last Updated**: 2025-11-14
**ARIA Version**: 1.0
