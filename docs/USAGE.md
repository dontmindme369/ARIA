# ARIA Usage Guide

## Quick Start

### Control Center (Recommended)

The control center provides a unified interface for both Teacher and Student ARIA:

```bash
python3 aria_control_center.py
```

**Main Menu Options**:
1. **Query Teacher ARIA** - Interactive query interface
2. **Start Student Watcher** - Begin corpus learning
3. **Stop Student Watcher** - Stop corpus learning
4. **View Corpus Stats** - Student learning statistics
5. **Run Flywheel Test** - System verification
6. **View Telemetry** - System logs
7. **Refresh Status** - Update dashboard
8. **Quit** - Exit

### Command Line Interface

For direct queries without the control center:

```bash
python3 aria_main.py "Your question here"
```

**Examples**:
```bash
# Basic query
python3 aria_main.py "How does gradient descent work?"

# With specific preset
python3 aria_main.py "Explain neural networks" --preset deep

# With custom config
python3 aria_main.py "What is quantum mechanics?" --config my_config.yaml

# With custom output directory
python3 aria_main.py "Machine learning basics" --output ./my_packs
```

---

## Teacher ARIA - Query & Retrieval

### Interactive Mode

```bash
python3 aria_control_center.py
# Select [1] Query Teacher ARIA
```

**Features**:
- Continuous query session
- Real-time statistics
- Pack path display
- Query timing
- Preset selection (automatic via bandit)

**Example Session**:
```
Query: How do transformers work in NLP?
⏳ Processing...

✓ Query completed in 1.23s
  • Preset: balanced
  • Run dir: aria_packs/nlp_transformers_1731596400
  • Chunks retrieved: 64
```

### Programmatic Usage

```python
from core.aria_core import ARIA

# Initialize
aria = ARIA(
    index_roots=["./datasets"],
    out_root="./aria_packs",
    enforce_session=False
)

# Query
result = aria.query("What is machine learning?")

# Result structure
{
    "query": "What is machine learning?",
    "preset": "balanced",
    "run_dir": "aria_packs/machine_learning_1731596400",
    "pack": "aria_packs/.../last_pack.json",
    "chunks_retrieved": 64,
    "files_used": 12,
    "perspective": "educational",
    "rotation_angle": 24.0
}
```

---

## Understanding Presets

ARIA uses **Thompson Sampling** (Bayesian bandit) to automatically select the best preset for each query.

### Preset Overview

| Preset     | Chunks | Rotations | Per-File | Best For                    |
|------------|--------|-----------|----------|----------------------------|
| **fast**   | 40     | 1         | 8        | Quick facts, simple queries |
| **balanced** | 64   | 2         | 6        | General questions          |
| **deep**   | 96     | 3         | 5        | Complex research, thorough |
| **diverse**| 80     | 2         | 4        | Broad exploration          |

### Automatic Selection

The bandit learns which preset works best for different query types:

**Exploration Phase** (first 20 queries):
- Tries all presets randomly
- Gathers performance data
- No preference yet

**Exploitation Phase** (after 20 queries):
- Uses learned preferences
- Balances exploration vs exploitation
- Adapts to your query patterns

### Manual Override

Force a specific preset:

```bash
python3 aria_main.py "Complex question" --preset deep
```

```python
result = aria.query("Complex question", preset_override="deep")
```

---

## Perspective-Aware Retrieval

ARIA detects query perspective and adjusts retrieval accordingly.

### 8 Perspectives

| Perspective      | Angle | Signal Words                      | Use Case                  |
|------------------|-------|-----------------------------------|---------------------------|
| Educational      | 30°   | what, explain, teach, learn       | Learning concepts         |
| Diagnostic       | 90°   | debug, fix, error, why broken     | Troubleshooting          |
| Security         | 45°   | vulnerability, attack, threat     | Security analysis        |
| Implementation   | 60°   | how to build, create, implement   | Building things          |
| Research         | 120°  | investigate, explore, discover    | Deep investigation       |
| Theoretical      | 75°   | theory, concept, principle        | Abstract concepts        |
| Practical        | 50°   | guide, tutorial, step-by-step     | Practical how-tos        |
| Reference        | 15°   | definition, what is, lookup       | Quick facts              |

### How Perspective Affects Retrieval

**Rotation Angle**:
- Larger angles = more aggressive exploration
- Smaller angles = focused, direct retrieval

**Example**:
```
Query: "Debug authentication error"
→ Perspective: Diagnostic (90°)
→ Aggressive rotation to find error-related content

Query: "What is authentication?"
→ Perspective: Reference (15°)
→ Minimal rotation, direct definition lookup
```

---

## Understanding Query Results

### Pack Structure

Each query generates a "pack" (JSON file) containing retrieved chunks:

```json
{
  "query": "How does gradient descent work?",
  "timestamp": "2025-11-14T12:00:00",
  "preset": "balanced",
  "perspective": "educational",
  "items": [
    {
      "path": "/path/to/source/file.txt",
      "start_line": 42,
      "end_line": 68,
      "content": "Gradient descent is an optimization...",
      "score": 0.89,
      "source_type": "text"
    }
    // ... more chunks
  ],
  "metadata": {
    "chunks_retrieved": 64,
    "files_used": 8,
    "rotation_angle": 24.0,
    "rotations": 2
  }
}
```

### Interpreting Scores

**Hybrid Score** = `0.3 × BM25 + 0.7 × Semantic`

- **0.8 - 1.0**: Excellent match
- **0.6 - 0.8**: Good match
- **0.4 - 0.6**: Moderate relevance
- **< 0.4**: Weak relevance (usually filtered)

---

## Student ARIA - Corpus Learning

Student ARIA learns from ALL your LM Studio conversations (not just ARIA queries).

### Starting the Watcher

**Via Control Center**:
```bash
python3 aria_control_center.py
# Select [2] Start Student Watcher
```

**Standalone**:
```bash
python3 conversation_watcher.py
```

### What Gets Captured

**Source**: `~/.lmstudio/conversations/`

**Captured Data**:
- All conversation turns
- Message timestamps
- Model used
- Conversation metadata

**Filtered Out**:
- System messages
- Empty messages
- Duplicate conversations

### Corpus Location

**Saved To**: `../training_data/conversation_corpus/`

**Format**: JSON files with conversation structure

### Viewing Statistics

```bash
python3 aria_control_center.py
# Select [4] View Corpus Stats
```

**Displays**:
- Conversations captured
- Total messages
- Corpus size (MB)
- Recent conversations
- Average messages per conversation

---

## Advanced Usage

### Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
paths:
  index_roots:
    - ~/my_knowledge
    - ./project_docs
  output_dir: ./my_aria_packs

retrieval:
  top_k: 128        # More chunks
  semantic_weight: 0.8  # Favor semantic over BM25

postfilter:
  max_per_source: 10    # More from each file
  diversity_filter: true
```

Use it:
```bash
python3 aria_main.py "Query" --config my_config.yaml
```

### Batch Queries

```python
from core.aria_core import ARIA

aria = ARIA(index_roots=["./datasets"], out_root="./packs")

queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain backpropagation"
]

results = []
for q in queries:
    result = aria.query(q)
    results.append(result)
    print(f"✓ {q}: {result['chunks_retrieved']} chunks")
```

### Monitoring Performance

**Via Control Center**:
```bash
python3 aria_control_center.py
# Select [6] View Telemetry
```

**Logs Location**: `../var/telemetry/`

**Available Logs**:
- `conversation_watcher.log` - Student ARIA activity
- `aria_queries.log` - Query history
- `bandit_performance.log` - Preset selection history

---

## Best Practices

### Query Formulation

**Good Queries**:
- "How does gradient descent optimize neural networks?"
- "Explain the transformer architecture"
- "Debug Python import error: ModuleNotFoundError"

**Avoid**:
- Single words: "gradient" (too vague)
- Yes/no: "Is Python good?" (use "What are Python's strengths?")
- Multiple unrelated: "Python and Java and C++" (split into separate queries)

### Knowledge Base Organization

**Recommended Structure**:
```
datasets/
├── programming/
│   ├── python/
│   └── javascript/
├── ml/
│   ├── theory/
│   └── practice/
└── docs/
    ├── api_references/
    └── tutorials/
```

**File Formats**:
- ✅ `.txt`, `.md`, `.json`, `.csv`
- ✅ Plain text is best
- ⚠️ Binary files not supported

### Preset Selection Strategy

**Let the bandit learn**:
- First 20+ queries: Let it explore
- Don't override unless necessary
- Monitor bandit stats to see what works

**When to override**:
- Testing specific strategies
- Known query type (reference lookup = fast)
- Performance requirements (fast for quick answers)

---

## Troubleshooting

### No Results Returned

**Possible Causes**:
1. Query too specific
2. Knowledge base doesn't contain relevant info
3. Semantic model not downloaded

**Solutions**:
```bash
# Broaden query
"gradient descent" → "optimization algorithms"

# Check knowledge base
ls -R datasets/

# Pre-download model
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Slow Queries

**Optimization**:
1. Use `fast` preset for simple queries
2. Reduce `top_k` in config
3. Use GPU acceleration (install PyTorch with CUDA)
4. Pre-compute embeddings (future feature)

### Student Watcher Not Capturing

**Check**:
```bash
# Verify LM Studio path
ls ~/.lmstudio/conversations/

# Check watcher is running
ps aux | grep conversation_watcher

# View logs
tail -f ../var/telemetry/conversation_watcher.log
```

---

## Examples by Use Case

### Research Assistant

```bash
python3 aria_main.py "Comprehensive overview of quantum computing" --preset deep
```

### Code Helper

```bash
python3 aria_main.py "Python async await examples" --preset balanced
```

### Quick Lookup

```bash
python3 aria_main.py "What is REST API?" --preset fast
```

### Debugging

```bash
python3 aria_main.py "Fix TypeScript type error: cannot assign undefined" --preset balanced
```

### Learning

```bash
python3 aria_main.py "Explain backpropagation step by step" --preset deep
```

---

## Integration with LLMs

### Using Packs with LLMs

```python
import json

# Load pack
with open("aria_packs/.../last_pack.json") as f:
    pack = json.load(f)

# Build context
context = "\n\n---\n\n".join([
    item["content"] for item in pack["items"]
])

# Send to LLM
prompt = f"""Context:
{context}

Question: {pack['query']}

Answer:"""

# Use with any LLM API (OpenAI, Anthropic, etc.)
```

### LM Studio Integration

ARIA works seamlessly with LM Studio:
1. Use ARIA to retrieve relevant context
2. Feed pack contents to LM Studio chat
3. Student ARIA learns from the conversation
4. Continuous improvement loop

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system internals
- See [API_REFERENCE.md](API_REFERENCE.md) for programmatic usage
- Check [QUATERNIONS.md](QUATERNIONS.md) for mathematical details
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to extend ARIA
