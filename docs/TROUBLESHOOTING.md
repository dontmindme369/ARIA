# ARIA Troubleshooting Guide

**Common Issues and Solutions**

## Quick Diagnosis

Having issues? Start here:

```bash
# Check system status
python -c "import torch; print('GPU:', torch.cuda.is_available())"
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
python -c "import numpy as np; print('NumPy:', np.__version__)"

# Verify installation
python -m pytest tests/test_aria_comprehensive.py -v

# Check configuration
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

---

## Installation Issues

### ImportError: No module named 'torch'

**Problem**: PyTorch not installed

**Solution**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or for GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### ImportError: No module named 'sentence_transformers'

**Problem**: Sentence transformers not installed

**Solution**:
```bash
pip install sentence-transformers
```

### CUDA not available despite having GPU

**Problem**: PyTorch CPU version installed

**Solution**:
```bash
# Uninstall CPU version
pip uninstall torch

# Install GPU version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### ImportError: No module named 'numpy'

**Problem**: NumPy not installed (required for quaternions)

**Solution**:
```bash
pip install numpy scipy  # scipy often needed too
```

---

## Configuration Issues

### FileNotFoundError: config.yaml not found

**Problem**: Working directory doesn't have config

**Solution**:
```bash
# Copy example config
cp config.yaml.example config.yaml

# Or create from scratch
python -c "from aria_config import create_default_config; create_default_config()"
```

### KeyError: 'data_dir' in config

**Problem**: Incomplete configuration

**Solution**:
Check config.yaml has all required fields:
```yaml
paths:
  data_dir: "./data"
  cache_dir: "./cache"
  output_dir: "./output"
  quaternion_state_path: "./state/quaternion_states.jsonl"
```

---

## Exploration System Issues

### "Exploration system unavailable"

**Symptoms**: Error message during initialization

**Diagnosis**:
```bash
# Check if exploration modules exist
ls src/quaternion_state.py
ls src/pca_exploration.py
ls src/aria_exploration.py

# Try importing
python -c "from src.aria_exploration import ExplorationManager; print('OK')"
```

**Solutions**:

1. **Files missing**:
```bash
# Ensure all 3 exploration files are in src/
# Download from repository if needed
```

2. **Import errors**:
```bash
# Check dependencies
pip install numpy scikit-learn scipy
```

3. **Path issues**:
```python
# Verify Python path includes src/
import sys
sys.path.append('./src')
```

### Quaternion state not persisting

**Symptoms**: State resets to [1, 0, 0, 0] every query

**Diagnosis**:
```bash
# Check state directory exists
ls -la ./state/exploration/quaternion/

# Check for state file
cat ./state/exploration/quaternion/quaternion_states.jsonl | head -5
```

**Solutions**:

1. **Directory doesn't exist**:
```bash
mkdir -p ./state/exploration/quaternion
chmod 755 ./state
```

2. **Permission errors**:
```bash
# Check ownership
ls -la ./state/
# Fix permissions
chmod -R 755 ./state/
```

3. **State file corrupted**:
```bash
# Backup and reset
mv ./state/exploration/quaternion/quaternion_states.jsonl ./state/backup.jsonl
# System will create fresh state file
```

### PCA exploration disabled

**Symptoms**: "PCA exploration disabled (corpus too small)" in logs

**Diagnosis**:
```python
# Check corpus size
from pathlib import Path
corpus_files = list(Path("./data").glob("**/*.*"))
print(f"Corpus files: {len(corpus_files)}")
```

**Solutions**:

1. **Corpus too small** (< 10 documents):
```bash
# Option A: Add more documents to ./data/
# Option B: Temporarily disable PCA
```

Update config:
```yaml
exploration:
  pca_enabled: false  # Disable PCA, keep quaternion + spiral
```

2. **Wrong data directory**:
```yaml
# Verify data_dir path
paths:
  data_dir: "./data"  # Must point to your documents
```

### Golden ratio spiral returns duplicates

**Symptoms**: Most spiral chunks are duplicates of initial retrieval

**Diagnosis**:
```python
# Check unique ratio in telemetry
unique_ratio = (
    result['exploration']['golden_ratio']['unique_chunks'] / 
    result['exploration']['golden_ratio']['chunks_retrieved']
)
print(f"Unique ratio: {unique_ratio:.2%}")
# Should be >60%
```

**Solutions**:

1. **Exploration radius too large**:
```yaml
exploration:
  exploration_radius: 0.2  # Reduce from 0.3
```

2. **Corpus too small**:
```bash
# Golden ratio works best with larger corpus
# Add more documents or accept some overlap
```

3. **Query too specific**:
```python
# Very narrow queries naturally have less to explore
# This is expected behavior
```

### Exploration latency too high

**Symptoms**: Exploration takes >300ms

**Diagnosis**:
```python
# Check breakdown in telemetry
latency = result['exploration']['latency_ms']
print(f"Exploration latency: {latency}ms")

# Check components
print(f"Quaternion: {result['exploration']['quaternion_state']['latency_ms']}ms")
print(f"PCA: {result['exploration']['pca']['latency_ms']}ms")
print(f"Spiral: {result['exploration']['golden_ratio']['latency_ms']}ms")
```

**Solutions**:

1. **Reduce golden ratio samples**:
```yaml
exploration:
  golden_ratio_samples: 8  # Reduce from 13
```

2. **Disable PCA** (if not helping):
```yaml
exploration:
  pca_enabled: false
```

3. **Enable GPU** (for embeddings):
```yaml
performance:
  use_gpu: true
```

4. **Cache embeddings**:
```yaml
performance:
  cache_embeddings: true
```

### SLERP numerical instability

**Symptoms**: `RuntimeWarning: invalid value encountered in arccos`

**Diagnosis**:
```python
# Check quaternion magnitude
q = result['exploration']['quaternion_state']['vector']
magnitude = np.sqrt(sum(x**2 for x in q))
print(f"Quaternion magnitude: {magnitude}")  # Should be 1.0
```

**Cause**: Numerical precision errors in quaternion normalization

**Solution**:
```python
# In quaternion_state.py, add epsilon to normalization:
def normalize_quaternion(q, epsilon=1e-8):
    norm = np.sqrt(np.sum(q**2)) + epsilon
    return q / norm
```

### Geodesic distance always large

**Symptoms**: Geodesic distances consistently >2.0 radians

**Diagnosis**:
```python
# Check if state is evolving properly
states = [run['exploration']['quaternion_state']['vector'] 
          for run in recent_runs]
for i, state in enumerate(states):
    print(f"Query {i}: {state}")
# Should show gradual evolution, not random jumps
```

**Cause**: State not persisting or momentum too high

**Solutions**:

1. **Verify state persistence** (see above)

2. **Reduce momentum decay**:
```yaml
exploration:
  momentum_decay: 0.8  # Increase from 0.5 (slower decay)
```

3. **Increase SLERP interpolation factor**:
```python
# In quaternion_state.py:
new_state = slerp(current_state, target, t=0.5)  # Was 0.3
```

### PCA variance explained too low

**Symptoms**: `variance_explained < 0.80` in telemetry

**Diagnosis**:
```python
# Check PCA components
result['exploration']['pca']['variance_explained']
result['exploration']['pca']['components']
```

**Cause**: Too few PCA components for corpus complexity

**Solution**:
```yaml
exploration:
  pca_components: 64  # Increase from 32
```

**Trade-off**: More components = better representation, slower computation

---

## Retrieval Issues

### No chunks retrieved (empty results)

**Symptoms**: Query returns 0 chunks

**Diagnosis**:
```python
from pathlib import Path
print(list(Path("./data").glob("**/*.*")))  # Check files exist
```

**Solutions**:
1. **No documents**: Add documents to `data_dir`
2. **Wrong path**: Update `config.yaml` paths
3. **Unsupported format**: Check file extensions (.txt, .md, .pdf, .docx)

### Very low retrieval scores (<0.3)

**Symptoms**: All chunks have low scores

**Causes**:
- Query mismatch with corpus
- Corpus too small
- Embeddings not cached

**Solutions**:
1. Expand corpus with relevant documents
2. Use query expansion strategy
3. Check embedding model quality

### Retrieval extremely slow (>5s)

**Symptoms**: Queries take several seconds

**Diagnosis**:
```python
# Time each component
import time

start = time.time()
chunks = retriever.retrieve(query, k=20)
print(f"Retrieval: {time.time() - start:.2f}s")
```

**Solutions**:
1. **Enable GPU**: Set `use_gpu: true` in config
2. **Cache embeddings**: Ensure cache is enabled
3. **Reduce corpus**: Remove irrelevant documents
4. **Lower k**: Retrieve fewer chunks

---

## Anchor Selector Issues

### Always selecting 'default_balanced' mode

**Symptoms**: No mode diversity

**Causes**:
- exemplars.txt missing
- exemplars.txt empty
- Patterns don't match queries

**Solutions**:
1. **Check exemplars exist**:
   ```bash
   wc -l data/exemplars.txt  # Should be ~746 lines
   ```

2. **Verify patterns load**:
   ```python
   from anchor_selector import AnchorSelector
   selector = AnchorSelector()
   print(len(selector.patterns))  # Should be >700
   ```

3. **Test manually**:
   ```python
   selector = AnchorSelector()
   print(selector.select_mode("Implement a binary search tree"))  # Should be 'technical'
   print(selector.select_mode("What is consciousness?"))  # Should be 'philosophical'
   ```

### Wrong anchor mode selected

**Symptoms**: Query gets inappropriate mode

**Solution**:
Add patterns to exemplars.txt:
```
your_domain_technical:: Technical queries in your domain → keywords, api, implementation
```

---

## Postfilter Issues

### Too many chunks filtered out

**Symptoms**: 20 chunks retrieved → 2 after postfilter

**Cause**: Quality threshold too high

**Solution**:
Lower `min_quality_score` in config:
```yaml
postfilter:
  min_quality_score: 0.3  # Lower from 0.5
```

### Pack lacks diversity (duplicate chunks)

**Symptoms**: Same content repeated

**Solution**:
Enable diversity filter:
```yaml
postfilter:
  enable_diversity: true
  max_duplication_ratio: 0.3
```

---

## Bandit Issues

### Bandit not learning (flat rewards)

**Symptoms**: Rewards don't improve over time

**Diagnosis**:
```python
import json
from pathlib import Path

runs = list(Path("output/rag_runs/aria").glob("*.json"))
rewards = [json.load(open(r))['quality']['composite_reward'] for r in runs]
print(f"Rewards: {rewards[-10:]}")  # Last 10
```

**Solutions**:
1. **Check reward signal**:
   - Verify quality scores are computed
   - Check telemetry is recorded
   
2. **Increase exploration**:
   ```yaml
   bandit:
     epsilon: 0.2  # More exploration
   ```

3. **Check strategy definitions**:
   - Ensure strategies are distinct
   - Verify strategies execute correctly

### Single strategy always selected

**Symptoms**: 100% exploitation, no exploration

**Causes**:
- Epsilon too low
- One strategy vastly better
- Bandit parameters not updating

**Solutions**:
1. **Increase epsilon**: `epsilon: 0.1` → `epsilon: 0.2`
2. **Reset bandit**: Delete state files, restart
3. **Check updates**: Verify alpha/beta parameters change

---

## Curiosity Engine Issues

### No gaps ever detected

**Symptoms**: Curiosity always silent

**Causes**:
- Gap threshold too high
- High-quality retrieval (no gaps!)
- Curiosity disabled

**Solutions**:
1. **Lower threshold**:
   ```yaml
   curiosity:
     gap_threshold: 0.2  # More sensitive
   ```

2. **Verify enabled**:
   ```yaml
   curiosity:
     enabled: true
   ```

3. **Test manually**:
   ```python
   from aria_curiosity import ARIACuriosity
   
   curiosity = ARIACuriosity(personality=9)  # Max curiosity
   result = curiosity.detect_gaps(query, chunks)
   print(result)
   ```

### Too many irrelevant questions

**Symptoms**: Generated questions off-topic

**Solution**:
Lower personality (less aggressive):
```yaml
curiosity:
  personality: 5  # More conservative
```

---

## Performance Issues

### High memory usage (>8GB)

**Causes**:
- Large embedding cache
- Many documents in memory
- Large reasoning model loaded

**Solutions**:
1. **Limit cache size**: Clear old embeddings
2. **Reduce batch size**: Lower `batch_size` in config
3. **Offload model**: Use smaller model

### System freezes during retrieval

**Causes**:
- OOM (out of memory)
- Infinite loop in code
- Deadlock in threading

**Diagnosis**:
```bash
# Monitor memory
watch -n 1 "free -h"

# Check processes
top -p $(pgrep -f aria)
```

**Solutions**:
1. Add memory limits
2. Enable debug logging
3. Check for infinite loops

---

## Data Issues

### Unicode errors reading documents

**Symptoms**: `UnicodeDecodeError`

**Solution**:
```python
# In aria_retrieval.py, use error handling:
text = path.read_text(encoding='utf-8', errors='ignore')
```

### PDF extraction returns gibberish

**Causes**:
- Scanned PDF (images, not text)
- Encrypted PDF
- Complex layout

**Solutions**:
1. **Use OCR**: Install `pytesseract` for scanned PDFs
2. **Try different library**: Switch from PyPDF2 to pdfplumber
3. **Pre-convert**: Convert to text externally

---

## Debug Mode

Enable verbose logging:

```yaml
logging:
  level: "DEBUG"
  debug_mode: true
```

Then run query and check output:
```bash
python aria_main.py 2>&1 | tee debug.log
```

Look for:
- `[ARIA] Exploration System enabled`
- `[ARIA] 🌀 Quaternion state loaded`
- `[ARIA] 🌀 Applying exploration`
- `[ARIA] ✅ Exploration complete`

---

## Getting Help

If issue persists:

1. **Check existing issues**: https://github.com/dontmindme369/aria/issues
2. **Create detailed issue**:
   - ARIA version
   - Python version
   - Operating system
   - Full error message
   - Steps to reproduce
   - Relevant config

3. **Include diagnostics**:
   ```bash
   python -c "
   import sys, torch, numpy as np
   from sentence_transformers import SentenceTransformer
   print('Python:', sys.version)
   print('PyTorch:', torch.__version__)
   print('NumPy:', np.__version__)
   print('CUDA:', torch.cuda.is_available())
   print('SentenceTransformers: OK')
   "
   ```

---

## Common Error Messages

### "RuntimeError: CUDA out of memory"

**Solution**:
```yaml
performance:
  batch_size: 16  # Reduce from 32
  use_gpu: false  # Use CPU instead
```

### "ValueError: Could not find anchor mode"

**Solution**:
Add mode to config:
```yaml
anchors:
  available_modes:
    - your_mode_name
```

### "KeyError: 'reward'"

**Solution**:
Update telemetry schema - old format incompatible

### "ZeroDivisionError in normalize_quaternion"

**Solution**:
Add epsilon to normalization (see SLERP issues above)

### "LinAlgError: SVD did not converge"

**Cause**: PCA fitting failed

**Solution**:
```yaml
exploration:
  pca_components: 16  # Reduce from 32
```

---

**Still stuck? Check the docs or open an issue!** 🔧
