# ARIA Unified Control Center

**Location**: [aria_control_center.py](aria_control_center.py)

## Overview

Unified command center for both Teacher and Student ARIA systems with integrated corpus learning.

## Features

### 👨‍🏫 Teacher ARIA - Query & Retrieval
- **Interactive query interface** with perspective-aware retrieval
- **Dynamic preset selection** via Thompson Sampling bandit
- **Quaternion semantic exploration** with golden ratio spiral
- **Real-time telemetry** and metrics tracking

### 🎓 Student ARIA - Corpus Learning
- **Conversation watcher** monitors LM Studio conversations
- **Automatic corpus building** from captured conversations
- **Cross-domain pattern learning** from all conversations
- **Corpus statistics** and growth tracking

## Usage

```bash
cd aria
python3 aria_control_center.py
```

### Menu Options

1. **Query Teacher ARIA** - Interactive query interface
2. **Start Student Watcher** - Begin monitoring LM Studio conversations
3. **Stop Student Watcher** - Stop corpus learning
4. **View Corpus Stats** - Detailed corpus statistics
5. **Run Flywheel Test** - Execute cross-domain acceleration test
6. **View Telemetry** - System logs and metrics
7. **Refresh Status** - Update system statistics
8. **Quit** - Exit control center

## Dynamic Preset Selection

ARIA uses **Thompson Sampling** (Bayesian multi-armed bandit) for adaptive preset selection:

### How It Works

1. **Query Features** - Extracted from each query:
   - Length, word count, complexity
   - Domain detection (concept, code, debug, etc.)
   - Technical density
   - Conversation depth

2. **Thompson Sampling** - Probabilistic selection:
   - Each preset has `alpha` (successes) and `beta` (failures)
   - Sample from Beta(α, β) distribution for each preset
   - Select preset with highest sample value
   - Balances exploration vs exploitation

3. **Compound Reward Signal** - Multi-factor learning:
   - **40%** Exemplar fit (style and citation matching)
   - **30%** Coverage score (semantic coverage)
   - **30%** Diversity (MMR diversity)
   - **-20%** penalty if issues detected

4. **Continuous Adaptation**:
   - After each query, reward updates the bandit
   - Alpha increases with reward (success)
   - Beta increases with (1 - reward) (failure)
   - System learns which presets work best for different query types

### Preset Options

- **fast**: Quick retrieval (40 chunks, 1 rotation)
- **balanced**: Standard retrieval (64 chunks, 2 rotations)
- **deep**: Thorough retrieval (96 chunks, 3 rotations)
- **diverse**: High variety (80 chunks, 2 rotations, limited per file)

### Exploration vs Exploitation

- **First 20 queries**: Exploration phase (tries all presets)
- **After 20 queries**: Exploitation phase (favors best performers)
- Thompson Sampling naturally balances trying new strategies vs using proven ones

## Perspective-Aware Retrieval

ARIA detects query perspective and applies appropriate rotation parameters:

- **Educational**: 30° (gentle, broad concepts)
- **Diagnostic**: 90° (aggressive, focused search)
- **Security**: 45° (moderate, threat analysis)
- **Implementation**: 60° (strong, code/building)
- **Research**: 120° (very aggressive, investigation)
- **Theoretical**: 75° (strong, abstract concepts)
- **Practical**: 50° (moderate, applied knowledge)
- **Reference**: 15° (minimal, factual lookup)

Rotation angles are scaled by confidence and user profile adjustments.

## Quaternion Exploration

Each retrieval uses quaternion rotations in semantic space:

1. **Golden ratio spiral** sampling for uniform coverage
2. **Multi-rotation exploration** for iterative refinement
3. **PCA alignment** with semantic structure
4. **φ-scaled angles** (golden ratio = 1.618...) for resonance avoidance

## Student ARIA Learning

The conversation watcher:

1. **Monitors** `~/.lmstudio/conversations/` for new conversations
2. **Captures** ALL conversations (not just ARIA queries)
3. **Learns** cross-domain synthesis patterns
4. **Builds** corpus of thinking patterns for future training

### Corpus Location

`../training_data/conversation_corpus/` (relative to project root)

## System Requirements

- Python 3.8+
- LM Studio (for Student ARIA corpus learning)
- Datasets folder at `../datasets/` (relative to aria/)

## Architecture

```
aria_control_center.py
├── TeacherARIA
│   ├── Initialize ARIA core
│   ├── Execute queries
│   └── Track statistics
├── StudentARIA
│   ├── Conversation watcher
│   ├── Corpus management
│   └── Learning statistics
└── ControlCenter
    ├── Unified menu
    ├── Status dashboard
    └── System coordination
```

## State Management

- **Bandit state**: `~/.aria/bandit_state.json`
- **Watcher state**: `../var/watcher_state.json`
- **Telemetry**: `../var/telemetry/`
- **Packs**: `aria_packs/`

## Integration

The control center integrates:
- `src/core/aria_core.py` - Main orchestrator
- `src/retrieval/local_rag_context_v7_guided_exploration.py` - Retrieval engine
- `src/intelligence/bandit_context.py` - Thompson Sampling
- `src/intelligence/aria_exploration.py` - Quaternion exploration
- `src/perspective/detector.py` - Perspective detection
- `../src/student/conversation_watcher.py` - Corpus learning

## Testing

Run the comprehensive test suite:

```bash
python3 tests/comprehensive_test_suite.py
```

Or run the flywheel test directly:

```bash
python3 ../aria_systems_test_and_analysis/stress_tests/test_real_data_flywheel.py
```

## Notes

- **Preset selection is fully dynamic** - Thompson Sampling adapts to query patterns
- **No manual preset specification needed** - Bandit learns optimal strategies
- **Compound reward signal** ensures multi-objective optimization
- **Perspective detection** biases retrieval towards appropriate semantic regions
- **Quaternion rotations** provide unique semantic exploration capabilities
- **Student learning** captures cross-domain patterns for continuous improvement

---

**Version**: 1.0
**Date**: 2025-11-13
**Status**: Production Ready
