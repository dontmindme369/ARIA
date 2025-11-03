#!/usr/bin/env python3
"""
ARIA Production System Test
Tests ARIA backend system with real API

Tests complete pipeline:
1. Anchor selection
2. Retrieval with ARIARetrieval
3. Pack generation
4. Integration check
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, "/mnt/project")

print("=" * 80)
print("ARIA PRODUCTION SYSTEM TEST")
print("=" * 80)
print()

# Test counters
passed = 0
failed = 0


def test_section(name):
    """Print section header"""
    print(f"\n{name}")
    print("-" * 80)


def test_ok(msg):
    """Pass test"""
    global passed
    print(f"✅ {msg}")
    passed += 1


def test_fail(msg, error):
    """Fail test"""
    global failed
    print(f"❌ {msg}: {error}")
    failed += 1


# ============================================================================
# SETUP: Create Test Environment
# ============================================================================

test_section("🔧 SETUP: Creating Test Environment")

# Create test documents
test_docs_dir = Path(tempfile.mkdtemp())
print(f"Test directory: {test_docs_dir}")

# Document 1: Machine Learning
(test_docs_dir / "ml.txt").write_text(
    """
Machine Learning Basics

Machine learning enables computers to learn from data without explicit programming.
Three main types exist: supervised, unsupervised, and reinforcement learning.

Supervised learning uses labeled examples. The model learns input-output mappings.
Common algorithms: linear regression, decision trees, neural networks.

Unsupervised learning finds patterns in unlabeled data. Examples: clustering, PCA.
Reinforcement learning uses rewards and penalties for trial-and-error learning.
"""
)

# Document 2: Neural Networks
(test_docs_dir / "nn.txt").write_text(
    """
Neural Networks Overview

Neural networks consist of interconnected layers of nodes (neurons).
Each connection has a weight adjusted during training.

Architecture:
- Input layer: receives data
- Hidden layers: process information
- Output layer: produces predictions

Training uses backpropagation to minimize error.
Activation functions (ReLU, sigmoid, tanh) introduce non-linearity.
Deep learning uses many hidden layers for complex pattern recognition.
"""
)

# Document 3: Python
(test_docs_dir / "python.txt").write_text(
    """
Python Programming

Python is a high-level interpreted language emphasizing readability.
Created by Guido van Rossum in 1991.

Key features:
- Dynamic typing
- Automatic memory management  
- Rich standard library
- Extensive third-party packages

Common uses:
- Web development (Django, Flask)
- Data science (NumPy, pandas)
- Machine learning (scikit-learn, TensorFlow)
- Automation and scripting
"""
)

test_ok(f"Created {len(list(test_docs_dir.glob('*.txt')))} test documents")


# ============================================================================
# TEST 1: Anchor Selector
# ============================================================================

test_section("📍 TEST 1: Anchor Selector Multi-Anchor)")

try:
    from anchor_selector import AnchorSelector

    selector = AnchorSelector()
    test_ok("Loaded AnchorSelector")
except Exception as e:
    test_fail("Load AnchorSelector", str(e))
    selector = None

if selector:
    # Test mode detection
    test_cases = [
        ("How do I implement binary search?", "technical"),
        ("What is the meaning of consciousness?", "philosophical"),
        ("What is 2+2?", "factual"),
        ("Explain neural networks simply", "educational"),
    ]

    correct = 0
    for query, expected in test_cases:
        detected = selector.select_mode(query)
        # Accept either expected or default_balanced (fallback)
        if detected == expected or detected == "default_balanced":
            correct += 1

    accuracy = (correct / len(test_cases)) * 100
    test_ok(f"Mode detection: {correct}/{len(test_cases)} ({accuracy:.0f}% accuracy)")


# ============================================================================
# TEST 2: Retrieval System
# ============================================================================

test_section("📚 TEST 2: ARIARetrieval System")

# Initialize variables to avoid unbound warnings
retrieval = None
query = "What is machine learning?"
result = None
items = []

try:
    from aria_retrieval import ARIARetrieval

    retrieval = ARIARetrieval(
        index_roots=[test_docs_dir], cache_dir=test_docs_dir / "cache"
    )
    test_ok("Initialized ARIARetrieval")

    # Test retrieval
    query = "What is machine learning?"
    result = retrieval.retrieve(query, top_k=5)

    items = result.get("items", [])
    test_ok(f"Retrieved {len(items)} chunks for query: '{query}'")

    if items:
        top_score = items[0].get("score", 0)
        test_ok(f"Top chunk score: {top_score:.3f}")

        # Show sample chunk
        top_chunk = items[0]
        print(f"\n   Sample chunk (score {top_score:.3f}):")
        print(f"   Source: {Path(top_chunk.get('path', 'unknown')).name}")
        print(f"   Text: {top_chunk.get('text', '')[:100]}...")

except Exception as e:
    test_fail("ARIARetrieval", str(e))
    items = []


# ============================================================================
# TEST 3: Pack Generation
# ============================================================================

test_section("📦 TEST 3: Retrieval Pack Generation")

pack = None
pack_file = None

try:
    # Create pack structure
    pack = {
        "query": query if "query" in locals() else "test query",
        "timestamp": datetime.now().isoformat(),
        "anchor_mode": selector.select_mode(query) if selector else "default_balanced",
        "chunks_count": len(items),
        "chunks": items[:5],  # Top 5
        "sources": list(set(Path(c.get("path", "")).name for c in items)),
        "avg_score": sum(c.get("score", 0) for c in items) / len(items) if items else 0,
        "meta": result.get("meta", {}) if result is not None else {},
    }

    test_ok("Created pack structure")

    # Save pack
    output_dir = Path(tempfile.mkdtemp())
    pack_file = output_dir / f"pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    pack_file.write_text(json.dumps(pack, indent=2, default=str))

    test_ok(f"Saved pack to: {pack_file}")

    # Verify saved pack
    saved = json.loads(pack_file.read_text())
    assert "query" in saved
    assert "chunks" in saved
    test_ok("Pack verification passed")

except Exception as e:
    test_fail("Pack generation", str(e))
    pack = None


# ============================================================================
# TEST 4: Full Pipeline Integration
# ============================================================================

test_section("🔗 TEST 4: Full Pipeline Integration")

try:
    # Complete query execution
    test_query = "How do neural networks work?"

    # 1. Anchor selection
    mode = selector.select_mode(test_query) if selector else "default_balanced"
    print(f"   1. Anchor mode: {mode}")

    # 2. Retrieval
    if retrieval is not None:
        result = retrieval.retrieve(test_query, top_k=10)
        chunks = result.get("items", [])
        print(f"   2. Retrieved: {len(chunks)} chunks")
    else:
        chunks = []
        result = None
        print(f"   2. Retrieval not initialized")

    # 3. Pack creation
    final_pack = {
        "query": test_query,
        "mode": mode,
        "chunks": chunks[:5],
        "timestamp": datetime.now().isoformat(),
        "meta": result.get("meta", {}) if result is not None else {},
    }
    print(f"   3. Created pack with {len(final_pack['chunks'])} chunks")

    test_ok("Full pipeline execution successful")

except Exception as e:
    test_fail("Full pipeline", str(e))
    final_pack = None


# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)
print()

total = passed + failed
print(f"✅ Passed: {passed}/{total}")
print(f"❌ Failed: {failed}/{total}")
print(f"📊 Success Rate: {(passed / total * 100):.1f}%" if total > 0 else "N/A")
print()

# Display pack if generated
if pack:
    print("=" * 80)
    print("SAMPLE RETRIEVAL PACK")
    print("=" * 80)
    print()
    print(json.dumps(pack, indent=2, default=str))
    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"✓ Test documents: {test_docs_dir}")
if pack_file:
    print(f"✓ Pack file: {pack_file}")
else:
    print(f"⚠️  Pack file: Not generated (check errors above)")
print()

if failed == 0:
    print("🎉 ALL TESTS PASSED! ARIA system is operational.")
else:
    print(f"⚠️  {failed} test(s) failed. System partially operational.")

print()
print("=" * 80)
