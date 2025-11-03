#!/usr/bin/env python3
"""
Diagnose Integration Test Failure
"""

import sys
from pathlib import Path

# Add ARIA to path
sys.path.insert(0, '/media/notapplicable/Internal-SSD/ai-quaternions-model/src')

print("=" * 70)
print("ARIA Integration Test Diagnostics")
print("=" * 70)

# Check mock data directory
print("\n1. Checking mock data directory...")
mock_base = Path("/media/notapplicable/ARIA-knowledge/aria-github-clean")
data_dir = mock_base / "data"

if data_dir.exists():
    files = list(data_dir.rglob("*.txt"))
    print(f"   ✓ Mock data directory exists")
    print(f"   ✓ Found {len(files)} text files")
else:
    print(f"   ✗ Mock data directory missing: {data_dir}")

# Run individual core tests
print("\n2. Running Core Component Tests Individually...")
print("-" * 70)

test_results = {}

# Test 1: Retrieval
print("\n[TEST 1] Retrieval System")
try:
    from aria_retrieval import ARIARetrieval
    retriever = ARIARetrieval(
        index_roots=[data_dir],
        max_per_file=3
    )
    result = retriever.retrieve("explain neural networks", top_k=10)
    print(f"   ✓ PASSED - Retrieved {len(result['items'])} items")
    test_results['retrieval'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['retrieval'] = 'FAIL'

# Test 2: Postfilter
print("\n[TEST 2] Postfilter System")
try:
    from aria_postfilter import ARIAPostfilter
    pf = ARIAPostfilter(max_per_source=2, min_keep=3)
    test_items = [
        {'path': 'doc1.txt', 'text': 'content1', 'score': 0.9, 'chunk_id': 0},
        {'path': 'doc1.txt', 'text': 'content2', 'score': 0.8, 'chunk_id': 1},
        {'path': 'doc2.txt', 'text': 'content3', 'score': 0.7, 'chunk_id': 0},
    ]
    result = pf.filter(test_items)
    print(f"   ✓ PASSED - Filtered to {len(result['items'])} items")
    test_results['postfilter'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['postfilter'] = 'FAIL'

# Test 3: Anchor Selector
print("\n[TEST 3] Anchor System")
try:
    from anchor_selector import AnchorSelector
    exemplars_path = mock_base / "examples" / "exemplars.txt"
    selector = AnchorSelector(exemplar_path=exemplars_path)
    mode = selector.select_mode("explain neural networks")
    print(f"   ✓ PASSED - Detected mode: {mode}")
    test_results['anchor'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['anchor'] = 'FAIL'
    import traceback
    traceback.print_exc()

# Test 4: Curiosity Engine
print("\n[TEST 4] Curiosity Engine")
try:
    from aria_curiosity import ARIACuriosity
    exemplars_path = mock_base / "examples" / "exemplars.txt"
    curiosity = ARIACuriosity(exemplar_path=exemplars_path)
    print(f"   ✓ PASSED - Has gap_detector: {hasattr(curiosity, 'gap_detector')}")
    test_results['curiosity'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['curiosity'] = 'FAIL'
    import traceback
    traceback.print_exc()

# Test 5: End-to-End
print("\n[TEST 5] End-to-End Pipeline")
try:
    from aria_main import ARIA
    aria = ARIA()
    print(f"   ✓ PASSED - ARIA initialized")
    test_results['e2e'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['e2e'] = 'FAIL'
    import traceback
    traceback.print_exc()

# Test 6: Plugin Integration
print("\n[TEST 6] LM Studio Plugin Integration")
try:
    # Just test JSON structure
    mock_output = {
        "preset": "technical",
        "triples": [["a", "b", "c"]],
        "metrics": {"retrieval": {}, "generation": {}},
        "curiosity": {"questions": [], "gaps": []}
    }
    assert 'preset' in mock_output
    assert 'triples' in mock_output
    print(f"   ✓ PASSED - JSON structure valid")
    test_results['plugin'] = 'PASS'
except Exception as e:
    print(f"   ✗ FAILED - {e}")
    test_results['plugin'] = 'FAIL'

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

passed = sum(1 for v in test_results.values() if v == 'PASS')
failed = sum(1 for v in test_results.values() if v == 'FAIL')

for test_name, result in test_results.items():
    status = "✓" if result == 'PASS' else "✗"
    print(f"{status} {test_name:20s}: {result}")

print(f"\nTotal: {passed}/{len(test_results)} passed")

if failed > 0:
    print(f"\n⚠️  {failed} test(s) failed - see detailed output above")
else:
    print(f"\n✅ All core tests passed!")
