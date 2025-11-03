#!/usr/bin/env python3
"""
Test Quaternion and PCA Exploration
====================================

Standalone test to verify the exploration modules work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Test corpus
CORPUS = [
    "Machine learning uses algorithms to learn patterns from data",
    "Neural networks are inspired by biological neurons in the brain",
    "Deep learning employs multiple layers to extract hierarchical features",
    "Natural language processing enables computers to understand human language",
    "Computer vision allows machines to interpret visual information",
    "Reinforcement learning trains agents through rewards and penalties",
    "Supervised learning requires labeled training data",
    "Unsupervised learning discovers hidden patterns without labels",
    "Transfer learning leverages pre-trained models for new tasks",
    "Quantum computing promises exponential speedups for certain problems"
]

def test_quaternion_state():
    """Test 1: Quaternion State Manager"""
    print("\n" + "="*60)
    print("TEST 1: Quaternion State Manager")
    print("="*60)
    
    try:
        from quaternion_state import QuaternionStateManager, s3_normalize
        
        # Create manager
        mgr = QuaternionStateManager(Path("/tmp/test_qstate"))
        print("✓ Created QuaternionStateManager")
        
        # Fit semantic model
        mgr.fit_semantic_model(CORPUS)
        print(f"✓ Fitted 4D semantic model on {len(CORPUS)} documents")
        print(f"  Current state: {mgr.q}")
        print(f"  State norm: {np.linalg.norm(mgr.q):.6f} (should be ~1.0)")
        
        # Query to 4D
        query = "what is deep learning?"
        query_4d = mgr.query_to_4d(query)
        print(f"\n✓ Query to 4D: '{query}'")
        print(f"  4D vector: {query_4d}")
        print(f"  Norm: {np.linalg.norm(query_4d):.6f}")
        
        # Update state
        doc_vecs = [mgr.query_to_4d(doc) for doc in CORPUS[:3]]
        mgr.update_state(doc_vecs)
        print(f"\n✓ Updated state with momentum")
        print(f"  New state: {mgr.q}")
        print(f"  Momentum: {mgr.momentum}")
        
        # Save state
        mgr.save_state("test_query", {"test": True})
        print(f"\n✓ Saved state to: {mgr.assoc_path}")
        
        # Recall similar
        similar = mgr.recall_similar_states("deep learning", top_k=1)
        if similar:
            print(f"\n✓ Recalled {len(similar)} similar state(s)")
            print(f"  Label: {similar[0].get('label')}")
        
        print("\n✅ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pca_exploration():
    """Test 2: PCA Exploration"""
    print("\n" + "="*60)
    print("TEST 2: PCA Exploration")
    print("="*60)
    
    try:
        from pca_exploration import PCAExplorer, random_rotation_matrix_3d, golden_ratio_angles
        
        # Test rotation matrix
        R = random_rotation_matrix_3d(max_degrees=10.0)
        print("✓ Created 3D rotation matrix")
        print(f"  Shape: {R.shape}")
        print(f"  Det(R): {np.linalg.det(R):.6f} (should be ~1.0)")
        
        # Test golden ratio angles
        angles = golden_ratio_angles(8)
        print(f"\n✓ Generated {len(angles)} golden ratio angles")
        print(f"  First 3 angles (radians): {angles[:3]}")
        
        # Create explorer
        explorer = PCAExplorer(
            n_rotations=5,
            subspace_dim=8,
            max_angle_deg=10.0,
            combine_mode="max"
        )
        print("\n✓ Created PCAExplorer")
        
        # Create dummy embeddings
        np.random.seed(42)
        query_vec = np.random.randn(128)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        doc_vecs = np.random.randn(len(CORPUS), 128)
        doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        
        print(f"\n✓ Created dummy embeddings")
        print(f"  Query shape: {query_vec.shape}")
        print(f"  Docs shape: {doc_vecs.shape}")
        
        # Explore
        scores, msg = explorer.explore(query_vec, doc_vecs, top_n=10)
        print(f"\n✓ Performed PCA exploration")
        print(f"  Message: {msg}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Top 3 scores: {scores[:3]}")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        print("\n✅ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test 3: Full Integration"""
    print("\n" + "="*60)
    print("TEST 3: Exploration Integration")
    print("="*60)
    
    try:
        from aria_exploration import create_exploration_manager
        
        # Create manager
        mgr = create_exploration_manager(
            state_dir=Path("/tmp/test_exploration"),
            corpus_texts=CORPUS
        )
        print("✓ Created ExplorationManager")
        print(f"  Quaternion enabled: {mgr.quaternion_mgr is not None}")
        print(f"  PCA enabled: {mgr.pca_explorer is not None}")
        
        # Mock retrieval results
        results = {
            "items": [
                {"text": CORPUS[0], "score": 0.8, "path": "doc1.txt", "chunk_id": 0},
                {"text": CORPUS[1], "score": 0.6, "path": "doc2.txt", "chunk_id": 0},
                {"text": CORPUS[2], "score": 0.5, "path": "doc3.txt", "chunk_id": 0},
            ],
            "meta": {"query": "machine learning", "scanned": 10, "kept": 3}
        }
        
        print("\n✓ Created mock retrieval results")
        print(f"  Items: {len(results['items'])}")
        
        # Test golden ratio spiral
        enhanced = mgr.explore(
            query="machine learning algorithms",
            retrieval_results=results.copy(),
            strategy="golden_ratio_spiral"
        )
        
        print(f"\n✓ Applied golden_ratio_spiral exploration")
        print(f"  Applied: {enhanced['exploration']['applied']}")
        print(f"  Strategy: {enhanced['exploration']['strategy']}")
        
        if enhanced['exploration']['applied']:
            print(f"  Items reranked: {len(enhanced['items'])}")
            print(f"  Top item score: {enhanced['items'][0]['score']:.4f}")
            if 'quaternion_score' in enhanced['items'][0]:
                print(f"  Quaternion score: {enhanced['items'][0]['quaternion_score']:.4f}")
        
        # Get stats
        stats = mgr.get_stats()
        print(f"\n✓ Exploration stats:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Quaternion updates: {stats['quaternion_updates']}")
        
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ARIA EXPLORATION TEST SUITE")
    print("="*60)
    
    results = {
        "quaternion_state": test_quaternion_state(),
        "pca_exploration": test_pca_exploration(),
        "integration": test_integration()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
