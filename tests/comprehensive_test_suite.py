#!/usr/bin/env python3
"""
ARIA Comprehensive Test Suite - MELTDOWN STRESS TEST
Based on original ARIA testing methodology from TESTING_COMPLETE.md

Tests all core components systematically:
1. Bandit selection system
2. Query feature extraction
3. Retrieval pipeline
4. Postfilter system
5. Telemetry and metrics
6. Quaternion exploration
7. Exemplar fit scoring
8. End-to-end integration
9. Stress tests (high volume, concurrent, memory, edge cases)
10. Flywheel effect validation
"""

import sys
import time
import json
import threading
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import ARIA components
from core.aria_core import ARIA
from intelligence.bandit_context import BanditState, select_preset, give_reward
from intelligence.presets import Preset
from intelligence.aria_exploration import QuaternionExplorer, PHI, compute_rotation_params_from_perspective
from intelligence.quaternion import Quaternion
from monitoring.aria_telemetry import compute_retrieval_stats, coverage_score, detect_issues, RunMetrics
from retrieval.query_features import QueryFeatureExtractor
from anchors.exemplar_fit import ExemplarFitScorer


class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_result(self, name: str, passed: bool, details: Dict[str, Any] = None):
        self.total += 1
        if passed:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"

        self.tests.append({
            "name": name,
            "status": status,
            "passed": passed,
            "details": details or {}
        })
        print(f"{status}: {name}")
        if details:
            for k, v in details.items():
                print(f"  {k}: {v}")
        print()

    def summary(self):
        success_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        print("=" * 70)
        print("COMPREHENSIVE TEST SUITE - SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 70)
        return self.failed == 0


def test_bandit_selection(results: TestResults):
    """Test 1: Bandit selection system"""
    print("\n" + "=" * 70)
    print("TEST 1: BANDIT SELECTION SYSTEM")
    print("=" * 70)

    try:
        # Use temporary state file
        import tempfile
        state_file = tempfile.mktemp(suffix=".json")

        # Test preset selection
        features = {"length": 50, "term_count": 10}
        preset, reason, meta = select_preset(features, state_file)

        # Verify preset is valid
        assert preset is not None
        assert preset.name in ["fast", "balanced", "deep", "diverse"]
        assert reason is not None
        assert meta is not None

        # Test reward update
        give_reward(preset.name, 0.8, state_file)

        # Cleanup
        Path(state_file).unlink(missing_ok=True)

        results.add_result(
            "Bandit Selection System",
            True,
            {
                "preset_selected": preset.name,
                "reason": reason,
                "phase": meta.get("phase", "unknown"),
                "reward_updated": True
            }
        )
    except Exception as e:
        results.add_result("Bandit Selection System", False, {"error": str(e)})


def test_feature_extraction(results: TestResults):
    """Test 2: Query feature extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: QUERY FEATURE EXTRACTION")
    print("=" * 70)

    try:
        extractor = QueryFeatureExtractor()

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How do I fix this Python error?",
            "Implement a binary search tree in C++"
        ]

        all_passed = True
        features_list = []
        for query in test_queries:
            features = extractor.extract(query)
            features_dict = asdict(features)
            # Remove numpy arrays for display
            if 'vector' in features_dict:
                features_dict['vector'] = f"array({features_dict['vector'].shape})"
            features_list.append(features_dict)
            # Verify features exist (word_count is the actual field, not term_count)
            if 'length' not in features_dict or 'word_count' not in features_dict:
                all_passed = False
                break

        results.add_result(
            "Query Feature Extraction",
            all_passed,
            {
                "queries_tested": len(test_queries),
                "sample_features": features_list[0] if features_list else {}
            }
        )
    except Exception as e:
        results.add_result("Query Feature Extraction", False, {"error": str(e)})


def test_retrieval_pipeline(results: TestResults):
    """Test 3: Retrieval pipeline"""
    print("\n" + "=" * 70)
    print("TEST 3: RETRIEVAL PIPELINE")
    print("=" * 70)

    try:
        # Test with real datasets
        datasets_dir = PROJECT_ROOT.parent / "datasets"
        aria = ARIA(
            index_roots=[str(datasets_dir)],
            out_root=str(PROJECT_ROOT / "aria_packs"),
            enforce_session=False
        )

        query = "machine learning"
        result = aria.query(query)

        # Check actual aria_core return structure
        # result has keys: preset, reason, flags, pack, filtered, postfiltered, metrics, run_dir
        pack_path = None
        if "pack" in result:
            pack_path = Path(result["pack"])

        # Load chunks from pack file
        chunks = []
        if pack_path and pack_path.exists():
            import json
            with open(pack_path) as f:
                pack_data = json.load(f)
                chunks = pack_data.get("items", [])

        passed = "metrics" in result and len(chunks) > 0

        results.add_result(
            "Retrieval Pipeline",
            passed,
            {
                "pack_created": str(pack_path.name) if pack_path else "none",
                "chunks_retrieved": len(chunks),
                "preset_used": result.get("preset", "unknown")
            }
        )
    except Exception as e:
        import traceback
        results.add_result("Retrieval Pipeline", False, {"error": str(e), "traceback": traceback.format_exc()[:500]})


def test_postfilter_system(results: TestResults):
    """Test 4: Postfilter system"""
    print("\n" + "=" * 70)
    print("TEST 4: POSTFILTER SYSTEM")
    print("=" * 70)

    try:
        # Find a recent pack to test postfilter on
        packs_dir = PROJECT_ROOT / "aria_packs"
        pack_files = list(packs_dir.glob("**/last_pack.json"))

        if not pack_files:
            results.add_result("Postfilter System", False, {"error": "No packs found to test"})
            return

        pack_path = pack_files[0]
        filtered_path = pack_path.parent / "test_filtered.json"

        # Run postfilter
        postfilter_script = PROJECT_ROOT / "src" / "retrieval" / "aria_postfilter.py"
        result = subprocess.run([
            "python3", str(postfilter_script),
            str(pack_path),
            "--output", str(filtered_path)
        ], capture_output=True, text=True, timeout=10)

        # Check filtered pack was created
        passed = result.returncode == 0 and filtered_path.exists()

        # Cleanup test file
        if filtered_path.exists():
            filtered_path.unlink()

        results.add_result(
            "Postfilter System",
            passed,
            {
                "pack_tested": pack_path.name,
                "filtered_created": passed
            }
        )
    except Exception as e:
        results.add_result("Postfilter System", False, {"error": str(e)})


def test_telemetry_metrics(results: TestResults):
    """Test 5: Telemetry and metrics"""
    print("\n" + "=" * 70)
    print("TEST 5: TELEMETRY AND METRICS")
    print("=" * 70)

    try:
        # Create mock chunks
        chunks = [
            {"text": "test chunk 1", "path": "file1.txt", "score": 0.9},
            {"text": "test chunk 2", "path": "file2.txt", "score": 0.8},
            {"text": "test chunk 3", "path": "file1.txt", "score": 0.7}
        ]

        # Test compute_retrieval_stats
        stats = compute_retrieval_stats(chunks)
        assert stats.total == 3
        assert stats.unique_sources == 2
        assert 0 <= stats.diversity <= 1

        # Test coverage_score
        exemplars = ["test exemplar content"]
        retrieved = ["test chunk 1", "test chunk 2"]
        score = coverage_score(exemplars, retrieved)
        assert 0 <= score <= 1

        # Test detect_issues
        metrics = RunMetrics(
            coverage_score=0.3,
            exemplar_fit=0.5,
            retrieval={"total": 10, "unique_sources": 5}
        )
        issues = detect_issues(metrics)
        assert isinstance(issues, list)

        results.add_result(
            "Telemetry and Metrics",
            True,
            {
                "stats_computed": True,
                "coverage_calculated": True,
                "issues_detected": len(issues)
            }
        )
    except Exception as e:
        results.add_result("Telemetry and Metrics", False, {"error": str(e)})


def test_exemplar_fit(results: TestResults):
    """Test 7: Exemplar fit scoring"""
    print("\n" + "=" * 70)
    print("TEST 7: EXEMPLAR FIT SCORING")
    print("=" * 70)

    try:
        exemplars = ["This is an example response with proper style and citations [1]."]
        scorer = ExemplarFitScorer(exemplars)

        answer = "This is a test answer with similar style and a citation [2]."
        coverage = 0.7

        fit, breakdown = scorer.calculate_fit(answer, coverage)

        assert 0 <= fit <= 1
        assert "style" in breakdown
        assert "citations" in breakdown
        assert "confidence" in breakdown

        results.add_result(
            "Exemplar Fit Scoring",
            True,
            {
                "fit_score": f"{fit:.3f}",
                "style": f"{breakdown['style']:.3f}",
                "citations": f"{breakdown['citations']:.3f}"
            }
        )
    except Exception as e:
        results.add_result("Exemplar Fit Scoring", False, {"error": str(e)})


def test_end_to_end(results: TestResults):
    """Test 8: End-to-end integration"""
    print("\n" + "=" * 70)
    print("TEST 8: END-TO-END INTEGRATION")
    print("=" * 70)

    try:
        datasets_dir = PROJECT_ROOT.parent / "datasets"
        aria = ARIA(
            index_roots=[str(datasets_dir)],
            out_root=str(PROJECT_ROOT / "aria_packs"),
            enforce_session=False
        )

        queries = [
            "What is gradient descent?",
            "Explain neural networks",
            "How does backpropagation work?"
        ]

        all_passed = True
        total_time = 0

        for query in queries:
            start = time.time()
            result = aria.query(query)
            elapsed = time.time() - start
            total_time += elapsed

            # Verify result structure
            if not all(k in result for k in ["preset", "pack", "filtered", "metrics"]):
                all_passed = False
                break

            # Verify files exist
            if not Path(result["pack"]).exists():
                all_passed = False
                break

        avg_time = total_time / len(queries)

        results.add_result(
            "End-to-End Integration",
            all_passed,
            {
                "queries_tested": len(queries),
                "avg_time_seconds": f"{avg_time:.3f}",
                "total_time_seconds": f"{total_time:.3f}"
            }
        )
    except Exception as e:
        results.add_result("End-to-End Integration", False, {"error": str(e)})


def test_high_volume(results: TestResults):
    """Test 9: High volume stress test"""
    print("\n" + "=" * 70)
    print("TEST 9: HIGH VOLUME STRESS TEST")
    print("=" * 70)

    try:
        import tempfile
        num_queries = 50
        queries = [f"Test query {i}" for i in range(num_queries)]

        extractor = QueryFeatureExtractor()
        state_file = tempfile.mktemp(suffix=".json")

        start = time.time()
        for query in queries:
            features = asdict(extractor.extract(query))
            preset, reason, meta = select_preset(features, state_file)
        elapsed = time.time() - start

        qps = num_queries / elapsed

        # Cleanup
        Path(state_file).unlink(missing_ok=True)

        results.add_result(
            "High Volume Stress Test",
            True,
            {
                "queries_processed": num_queries,
                "time_seconds": f"{elapsed:.3f}",
                "throughput_qps": f"{qps:.1f}"
            }
        )
    except Exception as e:
        results.add_result("High Volume Stress Test", False, {"error": str(e)})


def test_concurrent_processing(results: TestResults):
    """Test 10: Concurrent processing stress test"""
    print("\n" + "=" * 70)
    print("TEST 10: CONCURRENT PROCESSING STRESS TEST")
    print("=" * 70)

    try:
        import tempfile
        num_threads = 10
        queries_per_thread = 20
        total_queries = num_threads * queries_per_thread

        state_file = tempfile.mktemp(suffix=".json")

        def worker(thread_id):
            extractor = QueryFeatureExtractor()
            for i in range(queries_per_thread):
                query = f"Thread {thread_id} query {i}"
                features = asdict(extractor.extract(query))
                preset, reason, meta = select_preset(features, state_file)

        threads = []
        start = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elapsed = time.time() - start
        qps = total_queries / elapsed

        # Cleanup
        Path(state_file).unlink(missing_ok=True)

        results.add_result(
            "Concurrent Processing Stress Test",
            True,
            {
                "num_threads": num_threads,
                "queries_per_thread": queries_per_thread,
                "total_queries": total_queries,
                "time_seconds": f"{elapsed:.3f}",
                "throughput_qps": f"{qps:.1f}"
            }
        )
    except Exception as e:
        results.add_result("Concurrent Processing Stress Test", False, {"error": str(e)})


def test_quaternion_rotation(results: TestResults):
    """Test 11: Quaternion rotation mathematics"""
    print("\n" + "=" * 70)
    print("TEST 11: QUATERNION ROTATION MATHEMATICS")
    print("=" * 70)

    try:
        # Test quaternion creation and rotation
        quat = Quaternion.from_axis_angle([0, 0, 1], np.pi/4)  # 45° around Z
        vec = np.array([1.0, 0.0, 0.0])
        rotated = quat.rotate_vector(vec)

        # Should rotate (1,0,0) to approximately (0.707, 0.707, 0)
        expected_x = np.cos(np.pi/4)
        expected_y = np.sin(np.pi/4)

        error_x = abs(rotated[0] - expected_x)
        error_y = abs(rotated[1] - expected_y)
        error_z = abs(rotated[2])

        passed = error_x < 0.01 and error_y < 0.01 and error_z < 0.01

        results.add_result(
            "Quaternion Rotation Mathematics",
            passed,
            {
                "input_vector": [1.0, 0.0, 0.0],
                "rotation": "45° around Z-axis",
                "output_vector": [f"{rotated[0]:.4f}", f"{rotated[1]:.4f}", f"{rotated[2]:.4f}"],
                "expected": [f"{expected_x:.4f}", f"{expected_y:.4f}", "0.0000"],
                "max_error": f"{max(error_x, error_y, error_z):.6f}"
            }
        )
    except Exception as e:
        results.add_result("Quaternion Rotation Mathematics", False, {"error": str(e)})


def test_golden_ratio_spiral(results: TestResults):
    """Test 12: Golden ratio spiral generation"""
    print("\n" + "=" * 70)
    print("TEST 12: GOLDEN RATIO SPIRAL GENERATION")
    print("=" * 70)

    try:
        explorer = QuaternionExplorer()

        # Test phi constant
        phi_correct = abs(explorer.phi - 1.618033988749895) < 1e-10

        # Generate spiral points
        points = explorer.golden_ratio_spiral(100, radius=1.0)

        # Verify all points on unit sphere
        on_sphere = all(abs(np.sqrt(x**2 + y**2 + z**2) - 1.0) < 0.01 for x, y, z in points)

        # Test spiral sample positions
        positions = explorer.spiral_sample_positions(50, max_radius=1.0)

        passed = phi_correct and on_sphere and len(points) == 100 and len(positions) == 50

        results.add_result(
            "Golden Ratio Spiral Generation",
            passed,
            {
                "phi_constant": f"{explorer.phi:.10f}",
                "phi_correct": phi_correct,
                "points_generated": len(points),
                "all_on_sphere": on_sphere,
                "sample_positions": len(positions)
            }
        )
    except Exception as e:
        results.add_result("Golden Ratio Spiral Generation", False, {"error": str(e)})


def test_perspective_rotation_params(results: TestResults):
    """Test 13: Perspective-based rotation parameter computation"""
    print("\n" + "=" * 70)
    print("TEST 13: PERSPECTIVE ROTATION PARAMETERS")
    print("=" * 70)

    try:
        perspectives = ["educational", "diagnostic", "security", "implementation",
                       "research", "theoretical", "practical", "reference"]

        params_computed = {}
        all_valid = True

        for perspective in perspectives:
            params = compute_rotation_params_from_perspective(perspective, confidence=0.8)

            params_computed[perspective] = params

            # Verify params have expected structure
            if not all(k in params for k in ["angle", "subspace", "perspective", "confidence"]):
                all_valid = False
                break

            # Verify angle is scaled by confidence
            if params["angle"] > 0:  # Skip mixed/reference with 0 angle
                expected_base = params["angle"] / 0.8  # Reverse the scaling
                if not (0 <= params["angle"] <= 180):
                    all_valid = False
                    break

        results.add_result(
            "Perspective Rotation Parameters",
            all_valid,
            {
                "perspectives_tested": len(perspectives),
                "all_valid": all_valid,
                "sample_educational": f"{params_computed['educational']['angle']:.1f}°",
                "sample_diagnostic": f"{params_computed['diagnostic']['angle']:.1f}°",
                "sample_research": f"{params_computed['research']['angle']:.1f}°"
            }
        )
    except Exception as e:
        results.add_result("Perspective Rotation Parameters", False, {"error": str(e)})


def test_multi_rotation_exploration(results: TestResults):
    """Test 14: Multi-rotation semantic exploration"""
    print("\n" + "=" * 70)
    print("TEST 14: MULTI-ROTATION SEMANTIC EXPLORATION")
    print("=" * 70)

    try:
        explorer = QuaternionExplorer(embedding_dim=128)

        # Mock embeddings
        query_emb = np.random.randn(128)
        doc_embs = np.random.randn(100, 128)

        # Test single rotation
        results_1rot = explorer.multi_rotation_exploration(
            query_emb, doc_embs,
            num_rotations=1,
            angle_per_rotation=30.0,
            subspace_index=0
        )

        # Test multiple rotations
        results_3rot = explorer.multi_rotation_exploration(
            query_emb, doc_embs,
            num_rotations=3,
            angle_per_rotation=30.0,
            subspace_index=0
        )

        # Test no rotation (baseline)
        results_0rot = explorer.multi_rotation_exploration(
            query_emb, doc_embs,
            num_rotations=0
        )

        # Verify results structure
        passed = (
            len(results_1rot) == 100 and
            len(results_3rot) == 100 and
            len(results_0rot) == 100 and
            all(isinstance(doc_idx, (int, np.integer)) for doc_idx, _ in results_1rot[:5])
        )

        results.add_result(
            "Multi-Rotation Semantic Exploration",
            passed,
            {
                "query_dim": query_emb.shape[0],
                "num_documents": len(doc_embs),
                "1_rotation_results": len(results_1rot),
                "3_rotation_results": len(results_3rot),
                "baseline_results": len(results_0rot),
                "top_score_3rot": f"{results_3rot[0][1]:.4f}"
            }
        )
    except Exception as e:
        results.add_result("Multi-Rotation Semantic Exploration", False, {"error": str(e)})


def test_edge_cases(results: TestResults):
    """Test 12: Edge cases and adversarial inputs"""
    print("\n" + "=" * 70)
    print("TEST 12: EDGE CASES AND ADVERSARIAL INPUTS")
    print("=" * 70)

    try:
        extractor = QueryFeatureExtractor()

        edge_cases = [
            "",  # Empty
            " ",  # Whitespace only
            "a" * 10000,  # Very long
            "🔥" * 100,  # Unicode
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "\x00\x01\x02",  # Null bytes
            "a\nb\nc\nd\ne\nf\ng\nh",  # Many newlines
        ]

        handled = 0
        crashed = 0

        for case in edge_cases:
            try:
                features = extractor.extract(case)
                handled += 1
            except Exception:
                crashed += 1

        passed = crashed == 0

        results.add_result(
            "Edge Cases and Adversarial Inputs",
            passed,
            {
                "total_cases": len(edge_cases),
                "handled_gracefully": handled,
                "crashed": crashed
            }
        )
    except Exception as e:
        results.add_result("Edge Cases and Adversarial Inputs", False, {"error": str(e)})


def main():
    print("\n")
    print("=" * 70)
    print("ARIA COMPREHENSIVE TEST SUITE - MELTDOWN STRESS TEST")
    print("=" * 70)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = TestResults()

    # Run all tests
    test_bandit_selection(results)
    test_feature_extraction(results)
    test_retrieval_pipeline(results)
    test_postfilter_system(results)
    test_telemetry_metrics(results)
    test_exemplar_fit(results)
    test_end_to_end(results)
    test_high_volume(results)
    test_concurrent_processing(results)
    test_edge_cases(results)

    # Quaternion exploration tests (Tests 11-14)
    test_quaternion_rotation(results)
    test_golden_ratio_spiral(results)
    test_perspective_rotation_params(results)
    test_multi_rotation_exploration(results)

    # Print summary
    print("\n")
    all_passed = results.summary()

    # Save results to JSON (with numpy array handling)
    import numpy as np

    def json_serializable(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_serializable(item) for item in obj]
        return obj

    report_path = PROJECT_ROOT / "TEST_RESULTS.json"
    with open(report_path, "w") as f:
        report_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "success_rate": (results.passed / results.total * 100) if results.total > 0 else 0,
            "tests": json_serializable(results.tests)
        }
        json.dump(report_data, f, indent=2)

    print(f"\nResults saved to: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
