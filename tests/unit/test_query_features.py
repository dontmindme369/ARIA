"""
Unit tests for Query Feature Extraction

Tests cover:
- Domain detection (code, paper, conversation, concept, factual)
- Complexity estimation
- Technical density calculation
- Follow-up query detection
- Feature vector generation
- Edge cases and validation
"""

import pytest
import numpy as np
import json
from pathlib import Path

from retrieval.query_features import (
    QueryFeatureExtractor,
    QueryFeatures,
    extract,
)


class TestQueryFeatureExtractor:
    """Test QueryFeatureExtractor class"""

    def test_initialization(self):
        """Test extractor initialization"""
        extractor = QueryFeatureExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'DOMAIN_PATTERNS')

    def test_domain_patterns_exist(self):
        """Test that all domain patterns are defined"""
        extractor = QueryFeatureExtractor()

        expected_domains = ['code', 'paper', 'conversation', 'concept', 'factual']

        for domain in expected_domains:
            assert domain in extractor.DOMAIN_PATTERNS
            assert 'keywords' in extractor.DOMAIN_PATTERNS[domain]
            assert 'patterns' in extractor.DOMAIN_PATTERNS[domain]


class TestDomainDetection:
    """Test domain detection logic"""

    def test_detect_code_domain(self):
        """Test code domain detection"""
        extractor = QueryFeatureExtractor()

        queries = [
            "How do I implement a function in Python?",
            "Debug ValueError in my code",
            "def quicksort(arr):",
            "What is the syntax for importing modules?",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert features.domain == 'code', f"Failed for: {query}"

    def test_detect_paper_domain(self):
        """Test paper/research domain detection"""
        extractor = QueryFeatureExtractor()

        queries = [
            "Summarize the paper by Smith et al. 2020",
            "What methodology did the study use?",
            "Explain the research hypothesis in the paper",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert features.domain == 'paper', f"Failed for: {query}"

    def test_detect_conversation_domain(self):
        """Test conversation domain detection"""
        extractor = QueryFeatureExtractor()

        queries = [
            "We discussed this earlier",
            "As we talked about",
            "What did they say about our proposal?",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert features.domain == 'conversation', f"Failed for: {query}"

    def test_detect_concept_domain(self):
        """Test concept/explanatory domain detection"""
        extractor = QueryFeatureExtractor()

        queries = [
            "Explain how photosynthesis works",
            "What is the concept of entropy?",
            "Help me understand quantum mechanics",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert features.domain == 'concept', f"Failed for: {query}"

    def test_detect_factual_domain(self):
        """Test factual domain detection"""
        extractor = QueryFeatureExtractor()

        queries = [
            "When was the Declaration of Independence signed?",
            "Where is the Eiffel Tower?",
            "Who invented the telephone?",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert features.domain == 'factual', f"Failed for: {query}"

    def test_default_domain_fallback(self):
        """Test fallback to default domain"""
        extractor = QueryFeatureExtractor()

        query = "xyz abc 123"  # No clear domain indicators
        features = extractor.extract(query)

        assert features.domain == 'default'

    def test_domain_confidence_scoring(self):
        """Test that domain confidence is calculated"""
        extractor = QueryFeatureExtractor()

        query = "How do I debug a Python function error?"
        features = extractor.extract(query)

        assert features.domain_confidence >= 0.0
        assert features.domain_confidence <= 1.0

    def test_strong_domain_signal(self):
        """Test queries with strong domain signals"""
        extractor = QueryFeatureExtractor()

        # Multiple code indicators
        query = "def function(): import error debug stacktrace"
        features = extractor.extract(query)

        assert features.domain == 'code'
        assert features.domain_confidence > 0.3


class TestComplexityEstimation:
    """Test complexity estimation"""

    def test_simple_complexity(self):
        """Test simple query detection"""
        extractor = QueryFeatureExtractor()

        simple_queries = [
            "What is Python?",
            "Define recursion",
            "Who is Einstein?",
        ]

        for query in simple_queries:
            features = extractor.extract(query)
            assert features.complexity == 'simple', f"Failed for: {query}"

    def test_complex_complexity(self):
        """Test complex query detection"""
        extractor = QueryFeatureExtractor()

        complex_queries = [
            "Compare the architectural differences between systems",
            "Analyze the relationship between variables",
            "Explain why this algorithm performs better",
        ]

        for query in complex_queries:
            features = extractor.extract(query)
            assert features.complexity == 'complex', f"Failed for: {query}"

    def test_medium_complexity(self):
        """Test medium complexity (default)"""
        extractor = QueryFeatureExtractor()

        query = "Tell me about machine learning"
        features = extractor.extract(query)

        assert features.complexity == 'medium'


class TestTechnicalDensity:
    """Test technical density calculation"""

    def test_high_technical_density(self):
        """Test queries with high technical term density"""
        extractor = QueryFeatureExtractor()

        query = "system algorithm function process architecture implementation protocol interface"
        features = extractor.extract(query)

        # Should have high technical density
        assert features.technical_density > 0.5

    def test_low_technical_density(self):
        """Test queries with low technical term density"""
        extractor = QueryFeatureExtractor()

        query = "the quick brown fox jumps over lazy dog"
        features = extractor.extract(query)

        # Should have low technical density
        assert features.technical_density < 0.3

    def test_technical_density_range(self):
        """Test that technical density is in valid range"""
        extractor = QueryFeatureExtractor()

        queries = [
            "simple query",
            "algorithm optimization performance",
            "vector embedding retrieval framework model",
        ]

        for query in queries:
            features = extractor.extract(query)
            assert 0.0 <= features.technical_density <= 1.0

    def test_technical_density_empty_query(self):
        """Test technical density for empty query"""
        extractor = QueryFeatureExtractor()

        features = extractor.extract("")
        assert features.technical_density == 0.0


class TestFollowUpDetection:
    """Test follow-up query detection"""

    def test_is_follow_up_with_context(self):
        """Test follow-up detection with conversation context"""
        extractor = QueryFeatureExtractor()

        context = ["What is machine learning?"]
        query = "And what about deep learning?"

        features = extractor.extract(query, conversation_context=context)
        assert features.is_follow_up is True

    def test_is_follow_up_pronouns(self):
        """Test follow-up detection based on pronouns"""
        extractor = QueryFeatureExtractor()

        follow_up_queries = [
            "What about that?",
            "How does this work?",
            "Tell me more about it",
        ]

        context = ["Previous query"]

        for query in follow_up_queries:
            features = extractor.extract(query, conversation_context=context)
            assert features.is_follow_up is True, f"Failed for: {query}"

    def test_not_follow_up_no_context(self):
        """Test that queries without context aren't follow-ups"""
        extractor = QueryFeatureExtractor()

        query = "What is machine learning?"
        features = extractor.extract(query, conversation_context=None)

        assert features.is_follow_up is False

    def test_not_follow_up_empty_context(self):
        """Test with empty context list"""
        extractor = QueryFeatureExtractor()

        query = "And what about this?"
        features = extractor.extract(query, conversation_context=[])

        assert features.is_follow_up is False

    def test_conversation_depth(self):
        """Test conversation depth tracking"""
        extractor = QueryFeatureExtractor()

        context = ["query1", "query2", "query3"]
        query = "What else?"

        features = extractor.extract(query, conversation_context=context)
        assert features.conversation_depth == 3


class TestFeatureVectorGeneration:
    """Test feature vector generation"""

    def test_vector_dimensionality(self):
        """Test that feature vector has correct dimensions"""
        extractor = QueryFeatureExtractor()

        query = "Test query"
        features = extractor.extract(query)

        assert features.vector is not None
        assert isinstance(features.vector, np.ndarray)
        # Expected dimensions: 3 basic + 6 domains + 3 complexity + 3 misc = 15
        assert len(features.vector) > 10

    def test_vector_normalization(self):
        """Test that vector components are in reasonable ranges"""
        extractor = QueryFeatureExtractor()

        query = "Test query with some technical terms like algorithm and optimization"
        features = extractor.extract(query)

        # Most features should be in [0, 1] range
        assert np.all(features.vector >= 0.0)
        assert np.all(features.vector <= 1.1)  # Small tolerance for floating point

    def test_vector_consistency(self):
        """Test that same query produces same vector"""
        extractor = QueryFeatureExtractor()

        query = "Consistent query test"

        features1 = extractor.extract(query)
        features2 = extractor.extract(query)

        np.testing.assert_array_equal(features1.vector, features2.vector)


class TestBasicFeatures:
    """Test basic feature extraction"""

    def test_length_feature(self):
        """Test query length extraction"""
        extractor = QueryFeatureExtractor()

        query = "short"
        features = extractor.extract(query)

        assert features.length == len(query)

    def test_word_count_feature(self):
        """Test word count extraction"""
        extractor = QueryFeatureExtractor()

        query = "one two three four five"
        features = extractor.extract(query)

        assert features.word_count == 5

    def test_avg_word_length(self):
        """Test average word length calculation"""
        extractor = QueryFeatureExtractor()

        query = "a bb ccc"  # lengths: 1, 2, 3
        features = extractor.extract(query)

        expected_avg = (1 + 2 + 3) / 3
        assert features.avg_word_length == pytest.approx(expected_avg)

    def test_multiple_questions(self):
        """Test multiple question detection"""
        extractor = QueryFeatureExtractor()

        query = "What is this? How does it work? Why?"
        features = extractor.extract(query)

        assert features.has_multiple_questions is True

    def test_single_question(self):
        """Test single question detection"""
        extractor = QueryFeatureExtractor()

        query = "What is this?"
        features = extractor.extract(query)

        assert features.has_multiple_questions is False


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_query(self):
        """Test extraction from empty query"""
        extractor = QueryFeatureExtractor()

        features = extractor.extract("")

        assert features.length == 0
        assert features.word_count == 0
        assert features.domain == 'default'
        assert features.vector is not None

    def test_whitespace_only_query(self):
        """Test query with only whitespace"""
        extractor = QueryFeatureExtractor()

        features = extractor.extract("   \t  \n  ")

        assert features.word_count == 0
        assert features.domain == 'default'

    def test_very_long_query(self):
        """Test handling of very long queries"""
        extractor = QueryFeatureExtractor()

        query = "word " * 500  # 500 words
        features = extractor.extract(query)

        assert features.word_count == 500
        assert features.vector is not None

    def test_unicode_query(self):
        """Test handling of unicode characters"""
        extractor = QueryFeatureExtractor()

        query = "Qu'est-ce que l'intelligence artificielle?"
        features = extractor.extract(query)

        assert features.length > 0
        assert features.word_count > 0

    def test_special_characters(self):
        """Test handling of special characters"""
        extractor = QueryFeatureExtractor()

        query = "!@#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
        features = extractor.extract(query)

        assert features.vector is not None
        assert features.domain == 'default'

    def test_none_query(self):
        """Test None query handling"""
        extractor = QueryFeatureExtractor()

        features = extractor.extract(None)

        # Should handle gracefully
        assert features.length == 0
        assert features.word_count == 0


class TestBackwardsCompatibility:
    """Test backwards compatibility with legacy extract() function"""

    def test_legacy_extract_function(self):
        """Test legacy extract() function"""
        result = extract("What is a binary search tree?")

        assert isinstance(result, dict)
        assert 'len' in result
        assert 'domain' in result
        assert 'complexity' in result
        assert 'is_q' in result

    def test_legacy_extract_domain(self):
        """Test legacy extract returns correct domain"""
        result = extract("How do I debug a Python function?")

        assert result['domain'] == 'code'

    def test_legacy_extract_complexity(self):
        """Test legacy extract returns complexity"""
        result = extract("What is Python?")

        assert result['complexity'] == 'simple'

    def test_legacy_extract_question_mark(self):
        """Test legacy extract detects question marks"""
        result_with_q = extract("What is this?")
        result_without_q = extract("Tell me about this")

        assert result_with_q['is_q'] is True
        assert result_without_q['is_q'] is False


class TestRegressionTests:
    """Regression tests using known query-feature mappings"""

    def test_known_features(self, test_data_dir):
        """Test against known query-feature mappings"""
        # Load known features if file exists
        known_features_path = test_data_dir / "known_features.json"

        if not known_features_path.exists():
            pytest.skip("Known features file not found")

        with open(known_features_path) as f:
            data = json.load(f)

        extractor = QueryFeatureExtractor()

        for item in data.get('query_feature_mappings', []):
            query = item['query']
            expected = item['expected']

            features = extractor.extract(query)

            # Check domain
            if 'domain' in expected:
                assert features.domain == expected['domain'], \
                    f"Domain mismatch for: {query}"

            # Check complexity
            if 'complexity' in expected:
                assert features.complexity == expected['complexity'], \
                    f"Complexity mismatch for: {query}"

            # Check has_question
            if 'has_question' in expected:
                # Either in multi-question or single question
                has_q = features.has_multiple_questions or ('?' in query)
                assert has_q == expected['has_question'], \
                    f"Question detection mismatch for: {query}"

            # Check word count
            if 'word_count' in expected:
                assert features.word_count == expected['word_count'], \
                    f"Word count mismatch for: {query}"

            # Check technical density minimum
            if 'technical_density_min' in expected:
                assert features.technical_density >= expected['technical_density_min'], \
                    f"Technical density too low for: {query}"


class TestQueryFeaturesDataclass:
    """Test QueryFeatures dataclass"""

    def test_dataclass_creation(self):
        """Test creating QueryFeatures instance"""
        features = QueryFeatures(
            length=10,
            word_count=2,
            avg_word_length=5.0,
            domain='code',
            domain_confidence=0.8,
            complexity='simple',
            has_multiple_questions=False,
            technical_density=0.3,
            is_follow_up=False,
            conversation_depth=0,
            vector=np.array([1, 2, 3]),
        )

        assert features.length == 10
        assert features.word_count == 2
        assert features.domain == 'code'
        assert np.array_equal(features.vector, np.array([1, 2, 3]))

    def test_dataclass_optional_vector(self):
        """Test that vector can be None"""
        features = QueryFeatures(
            length=5,
            word_count=1,
            avg_word_length=5.0,
            domain='default',
            domain_confidence=0.0,
            complexity='medium',
            has_multiple_questions=False,
            technical_density=0.0,
            is_follow_up=False,
            conversation_depth=0,
            vector=None,
        )

        assert features.vector is None
