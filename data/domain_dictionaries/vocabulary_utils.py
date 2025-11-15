#!/usr/bin/env python3
"""
Vocabulary Utilities for Enhanced Semantic Network v2.0
Provides functions to load and work with v2 vocabularies in various formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict


class VocabularyV2Loader:
    """Load and work with Enhanced Semantic Network v2.0 vocabularies"""

    def __init__(self, dict_dir: Optional[str] = None):
        """
        Initialize vocabulary loader.

        Args:
            dict_dir: Directory containing vocabularies. Defaults to current file's directory.
        """
        if dict_dir is None:
            self.dict_dir = Path(__file__).parent
        else:
            self.dict_dir = Path(dict_dir)
        self.index_file = self.dict_dir / "domain_vocabulary_index_v2.json"
        self.index = None
        self.vocabularies = {}

    def load_index(self) -> Dict:
        """Load the v2 vocabulary index"""
        if self.index is None:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        return self.index

    def load_vocabulary(self, domain: str) -> Dict:
        """
        Load a specific domain vocabulary.

        Args:
            domain: Domain name (e.g., 'philosophy', 'engineering')

        Returns:
            Complete vocabulary dictionary
        """
        if domain in self.vocabularies:
            return self.vocabularies[domain]

        index = self.load_index()

        if domain not in index['vocabularies']:
            raise ValueError(f"Domain '{domain}' not found in index")

        vocab_path = self.dict_dir / index['vocabularies'][domain]['file']

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)

        self.vocabularies[domain] = vocabulary
        return vocabulary

    def load_all_vocabularies(self) -> Dict[str, Dict]:
        """Load all vocabularies listed in index"""
        index = self.load_index()

        for domain in index['vocabularies'].keys():
            if domain not in self.vocabularies:
                self.load_vocabulary(domain)

        return self.vocabularies

    def get_all_terms(self, domain: str) -> List[str]:
        """
        Get all concept terms from a domain vocabulary.

        Args:
            domain: Domain name

        Returns:
            List of all terms
        """
        vocab = self.load_vocabulary(domain)
        return list(vocab.get('concepts', {}).keys())

    def get_terms_with_synonyms(self, domain: str) -> List[str]:
        """
        Get all terms including synonyms.

        Args:
            domain: Domain name

        Returns:
            List of terms including all synonyms
        """
        vocab = self.load_vocabulary(domain)
        terms = []

        for concept_id, concept in vocab.get('concepts', {}).items():
            terms.append(concept.get('term', concept_id))
            terms.extend(concept.get('synonyms', []))

        return terms

    def get_terms_by_category(self, domain: str, category: str) -> List[str]:
        """
        Get terms filtered by category.

        Args:
            domain: Domain name
            category: Category name

        Returns:
            List of terms in that category
        """
        vocab = self.load_vocabulary(domain)
        terms = []

        for concept_id, concept in vocab.get('concepts', {}).items():
            if concept.get('category') == category:
                terms.append(concept.get('term', concept_id))

        return terms

    def get_terms_by_complexity(self, domain: str, complexity: str) -> List[str]:
        """
        Get terms filtered by complexity level.

        Args:
            domain: Domain name
            complexity: Complexity level (basic, intermediate, advanced, expert)

        Returns:
            List of terms at that complexity
        """
        vocab = self.load_vocabulary(domain)
        terms = []

        for concept_id, concept in vocab.get('concepts', {}).items():
            if concept.get('complexity') == complexity:
                terms.append(concept.get('term', concept_id))

        return terms

    def get_detection_patterns(self, domain: str) -> List[str]:
        """
        Get all detection patterns from a domain.

        Args:
            domain: Domain name

        Returns:
            List of all detection patterns
        """
        vocab = self.load_vocabulary(domain)
        patterns = []

        for concept in vocab.get('concepts', {}).values():
            patterns.extend(concept.get('detection_patterns', []))

        return list(set(patterns))  # Remove duplicates

    def get_reasoning_heuristics(self, domain: str) -> List[Dict]:
        """
        Get reasoning heuristics for a domain.

        Args:
            domain: Domain name

        Returns:
            List of reasoning heuristic dictionaries
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('reasoning_heuristics', [])

    def get_common_errors(self, domain: str) -> List[Dict]:
        """
        Get common errors for a domain.

        Args:
            domain: Domain name

        Returns:
            List of common error dictionaries
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('common_errors', [])

    def get_mental_models(self, domain: str) -> List[Dict]:
        """
        Get mental models for a domain.

        Args:
            domain: Domain name

        Returns:
            List of mental model dictionaries
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('mental_models', [])

    def get_concept(self, domain: str, concept_id: str) -> Optional[Dict]:
        """
        Get a specific concept by ID.

        Args:
            domain: Domain name
            concept_id: Concept identifier

        Returns:
            Concept dictionary or None
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('concepts', {}).get(concept_id)

    def get_related_concepts(self, domain: str, concept_id: str) -> List[str]:
        """
        Get concepts related to a given concept.

        Args:
            domain: Domain name
            concept_id: Concept identifier

        Returns:
            List of related concept IDs
        """
        concept = self.get_concept(domain, concept_id)
        if concept:
            return concept.get('related_concepts', [])
        return []

    def build_concept_graph(self, domain: str) -> Dict[str, List[str]]:
        """
        Build a graph of concept relationships.

        Args:
            domain: Domain name

        Returns:
            Dictionary mapping concept IDs to related concept IDs
        """
        vocab = self.load_vocabulary(domain)
        graph = {}

        for concept_id, concept in vocab.get('concepts', {}).items():
            graph[concept_id] = concept.get('related_concepts', [])

        return graph

    def to_legacy_format(self, domain: str) -> Dict[str, List[str]]:
        """
        Convert v2 vocabulary to legacy format for backward compatibility.
        Legacy format: {subdomain: [term1, term2, ...]}

        Args:
            domain: Domain name

        Returns:
            Dictionary in legacy format
        """
        vocab = self.load_vocabulary(domain)
        legacy = defaultdict(list)

        for concept_id, concept in vocab.get('concepts', {}).items():
            category = concept.get('category', 'general')
            term = concept.get('term', concept_id)

            legacy[category].append(term)

            # Also add synonyms
            for synonym in concept.get('synonyms', []):
                legacy[category].append(synonym)

        return dict(legacy)

    def get_all_legacy_format(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get all vocabularies in legacy format.

        Returns:
            Dictionary mapping domain to legacy format vocabulary
        """
        index = self.load_index()
        legacy_all = {}

        for domain in index['vocabularies'].keys():
            legacy_all[domain] = self.to_legacy_format(domain)

        return legacy_all

    def get_anchor_alignment(self, domain: str) -> str:
        """
        Get the anchor framework this domain aligns with.

        Args:
            domain: Domain name

        Returns:
            Anchor framework name
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('anchor_alignment', 'unknown')

    def get_domains_by_anchor(self, anchor: str) -> List[str]:
        """
        Get all domains that align with a specific anchor.

        Args:
            anchor: Anchor framework name

        Returns:
            List of domain names
        """
        index = self.load_index()
        domains = []

        for domain, info in index['vocabularies'].items():
            if info.get('anchor_alignment') == anchor:
                domains.append(domain)

        return domains

    def get_epistemic_standards(self, domain: str) -> Dict:
        """
        Get epistemic standards for a domain.

        Args:
            domain: Domain name

        Returns:
            Epistemic standards dictionary
        """
        vocab = self.load_vocabulary(domain)
        return vocab.get('epistemic_standards', {})

    def search_concepts(self, domain: str, query: str,
                       search_fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Search concepts by keyword in various fields.

        Args:
            domain: Domain name
            query: Search query (case-insensitive)
            search_fields: Fields to search in (default: term, definition, synonyms)

        Returns:
            List of matching concepts
        """
        if search_fields is None:
            search_fields = ['term', 'definition', 'synonyms']

        vocab = self.load_vocabulary(domain)
        query_lower = query.lower()
        matches = []

        for concept_id, concept in vocab.get('concepts', {}).items():
            for field in search_fields:
                value = concept.get(field, '')

                # Handle both string and list fields
                if isinstance(value, str):
                    if query_lower in value.lower():
                        matches.append({'id': concept_id, **concept})
                        break
                elif isinstance(value, list):
                    if any(query_lower in str(item).lower() for item in value):
                        matches.append({'id': concept_id, **concept})
                        break

        return matches

    def get_statistics(self, domain: str) -> Dict:
        """
        Get statistics about a vocabulary.

        Args:
            domain: Domain name

        Returns:
            Statistics dictionary
        """
        vocab = self.load_vocabulary(domain)
        concepts = vocab.get('concepts', {})

        # Count by complexity
        complexity_counts = defaultdict(int)
        category_counts = defaultdict(int)

        for concept in concepts.values():
            complexity_counts[concept.get('complexity', 'unknown')] += 1
            category_counts[concept.get('category', 'unknown')] += 1

        return {
            'domain': domain,
            'total_concepts': len(concepts),
            'total_heuristics': len(vocab.get('reasoning_heuristics', [])),
            'total_errors': len(vocab.get('common_errors', [])),
            'total_mental_models': len(vocab.get('mental_models', [])),
            'complexity_distribution': dict(complexity_counts),
            'category_distribution': dict(category_counts),
            'anchor_alignment': vocab.get('anchor_alignment', 'unknown'),
            'version': vocab.get('version', 'unknown'),
            'last_updated': vocab.get('last_updated', 'unknown')
        }

    def get_all_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all vocabularies.

        Returns:
            Dictionary mapping domain to statistics
        """
        index = self.load_index()
        stats = {}

        for domain in index['vocabularies'].keys():
            stats[domain] = self.get_statistics(domain)

        return stats


# Convenience functions for common operations

def load_vocabulary(domain: str, dict_dir: Optional[str] = None) -> Dict:
    """Quick function to load a single vocabulary"""
    loader = VocabularyV2Loader(dict_dir)
    return loader.load_vocabulary(domain)


def get_all_terms(domain: str, dict_dir: Optional[str] = None) -> List[str]:
    """Quick function to get all terms from a domain"""
    loader = VocabularyV2Loader(dict_dir)
    return loader.get_all_terms(domain)


def get_legacy_format(domain: str, dict_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Quick function to get vocabulary in legacy format"""
    loader = VocabularyV2Loader(dict_dir)
    return loader.to_legacy_format(domain)


def print_vocabulary_stats(domain: str, dict_dir: Optional[str] = None):
    """Print statistics for a vocabulary"""
    loader = VocabularyV2Loader(dict_dir)
    stats = loader.get_statistics(domain)

    print(f"\n{'='*60}")
    print(f"Vocabulary Statistics: {stats['domain']}")
    print(f"{'='*60}")
    print(f"Version: {stats['version']}")
    print(f"Last Updated: {stats['last_updated']}")
    print(f"Anchor Alignment: {stats['anchor_alignment']}")
    print(f"\nContent:")
    print(f"  Concepts: {stats['total_concepts']}")
    print(f"  Reasoning Heuristics: {stats['total_heuristics']}")
    print(f"  Common Errors: {stats['total_errors']}")
    print(f"  Mental Models: {stats['total_mental_models']}")
    print(f"\nComplexity Distribution:")
    for level, count in stats['complexity_distribution'].items():
        print(f"  {level:15s}: {count:3d}")
    print(f"\nCategory Distribution:")
    for category, count in sorted(stats['category_distribution'].items()):
        print(f"  {category:30s}: {count:3d}")


if __name__ == '__main__':
    import sys

    # Test the loader
    loader = VocabularyV2Loader()

    print("="*60)
    print("Enhanced Semantic Network v2.0 - Vocabulary Utilities")
    print("="*60)

    # Load index
    index = loader.load_index()
    print(f"\nLoaded index with {index['coverage']['total_vocabularies']} vocabularies")
    print(f"Total concepts: {index['coverage']['total_concepts']}")
    print(f"Anchors supported: {len(index['coverage']['anchor_frameworks_supported'])}")

    # Show statistics for each vocabulary
    print("\n" + "="*60)
    print("Vocabulary Statistics Summary")
    print("="*60)

    for domain in index['vocabularies'].keys():
        stats = loader.get_statistics(domain)
        print(f"\n{domain:20s}: {stats['total_concepts']:3d} concepts, "
              f"{stats['total_heuristics']:2d} heuristics, "
              f"{stats['total_errors']:2d} errors, "
              f"{stats['total_mental_models']:2d} models")

    # If domain specified, show detailed stats
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        print_vocabulary_stats(domain)

        # Show sample concepts
        print(f"\nSample Concepts:")
        terms = loader.get_all_terms(domain)
        for term in terms[:5]:
            print(f"  - {term}")
