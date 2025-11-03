#!/usr/bin/env python3
"""
Conversation State Graph - Wave 2: Advanced Reasoning

Maintains a dynamic graph of entities, claims, and relationships discussed
across multi-turn conversations.

Integration with Current ARIA:
- Extends aria_session.py with structured knowledge tracking
- Feeds into multi_hop_reasoner.py for context-aware query decomposition
- Used by contradiction_detector.py to find inconsistencies
- Enables temporal_tracker.py to detect topic drift

Key Features:
1. Entity extraction and tracking (people, concepts, systems)
2. Claim/statement graph (who said what, when, with what confidence)
3. Relationship mapping (X relates to Y, X causes Z)
4. Temporal evolution (how entities/claims change over conversation)
5. Cross-reference tracking (which chunks support which claims)
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Types of entities we track"""
    PERSON = "person"
    CONCEPT = "concept"
    SYSTEM = "system"
    TECHNOLOGY = "technology"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities"""
    IS_A = "is_a"  # X is a type of Y
    HAS_A = "has_a"  # X has property Y
    CAUSES = "causes"  # X causes Y
    RELATES_TO = "relates_to"  # X relates to Y
    CONTRADICTS = "contradicts"  # X contradicts Y
    SUPPORTS = "supports"  # X supports Y
    MENTIONED_WITH = "mentioned_with"  # X mentioned with Y
    PRECEDES = "precedes"  # X precedes Y (temporal)


@dataclass
class Entity:
    """An entity mentioned in conversation"""
    id: str  # Unique ID (hash of canonical name)
    canonical_name: str  # Normalized name
    aliases: Set[str] = field(default_factory=set)  # Other names
    entity_type: EntityType = EntityType.UNKNOWN
    
    # When first/last mentioned
    first_mention: Optional[str] = None  # ISO timestamp
    last_mention: Optional[str] = None
    mention_count: int = 0
    
    # Associated attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Which turns mentioned this
    turn_indices: List[int] = field(default_factory=list)
    
    # Confidence in entity extraction
    confidence: float = 1.0


@dataclass
class Claim:
    """A claim or statement made in conversation"""
    id: str  # Unique ID
    text: str  # The claim text
    
    # Who/what made this claim
    source: str  # "user", "assistant", or chunk source
    turn_index: int
    timestamp: str
    
    # Entities involved in this claim
    entities: List[str] = field(default_factory=list)  # Entity IDs
    
    # Relationships this claim establishes
    relationships: List[Tuple[str, RelationType, str]] = field(default_factory=list)
    
    # Supporting evidence
    evidence: List[str] = field(default_factory=list)  # Chunk IDs or sources
    
    # Confidence and validation
    confidence: float = 1.0
    validated: bool = False
    contradicted_by: List[str] = field(default_factory=list)  # Other claim IDs


@dataclass
class Relationship:
    """A relationship between two entities"""
    id: str
    entity1_id: str
    relation_type: RelationType
    entity2_id: str
    
    # When established
    first_mentioned: str  # ISO timestamp
    last_mentioned: str
    mention_count: int = 0
    
    # Supporting claims
    claim_ids: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 1.0


class ConversationStateGraph:
    """
    Maintains structured knowledge graph of conversation
    
    Graph Structure:
    - Nodes: Entities + Claims
    - Edges: Relationships between entities
    - Metadata: Temporal info, confidence, sources
    """
    
    def __init__(self, state_path: Optional[Path] = None):
        """
        Args:
            state_path: Path to save/load state
        """
        self.state_path = state_path or Path.home() / '.aria_conversation_graph.json'
        
        # Graph components
        self.entities: Dict[str, Entity] = {}
        self.claims: Dict[str, Claim] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Indices for fast lookup
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.entities_by_turn: Dict[int, List[str]] = defaultdict(list)
        self.claims_by_turn: Dict[int, List[str]] = defaultdict(list)
        
        # Conversation metadata
        self.current_turn: int = 0
        self.conversation_start: Optional[str] = None
        
        # Entity extraction patterns
        self._build_extraction_patterns()
        
        # Load state
        self._load_state()
    
    def _build_extraction_patterns(self):
        """Build regex patterns for entity extraction"""
        
        # Technology/system patterns
        self.tech_patterns = [
            r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b',  # CamelCase
            r'\b[A-Z]{2,}\b',  # ACRONYMS
            r'\b(algorithm|model|system|framework|library|protocol)\b',
        ]
        
        # Concept patterns
        self.concept_patterns = [
            r'\b(concept of|theory of|principle of)\s+(\w+)',
            r'\b(\w+)\s+(approach|method|technique|strategy)\b',
        ]
        
        # Person patterns (capitalized names)
        self.person_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # First Last
        ]
    
    def add_turn(
        self,
        turn_index: int,
        user_message: str,
        assistant_response: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Process a conversation turn and update graph
        
        Args:
            turn_index: Turn number
            user_message: User's message
            assistant_response: Assistant's response
            retrieved_chunks: Optional chunks that informed response
        """
        self.current_turn = turn_index
        
        if self.conversation_start is None:
            self.conversation_start = datetime.now().isoformat()
        
        timestamp = datetime.now().isoformat()
        
        # Extract entities from user message
        user_entities = self._extract_entities(user_message, turn_index, "user")
        
        # Extract entities from assistant response
        assistant_entities = self._extract_entities(assistant_response, turn_index, "assistant")
        
        # Extract claims from assistant response
        claims = self._extract_claims(assistant_response, turn_index, timestamp, assistant_entities)
        
        # If we have chunks, extract entities from them too
        if retrieved_chunks:
            for chunk in retrieved_chunks:
                chunk_text = chunk.get('text', '')
                self._extract_entities(chunk_text, turn_index, chunk.get('file', 'chunk'))
        
        # Update relationships based on co-occurrence
        self._update_cooccurrence_relationships(
            user_entities + assistant_entities, 
            turn_index, 
            timestamp
        )
        
        # Detect explicit relationships in text
        self._detect_explicit_relationships(
            user_message + " " + assistant_response,
            turn_index,
            timestamp
        )
    
    def _extract_entities(
        self,
        text: str,
        turn_index: int,
        source: str
    ) -> List[str]:
        """
        Extract entities from text
        
        Returns:
            List of entity IDs
        """
        extracted = []
        timestamp = datetime.now().isoformat()
        
        # Extract technical terms (CamelCase, ACRONYMS)
        for pattern in self.tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match not in ['The', 'And', 'But', 'For']:
                    entity_id = self._add_entity(
                        match,
                        EntityType.TECHNOLOGY,
                        turn_index,
                        timestamp
                    )
                    extracted.append(entity_id)
        
        # Extract concepts
        for pattern in self.concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept = match.group(2) if (match.lastindex and match.lastindex >= 2) else match.group(0)
                entity_id = self._add_entity(
                    concept,
                    EntityType.CONCEPT,
                    turn_index,
                    timestamp
                )
                extracted.append(entity_id)
        
        # Extract person names
        for pattern in self.person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entity_id = self._add_entity(
                    match,
                    EntityType.PERSON,
                    turn_index,
                    timestamp
                )
                extracted.append(entity_id)
        
        return extracted
    
    def _add_entity(
        self,
        name: str,
        entity_type: EntityType,
        turn_index: int,
        timestamp: str
    ) -> str:
        """
        Add or update an entity
        
        Returns:
            Entity ID
        """
        # Normalize name
        canonical_name = name.strip().lower()
        
        # Check if entity exists
        if canonical_name in self.entity_by_name:
            entity_id = self.entity_by_name[canonical_name]
            entity = self.entities[entity_id]
            
            # Update existing entity
            entity.last_mention = timestamp
            entity.mention_count += 1
            if turn_index not in entity.turn_indices:
                entity.turn_indices.append(turn_index)
            entity.aliases.add(name)
            
            return entity_id
        
        # Create new entity
        entity_id = hashlib.md5(canonical_name.encode()).hexdigest()[:16]
        
        entity = Entity(
            id=entity_id,
            canonical_name=canonical_name,
            aliases={name},
            entity_type=entity_type,
            first_mention=timestamp,
            last_mention=timestamp,
            mention_count=1,
            turn_indices=[turn_index]
        )
        
        self.entities[entity_id] = entity
        self.entity_by_name[canonical_name] = entity_id
        self.entities_by_turn[turn_index].append(entity_id)
        
        return entity_id
    
    def _extract_claims(
        self,
        text: str,
        turn_index: int,
        timestamp: str,
        entity_ids: List[str]
    ) -> List[str]:
        """
        Extract claims from text
        
        Returns:
            List of claim IDs
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Create claim
            claim_id = hashlib.md5(
                f"{turn_index}:{sentence}".encode()
            ).hexdigest()[:16]
            
            claim = Claim(
                id=claim_id,
                text=sentence,
                source="assistant",
                turn_index=turn_index,
                timestamp=timestamp,
                entities=entity_ids,
                confidence=0.8  # Default confidence
            )
            
            self.claims[claim_id] = claim
            self.claims_by_turn[turn_index].append(claim_id)
            claims.append(claim_id)
        
        return claims
    
    def _update_cooccurrence_relationships(
        self,
        entity_ids: List[str],
        turn_index: int,
        timestamp: str
    ):
        """
        Update relationships based on entity co-occurrence
        """
        # Create "mentioned_with" relationships for entities in same turn
        for i, entity1_id in enumerate(entity_ids):
            for entity2_id in entity_ids[i+1:]:
                self._add_relationship(
                    entity1_id,
                    RelationType.MENTIONED_WITH,
                    entity2_id,
                    timestamp
                )
    
    def _detect_explicit_relationships(
        self,
        text: str,
        turn_index: int,
        timestamp: str
    ):
        """
        Detect explicit relationship statements
        
        Examples:
        - "X is a Y"
        - "X causes Y"
        - "X relates to Y"
        """
        # Pattern: X is a Y
        is_a_pattern = r'(\w+(?:\s+\w+)?)\s+is\s+a\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(is_a_pattern, text, re.IGNORECASE):
            entity1_name = match.group(1).lower()
            entity2_name = match.group(2).lower()
            
            entity1_id = self.entity_by_name.get(entity1_name)
            entity2_id = self.entity_by_name.get(entity2_name)
            
            if entity1_id and entity2_id:
                self._add_relationship(
                    entity1_id,
                    RelationType.IS_A,
                    entity2_id,
                    timestamp
                )
        
        # Pattern: X causes Y
        causes_pattern = r'(\w+(?:\s+\w+)?)\s+causes?\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(causes_pattern, text, re.IGNORECASE):
            entity1_name = match.group(1).lower()
            entity2_name = match.group(2).lower()
            
            entity1_id = self.entity_by_name.get(entity1_name)
            entity2_id = self.entity_by_name.get(entity2_name)
            
            if entity1_id and entity2_id:
                self._add_relationship(
                    entity1_id,
                    RelationType.CAUSES,
                    entity2_id,
                    timestamp
                )
    
    def _add_relationship(
        self,
        entity1_id: str,
        relation_type: RelationType,
        entity2_id: str,
        timestamp: str
    ) -> str:
        """
        Add or update a relationship
        
        Returns:
            Relationship ID
        """
        # Create relationship ID (order-independent for symmetric relations)
        if relation_type in [RelationType.MENTIONED_WITH, RelationType.RELATES_TO]:
            ids_sorted = sorted([entity1_id, entity2_id])
            rel_key = f"{ids_sorted[0]}:{relation_type.value}:{ids_sorted[1]}"
        else:
            rel_key = f"{entity1_id}:{relation_type.value}:{entity2_id}"
        
        rel_id = hashlib.md5(rel_key.encode()).hexdigest()[:16]
        
        if rel_id in self.relationships:
            # Update existing
            rel = self.relationships[rel_id]
            rel.last_mentioned = timestamp
            rel.mention_count += 1
        else:
            # Create new
            rel = Relationship(
                id=rel_id,
                entity1_id=entity1_id,
                relation_type=relation_type,
                entity2_id=entity2_id,
                first_mentioned=timestamp,
                last_mentioned=timestamp,
                mention_count=1
            )
            self.relationships[rel_id] = rel
        
        return rel_id
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """
        Get full context for an entity
        
        Returns:
            Dict with entity info, relationships, claims
        """
        canonical = entity_name.strip().lower()
        entity_id = self.entity_by_name.get(canonical)
        
        if not entity_id:
            return {'found': False}
        
        entity = self.entities[entity_id]
        
        # Get relationships
        related = []
        for rel in self.relationships.values():
            if rel.entity1_id == entity_id:
                related.append({
                    'type': rel.relation_type.value,
                    'target': self.entities[rel.entity2_id].canonical_name,
                    'mentions': rel.mention_count
                })
            elif rel.entity2_id == entity_id:
                related.append({
                    'type': rel.relation_type.value,
                    'source': self.entities[rel.entity1_id].canonical_name,
                    'mentions': rel.mention_count
                })
        
        # Get associated claims
        associated_claims = [
            {'text': claim.text, 'confidence': claim.confidence}
            for claim in self.claims.values()
            if entity_id in claim.entities
        ]
        
        return {
            'found': True,
            'entity': {
                'name': entity.canonical_name,
                'type': entity.entity_type.value,
                'aliases': list(entity.aliases),
                'mentions': entity.mention_count,
                'first_seen': entity.first_mention,
                'last_seen': entity.last_mention
            },
            'relationships': related,
            'claims': associated_claims[:10]  # Top 10
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get high-level conversation summary"""
        # Top entities by mention count
        top_entities = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True
        )[:10]
        
        # Relationship distribution
        rel_counts = defaultdict(int)
        for rel in self.relationships.values():
            rel_counts[rel.relation_type.value] += 1
        
        return {
            'total_turns': self.current_turn + 1,
            'total_entities': len(self.entities),
            'total_claims': len(self.claims),
            'total_relationships': len(self.relationships),
            'top_entities': [
                {'name': e.canonical_name, 'mentions': e.mention_count}
                for e in top_entities
            ],
            'relationship_types': dict(rel_counts),
            'started': self.conversation_start
        }
    
    def find_contradictions(self) -> List[Dict[str, Any]]:
        """
        Find potential contradictions between claims
        
        Returns:
            List of potential contradictions
        """
        contradictions = []
        
        # Look for claims about same entities with different assertions
        entity_claims = defaultdict(list)
        for claim in self.claims.values():
            for entity_id in claim.entities:
                entity_claims[entity_id].append(claim)
        
        # For each entity, check if claims contradict
        for entity_id, claims in entity_claims.items():
            if len(claims) < 2:
                continue
            
            entity = self.entities[entity_id]
            
            # Simple contradiction detection: opposite sentiment or negation
            for i, claim1 in enumerate(claims):
                for claim2 in claims[i+1:]:
                    # Check for negation patterns
                    if ('not' in claim1.text.lower() and 'not' not in claim2.text.lower()) or \
                       ('not' in claim2.text.lower() and 'not' not in claim1.text.lower()):
                        contradictions.append({
                            'entity': entity.canonical_name,
                            'claim1': claim1.text,
                            'claim2': claim2.text,
                            'confidence': 0.6
                        })
        
        return contradictions
    
    def _save_state(self):
        """Save graph state to disk"""
        try:
            state = {
                'version': '1.0',
                'current_turn': self.current_turn,
                'conversation_start': self.conversation_start,
                'entities': {
                    eid: {
                        'canonical_name': e.canonical_name,
                        'aliases': list(e.aliases),
                        'entity_type': e.entity_type.value,
                        'first_mention': e.first_mention,
                        'last_mention': e.last_mention,
                        'mention_count': e.mention_count,
                        'turn_indices': e.turn_indices,
                        'confidence': e.confidence
                    }
                    for eid, e in self.entities.items()
                },
                'relationships': {
                    rid: {
                        'entity1_id': r.entity1_id,
                        'relation_type': r.relation_type.value,
                        'entity2_id': r.entity2_id,
                        'first_mentioned': r.first_mentioned,
                        'last_mentioned': r.last_mentioned,
                        'mention_count': r.mention_count,
                        'confidence': r.confidence
                    }
                    for rid, r in self.relationships.items()
                },
                'claims': {
                    cid: {
                        'text': c.text,
                        'source': c.source,
                        'turn_index': c.turn_index,
                        'timestamp': c.timestamp,
                        'entities': c.entities,
                        'confidence': c.confidence
                    }
                    for cid, c in list(self.claims.items())[-100:]  # Keep last 100 claims
                }
            }
            
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            print(f"Warning: Could not save conversation graph: {e}")
    
    def _load_state(self):
        """Load graph state from disk"""
        if not self.state_path.exists():
            return
        
        try:
            state = json.loads(self.state_path.read_text())
            
            self.current_turn = state.get('current_turn', 0)
            self.conversation_start = state.get('conversation_start')
            
            # Restore entities
            for eid, e_data in state.get('entities', {}).items():
                entity = Entity(
                    id=eid,
                    canonical_name=e_data['canonical_name'],
                    aliases=set(e_data['aliases']),
                    entity_type=EntityType(e_data['entity_type']),
                    first_mention=e_data['first_mention'],
                    last_mention=e_data['last_mention'],
                    mention_count=e_data['mention_count'],
                    turn_indices=e_data['turn_indices'],
                    confidence=e_data['confidence']
                )
                self.entities[eid] = entity
                self.entity_by_name[entity.canonical_name] = eid
            
            # Restore relationships
            for rid, r_data in state.get('relationships', {}).items():
                rel = Relationship(
                    id=rid,
                    entity1_id=r_data['entity1_id'],
                    relation_type=RelationType(r_data['relation_type']),
                    entity2_id=r_data['entity2_id'],
                    first_mentioned=r_data['first_mentioned'],
                    last_mentioned=r_data['last_mentioned'],
                    mention_count=r_data['mention_count'],
                    confidence=r_data['confidence']
                )
                self.relationships[rid] = rel
            
            print(f"✓ Loaded conversation graph: {len(self.entities)} entities, {len(self.relationships)} relationships")
        
        except Exception as e:
            print(f"Warning: Could not load conversation graph: {e}")


# ============================================================================
# Integration with aria_session.py
# ============================================================================

def integrate_with_aria_session():
    """
    Example: Add conversation graph to aria_session.py
    
    The graph tracks structured knowledge while aria_session tracks raw messages
    """
    
    # Initialize graph
    graph = ConversationStateGraph(
        state_path=Path.home() / '.aria_conversation_graph.json'
    )
    
    # Example: Process a conversation turn
    user_msg = "How does Thompson Sampling work in the context of multi-armed bandits?"
    assistant_resp = "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It maintains a probability distribution over the reward of each arm and samples from these distributions to make decisions."
    
    # Mock retrieved chunks
    chunks = [
        {'text': 'Thompson Sampling uses Beta distributions...', 'file': 'bandits.pdf'},
        {'text': 'Multi-armed bandits balance exploration and exploitation...', 'file': 'rl_book.pdf'}
    ]
    
    # Add turn to graph
    graph.add_turn(
        turn_index=0,
        user_message=user_msg,
        assistant_response=assistant_resp,
        retrieved_chunks=chunks
    )
    
    # Query entity context
    context = graph.get_entity_context("thompson sampling")
    print(f"\nEntity Context for 'Thompson Sampling':")
    print(f"  Found: {context['found']}")
    if context['found']:
        print(f"  Type: {context['entity']['type']}")
        print(f"  Mentions: {context['entity']['mentions']}")
        print(f"  Relationships: {len(context['relationships'])}")
    
    # Get conversation summary
    summary = graph.get_conversation_summary()
    print(f"\nConversation Summary:")
    print(f"  Turns: {summary['total_turns']}")
    print(f"  Entities tracked: {summary['total_entities']}")
    print(f"  Relationships: {summary['total_relationships']}")
    print(f"  Top entities: {[e['name'] for e in summary['top_entities'][:3]]}")
    
    # Check for contradictions
    contradictions = graph.find_contradictions()
    if contradictions:
        print(f"\n⚠️  Found {len(contradictions)} potential contradictions")


if __name__ == '__main__':
    integrate_with_aria_session()
