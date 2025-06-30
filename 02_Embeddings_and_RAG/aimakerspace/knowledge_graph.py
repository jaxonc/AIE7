import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any
import re
from dataclasses import dataclass, field
import asyncio


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    frequency: int = 0
    contexts: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    contexts: List[str] = field(default_factory=list)


class KnowledgeGraphBuilder:
    """Builds and manages a knowledge graph from text chunks."""
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        self.graph = nx.Graph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.text_chunks: List[str] = []
        self.chunk_entities: Dict[int, Set[str]] = defaultdict(set)
        
    def extract_entities_basic(self, text: str) -> List[Tuple[str, str]]:
        """Basic entity extraction without external dependencies."""
        entities = []
        
        # Extract capitalized words/phrases as potential entities
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, text)
        
        for match in matches:
            if len(match) > 2 and not match.lower() in ['The', 'This', 'That', 'Part', 'When', 'What', 'How']:
                entities.append((match, "ENTITY"))
        
        # Extract common business terms
        business_terms = [
            "CEO", "CTO", "CFO", "VP", "startup", "company", "business", 
            "funding", "venture capital", "hiring", "executive", "product",
            "market", "customer", "revenue", "strategy", "management", "team",
            "investor", "board", "employee", "candidate", "interview"
        ]
        
        text_lower = text.lower()
        for term in business_terms:
            if term.lower() in text_lower:
                entities.append((term.title(), "BUSINESS_TERM"))
                
        return entities
    
    def extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relations between entities in the text."""
        relations = []
        
        # Simple co-occurrence based relations
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities co-occur in the same sentence
                sentences = text.split('.')
                for sentence in sentences:
                    if entity1.lower() in sentence.lower() and entity2.lower() in sentence.lower():
                        # Determine relation type based on keywords
                        sentence_lower = sentence.lower()
                        if any(word in sentence_lower for word in ['hire', 'recruit', 'manage']):
                            relations.append((entity1, entity2, "MANAGES"))
                        elif any(word in sentence_lower for word in ['fund', 'invest', 'capital']):
                            relations.append((entity1, entity2, "FUNDS"))
                        elif any(word in sentence_lower for word in ['compete', 'vs', 'against']):
                            relations.append((entity1, entity2, "COMPETES_WITH"))
                        else:
                            relations.append((entity1, entity2, "RELATED_TO"))
                        break
        
        return relations
    
    def build_graph_from_texts(self, texts: List[str]) -> None:
        """Build knowledge graph from list of text chunks."""
        self.text_chunks = texts
        
        print(f"🔗 Building knowledge graph from {len(texts)} text chunks...")
        
        # Extract entities and relations from each chunk
        for chunk_idx, text in enumerate(texts):
            # Extract entities
            entity_tuples = self.extract_entities_basic(text)
            chunk_entities = []
            
            for entity_text, entity_type in entity_tuples:
                entity_text = entity_text.strip()
                if len(entity_text) > 2:
                    chunk_entities.append(entity_text)
                    
                    # Add or update entity
                    if entity_text not in self.entities:
                        self.entities[entity_text] = Entity(
                            name=entity_text,
                            entity_type=entity_type,
                            frequency=1,
                            contexts=[text[:200]]
                        )
                    else:
                        self.entities[entity_text].frequency += 1
                        if len(self.entities[entity_text].contexts) < 3:
                            self.entities[entity_text].contexts.append(text[:200])
                    
                    # Add entity to graph
                    self.graph.add_node(entity_text, 
                                      entity_type=entity_type,
                                      frequency=self.entities[entity_text].frequency)
            
            # Store chunk-entity mapping
            self.chunk_entities[chunk_idx] = set(chunk_entities)
            
            # Extract relations
            relations = self.extract_relations(text, chunk_entities)
            for source, target, relation_type in relations:
                self.relations.append(Relation(source, target, relation_type, contexts=[text[:200]]))
                
                # Add edge to graph
                if self.graph.has_edge(source, target):
                    self.graph[source][target]['weight'] += 1
                else:
                    self.graph.add_edge(source, target, 
                                      relation_type=relation_type, 
                                      weight=1)
        
        print(f"📊 Knowledge graph built: {len(self.graph.nodes)} entities, {len(self.graph.edges)} relations")
    
    def simple_clustering(self, num_clusters: int = 4) -> Dict[str, int]:
        """Simple clustering based on entity types and connectivity."""
        if len(self.graph.nodes) == 0:
            return {}
        
        entity_names = list(self.entities.keys())
        
        # If fewer entities than clusters, assign each to its own cluster
        if len(entity_names) < num_clusters:
            return {name: i for i, name in enumerate(entity_names)}
        
        # Group entities by type first
        type_groups: Dict[str, List[str]] = defaultdict(list)
        for entity_name in entity_names:
            entity_type = self.entities[entity_name].entity_type
            type_groups[entity_type].append(entity_name)
        
        # Assign clusters based on entity types
        entity_to_cluster = {}
        cluster_id = 0
        
        for entity_type, entities in type_groups.items():
            # If this type has many entities, split them into multiple clusters
            if len(entities) > num_clusters // 2:
                # Split by frequency - high frequency entities get separate clusters
                entities_by_freq = [(e, self.entities[e].frequency) for e in entities]
                entities_by_freq.sort(key=lambda x: x[1], reverse=True)
                
                # Assign in round-robin fashion
                for i, (entity, _) in enumerate(entities_by_freq):
                    entity_to_cluster[entity] = (cluster_id + i) % num_clusters
                cluster_id = (cluster_id + len(entities)) % num_clusters
            else:
                # Assign all entities of this type to the same cluster
                for entity in entities:
                    entity_to_cluster[entity] = cluster_id % num_clusters
                cluster_id += 1
        
        return entity_to_cluster
    
    def assign_chunk_categories(self, num_clusters: int = 4) -> Dict[int, int]:
        """Assign categories to text chunks based on their entities' clusters."""
        entity_clusters = self.simple_clustering(num_clusters)
        chunk_categories = {}
        
        for chunk_idx, chunk_entities in self.chunk_entities.items():
            if not chunk_entities:
                chunk_categories[chunk_idx] = 0  # Default category
                continue
            
            # Assign chunk to the most common cluster among its entities
            cluster_votes: Dict[int, int] = defaultdict(int)
            for entity in chunk_entities:
                if entity in entity_clusters:
                    cluster_votes[entity_clusters[entity]] += 1
            
            if cluster_votes:
                chunk_categories[chunk_idx] = max(cluster_votes.items(), key=lambda x: x[1])[0]
            else:
                chunk_categories[chunk_idx] = 0
        
        return chunk_categories
    
    def get_cluster_descriptions(self, num_clusters: int = 4) -> Dict[int, Dict[str, Any]]:
        """Get descriptions for each cluster based on top entities and keywords."""
        entity_clusters = self.simple_clustering(num_clusters)
        cluster_info: Dict[int, Dict[str, Any]] = defaultdict(lambda: {'entities': [], 'keywords': [], 'size': 0})
        
        # Group entities by cluster
        for entity, cluster_id in entity_clusters.items():
            entity_info = {
                'name': entity,
                'frequency': self.entities[entity].frequency,
                'type': self.entities[entity].entity_type
            }
            cluster_info[cluster_id]['entities'].append(entity_info)
            cluster_info[cluster_id]['size'] += 1
        
        # Sort entities by frequency and get top keywords
        for cluster_id in cluster_info:
            entities_list = cluster_info[cluster_id]['entities']
            entities_list.sort(key=lambda x: x['frequency'], reverse=True)
            
            # Extract keywords from top entities' contexts
            top_entities = entities_list[:5]
            all_contexts = []
            for entity_info in top_entities:
                entity_name = entity_info['name']
                all_contexts.extend(self.entities[entity_name].contexts)
            
            if all_contexts:
                # Simple keyword extraction
                text = ' '.join(all_contexts).lower()
                words = re.findall(r'\b[a-z]+\b', text)
                word_freq = Counter(words)
                # Filter out common words
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                keywords = [word for word, freq in word_freq.most_common(10) 
                          if word not in stop_words and len(word) > 3]
                cluster_info[cluster_id]['keywords'] = keywords[:5]
        
        return dict(cluster_info)
    
    def get_related_entities(self, entity: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """Get entities related to the given entity."""
        if entity not in self.graph:
            return []
        
        related = []
        for neighbor in self.graph.neighbors(entity):
            weight = float(self.graph[entity][neighbor].get('weight', 1))
            related.append((neighbor, weight))
        
        return sorted(related, key=lambda x: x[1], reverse=True)[:max_results]
    
    def expand_query_with_graph(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with related entities from the knowledge graph."""
        query_entities = self.extract_entities_basic(query)
        expanded_terms = [query]
        
        for entity_text, _ in query_entities:
            related_entities = self.get_related_entities(entity_text, max_expansions)
            for related_entity, _ in related_entities:
                expanded_terms.append(f"{query} {related_entity}")
        
        return expanded_terms[:max_expansions + 1]


class KnowledgeGraphEnhancedVectorDB:
    """Vector database enhanced with knowledge graph capabilities."""
    
    def __init__(self, embedding_model=None):
        from aimakerspace.vectordatabase import VectorDatabase
        self.vector_db = VectorDatabase(embedding_model)
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.cluster_categories: Dict[int, Dict[str, Any]] = {}
        self.chunk_to_category: Dict[int, int] = {}
        
    async def build_from_list(self, texts: List[str], num_categories: int = 4) -> "KnowledgeGraphEnhancedVectorDB":
        """Build both vector database and knowledge graph from texts."""
        print("🚀 Building enhanced vector database with knowledge graph...")
        
        # Build knowledge graph
        self.knowledge_graph.build_graph_from_texts(texts)
        
        # Get cluster-based categories
        chunk_categories = self.knowledge_graph.assign_chunk_categories(num_categories)
        cluster_descriptions = self.knowledge_graph.get_cluster_descriptions(num_categories)
        
        # Store category information
        self.cluster_categories = cluster_descriptions
        self.chunk_to_category = chunk_categories
        
        # Build vector database with graph-based categories
        embeddings = await self.vector_db.embedding_model.async_get_embeddings(texts)
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector = np.array(embedding)
            category_id = chunk_categories.get(i, 0)
            
            # Create rich metadata with graph information
            metadata = {
                'category_id': category_id,
                'category_name': f"Cluster_{category_id}",
                'entities': list(self.knowledge_graph.chunk_entities.get(i, set())),
                'chunk_index': i
            }
            
            self.vector_db.insert(text, vector, metadata)
        
        print(f"✅ Enhanced database built with {num_categories} discovered categories")
        self._print_category_summary()
        
        return self
    
    def _print_category_summary(self):
        """Print summary of discovered categories."""
        print("\n📊 Discovered Categories:")
        print("=" * 60)
        
        for category_id, info in self.cluster_categories.items():
            top_entities = [e['name'] for e in info['entities'][:3]]
            keywords = info['keywords'][:3]
            
            print(f"🏷️  Category {category_id} ({info['size']} entities):")
            print(f"   Top entities: {', '.join(top_entities)}")
            print(f"   Keywords: {', '.join(keywords)}")
            print()
    
    def search_with_graph_expansion(self, query: str, k: int = 5, 
                                  category_filter: Optional[int] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search with query expansion using knowledge graph."""
        # Expand query using graph
        expanded_queries = self.knowledge_graph.expand_query_with_graph(query)
        
        all_results = []
        for expanded_query in expanded_queries:
            if category_filter is not None:
                # Filter by category
                filtered_items = [
                    (key, vector) for key, vector in self.vector_db.vectors.items()
                    if self.vector_db.metadata.get(key, {}).get('category_id') == category_filter
                ]
            else:
                filtered_items = list(self.vector_db.vectors.items())
            
            # Get embedding for expanded query
            query_vector = np.array(self.vector_db.embedding_model.get_embedding(expanded_query))
            
            # Calculate similarities
            from aimakerspace.vectordatabase import cosine_similarity
            scores = [
                (key, cosine_similarity(query_vector, vector), self.vector_db.metadata.get(key, {}))
                for key, vector in filtered_items
            ]
            all_results.extend(scores)
        
        # Remove duplicates and sort
        unique_results: Dict[str, Tuple[str, float, Dict[str, Any]]] = {}
        for text, score, metadata in all_results:
            if text not in unique_results or score > unique_results[text][1]:
                unique_results[text] = (text, score, metadata)
        
        final_results = list(unique_results.values())
        return sorted(final_results, key=lambda x: x[1], reverse=True)[:k]
    
    def get_category_info(self) -> Dict[int, Dict[str, Any]]:
        """Get information about discovered categories."""
        return self.cluster_categories
    
    def get_categories(self) -> List[str]:
        """Get list of category names."""
        return [f"Cluster_{i}" for i in self.cluster_categories.keys()]
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get counts per category."""
        counts: Dict[str, int] = defaultdict(int)
        for category_id in self.chunk_to_category.values():
            counts[f"Cluster_{category_id}"] += 1
        return dict(counts)
    
    def get_related_entities_for_query(self, query: str) -> List[Tuple[str, float]]:
        """Get entities related to the query."""
        query_entities = self.knowledge_graph.extract_entities_basic(query)
        all_related = []
        
        for entity_text, _ in query_entities:
            related = self.knowledge_graph.get_related_entities(entity_text)
            all_related.extend(related)
        
        # Remove duplicates and sort
        unique_related: Dict[str, float] = {}
        for entity, weight in all_related:
            if entity not in unique_related or weight > unique_related[entity]:
                unique_related[entity] = weight
        
        return sorted(unique_related.items(), key=lambda x: x[1], reverse=True)[:10] 