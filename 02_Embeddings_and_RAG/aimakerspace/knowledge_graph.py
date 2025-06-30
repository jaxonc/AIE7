import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, Any
import re
from dataclasses import dataclass, field
import asyncio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
        self._cached_entity_clusters: Optional[Dict[str, int]] = None
        self._cached_num_clusters: Optional[int] = None
        
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
    
    def create_graph_embeddings(self, num_features: int = 8) -> Dict[str, np.ndarray]:
        """Create graph embeddings for entities based on graph structure and properties."""
        if len(self.graph.nodes) == 0:
            return {}
        
        embeddings = {}
        
        # Calculate graph metrics
        pagerank_scores = nx.pagerank(self.graph, weight='weight') if len(self.graph.edges) > 0 else {}
        clustering_coeffs = nx.clustering(self.graph, weight='weight')
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Entity type encoding
        entity_types = list(set(entity.entity_type for entity in self.entities.values()))
        type_to_idx = {t: i for i, t in enumerate(entity_types)}
        
        print(f"🎯 Creating graph embeddings with {num_features} features for {len(self.entities)} entities...")
        
        for entity_name, entity in self.entities.items():
            if entity_name not in self.graph:
                # Create default embedding for isolated entities
                embedding = np.zeros(num_features)
                embedding[0] = entity.frequency  # Frequency as first feature
                embedding[1] = len(entity_types) if entity.entity_type not in type_to_idx else type_to_idx[entity.entity_type]
                embeddings[entity_name] = embedding
                continue
            
            # Feature vector construction
            features = []
            
            # 1. Node degree (normalized)
            degree = len(list(self.graph.neighbors(entity_name)))
            features.append(float(degree))
            
            # 2. PageRank score
            features.append(pagerank_scores.get(entity_name, 0.0))
            
            # 3. Clustering coefficient
            features.append(clustering_coeffs.get(entity_name, 0.0))
            
            # 4. Degree centrality
            features.append(degree_centrality.get(entity_name, 0.0))
            
            # 5. Entity frequency (log-scaled)
            features.append(np.log1p(entity.frequency))
            
            # 6. Entity type (one-hot encoded - using index)
            features.append(float(type_to_idx.get(entity.entity_type, 0)))
            
            # 7. Number of contexts (indicates how widely used the entity is)
            features.append(float(len(entity.contexts)))
            
            # 8. Average edge weight (indicates strength of relationships)
            if entity_name in self.graph:
                edge_weights = [self.graph[entity_name][neighbor].get('weight', 1.0) 
                              for neighbor in self.graph.neighbors(entity_name)]
                avg_edge_weight = np.mean(edge_weights) if edge_weights else 0.0
            else:
                avg_edge_weight = 0.0
            features.append(avg_edge_weight)
            
            # Pad or truncate to desired number of features
            if len(features) < num_features:
                features.extend([0.0] * (num_features - len(features)))
            else:
                features = features[:num_features]
            
            embeddings[entity_name] = np.array(features)
        
        return embeddings
    
    def kmeans_clustering(self, num_clusters: int = 4) -> Dict[str, int]:
        """K-means clustering based on graph embeddings."""
        # Check if we already have cached results for this number of clusters
        if (self._cached_entity_clusters is not None and 
            self._cached_num_clusters == num_clusters):
            return self._cached_entity_clusters
            
        if len(self.graph.nodes) == 0:
            return {}
        
        entity_names = list(self.entities.keys())
        
        # If fewer entities than clusters, assign each to its own cluster
        if len(entity_names) < num_clusters:
            result = {name: i for i, name in enumerate(entity_names)}
            self._cached_entity_clusters = result
            self._cached_num_clusters = num_clusters
            return result
        
        print(f"🧠 Applying K-means clustering with graph embeddings...")
        
        # Create graph embeddings
        embeddings_dict = self.create_graph_embeddings()
        
        # Convert to matrix for scikit-learn
        embedding_matrix = np.array([embeddings_dict[name] for name in entity_names])
        
        # Standardize features
        scaler = StandardScaler()
        embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix_scaled)
                
        # Create entity to cluster mapping
        entity_to_cluster = {entity_names[i]: int(cluster_labels[i]) for i in range(len(entity_names))}
        
        # Cache the results
        self._cached_entity_clusters = entity_to_cluster
        self._cached_num_clusters = num_clusters
        
        print(f"📊 K-means clustering completed: {len(set(cluster_labels))} clusters formed")
        
        return entity_to_cluster
    
    def assign_chunk_categories(self, num_clusters: int = 4) -> Dict[int, int]:
        """Assign categories to text chunks based on their entities' clusters."""
        entity_clusters = self.kmeans_clustering(num_clusters)
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
    
    def get_semantic_cluster_name(self, cluster_id: int, cluster_info: Dict[str, Any]) -> str:
        """Generate a semantic name for a cluster based on its top entities and keywords."""
        if not cluster_info or not cluster_info.get('entities'):
            return f"Cluster_{cluster_id}"
        
        # Get top entities (max 3)
        top_entities = [e['name'] for e in cluster_info['entities'][:3]]
        keywords = cluster_info.get('keywords', [])[:2]  # Max 2 keywords
        
        # Create semantic name based on most frequent entity
        primary_entity = top_entities[0] if top_entities else "Unknown"
        
        # Generate descriptive name based on primary entity and keywords
        if any(term in primary_entity.lower() for term in ['ceo', 'executive', 'hiring', 'manage']):
            semantic_name = "Executive & Hiring"
        elif any(term in primary_entity.lower() for term in ['startup', 'company', 'business']):
            semantic_name = "Business & Strategy"
        elif any(term in primary_entity.lower() for term in ['fund', 'invest', 'capital', 'market']):
            semantic_name = "Funding & Markets"
        elif any(term in primary_entity.lower() for term in ['product', 'team', 'operation']):
            semantic_name = "Operations & Product"
        else:
            # Fallback: use primary entity as base
            semantic_name = primary_entity.replace('_', ' ').title()
        
        # Add top entities for context
        entity_context = ', '.join(top_entities)
        return f"{semantic_name} ({entity_context})"
    
    def get_cluster_descriptions(self, num_clusters: int = 4) -> Dict[int, Dict[str, Any]]:
        """Get descriptions for each cluster based on top entities and keywords."""
        entity_clusters = self.kmeans_clustering(num_clusters)
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
            cluster_info = cluster_descriptions.get(category_id, {})
            metadata = {
                'category_id': category_id,
                'category_name': self.knowledge_graph.get_semantic_cluster_name(category_id, cluster_info),
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
        return [self.knowledge_graph.get_semantic_cluster_name(category_id, self.cluster_categories.get(category_id, {})) for category_id in self.cluster_categories.keys()]
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get counts per category."""
        counts: Dict[str, int] = defaultdict(int)
        for category_id in self.chunk_to_category.values():
            cluster_info = self.cluster_categories.get(category_id, {})
            counts[self.knowledge_graph.get_semantic_cluster_name(category_id, cluster_info)] += 1
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