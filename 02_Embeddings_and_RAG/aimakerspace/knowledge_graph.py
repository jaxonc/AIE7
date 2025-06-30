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
        
    def _get_entity_stop_words(self) -> Set[str]:
        """Get stop words specifically for entity extraction."""
        return {
            # Common words that get capitalized but aren't entities
            'the', 'this', 'that', 'these', 'those', 'a', 'an',
            'when', 'what', 'where', 'why', 'how', 'who', 'which',
            'part', 'section', 'chapter', 'page', 'line', 'paragraph',
            'first', 'second', 'third', 'last', 'next', 'previous',
            'many', 'most', 'some', 'all', 'each', 'every', 'any',
            'new', 'old', 'big', 'small', 'large', 'great', 'good', 'best',
            'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there',
            'maybe', 'perhaps', 'however', 'therefore', 'meanwhile',
            'according', 'during', 'through', 'within', 'without',
            'another', 'other', 'others', 'such', 'same', 'different',
            'following', 'above', 'below', 'around', 'between', 'among',
            'including', 'regarding', 'concerning', 'considering',
            'unfortunately', 'fortunately', 'obviously', 'certainly',
            'probably', 'possibly', 'actually', 'really', 'truly',
            'especially', 'particularly', 'specifically', 'generally'
        }

    def _get_expanded_business_terms(self) -> Dict[str, str]:
        """Get comprehensive business terms with their entity types."""
        return {
            # Executive roles
            'ceo': 'EXECUTIVE', 'chief executive officer': 'EXECUTIVE',
            'cto': 'EXECUTIVE', 'chief technology officer': 'EXECUTIVE', 
            'cfo': 'EXECUTIVE', 'chief financial officer': 'EXECUTIVE',
            'coo': 'EXECUTIVE', 'chief operating officer': 'EXECUTIVE',
            'cmo': 'EXECUTIVE', 'chief marketing officer': 'EXECUTIVE',
            'vp': 'EXECUTIVE', 'vice president': 'EXECUTIVE',
            'director': 'EXECUTIVE', 'managing director': 'EXECUTIVE',
            'founder': 'EXECUTIVE', 'co-founder': 'EXECUTIVE',
            'president': 'EXECUTIVE', 'chairman': 'EXECUTIVE',
            
            # Business entities
            'startup': 'BUSINESS', 'company': 'BUSINESS', 'corporation': 'BUSINESS',
            'business': 'BUSINESS', 'enterprise': 'BUSINESS', 'firm': 'BUSINESS',
            'organization': 'BUSINESS', 'venture': 'BUSINESS', 'partnership': 'BUSINESS',
            'unicorn': 'BUSINESS', 'scale-up': 'BUSINESS', 'spin-off': 'BUSINESS',
            
            # Funding and finance
            'funding': 'FINANCE', 'investment': 'FINANCE', 'capital': 'FINANCE',
            'venture capital': 'FINANCE', 'angel investor': 'FINANCE',
            'series a': 'FINANCE', 'series b': 'FINANCE', 'series c': 'FINANCE',
            'ipo': 'FINANCE', 'valuation': 'FINANCE', 'revenue': 'FINANCE',
            'profit': 'FINANCE', 'equity': 'FINANCE', 'shares': 'FINANCE',
            'round': 'FINANCE', 'seed funding': 'FINANCE', 'pre-seed': 'FINANCE',
            
            # People and roles
            'employee': 'PERSON', 'staff': 'PERSON', 'team': 'PERSON',
            'candidate': 'PERSON', 'hire': 'PERSON', 'recruit': 'PERSON',
            'manager': 'PERSON', 'executive': 'PERSON', 'leader': 'PERSON',
            'consultant': 'PERSON', 'advisor': 'PERSON', 'mentor': 'PERSON',
            'intern': 'PERSON', 'contractor': 'PERSON', 'freelancer': 'PERSON',
            
            # Business operations
            'strategy': 'CONCEPT', 'management': 'CONCEPT', 'leadership': 'CONCEPT',
            'operations': 'CONCEPT', 'marketing': 'CONCEPT', 'sales': 'CONCEPT',
            'product': 'CONCEPT', 'service': 'CONCEPT', 'platform': 'CONCEPT',
            'technology': 'CONCEPT', 'innovation': 'CONCEPT', 'disruption': 'CONCEPT',
            'growth': 'CONCEPT', 'scale': 'CONCEPT', 'expansion': 'CONCEPT',
            'acquisition': 'CONCEPT', 'merger': 'CONCEPT', 'exit': 'CONCEPT',
            
            # Market and industry
            'market': 'MARKET', 'industry': 'MARKET', 'sector': 'MARKET',
            'customer': 'MARKET', 'client': 'MARKET', 'user': 'MARKET',
            'competition': 'MARKET', 'competitor': 'MARKET', 'partnership': 'MARKET',
            'ecosystem': 'MARKET', 'landscape': 'MARKET', 'segment': 'MARKET',
            
            # Process and methods
            'hiring': 'PROCESS', 'recruitment': 'PROCESS', 'interview': 'PROCESS',
            'onboarding': 'PROCESS', 'training': 'PROCESS', 'development': 'PROCESS',
            'planning': 'PROCESS', 'execution': 'PROCESS', 'implementation': 'PROCESS',
            'launch': 'PROCESS', 'pivot': 'PROCESS', 'iteration': 'PROCESS'
        }

    def extract_entities_enhanced(self, text: str) -> List[Tuple[str, str]]:
        """Enhanced entity extraction with robust filtering and comprehensive business terms."""
        entities = []
        entity_stop_words = self._get_entity_stop_words()
        business_terms = self._get_expanded_business_terms()
        
        # 1. Extract capitalized phrases with better patterns
        patterns = [
            # Multi-word proper nouns (e.g., "John Smith", "Google Inc")
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
            # Single capitalized words (more selective)
            r'\b[A-Z][a-z]{2,}\b',
            # Acronyms (2-5 capital letters)
            r'\b[A-Z]{2,5}\b',
            # Title case with numbers (e.g., "Series A", "Web 2.0")
            r'\b[A-Z][a-z]+\s*[0-9]+(?:\.[0-9]+)?\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                match_clean = match.strip()
                if (
                    len(match_clean) > 2 and
                    match_clean.lower() not in entity_stop_words and
                    not match_clean.lower() in text.lower()[:30] and  # Avoid sentence starters
                    (not match_clean.isupper() or len(match_clean) <= 5) and  # Allow short acronyms only
                    not match_clean.endswith('.') and  # Avoid abbreviations
                    not any(char.isdigit() for char in match_clean if '.' not in match_clean) and  # Allow version numbers
                    not match_clean.lower() in ['engineering', 'series', 'capital', 'the', 'this', 'that'] and  # Additional common false positives
                    not (len(match_clean) < 5 and match_clean.lower() in ['a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'we'])
                ):
                    # Determine entity type based on context
                    entity_type = "ENTITY"
                    if any(role in match_clean.lower() for role in ['ceo', 'cto', 'cfo', 'founder', 'director']):
                        entity_type = "EXECUTIVE"
                    elif any(org in match_clean.lower() for org in ['inc', 'corp', 'ltd', 'llc', 'company']):
                        entity_type = "BUSINESS"
                    elif match_clean.isupper() and 2 <= len(match_clean) <= 5:
                        entity_type = "ACRONYM"
                    
                    entities.append((match_clean, entity_type))
        
        # 2. Extract business terms with context awareness
        text_lower = text.lower()
        for term, term_type in business_terms.items():
            if term in text_lower:
                # Check context to ensure it's used as an entity, not just mentioned
                term_positions = [m.start() for m in re.finditer(re.escape(term), text_lower)]
                for pos in term_positions:
                    # Extract context around the term
                    context_start = max(0, pos - 20)
                    context_end = min(len(text), pos + len(term) + 20)
                    context = text[context_start:context_end].lower()
                    
                    # Entity validation based on context
                    if (
                        # Not in the middle of another word
                        (pos == 0 or not text[pos-1].isalnum()) and
                        (pos + len(term) >= len(text) or not text[pos + len(term)].isalnum()) and
                        # Avoid overly generic usage
                        not any(generic in context for generic in ['in general', 'for example', 'such as', 'like']) and
                        # Prioritize specific business contexts
                        (any(business in context for business in ['hire', 'manage', 'lead', 'fund', 'invest', 'startup']) or 
                         term_type in ['EXECUTIVE', 'FINANCE', 'PROCESS'])
                    ):
                        entities.append((term.title(), term_type))
                        break  # Only add once per term per text
        
        # 3. Remove duplicates and apply final quality filters
        seen = set()
        filtered_entities = []
        for entity_text, entity_type in entities:
            entity_key = entity_text.lower()
            if (
                entity_key not in seen and
                len(entity_text) >= 2 and
                len(entity_text) <= 50 and  # Reasonable length limits
                not entity_text.lower() in ['we', 'our', 'us', 'you', 'your', 'they', 'them'] and
                entity_text not in ['In', 'On', 'At', 'By', 'For', 'With', 'From', 'To']  # Common false positives
            ):
                seen.add(entity_key)
                filtered_entities.append((entity_text, entity_type))
        
        return filtered_entities

    def extract_entities_basic(self, text: str) -> List[Tuple[str, str]]:
        """Enhanced entity extraction - now calls the robust version."""
        return self.extract_entities_enhanced(text)
    
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
                # Extract robust keywords
                keywords = self._extract_robust_keywords(all_contexts)
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
    
    def _get_comprehensive_stop_words(self) -> Set[str]:
        """Get a comprehensive set of stop words for keyword filtering."""
        return {
            # Articles
            'a', 'an', 'the',
            
            # Conjunctions
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            
            # Prepositions
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 
            'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by', 
            'during', 'except', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 
            'off', 'on', 'outside', 'over', 'through', 'to', 'toward', 'under', 'until', 
            'up', 'upon', 'with', 'within', 'without',
            
            # Pronouns
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            
            # Verbs (common forms)
            'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
            'was', 'were', 'get', 'gets', 'got', 'getting', 'go', 'goes', 'went', 'going',
            'make', 'makes', 'made', 'making', 'take', 'takes', 'took', 'taking',
            'come', 'comes', 'came', 'coming', 'see', 'sees', 'saw', 'seeing', 'know', 
            'knows', 'knew', 'knowing', 'think', 'thinks', 'thought', 'thinking',
            
            # Adverbs
            'very', 'really', 'quite', 'too', 'so', 'just', 'only', 'also', 'still', 'even',
            'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'than', 'too', 'well', 'as',
            
            # Question words
            'what', 'where', 'when', 'why', 'who', 'whom', 'whose', 'which', 'how',
            
            # Common business filler words
            'said', 'says', 'say', 'saying', 'tells', 'told', 'telling', 'ask', 'asked', 'asking',
            'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next', 'new', 'old',
            'good', 'great', 'best', 'better', 'bad', 'worse', 'worst', 'big', 'small', 'large',
            'little', 'long', 'short', 'high', 'low', 'right', 'left', 'early', 'late',
            
            # Time and frequency words
            'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'time', 'day', 'always',
            'never', 'sometimes', 'often', 'usually', 'rarely', 'frequently', 'occasionally',
            
            # Common verbs that add little meaning
            'use', 'used', 'using', 'work', 'works', 'worked', 'working', 'help', 'helps', 
            'helped', 'helping', 'find', 'finds', 'found', 'finding', 'look', 'looks', 
            'looked', 'looking', 'give', 'gives', 'gave', 'giving', 'put', 'puts', 'putting',
            'keep', 'keeps', 'kept', 'keeping', 'let', 'lets', 'letting', 'try', 'tries', 
            'tried', 'trying', 'seem', 'seems', 'seemed', 'seeming', 'feel', 'feels', 'felt',
            'feeling', 'become', 'becomes', 'became', 'becoming', 'leave', 'leaves', 'left',
            'leaving', 'move', 'moves', 'moved', 'moving', 'live', 'lives', 'lived', 'living',
            'show', 'shows', 'showed', 'showing', 'hear', 'hears', 'heard', 'hearing',
            'play', 'plays', 'played', 'playing', 'run', 'runs', 'ran', 'running', 'turn',
            'turns', 'turned', 'turning', 'start', 'starts', 'started', 'starting', 'end',
            'ends', 'ended', 'ending', 'stop', 'stops', 'stopped', 'stopping', 'follow',
            'follows', 'followed', 'following', 'set', 'sets', 'setting', 'sit', 'sits',
            'sat', 'sitting', 'stand', 'stands', 'stood', 'standing', 'meet', 'meets', 
            'met', 'meeting', 'bring', 'brings', 'brought', 'bringing', 'happen', 'happens',
            'happened', 'happening', 'write', 'writes', 'wrote', 'writing', 'provide',
            'provides', 'provided', 'providing', 'serve', 'serves', 'served', 'serving',
            
            # Numbers and ordinals that are not meaningful as keywords
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            
            # Common adjectives that are too general
            'different', 'same', 'important', 'possible', 'available', 'free', 'able', 'sure',
            'certain', 'clear', 'easy', 'hard', 'difficult', 'simple', 'complex', 'basic',
            'advanced', 'general', 'specific', 'special', 'particular', 'main', 'major', 'minor',
            'whole', 'full', 'empty', 'open', 'close', 'closed', 'public', 'private', 'personal',
            'social', 'final', 'current', 'recent', 'former', 'previous', 'original', 'additional',
            
            # Modal and auxiliary words
            'might', 'may', 'could', 'would', 'should', 'must', 'ought', 'need', 'needs', 'needed', 'needing'
        }
        
    def _extract_robust_keywords(self, contexts: List[str], max_keywords: int = 5) -> List[str]:
        """Extract high-quality keywords from contexts with robust filtering."""
        if not contexts:
            return []
        
        # Combine all contexts
        text = ' '.join(contexts).lower()
        
        # Extract words (alphanumeric with optional hyphens)
        words = re.findall(r'\b[a-z][a-z0-9\-]*[a-z0-9]\b|\b[a-z]\b', text)
        
        # Get comprehensive stop words
        stop_words = self._get_comprehensive_stop_words()
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Filter words based on multiple criteria
        filtered_words = []
        for word, freq in word_freq.most_common(20):  # Consider top 20 to have more options
            if (
                # Basic filters
                word not in stop_words and
                len(word) >= 3 and  # Minimum length
                len(word) <= 20 and  # Maximum length (avoid weird artifacts)
                freq >= 2 and  # Must appear at least twice
                
                # Content quality filters
                not word.isdigit() and  # Not just numbers
                not word.startswith('http') and  # Not URLs
                not word.endswith('.com') and  # Not domain names
                not word.endswith('.org') and
                not word.endswith('.net') and
                word.count('-') <= 2 and  # Not overly hyphenated
                
                # Business domain relevance filters
                not word in {'things', 'stuff', 'something', 'anything', 'everything', 'nothing'} and
                (not word.startswith('re-') or word in {'revenue', 'resources', 'recruitment', 'research', 'requirements'}) and  # Allow meaningful re- words
                
                # Avoid overly generic business terms
                word not in {'business', 'company', 'organization', 'industry', 'sector', 'market', 'field', 'area'}
            ):
                # Prioritize business-relevant terms
                business_boost = 1
                if any(domain in word for domain in ['tech', 'startup', 'fund', 'invest', 'hire', 'exec', 'manage', 'lead', 'strategy', 'growth']):
                    business_boost = 2
                
                filtered_words.append((word, freq * business_boost))
        
        # Sort by boosted frequency and return top keywords
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in filtered_words[:max_keywords]]


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