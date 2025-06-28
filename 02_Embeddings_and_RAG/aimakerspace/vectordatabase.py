import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Optional, Union
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
import re


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def categorize_text_chunk(text: str) -> str:
    """
    Categorizes a text chunk based on its content into one of the PMarca Blog categories.
    
    Args:
        text: The text chunk to categorize
        
    Returns:
        Category string: "Startups", "Hiring", "Big Companies", or "Other"
    """
    text_lower = text.lower()
    
    # Startup-related keywords and patterns
    startup_keywords = [
        "startup", "entrepreneur", "founding", "founder", "venture capital", "vc", "funding",
        "business plan", "pitch", "mvp", "minimum viable product", "seed funding",
        "series a", "series b", "valuation", "burn rate", "runway", "pivot",
        "product market fit", "pmf", "market validation", "customer development"
    ]
    
    # Hiring-related keywords and patterns  
    hiring_keywords = [
        "hiring", "recruit", "interview", "candidate", "resume", "cv", "job",
        "employee", "staff", "team building", "onboarding", "hr", "human resources",
        "executive", "ceo", "cto", "cfo", "vp", "director", "manager",
        "performance review", "compensation", "salary", "promotion", "firing"
    ]
    
    # Big companies-related keywords
    big_company_keywords = [
        "big company", "large company", "corporation", "enterprise", "fortune 500",
        "public company", "ipo", "acquisition", "merger", "corporate", "bureaucracy",
        "organizational", "division", "department", "enterprise software", "legacy"
    ]
    
    # Check for section headers first (most reliable)
    if "guide to startups" in text_lower or "pmarca guide to startups" in text_lower:
        return "Startups"
    elif "guide to hiring" in text_lower or "pmarca guide to hiring" in text_lower:
        return "Hiring"  
    elif "guide to big companies" in text_lower or "pmarca guide to big companies" in text_lower:
        return "Big Companies"
    
    # Count keyword matches
    startup_count = sum(1 for keyword in startup_keywords if keyword in text_lower)
    hiring_count = sum(1 for keyword in hiring_keywords if keyword in text_lower)
    big_company_count = sum(1 for keyword in big_company_keywords if keyword in text_lower)
    
    # Determine category based on highest count
    max_count = max(startup_count, hiring_count, big_company_count)
    
    if max_count == 0:
        return "Other"
    elif startup_count == max_count:
        return "Startups"
    elif hiring_count == max_count:
        return "Hiring"
    elif big_company_count == max_count:
        return "Big Companies"
    else:
        return "Other"


class VectorDatabase:
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, str]] = {}
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.ndarray, metadata: Optional[Dict[str, str]] = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def insert_with_category(self, key: str, vector: np.ndarray, category: Optional[str] = None) -> None:
        """Insert a vector with automatic or manual category assignment."""
        if category is None:
            category = categorize_text_chunk(key)
        
        metadata = {"category": category}
        self.insert(key, vector, metadata)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        distance_measure: Callable = cosine_similarity,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors with optional category filtering."""
        if category_filter:
            # Filter vectors by category
            filtered_items = [
                (key, vector) for key, vector in self.vectors.items()
                if self.metadata.get(key, {}).get("category") == category_filter
            ]
        else:
            filtered_items = list(self.vectors.items())
        
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in filtered_items
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        category_filter: Optional[str] = None,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Search by text with optional category filtering."""
        query_vector = np.array(self.embedding_model.get_embedding(query_text))
        results = self.search(query_vector, k, distance_measure, category_filter)
        return [result[0] for result in results] if return_as_text else results

    def search_with_metadata(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, str]]]:
        """Search by text and return results with metadata."""
        query_vector = np.array(self.embedding_model.get_embedding(query_text))
        results = self.search(query_vector, k, distance_measure, category_filter)
        
        return [
            (text, score, self.metadata.get(text, {}))
            for text, score in results
        ]

    def get_categories(self) -> List[str]:
        """Get all unique categories in the database."""
        categories = set()
        for metadata in self.metadata.values():
            if "category" in metadata:
                categories.add(metadata["category"])
        return sorted(list(categories))

    def get_category_counts(self) -> Dict[str, int]:
        """Get counts of entries per category."""
        category_counts = defaultdict(int)
        for metadata in self.metadata.values():
            category = metadata.get("category", "Unknown")
            category_counts[category] += 1
        return dict(category_counts)

    def retrieve_from_key(self, key: str) -> Optional[np.ndarray]:
        """Retrieve vector by key."""
        return self.vectors.get(key, None)

    def retrieve_metadata_from_key(self, key: str) -> Dict[str, str]:
        """Retrieve metadata by key."""
        return self.metadata.get(key, {})

    async def abuild_from_list(self, list_of_text: List[str], auto_categorize: bool = True) -> "VectorDatabase":
        """Build database from list of texts with optional automatic categorization."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            vector = np.array(embedding)
            if auto_categorize:
                self.insert_with_category(text, vector)
            else:
                self.insert(text, vector)
        return self

    async def abuild_from_list_with_categories(
        self, 
        list_of_text: List[str], 
        categories: Optional[List[str]] = None
    ) -> "VectorDatabase":
        """Build database from list of texts with provided categories."""
        if categories and len(categories) != len(list_of_text):
            raise ValueError("Number of categories must match number of texts")
        
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            vector = np.array(embedding)
            category = categories[i] if categories else None
            self.insert_with_category(text, vector, category)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)

    # New metadata functionality demonstration
    print("\nMetadata functionality:")
    print("Categories:", vector_db.get_categories())
    print("Category counts:", vector_db.get_category_counts())
    
    # Search with metadata
    results_with_metadata = vector_db.search_with_metadata("I think fruit is awesome!", k=k)
    print(f"Results with metadata: {results_with_metadata}")
