from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import (
    CACHE_DIR,
    load_movies)

movies_cache_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def load_or_create_embeddings(self, documents):
        try:
            self.embeddings = np.load(movies_cache_path)
            self.documents = documents
            for id, doc in enumerate(documents):
                self.document_map[id] = doc
            if len(self.embeddings) != len(documents):
                raise ValueError("Cached embeddings do not match the number of documents.")
            return self.embeddings
        except FileNotFoundError:
            return self.build_embeddings(documents)    
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, self.document_map[idx]))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:limit]
    
    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for id, doc in enumerate(documents):
            self.document_map[id] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save(movies_cache_path, self.embeddings)
        return self.embeddings

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
        embeddings = self.model.encode([text])
        return embeddings[0]


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text):
    embedding_instance = SemanticSearch()
    embedding = embedding_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    embedding_instance = SemanticSearch()
    embedding = embedding_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def verify_embeddings():
    search_instance = SemanticSearch()
    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(search_instance.documents)}")
    print(f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions")