#pinecone_service.py
from pinecone import Pinecone, ServerlessSpec
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
import time
from .config import EMBEDDING_DIMENSION, MAX_RETRIES

load_dotenv()

class PineconeService:
    def __init__(self):
        """Initialize Pinecone service with 1536 dimensions."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
            
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "default-index")
        self.dimension = EMBEDDING_DIMENSION
        self.max_retries = MAX_RETRIES
        self.logger = logging.getLogger(__name__)
        self._initialize_index()
    
    
    def _initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)
    
    def upsert_vectors(self, vectors: List[tuple]):
        """Upsert vectors to Pinecone index."""
        self.index.upsert(vectors=vectors)
    
    def query_vectors(self, vector: List[float], top_k: int = 3) -> List[Dict]:
        """Query vectors from Pinecone index."""
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        return [
            {
                'text': match.metadata['text'],
                'score': match.score
            }
            for match in results.matches
        ]
