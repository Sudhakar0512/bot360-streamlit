#embedding_service.py
from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv
import logging
from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL

load_dotenv()

class EmbeddingService:
    # def __init__(self):
    #     """Initialize embedding service with 1536 dimensions."""
    #     self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #     self.dimension = EMBEDDING_DIMENSION
    #     self.model = EMBEDDING_MODEL
    #     self.logger = logging.getLogger(__name__)
    def __init__(self):
        """Initialize embedding service with 1536 dimensions."""
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0  # Adjust timeout if needed
        )
        self.dimension = EMBEDDING_DIMENSION
        self.model = EMBEDDING_MODEL
        self.logger = logging.getLogger(__name__)
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        if not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return [0] * self.dimension
            
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise