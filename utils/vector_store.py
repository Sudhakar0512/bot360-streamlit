#vector_store.py
from typing import List, Dict, Any
from .embedding_service import EmbeddingService
from .pinecone_service import PineconeService
import concurrent.futures
import logging
from .config import EMBEDDING_DIMENSION, DEFAULT_BATCH_SIZE

class OptimizedVectorStore:
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Initialize vector store with 1536 dimensions."""
        self.dimension = EMBEDDING_DIMENSION
        self.batch_size = batch_size
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.logger = logging.getLogger(__name__)
    
    
    def _process_batch(self, batch: List[Dict]) -> List[tuple]:
        """Process a batch of chunks into vectors with metadata."""
        try:
            embeddings = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Process embeddings in parallel
                future_to_text = {
                    executor.submit(self.embedding_service.get_embedding, chunk["text"]): chunk
                    for chunk in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_text):
                    chunk = future_to_text[future]
                    try:
                        embedding = future.result()
                        embeddings.append((chunk, embedding))
                    except Exception as e:
                        self.logger.error(f"Error processing chunk: {str(e)}")
                        continue
            
            return [
                (
                    f"chunk_{chunk['metadata']['chunk_index']}",
                    embedding,
                    {
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
                for chunk, embedding in embeddings
            ]
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            return []
    
    def store_embeddings(self, chunks: List[Dict]):
        """Store text chunks and their embeddings in batches."""
        try:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                vectors = self._process_batch(batch)
                if vectors:
                    self.pinecone_service.upsert_vectors(vectors)
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search with score threshold filtering.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of filtered search results
        """
        try:
            query_embedding = self.embedding_service.get_embedding(query)
            results = self.pinecone_service.query_vectors(query_embedding, top_k * 2)
            
            filtered_results = [
                result for result in results
                if result['score'] >= score_threshold
            ]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []
