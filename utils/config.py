"""Configuration settings for the application."""

# Embedding configuration
EMBEDDING_DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"  # This model outputs 1536 dimensions

# Batch processing configuration
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3

# Chunking configuration
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 100