"""
Embedding Service

Converts text chunks into vector embeddings using 
sentence-transformers (runs locally, no API costs).

Why sentence-transformers?
- Free (no per-request costs)
- Fast (runs locally on CPU/GPU)
- Privacy (data never leaves your server)
- Production quality (used by major companies)

Trade-off vs OpenAI embeddings:
- Slightly lower quality (~5-10% worse on benchmarks)
- But 1000x cheaper at scale
- Good enough for 90% of use cases
"""

from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingService:
    """Generates vector embeddings for text using sentence-transformers."""
    
    _model = None  # class-level cache
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Lazy-load the model (singleton pattern).
        
        Why singleton?
        - Loading the model takes 5-10 seconds
        - Loading once at startup is fine
        - Loading per-request would be terrible UX
        """
        if cls._model is None:
            print(f"📦 Loading embedding model: {settings.embedding_model}")
            cls._model = SentenceTransformer(settings.embedding_model)
            print(f"✅ Model loaded! Dimension: {cls._model.get_sentence_embedding_dimension()}")
        return cls._model
    
    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        """
        Convert a single text into a vector embedding.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding (typically 384 dimensions)
        """
        model = cls.get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    @classmethod
    def embed_batch(cls, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embeddings (batch processing).
        
        Why batch?
        - 10x faster than embedding one at a time
        - GPU/CPU is more efficient with batches
        - Critical for ingesting large documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (one per input text)
        """
        model = cls.get_model()
        embeddings = model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,  # nice UX for large batches
            batch_size=32  # process 32 at a time (adjust based on RAM)
        )
        return embeddings.tolist()