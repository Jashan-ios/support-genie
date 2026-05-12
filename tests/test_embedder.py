"""Test the embedding service."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedder import EmbeddingService


def test_single_embedding():
    """Test embedding a single text."""
    text = "How do I cancel my account?"
    embedding = EmbeddingService.embed_text(text)
    
    print(f"\n✅ Single embedding")
    print(f"   Text: {text}")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    assert len(embedding) == 384, "Expected 384 dimensions"


def test_batch_embedding():
    """Test embedding multiple texts at once."""
    texts = [
        "How do I delete my profile?",
        "Steps to remove my account",
        "What's the weather today?"
    ]
    embeddings = EmbeddingService.embed_batch(texts)
    
    print(f"\n✅ Batch embedding")
    print(f"   Input: {len(texts)} texts")
    print(f"   Output: {len(embeddings)} embeddings")
    print(f"   Each dimension: {len(embeddings[0])}")
    
    assert len(embeddings) == 3


def test_similarity():
    """Test that similar meanings give similar embeddings."""
    import numpy as np
    
    e1 = EmbeddingService.embed_text("I love hiking")
    e2 = EmbeddingService.embed_text("I enjoy trekking")
    e3 = EmbeddingService.embed_text("I love cooking")
    
    # cosine similarity
    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine(e1, e2)
    sim_1_3 = cosine(e1, e3)
    
    print(f"\n✅ Similarity test")
    print(f"   'hiking' vs 'trekking':  {sim_1_2:.3f}  (should be high)")
    print(f"   'hiking' vs 'cooking':   {sim_1_3:.3f}  (should be lower)")
    
    assert sim_1_2 > sim_1_3, "Similar texts should have higher similarity"


if __name__ == "__main__":
    print("Testing EmbeddingService...")
    test_single_embedding()
    test_batch_embedding()
    test_similarity()
    print("\n🎉 All embedding tests passed!")