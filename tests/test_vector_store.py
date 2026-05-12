"""Test the vector store service."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService


def test_full_flow():
    """Test the complete flow: add chunks, then search."""
    
    collection_name = "test_business"
    
    # cleanup from previous runs
    try:
        VectorStoreService.delete_collection(collection_name)
    except:
        pass  # collection didn't exist, that's fine
    
    # sample chunks
    chunks = [
        "To delete your account, go to Settings → Privacy → Delete Account.",
        "Premium membership costs 999 rupees per month with unlimited likes.",
        "You can report a user by tapping the three dots on their profile.",
        "To reset your password click on forgot password on the login screen.",
        "Blocking a user prevents them from messaging you ever again.",
    ]
    
    # embed them all in batch
    print("\n📊 Embedding chunks...")
    embeddings = EmbeddingService.embed_batch(chunks)
    
    # build metadata
    metadatas = [
        {"source": "faq.md", "section": "Account"},
        {"source": "faq.md", "section": "Premium"},
        {"source": "faq.md", "section": "Safety"},
        {"source": "faq.md", "section": "Account"},
        {"source": "faq.md", "section": "Safety"},
    ]
    
    # unique ids
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # add to vector store
    VectorStoreService.add_chunks(
        collection_name=collection_name,
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    # check stats
    stats = VectorStoreService.collection_stats(collection_name)
    print(f"\n📈 Collection stats: {stats}")
    
    # now test searching!
    print("\n🔍 Test 1: Search for account deletion")
    query = "How can I remove my profile?"
    query_embedding = EmbeddingService.embed_text(query)
    
    results = VectorStoreService.search(
        collection_name=collection_name,
        query_embedding=query_embedding,
        n_results=2
    )
    
    print(f"   Query: '{query}'")
    print(f"   Top results:")
    for i, (doc, dist) in enumerate(zip(results["documents"], results["distances"])):
        print(f"   {i+1}. (distance: {dist:.3f}) {doc[:80]}...")
    
    # test metadata filtering
    print("\n🔍 Test 2: Search only in Safety section")
    safety_query = "How do I report someone bothering me?"
    safety_embedding = EmbeddingService.embed_text(safety_query)
    
    results = VectorStoreService.search(
        collection_name=collection_name,
        query_embedding=safety_embedding,
        n_results=3,
        where={"section": "Safety"}  # filter by metadata!
    )
    
    print(f"   Query: '{safety_query}'")
    print(f"   Filter: only 'Safety' section")
    print(f"   Results found: {len(results['documents'])}")
    for i, doc in enumerate(results["documents"]):
        print(f"   {i+1}. {doc[:80]}...")


if __name__ == "__main__":
    print("Testing VectorStoreService...\n")
    test_full_flow()
    print("\n🎉 All vector store tests passed!")