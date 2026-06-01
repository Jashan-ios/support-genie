from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.hybrid_search import HybridSearchService
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService

def comapre_search_methods():
    """Comapre pure semantic vs hybrid search."""

    collection =  "trulymadly_faq"

    queries = [
        "How do I cancel my premium",
        "Someone Keeps messaging me",
        "What payment methods do you accpet"
    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"? QUERY: '{query}")
        print(f"{'='*60}")

        #Pure semantic
        print("\n🔍 PURE SEMANTIC SEARCH:")
        query_emb = EmbeddingService.embed_text(query)
        semantic = VectorStoreService.search(
            collection_name=collection,
            query_embedding=query_emb,
            n_results=3
        )
        for i, doc in enumerate(semantic["documents"], 1):
            print(f" {i}. {doc[:80]}...")

        #Hybrid
        print("\n HYBRID SEARCH (50/50)") 
        hybrid = HybridSearchService.search(
            collection_name=collection,
            query=query,
            n_results=3,
            semantic_weight=0.5
        )   
        for i, detail in enumerate(hybrid["details"], 1):
            print(f" {i}. {detail['document'][:80]}...")
            print(f"  sem: {detail['semantic_score']} | "
                  f"Kw: {detail['keyword_score']} | "
                  f"final: {detail['final_score']}")

if __name__ == "__main__":
    print("Comparing search methods")
    comapre_search_methods()          