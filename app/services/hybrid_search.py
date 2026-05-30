"""
Hybrid Search Service

Combines semantic search (embeddings) with keyword search (BM25)
for production-grade retrieval.

Why hybrid?
- Pure semantic: misses exact codes, names, acronyms
- Pure keyword: misses semantic similarity
- Hybrid: covers both blind spots

Algorithm: Reciprocal Rank Fusion (RRF)
- Get ranked list from both methods
- Combine rankings using RRF formula
- Higher final rank = better match
"""

from typing import Dict, List, Tuple
from rank_bm25 import BM25Okapi
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService


class HybridSearchService:
    """Combines semantic and keyword search for better retrieval."""

    @classmethod
    def search(
        cls,
        collection_name: str,
        query: str,
        n_results: int = 5,
        semantic_weight: float = 0.5
    ) -> Dict:
        """
        Perform hybrid search combining semantic + keyword.

        Args:
            collection_name: Which collection to search
            query: User's question
            n_results: How many final results to return
            semantic_weight: 0.0 = pure keyword, 1.0 = pure semantic
                            (0.5 = balanced, our default)

        Returns:
            Dict with documents, metadatas, scores
        """
        # Get more candidates than needed (we'll filter down)
        candidates_count = n_results * 3

        # Step 1: Semantic search
        semantic_results = cls._semantic_search(
            collection_name, query, candidates_count
        )

        # Step 2: Keyword search (BM25) on the same candidates
        keyword_scores = cls._keyword_search(
            query, semantic_results["documents"]
        )

        # Step 3: Combine using weighted fusion
        final_results = cls._combine_scores(
            semantic_results=semantic_results,
            keyword_scores=keyword_scores,
            semantic_weight=semantic_weight,
            n_results=n_results
        )

        return final_results

    @staticmethod
    def _semantic_search(
        collection_name: str,
        query: str,
        n_results: int
    ) -> Dict:
        """Run vector similarity search."""
        query_embedding = EmbeddingService.embed_text(query)
        return VectorStoreService.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=n_results
        )

    @staticmethod
    def _keyword_search(query: str, documents: List[str]) -> List[float]:
        """
        Run BM25 keyword search on documents.
        
        Returns normalized scores (0 to 1) for each doc.
        """
        # Tokenize documents and query (simple: lowercase + split)
        tokenized_docs = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        # Score each document against the query
        scores = bm25.get_scores(tokenized_query)

        # Normalize to 0-1 range
        max_score = max(scores) if scores.size > 0 and max(scores) > 0 else 1
        return [score / max_score for score in scores]

    @staticmethod
    def _combine_scores(
        semantic_results: Dict,
        keyword_scores: List[float],
        semantic_weight: float,
        n_results: int
    ) -> Dict:
        """
        Combine semantic and keyword scores with weighted fusion.

        Final score = (semantic_weight × semantic_score) + 
                      ((1 - semantic_weight) × keyword_score)
        """
        documents = semantic_results["documents"]
        metadatas = semantic_results["metadatas"]
        distances = semantic_results["distances"]

        # Convert distances to similarity scores (1 - distance, capped at 0)
        semantic_scores = [max(0, 1 - d) for d in distances]

        # Compute combined scores
        combined = []
        for i in range(len(documents)):
            sem_score = semantic_scores[i]
            kw_score = keyword_scores[i] if i < len(keyword_scores) else 0
            final_score = (
                semantic_weight * sem_score +
                (1 - semantic_weight) * kw_score
            )
            combined.append({
                "document": documents[i],
                "metadata": metadatas[i],
                "semantic_score": round(sem_score, 3),
                "keyword_score": round(kw_score, 3),
                "final_score": round(final_score, 3)
            })

        # Sort by final score (highest first)
        combined.sort(key=lambda x: x["final_score"], reverse=True)

        # Return top N
        top_results = combined[:n_results]

        return {
            "documents": [r["document"] for r in top_results],
            "metadatas": [r["metadata"] for r in top_results],
            "scores": [r["final_score"] for r in top_results],
            "details": top_results  # full breakdown for debugging
        }