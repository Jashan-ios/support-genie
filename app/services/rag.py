"""
RAG Service

Orchestrates the full retrieval-augmented generation flow:
1. Run hybrid search (semantic + keyword) on user question
2. Build prompt with retrieved context
3. Generate answer with LLM
4. Return answer with source citations

This is the main interface used by API endpoints.
"""

from typing import Dict, List
from app.services.hybrid_search import HybridSearchService
from app.services.llm import LLMService


class RAGService:
    """Main RAG orchestrator."""

    SYSTEM_PROMPT = """You are a helpful customer support assistant.

Answer the user's question using ONLY the context provided below.

Rules:
- If the context doesn't contain the answer, say "I don't have information about that in my knowledge base."
- Keep answers concise and friendly.
- Don't make up information.
- Don't mention "context" or "chunks" in your answer — talk naturally."""

    @classmethod
    def ask(
        cls,
        question: str,
        collection_name: str,
        n_results: int = 3
    ) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: The user's question
            collection_name: Which business's collection to search
            n_results: How many chunks to retrieve

        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n❓ Question: {question}")

        # Step 1-2: Hybrid search (handles embedding internally)
        print("[1/3] Running hybrid search...")
        search_results = HybridSearchService.search(
            collection_name=collection_name,
            query=question,
            n_results=n_results,
            semantic_weight=0.5
        )

        retrieved_chunks = search_results["documents"]
        sources = search_results["metadatas"]
        scores = search_results["scores"]

        # Step 3: Build prompt with context
        print("[2/3] Building prompt...")
        context = cls._format_context(retrieved_chunks)
        user_message = f"""Context:
{context}

Question: {question}"""

        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Step 4: Generate answer with LLM
        print("[3/3] Generating answer...")
        answer = LLMService.generate(messages)

        return {
            "question": question,
            "answer": answer,
            "sources": cls._format_sources(sources, scores),
            "chunks_used": len(retrieved_chunks)
        }

    @staticmethod
    def _format_context(chunks: List[str]) -> str:
        """Format retrieved chunks as numbered context."""
        return "\n\n".join(
            f"[Source {i+1}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )

    @staticmethod
    def _format_sources(metadatas: List[Dict], scores: List[float]) -> List[Dict]:
        """Format source citations for the response."""
        return [
            {
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "N/A"),
                "relevance": round(score, 3)
            }
            for meta, score in zip(metadatas, scores)
        ]