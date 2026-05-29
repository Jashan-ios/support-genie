"""
RAG service
Orchesterates the full retrieval augmented generation flow:
1. embedd user question 
2. search vector store for releveant chunks 
3. build prompt with retrieved context
4. return answer with source citabations

This is the main interface used by API endpoints
"""

from typing import Dict, List
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService

class RAGService:
    """ Main RAG orchestrator. """
    SYSTEM_PROMPT = """ You are a helpful customer support assistant.
    Answer the user's question using ONLY the context provided below
    RULES:
    - If the conext doesn't contain the answer, say "I don't have the information about that in my knowledge base."
    - Keep ansers concise and friendly
    - dont make up information
    - dont mention "context" and "chunks" in your answer - talk naturally
    """

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
        print(f"\n? Question: {question}")

        print("[1/4] embedding question..")
        quesry_embedding = EmbeddingService.embed_text(question)

        print("[2/4] searching base")
        search_results = VectorStoreService.search(
            collection_name=collection_name,
            query_embedding=quesry_embedding,
            n_results=n_results
        )
        retrieved_chunks = search_results["documents"]
        sources =search_results["metadatas"]
        distances = search_results["distances"]

        print("[3/4] building prompt")
        context = cls._format_context(retrieved_chunks)
        user_message = f"""Context:
{context}
Question: {question}"""
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        print("[4/4] generating answer...")
        answer = LLMService.generate(messages)

        return {
            "question": question,
            "answer": answer,
            "sources": cls._format_sources(sources, distances),
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
    def _format_sources(metadatas: List[Dict], distances: List[float]) -> List[Dict]:
        """Format source citations for the response."""
        return [
            {
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", "N/A"),
                "relevance": round(1 - dist, 3)  # convert distance to relevance score
            }
            for meta, dist in zip(metadatas, distances)
        ]
         


