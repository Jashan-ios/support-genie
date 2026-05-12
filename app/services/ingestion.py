"""
Ingestion PipeLine

The orchaster that combines:
- document loader (read file)
- chunking service (split smartly)
- embedding service (convert to vectors)
- vector store(presist with metadata)

This is the main entry point for adding documnets to the knowledge base.
"""
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from app.services.document_loader import DocumentLoader
from app.services.chunker import ChunkingService
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService

class IngestionPipeline:
    """Orchestrates the full document ingestion workflow"""

    @staticmethod 
    def ingest_document(
        file_path: Path,
        collection_name: str, 
        chunking_strategy: str  =   "recursive",
        extra_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Ingest a document end-to-end into the vector store.

        Args:
            file_path: Path to the document
            collection_name: Which business's collection to add to
            chunking_strategy: "recursive", "fixed", or "markdown"
            extra_metadata: Additional metadata to attach to all chunks

        Returns:
            Dict with ingestion stats (chunks added, file info, etc.)
        """

        print(f"\n📥 Ingesting: {file_path.name}")
        print(f"   Collection: {collection_name}")
        print(f"   Strategy: {chunking_strategy}")

        #step 1: Load the document
        print("\n[1/4] Loading document...")
        text = DocumentLoader.load(file_path)
        print(f"      Loaded {len(text)} characters")

        #step2: Chunk it using chosen strategy
        print("\n[2/4] Chunking text...")
        chunks = IngestionPipeline._chunk_text(text, chunking_strategy)
        print(f"      Created {len(chunks)} chunks")

        #step3: Generate embeddings 
        print("\n[3/4] Generating embeddings...")
        embeddings = EmbeddingService.embed_batch(chunks)
        print(f"      Generated {len(embeddings)} embeddings")

        #step4: Build metadata and store
        print("\n[4/4] Storing in vector database...")
        metadatas = IngestionPipeline._build_metadata(
            chunks, file_path, extra_metadata
        )
        ids = [f"{file_path.stem}_{uuid4().hex[:8]}_{i}" for i in range(len(chunks))]

        VectorStoreService.add_chunks(
            collection_name=collection_name,
            chunks=chunks, 
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        #returns stats
        stats = VectorStoreService.collection_stats(collection_name)

        return {
            "file": file_path.name,
            "chunks_added": len(chunks),
            "collection": collection_name,
            "total_chunks_in_collection": stats["count"],
            "strategy": chunking_strategy
        }
    
    @staticmethod
    def _chunk_text(text: str, strategy: str)-> list:
        if strategy == "recursive":
            return ChunkingService.recursive_chunks(text)
        elif strategy == "fixed":
            return ChunkingService.fixed_size_chunks(text)
        elif strategy == "markdown":
            return ChunkingService.markdown_chunks(text)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Use 'recursive', 'fixed', or 'markdown'."
            )
        
    @staticmethod
    def _build_metadata(
        chunks: list,
        file_path: Path,
        extra: Optional[Dict] = None
    ) -> list:
        """
        Build metadata dict for each chunk.
        
        Standard fields:
        - source: filename
        - file_type: extension
        - chunk_index: position in document
        + any extra metadata passed in
        """
        metadatas = []
        for i, _ in enumerate(chunks):
            meta = {
                "source": file_path.name,
                "file_type": file_path.suffix,
                "chunk_index": i
            }
            if extra:
                meta.update(extra)
            metadatas.append(meta)
        return metadatas    

