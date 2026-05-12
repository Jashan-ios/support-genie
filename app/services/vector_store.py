"""
Vector Store Service
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings


class VectorStoreService:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return cls._client

    @classmethod
    def get_or_create_collection(cls, collection_name: str):
        client = cls.get_client()
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def add_chunks(cls, collection_name, chunks, embeddings, metadatas, ids):
        collection = cls.get_or_create_collection(collection_name)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(chunks)} chunks to '{collection_name}'")

    @classmethod
    def search(cls, collection_name, query_embedding, n_results=3, where=None):
        collection = cls.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }

    @classmethod
    def collection_stats(cls, collection_name):
        collection = cls.get_or_create_collection(collection_name)
        return {
            "name": collection_name,
            "count": collection.count()
        }

    @classmethod
    def delete_collection(cls, collection_name):
        client = cls.get_client()
        client.delete_collection(collection_name)
        print(f"🗑️  Deleted collection '{collection_name}'")