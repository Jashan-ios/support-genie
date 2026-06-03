"""
Re- Ranker Services
Re-orders retrieved  chunks by TRUE relevance using a cross-encoder model.

Why re-ranking?
- hybrid search returns chunks by score 

"""

from typing import Dict, List
from sentence_transformers import CrossEncoder
import numpy as np

class ReRankerService:
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model = None

    @classmethod
    def get_model(cls)-> CrossEncoder:
        if cls._model is None:
            print(f"Loading cross-encoder:{cls.MODEL_NAME}")
            cls._model = CrossEncoder(cls.MODEL_NAME)
        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: Dict,
        top_n: int = 3
    )-> Dict:
        documents = candidates["documents"]
        metadatas = candidates["metadatas"]
        original_scores = candidates["scores"]

        if not documents:
            return candidates

        pairs = [(query, doc) for doc in documents] 
        model = cls.get_model()
        relevance_score = model.predict(pairs)

        scored_docs = list(zip(
            documents,
            metadatas,
            original_scores,
            relevance_score
        ))   
        scored_docs.sort(key=lambda x: x[3], reverse=True)

        top_docs = scored_docs[:top_n]
        def sigmoid(x):
            return 1/(1+ np.exp(-x))
        normalized_scores = [sigmoid(d[3]) for d in top_docs]

        return {
            "documents": [d[0] for d in top_docs],
            "metadatas": [d[1]for d in top_docs],
            "scores": [float(s) for s in normalized_scores],
            "original_score": [d[2] for d in top_docs]


        }