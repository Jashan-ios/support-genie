"""
RAG API Routes
Endpoints  for asking question to the knowledge base
"""

from fastapi import APIRouter, HTTPException
from app.models.schema import AskRequest, AskResponse, HistoryResponse
from app.services.rag import RAGService
from app.services.question_logger import QuestionLogger

router = APIRouter(prefix="/api", tags=["RAG"])

@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        results = RAGService.ask(
            question=request.question,
            collection_name=request.collection_name,
            n_results=request.n_results
        )
        return AskResponse(**results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )
    

@router.get("/history", response_model=HistoryResponse)
def question_history(n: int = 10) -> HistoryResponse:
    try:
        results = QuestionLogger.get_recent(n)
        return HistoryResponse(entries=results,
                                count=QuestionLogger.count())
    
    except Exception as e:
        raise HTTPException(
        status_code=500,
        detail=f"Failed to answer question: {str(e)}"
        )

 


   
