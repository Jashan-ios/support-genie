"""
RAG API Routes
Endpoints  for asking question to the knowledge base
"""

from fastapi import APIRouter, HTTPException
from app.models.schema import AskRequest, AskResponse, HistoryResponse
from app.services.rag import RAGService
from app.services.question_logger import QuestionLogger
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["RAG"])

@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        results = RAGService.ask(
            question=request.question,
            collection_name=request.collection_name,
            n_results=request.n_results,
            filter_by_section=request.filter_by_section
        )
        return AskResponse(**results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )
    
@router.post("/ask-stream")    
def ask_question_stream(request: AskRequest):
    def generate():
         for chunk in RAGService.ask_stream(
              question=request.question,
              collection_name=request.collection_name,
              n_results=request.n_results,
              filter_by_section=request.filter_by_section
         ):
             yield chunk

    return StreamingResponse(generate(), media_type="text/plain")          

    

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

 


   
