"""
Ingestion  API routes
Endpoints or uploading and processing documents 
"""

import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schema import IngestResponse,CollectionStats
from app.services.ingestion import IngestionPipeline
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["Ingestion"])

@router.post("/ingest", response_model= IngestResponse)
async def ingest_document(
collection_name: str = Form(...),
chunking_strategy: str = Form("recursive"),
file: UploadFile = File(...)
) -> IngestResponse :
    allowed_extensions = [".pdf", ".md", ".txt"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. ALlowed: {allowed_extensions}"
        )
    upload_path = settings.upload_dir/ file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = IngestionPipeline.ingest_document(
            file_path=upload_path,
            collection_name=collection_name,
            chunking_strategy=chunking_strategy
        )
        return IngestResponse(**result) 
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )
    
@router.get("/collection/{name}", response_model=CollectionStats)
def get_collection_stats(name: str)-> CollectionStats:
    try:
        stats = VectorStoreService.collection_stats(name)
        return CollectionStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"collection not found: {str(e)}")
    
    

    



