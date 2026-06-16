"""
Pydantic schemas for API requestsnd responses!!
Why Pydantic?
- Auto-validates incoming JSON
- Auto-generates API docs
- Catches bad input before it reaches your code
- Type hints = better IDE experience
"""

from typing import List, Optional
from pydantic import BaseModel, Field

#---REQUEST SCHEMAS---------

class AskRequest(BaseModel):
    "Request body for the /ask endpoints"

    question: str = Field(..., min_length=1, description="The user's question")
    collection_name: str = Field(..., description="which Knowledge base to query")
    n_results: int = Field(default=3, ge=1, le=10, description="How many chunks to retrieve")
    filter_by_section: Optional[str] = Field(default=None, description="Optional: filter chunks by section")


#-- RESPONSE SCHEMAS-------
class SourceCitation(BaseModel):
    "A single source citation in the response."
    source: str
    section: str
    relevance: float

class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    question: str
    answer: str
    sources: List[SourceCitation]
    chunks_used: int

class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""
    file: str
    chunks_added: int
    collection: str
    total_chunks_in_collection: int
    strategy: str

class CollectionStats(BaseModel):
    """Stats about a collection."""
    name: str
    count: int

class Healthresponse(BaseModel):
    """Health check response."""
    status: str
    service: str = "support-genie"

class LogEntry(BaseModel):
        question: str
        answer: str   

class HistoryResponse(BaseModel):
        entries: list[LogEntry]  
        count: int          

   

 


         



    

