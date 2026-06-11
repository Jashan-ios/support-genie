"""
SupportGenie API

Production-grade RAG-powered customer support API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schema import Healthresponse
from app.api import routes_rag, routes_ingest


# Create FastAPI app
app = FastAPI(
    title="SupportGenie API",
    description="RAG-powered customer support for any business",
    version="1.0.0"
)


# CORS — allow frontend apps to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # production: restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(routes_rag.router)
app.include_router(routes_ingest.router)


# Health check
@app.get("/", response_model=Healthresponse)
def health_check() -> Healthresponse:
    """Health check endpoint."""
    return Healthresponse(status="running")