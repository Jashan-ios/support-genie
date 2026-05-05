from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings loaded from .env file.
    
    Why pydantic-settings instead of os.getenv?
    - Type validation (catches bugs early)
    - Auto-loads from .env
    - Production standard pattern
    """

    groq_api_key: str = ""

    environment: str = "development"
    
    # Storage paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    processed_dir: Path = base_dir / "data" / "processed"
    chroma_persist_dir: Path = base_dir / "chroma_db"

    chunk_size:int = 500
    chunk_overlap:int = 50
    embedding_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()        