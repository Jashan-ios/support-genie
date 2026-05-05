"""
Document Loader Service
"""
from pathlib import Path
from pypdf import PdfReader


class DocumentLoader:
    """Loads documents from various formats into clean text."""
    
    @staticmethod
    def load(file_path: Path) -> str:
        """Load a document and return its text content."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            return DocumentLoader._load_pdf(file_path)
        elif extension == ".md":
            return DocumentLoader._load_markdown(file_path)
        elif extension == ".txt":
            return DocumentLoader._load_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported: .pdf, .md, .txt"
            )
    
    @staticmethod
    def _load_pdf(file_path: Path) -> str:
        """Extract text from PDF, page by page."""
        reader = PdfReader(file_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """Load markdown file as plain text."""
        return file_path.read_text(encoding="utf-8")
    
    @staticmethod
    def _load_text(file_path: Path) -> str:
        """Load plain text file."""
        return file_path.read_text(encoding="utf-8")