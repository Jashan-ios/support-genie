"""
Chunking Service

Implementing multiple chunking strategies for comparison and production use.
Each strategies is appropriate for different document types.
"""

#Imports
from typing import List
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

class ChunkingService:
    """Provides multiple chunking startegies for douments."""

    @staticmethod
    def fixed_size_chunks(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        strategy1: fixed size chunking

        Splits text into fixed-size chunks regardless of content.
        Simple but can break sentences/paragraphs awkardly.
        
        Use when: you need predictable chunk sizes and content and structure doesn't matter much
        """

        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator=""
        )
        return splitter.split_text(text)


    @staticmethod
    def recursive_chunks(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    )-> List[str]:
        """
        Strategy 2: Recursive chunking (the smart default)
        
        Tries to split at natural boundaries:
        paragraphs → sentences → words → characters
        
        Use when: You don't know the document structure 
        or it's unstructured prose.

        """   
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        ) 
        return splitter.split_text(text)
    
    @staticmethod
    def markdown_chunks(text: str)-> List[str]:
         """
        Strategy 3: Markdown-aware chunking
        
        Splits at markdown headers, preserving document structure.
        Each chunk = one section.
        
        Use when: Document has clear markdown structure 
        (docs, blogs, READMEs).

        """
         headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
         ]
         splitter = MarkdownHeaderTextSplitter(
             headers_to_split_on=headers_to_split_on
         )
         docs = splitter.split_text(text)

         enriched_chunks = []
         for doc in docs:
             headers = []
             if doc.metadata.get("Header 1"):
                 headers.append(doc.metadata["Header 1"])
             if doc.metadata.get("Header 2"):
                 headers.append(doc.metadata["Header 2"])
             if doc.metadata.get("Header 3"):
                 headers.append(doc.metadata["Header 3"])

             header_path = " > ".join(headers)

             if header_path:
                enriched = f"{header_path}\n\n{doc.page_content}"
             else:
               enriched = doc.page_content

             enriched_chunks.append(enriched)  
             
         return enriched_chunks
     
    @staticmethod
    def choose_chunking_strategy(filename: str, text: str) -> str: 
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "md":
           return "markdown"

        header_lines = sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))
        if header_lines >= 3:
            return "markdown"

        return "recursive"
    
  
