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

class ChukningService:
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

    @staticmethod
    def recursive_chunks(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    )-> list[str]:
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
         return [doc.page_content for doc  in docs]