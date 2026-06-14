"""
LLM Service

Wraps the Groq API for generating text responses.

Why a separate service?
- Hide API specifics (Groq today, OpenAI tomorrow — easy swap)
- Centralized error handling and retries
- Singleton client (avoid re-initializing on every call)
- Consistent interface across the app 
"""

from typing import List, Dict, Optional
from groq import Groq
from app.core.config import settings

class LLMService:
    "Handle LLM calls via groq API"
    _client: Optional[Groq] = None
    DEFAULT_MODEL = "llama-3.1-8b-instant"

    @classmethod
    def get_client(cls) -> Groq:
        """Lazy-load the groq client"""
        if cls._client is None:
            if not settings.groq_api_key:
                raise ValueError (
                     "GROQ api is not present in .env file"
                )
            cls._client = Groq(api_key=settings.groq_api_key)
        return cls._client
    
    @classmethod
    def generate(
        cls,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response from the llm
        
        Args:
        messages: List of messages dict
        model: which groq model to use
        temp: 0.0 = deterministic, 1.0 = creative 
        max_token = Max length of response 

        Returns:
        The generated text response 
        """
        client = cls.get_client()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM API error: {e}")
            raise
    
    @classmethod
    def generate_stream(cls, messages, model=DEFAULT_MODEL, temperature=0.3, max_tokens=1024):
        client = cls.get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        try:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"❌ LLM API error: {e}")
            raise

                
   
            

            
               

   