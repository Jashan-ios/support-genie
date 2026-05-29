"""Test the complete RAG service end-to-end."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import RAGService


def run_rag_demo():
    """Ask real questions to the RAG system."""
    
    collection = "trulymadly_faq"
    
    questions = [
        "How do I cancel my premium membership?",
        "Someone keeps messaging me, how can I stop them?",
        "What payment methods do you accept?",
        "Can I get a refund?",
        "What's the meaning of life?",  # not in our docs!
    ]
    
    for question in questions:
        result = RAGService.ask(
            question=question,
            collection_name=collection
        )
        
        print("\n" + "="*60)
        print(f"💬 ANSWER:")
        print(f"   {result['answer']}")
        print(f"\n📚 SOURCES:")
        for source in result['sources']:
            print(f"   • {source['source']} ({source['section']}) "
                  f"— {source['relevance']*100:.1f}% relevant")
        print("="*60)


if __name__ == "__main__":
    print("🚀 RAG System Demo\n")
    run_rag_demo()
    print("\n🎉 Your complete RAG system is alive!")