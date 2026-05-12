"""Test the full ingestion pipeline end-to-end."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ingestion import IngestionPipeline
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService


# Sample customer support doc (markdown formatted)
SAMPLE_DOC = """
# TrulyMadly Customer Support FAQ

## Account Management

### Creating an Account
To create a TrulyMadly account, download the app from App Store or Play Store. Sign up with your phone number and verify with OTP. Complete your profile with photos and bio.

### Deleting an Account
To delete your account permanently, go to Settings → Privacy → Delete Account. This action cannot be undone. All your matches and messages will be lost.

### Account Recovery
If you accidentally deleted your account, contact support@trulymadly.com within 30 days. We can restore your account if requested in time.

## Premium Membership

### Pricing
Premium membership costs 999 rupees per month. Annual subscription is 9999 rupees (saves 17%). All plans include unlimited likes, profile boost, and read receipts.

### Cancellation
You can cancel premium anytime from Settings → Subscription → Cancel. Cancellation takes effect at the end of your billing cycle. No refunds for partial months.

### Payment Methods
We accept all major credit cards, debit cards, UPI, and net banking. International users can pay via PayPal.

## Safety and Privacy

### Reporting Users
You can report a user by tapping the three dots menu on their profile. Select Report and choose the reason. Our team reviews reports within 24 hours.

### Blocking Users
Blocking a user prevents them from messaging you or seeing your profile. They won't be notified that you blocked them. You can unblock from Settings → Blocked Users.

### Data Privacy
Your personal data is encrypted and never shared with third parties. We comply with GDPR and Indian data protection laws.
"""


def test_full_ingestion():
    """Test the complete ingestion pipeline."""
    
    collection_name = "trulymadly_faq"
    
    # cleanup any previous test runs
    try:
        VectorStoreService.delete_collection(collection_name)
    except Exception:
        pass
    
    # write sample doc to a file
    test_file = Path("data/uploads/test_faq.md")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(SAMPLE_DOC)
    
    # 🚀 INGEST WITH ONE CALL!
    print("=" * 60)
    print("🎯 INGESTING DOCUMENT")
    print("=" * 60)
    
    stats = IngestionPipeline.ingest_document(
        file_path=test_file,
        collection_name=collection_name,
        chunking_strategy="markdown",
        extra_metadata={"business": "trulymadly", "category": "faq"}
    )
    
    print("\n" + "=" * 60)
    print("📊 INGESTION COMPLETE")
    print("=" * 60)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # now test searching the ingested content!
    print("\n" + "=" * 60)
    print("🔍 TESTING RETRIEVAL")
    print("=" * 60)
    
    test_queries = [
        "How can I remove my profile?",
        "How much is premium?",
        "Someone is bothering me, what do I do?",
        "Can I get my money back?",
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: '{query}'")
        query_embedding = EmbeddingService.embed_text(query)
        
        results = VectorStoreService.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            n_results=1
        )
        
        if results["documents"]:
            top_match = results["documents"][0]
            distance = results["distances"][0]
            metadata = results["metadatas"][0]
            
            print(f"   ✅ Match (distance: {distance:.3f})")
            print(f"   📄 Source: {metadata.get('source')}")
            print(f"   💬 {top_match[:150]}...")
    
    # test metadata filtering
    print("\n" + "=" * 60)
    print("🎯 TESTING METADATA FILTER")
    print("=" * 60)
    
    query = "What payment methods are accepted?"
    print(f"\n❓ Query: '{query}'")
    print(f"   Filter: only premium-related chunks")
    
    query_embedding = EmbeddingService.embed_text(query)
    results = VectorStoreService.search(
        collection_name=collection_name,
        query_embedding=query_embedding,
        n_results=2,
        where={"business": "trulymadly"}
    )
    
    for i, doc in enumerate(results["documents"], 1):
        print(f"\n   {i}. {doc[:150]}...")


if __name__ == "__main__":
    print("🚀 Testing Full RAG Ingestion Pipeline\n")
    test_full_ingestion()
    print("\n\n🎉 PIPELINE COMPLETE! Your RAG system is alive!")