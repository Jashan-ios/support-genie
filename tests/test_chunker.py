"""
Comapre chunking strategies on the same document.
This shows WHY strategy choice matters
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunker import ChunkingService

# Sample document with markdown structure
SAMPLE_DOC = """
# TrulyMadly FAQ

## Account Management

### Creating an Account
To create an account, download the app and sign up with your phone number. You'll need to verify with OTP.

### Deleting an Account
To delete your account, go to Settings → Privacy → Delete Account. This action is permanent.

## Premium Features

### Pricing
Premium membership costs 999 rupees per month. Annual plans are 9999 rupees.

### Cancellation
You can cancel anytime from Settings → Subscription. Refunds are not available once purchased.

## Safety

### Reporting Users
You can report a user by tapping the three dots on their profile and selecting Report.

### Blocking
Blocking a user prevents them from messaging you. They won't know they're blocked.
"""

def print_chunks(strategy_name: str, chunks: list):
    """Pretty print chunks for comparison."""
    print(f"\n{'='*60}")
    print(f"🔹 {strategy_name}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200])
        if len(chunk)> 200:
            print("... (truncated)")
        print() 

if __name__ == "__main__":
    print("📊 Comparing chunking strategies on TrulyMadly FAQ\n")
    print(f"Original doc length: {len(SAMPLE_DOC)} chars")

     # Strategy 1: Fixed-size
    chunks_1 = ChunkingService.fixed_size_chunks(
        SAMPLE_DOC, chunk_size=200, overlap=20
    )           
    print_chunks("FIXED-SIZE (chunk_size=200)", chunks_1)

    chunks_2 = ChunkingService.recursive_chunks(
        SAMPLE_DOC, chunk_size=200, overlap=20
    )
    print_chunks("RECURSIVE (chunk_size=200)", chunks_2)

    chunks_3 = ChunkingService.markdown_chunks(SAMPLE_DOC)
    print_chunks("MARKDOWN-AWARE", chunks_3)

    print("\n🎯 Compare them above. Notice:")
    print("   - Fixed-size: breaks awkwardly mid-sentence")
    print("   - Recursive:  respects paragraphs")
    print("   - Markdown:   each chunk = one topic!")
