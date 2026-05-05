"""Test the document loader with sample files"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.document_loader import DocumentLoader

def test_load_text():
    """Test loading a plain text file."""
    test_file = Path("data/uploads/test.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("This is a test document about TrulyMadly")

    text = DocumentLoader.load(test_file)
    print(f"✅ Text loaded: {text}")
    assert "TrulyMadly" in text

def test_load_markdown():
    test_file = Path("data/uploads/test.md") 
    test_file.write_text("# Hello\n\nThis is **markdown**.")

    text = DocumentLoader.load(test_file)
    print(f"✅ Markdown loaded: {text}")
    assert "Hello" in text


def test_unsupported_format():
    test_file = Path("data/uploads/test.xyz")
    test_file.write_text("This shouldn't load")

    try:
        DocumentLoader.load(test_file)
        print("should have raised an error") 
    except ValueError as e:
        print(f"✅ Correctly rejected unsupported format: {e}")       


if __name__ == "__main__":
    print("Running document loader tests...\n")
    test_load_text()
    test_load_markdown()
    test_unsupported_format()
    print("\n🎉 All tests passed!")        