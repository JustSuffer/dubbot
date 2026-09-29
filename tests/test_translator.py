"""
Unit tests for Translator (Step 3).
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dubbot.translator import Translator


def main():
    print("1. Initializing Translator...")
    translator = Translator(device="auto")
    print(f"   -> Translator device: {translator.device}, dtype: {translator.torch_dtype}")
    
    print("2. Testing batch translation (English -> Turkish)...")
    texts = [
        "Hello and welcome to our video tutorial.",
        "Artificial intelligence is transforming open source software.",
    ]
    translations = translator.translate_batch(texts, source_lang="en", target_lang="tr")
    
    assert len(translations) == len(texts), "Translation count mismatch!"
    for orig, trans in zip(texts, translations):
        print(f"   [EN]: {orig}")
        print(f"   [TR]: {trans}")
        assert len(trans) > 0, "Translation result was empty!"

    print("3. Testing segment translation...")
    segments = [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world!"},
    ]
    trans_segs = translator.translate_segments(segments, source_lang="en", target_lang="tr")
    assert "source_text" in trans_segs[0], "Missing source_text in segment!"
    assert trans_segs[0]["source_text"] == "Hello world!"
    print(f"   Segment translated: '{trans_segs[0]['text']}'")

    print("\nALL TRANSLATOR (STEP 3) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
