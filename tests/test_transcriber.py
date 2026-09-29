"""
Unit and integration tests for Transcriber (Step 2).
"""

import os
import subprocess
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dubbot.transcriber import Transcriber, TranscriptionError


def generate_speech_wav_windows(output_wav: Path, text: str = "Hello and welcome to dubbot. This is a local AI transcription test."):
    """Uses Windows built-in SAPI to generate a spoken audio test wav file."""
    escaped_text = text.replace('"', '`"')
    ps_command = f"""
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SetOutputToWaveFile('{output_wav}')
$speak.Speak('{escaped_text}')
$speak.Dispose()
"""
    subprocess.run(["powershell", "-Command", ps_command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_wav


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = test_dir / "sample_speech.wav"

    print("1. Generating spoken audio via Windows SpeechSynthesizer...")
    generate_speech_wav_windows(sample_wav)
    assert sample_wav.is_file() and sample_wav.stat().st_size > 0, "Failed to create sample speech file!"
    print(f"   -> Sample speech audio created: {sample_wav} ({sample_wav.stat().st_size} bytes)")

    print("2. Initializing Transcriber (tiny or base model on CUDA)...")
    transcriber = Transcriber(model_size="tiny", device="auto", compute_type="auto")
    print(f"   -> Configured device: {transcriber.device}, compute_type: {transcriber.compute_type}")

    print("3. Running transcription with timestamps...")
    segments = transcriber.transcribe(sample_wav, word_timestamps=True)
    print(f"   -> Extracted {len(segments)} segments.")

    assert len(segments) > 0, "Should have transcribed at least 1 segment!"
    for seg in segments:
        print(f"   [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
        assert "start" in seg and "end" in seg and "text" in seg, "Segment dict missing keys!"
        assert seg["end"] > seg["start"], "Segment end timestamp must be greater than start!"
        assert len(seg["text"]) > 0, "Transcribed text should not be empty!"

    print("\nALL TRANSCRIBER (STEP 2) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
