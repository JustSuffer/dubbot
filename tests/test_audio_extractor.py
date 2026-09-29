"""
Unit and integration tests for AudioExtractor (Step 1).
"""

import os
import subprocess
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dubbot.audio_extractor import AudioExtractor, AudioExtractionError


def create_synthetic_video(output_path: Path, duration: int = 3) -> Path:
    """Generate a valid test video with synthetic video and audio tracks via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=25",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    video_file = test_dir / "test_synth.mp4"
    audio_output = test_dir / "test_extracted.wav"

    print(f"1. Creating synthetic test video: {video_file}...")
    create_synthetic_video(video_file, duration=3)
    assert video_file.is_file(), "Test video was not created!"
    print("   -> Test video created successfully.")

    print("2. Initializing AudioExtractor (16kHz mono)...")
    extractor = AudioExtractor(sample_rate=16000, channels=1)

    print("3. Testing media probing...")
    duration = extractor.get_duration(video_file)
    has_audio = extractor.has_audio_stream(video_file)
    print(f"   -> Probed duration: {duration:.2f}s, Has audio: {has_audio}")
    assert has_audio is True, "Audio stream should be detected!"
    assert 2.8 <= duration <= 3.2, f"Expected ~3s duration, got {duration}"

    print("4. Extracting audio...")
    extracted_path = extractor.extract(video_file, audio_output, overwrite=True)
    assert extracted_path.is_file(), "Extracted WAV file does not exist!"
    assert extracted_path.stat().st_size > 0, "Extracted WAV is empty!"
    print(f"   -> Extracted file: {extracted_path} ({extracted_path.stat().st_size} bytes)")

    print("5. Verifying extracted WAV attributes...")
    wav_info = extractor.probe(extracted_path)
    audio_stream = next(s for s in wav_info["streams"] if s["codec_type"] == "audio")
    assert audio_stream["sample_rate"] == "16000", f"Expected 16000 Hz, got {audio_stream['sample_rate']}"
    assert audio_stream["channels"] == 1, f"Expected 1 channel, got {audio_stream['channels']}"
    assert audio_stream["codec_name"] == "pcm_s16le", f"Expected pcm_s16le, got {audio_stream['codec_name']}"
    print(f"   -> Confirmed 16000Hz mono pcm_s16le audio!")

    print("\nALL AUDIO EXTRACTOR (STEP 1) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
