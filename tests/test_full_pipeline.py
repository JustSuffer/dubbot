"""
End-to-End Pipeline Integration Test for Dubbot.
"""

import subprocess
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dubbot.audio_extractor import AudioExtractor
from dubbot.pipeline import DubbingPipeline


def create_sample_speaking_video(output_mp4: Path) -> Path:
    """Creates a sample video with spoken English audio using Windows SAPI and FFmpeg."""
    temp_dir = output_mp4.parent
    wav_path = temp_dir / "temp_input_speech.wav"
    
    # 1. Generate spoken audio
    text = "Hello and welcome to dubbot. This is a complete test of the local AI dubbing pipeline."
    ps_cmd = f"""
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SetOutputToWaveFile('{wav_path}')
$speak.Speak('{text}')
$speak.Dispose()
"""
    subprocess.run(["powershell", "-Command", ps_cmd], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 2. Get audio duration
    extractor = AudioExtractor()
    dur = extractor.get_duration(wav_path)
    
    # 3. Create MP4 with test video pattern and generated speech audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={dur:.2f}:size=640x360:rate=25",
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", str(output_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_mp4


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    input_video = test_dir / "pipeline_input_video.mp4"
    output_video = test_dir / "pipeline_dubbed_output_tr.mp4"
    
    print("1. Creating input test video with spoken speech...")
    create_sample_speaking_video(input_video)
    assert input_video.is_file(), "Input video creation failed!"
    print(f"   -> Input video created: {input_video} ({input_video.stat().st_size} bytes)")

    print("2. Initializing DubbingPipeline...")
    pipeline = DubbingPipeline(
        whisper_model="tiny",  # Use tiny for ultra-fast integration testing
        device="cuda",
        keep_temp=True,
    )

    print("3. Running DubbingPipeline (English -> Turkish)...")
    result = pipeline.run(
        video_path=input_video,
        target_lang="tr",
        source_lang="en",
        output_video_path=output_video,
    )

    print("4. Verifying pipeline outputs...")
    assert Path(result["output_video"]).is_file(), "Dubbed output video not found!"
    assert Path(result["output_video"]).stat().st_size > 1000, "Output video is suspiciously small!"
    
    # Verify streams in output video
    extractor = AudioExtractor()
    info = extractor.probe(result["output_video"])
    streams = info.get("streams", [])
    has_video = any(s["codec_type"] == "video" for s in streams)
    has_audio = any(s["codec_type"] == "audio" for s in streams)
    assert has_video and has_audio, "Output video missing video or audio stream!"

    print(f"   -> Final video verified: {result['output_video']}")
    print(f"   -> Total duration: {result['total_duration_sec']:.2f}s")
    print(f"   -> Processed segments: {result['total_segments']}")
    print(f"   -> Elapsed time: {result['elapsed_time_sec']:.2f}s")

    print("\n=======================================================")
    print("FULL END-TO-END PIPELINE TEST PASSED WITH FLYING COLORS!")
    print("=======================================================")


if __name__ == "__main__":
    main()
