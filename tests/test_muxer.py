"""
Unit tests for VideoMuxer (Step 6).
"""

import subprocess
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf
from dubbot.audio_extractor import AudioExtractor
from dubbot.muxer import VideoMuxer


def create_test_video(output_path: Path, duration: int = 3):
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=25",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest", str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate test video
    src_video = test_dir / "mux_src.mp4"
    create_test_video(src_video, duration=4)
    
    # 2. Generate a 1-second audio clip
    clip_wav = test_dir / "mux_clip.wav"
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clip_data = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    sf.write(str(clip_wav), clip_data, sr, subtype="PCM_16")
    
    muxer = VideoMuxer(sample_rate=16000)
    
    # 3. Assemble audio canvas
    segments = [
        {"start": 1.0, "end": 2.0, "aligned_audio_path": str(clip_wav)},
        {"start": 2.5, "end": 3.5, "aligned_audio_path": str(clip_wav)},
    ]
    assembled_wav = test_dir / "mux_assembled.wav"
    muxer.assemble_audio(segments, total_duration_sec=4.0, output_audio_path=assembled_wav)
    assert assembled_wav.is_file(), "Assembled audio not found!"
    
    # 4. Mux into final video
    out_video = test_dir / "mux_final.mp4"
    final_video = muxer.mux(src_video, assembled_wav, out_video, overwrite=True)
    assert final_video.is_file(), "Muxed video not found!"
    
    # 5. Verify duration & streams using AudioExtractor
    extractor = AudioExtractor()
    info = extractor.probe(final_video)
    streams = info.get("streams", [])
    v_stream = next((s for s in streams if s["codec_type"] == "video"), None)
    a_stream = next((s for s in streams if s["codec_type"] == "audio"), None)
    
    assert v_stream is not None, "Missing video stream!"
    assert a_stream is not None, "Missing audio stream!"
    print(f"Muxed video verified: video codec={v_stream['codec_name']}, audio codec={a_stream['codec_name']}")
    print(f"File size: {final_video.stat().st_size} bytes")

    print("\nALL VIDEO MUXER (STEP 6) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
