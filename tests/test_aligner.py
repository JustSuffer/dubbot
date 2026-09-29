"""
Unit tests for AudioAligner (Step 5).
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf
from dubbot.aligner import AudioAligner


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    # Generate 2.0 second pure tone
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    input_wav = test_dir / "align_test_in.wav"
    sf.write(str(input_wav), tone, sr, subtype="PCM_16")
    
    aligner = AudioAligner(target_sample_rate=sr)
    
    # Test 1: Compress 2.0s -> 1.5s
    out_15 = test_dir / "align_test_15.wav"
    aligner.time_stretch(input_wav, target_duration=1.5, output_path=out_15)
    dur_15 = aligner.get_audio_duration(out_15)
    print(f"Test 1: target=1.5s, actual={dur_15:.3f}s")
    assert abs(dur_15 - 1.5) < 0.05, f"Expected ~1.5s, got {dur_15}"
    
    # Test 2: Stretch 2.0s -> 2.6s
    out_26 = test_dir / "align_test_26.wav"
    aligner.time_stretch(input_wav, target_duration=2.6, output_path=out_26)
    dur_26 = aligner.get_audio_duration(out_26)
    print(f"Test 2: target=2.6s, actual={dur_26:.3f}s")
    assert abs(dur_26 - 2.6) < 0.05, f"Expected ~2.6s, got {dur_26}"

    print("\nALL AUDIO ALIGNER (STEP 5) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
