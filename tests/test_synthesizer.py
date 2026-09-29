"""
Unit tests for VoiceSynthesizer (Step 4 - Coqui XTTSv2).
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf
from dubbot.synthesizer import VoiceSynthesizer


def main():
    test_dir = Path(__file__).resolve().parent / "temp_fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    speaker_wav = test_dir / "sample_speech.wav"
    assert speaker_wav.is_file(), f"Reference speaker wav not found: {speaker_wav}"
    
    print("1. Initializing VoiceSynthesizer (Coqui XTTSv2 on CUDA)...")
    synthesizer = VoiceSynthesizer(device="auto")
    print(f"   -> Use GPU: {synthesizer.use_gpu}")
    
    print("2. Synthesizing Turkish speech with cloned voice...")
    out_wav = test_dir / "synth_test_output.wav"
    text = "Merhaba dunya, bu bir yerel ses klonlama testidir."
    
    generated_path = synthesizer.synthesize(
        text=text,
        speaker_wav=speaker_wav,
        language="tr",
        output_path=out_wav,
    )
    
    assert generated_path.is_file(), "Synthesized WAV was not generated!"
    assert generated_path.stat().st_size > 0, "Synthesized WAV is empty!"
    
    info = sf.info(str(generated_path))
    print(f"   -> Synthesized audio duration: {info.duration:.2f}s, sample rate: {info.samplerate}Hz")
    assert info.duration > 0.5, f"Audio too short: {info.duration}s"

    print("\nALL VOICE SYNTHESIZER (STEP 4) TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
