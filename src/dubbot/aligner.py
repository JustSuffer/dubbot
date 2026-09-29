"""
Audio Alignment & Time-Stretching Module for Dubbot.

Dynamically stretches or compresses synthesized speech to perfectly fit
original timestamp windows without pitch distortion using librosa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AlignmentError(Exception):
    """Raised when audio alignment or time-stretching fails."""
    pass


class AudioAligner:
    """
    Adjusts TTS audio durations to align with original speech segment intervals.
    """

    def __init__(
        self,
        min_rate: float = 0.65,
        max_rate: float = 1.65,
        target_sample_rate: int = 16000,
    ):
        """
        Initialize the AudioAligner.

        Args:
            min_rate: Minimum speed multiplier (slowing down limit, prevents unnatural drags).
            max_rate: Maximum speed multiplier (speeding up limit, prevents unintelligible chips).
            target_sample_rate: Target audio sample rate.
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_sample_rate = target_sample_rate

    def get_audio_duration(self, audio_path: str | Path) -> float:
        """Return the duration of an audio file in seconds."""
        info = sf.info(str(audio_path))
        return info.duration

    def time_stretch(
        self,
        audio_path: str | Path,
        target_duration: float,
        output_path: str | Path,
    ) -> Path:
        """
        Speed up or slow down audio to match target_duration, preserving original pitch.

        Args:
            audio_path: Path to the input WAV audio.
            target_duration: Desired duration in seconds.
            output_path: Path to save the stretched audio.

        Returns:
            Path to the output aligned audio file.
        """
        src = Path(audio_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Input audio file not found: {src}")

        dst = Path(output_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if target_duration <= 0.05:
            logger.warning("Target duration (%.3fs) is too short. Setting to minimum 0.1s.", target_duration)
            target_duration = 0.1

        # Load audio
        y, sr = librosa.load(str(src), sr=self.target_sample_rate, mono=True)
        current_duration = len(y) / sr

        if current_duration <= 0.01:
            raise AlignmentError(f"Audio file is practically empty: {src}")

        # Compute required speed factor:
        # rate > 1.0 speeds up (shorter), rate < 1.0 slows down (longer)
        raw_rate = current_duration / target_duration
        clamped_rate = float(np.clip(raw_rate, self.min_rate, self.max_rate))

        logger.debug(
            "Aligning audio '%s': current=%.2fs -> target=%.2fs (rate=%.2f, clamped=%.2f)",
            src.name, current_duration, target_duration, raw_rate, clamped_rate
        )

        try:
            # Apply pitch-preserving time stretch
            if abs(clamped_rate - 1.0) > 0.02:
                # time_stretch in librosa
                y_stretched = librosa.effects.time_stretch(y, rate=clamped_rate)
            else:
                y_stretched = y

            target_samples = int(target_duration * sr)
            actual_samples = len(y_stretched)

            # Post-process to ensure exact target length
            if actual_samples < target_samples:
                # Pad with silence to match exact duration
                pad_width = target_samples - actual_samples
                y_final = np.pad(y_stretched, (0, pad_width), mode="constant")
            elif actual_samples > target_samples:
                # Apply short fade-out on the tail to prevent clipping clicks, then trim
                fade_len = min(int(0.02 * sr), target_samples)  # 20ms fade
                y_trimmed = y_stretched[:target_samples].copy()
                if fade_len > 0:
                    fade_curve = np.linspace(1.0, 0.0, fade_len)
                    y_trimmed[-fade_len:] *= fade_curve
                y_final = y_trimmed
            else:
                y_final = y_stretched

            # Save the final aligned audio
            sf.write(str(dst), y_final, sr, subtype="PCM_16")

        except Exception as err:
            raise AlignmentError(f"Failed to time-stretch audio '{src}': {err}") from err

        return dst

    def align_segments(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Align all synthesized segments with their corresponding Whisper time windows.

        Args:
            segments: List of segment dicts with 'tts_audio_path', 'start', 'end'.
            output_dir: Directory to save the aligned audio clips.

        Returns:
            List of segment dicts updated with 'aligned_audio_path' and 'duration'.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        aligned_segments: List[Dict[str, Any]] = []

        for i, seg in enumerate(segments):
            tts_path = seg.get("tts_audio_path")
            if not tts_path or not Path(tts_path).is_file():
                logger.warning("Segment %d has no valid tts_audio_path. Skipping alignment.", i)
                continue

            target_duration = max(0.1, seg["end"] - seg["start"])
            aligned_path = out_dir / f"aligned_segment_{i:04d}.wav"

            self.time_stretch(
                audio_path=tts_path,
                target_duration=target_duration,
                output_path=aligned_path,
            )

            updated = dict(seg)
            updated["aligned_audio_path"] = str(aligned_path)
            updated["target_duration"] = target_duration
            aligned_segments.append(updated)

        logger.info("Successfully aligned %d segments.", len(aligned_segments))
        return aligned_segments
