"""
Audio Extraction Module for Dubbot.

Extracts high-quality 16kHz mono PCM WAV audio from input video files
using ffmpeg-python.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import ffmpeg

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction or probing fails."""
    pass


class AudioExtractor:
    """
    Extracts audio tracks from video files formatted for speech recognition and TTS.

    Defaults to 16kHz mono PCM 16-bit LE WAV format (optimal for Whisper and XTTS).
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the AudioExtractor.

        Args:
            sample_rate: Target audio sample rate in Hz (default: 16000).
            channels: Target number of audio channels (1 for mono, 2 for stereo, default: 1).
        """
        self.sample_rate = sample_rate
        self.channels = channels

    def probe(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Probe a media file using ffprobe and return metadata.

        Args:
            file_path: Path to the media file.

        Returns:
            Dictionary containing metadata and stream information.

        Raises:
            FileNotFoundError: If the media file does not exist.
            AudioExtractionError: If probing fails.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")

        try:
            probe_data = ffmpeg.probe(str(path))
            return probe_data
        except ffmpeg.Error as err:
            stderr = err.stderr.decode("utf-8", errors="replace") if err.stderr else str(err)
            raise AudioExtractionError(f"Failed to probe file '{path}': {stderr}") from err

    def get_duration(self, file_path: str | Path) -> float:
        """
        Get the duration of a media file in seconds.

        Args:
            file_path: Path to the media file.

        Returns:
            Duration in seconds as a float.
        """
        info = self.probe(file_path)
        format_info = info.get("format", {})
        if "duration" in format_info:
            return float(format_info["duration"])

        # Fallback to streams
        for stream in info.get("streams", []):
            if "duration" in stream:
                return float(stream["duration"])

        raise AudioExtractionError(f"Could not determine duration for: {file_path}")

    def has_audio_stream(self, file_path: str | Path) -> bool:
        """
        Check if the media file contains at least one audio stream.

        Args:
            file_path: Path to the media file.

        Returns:
            True if audio stream exists, False otherwise.
        """
        info = self.probe(file_path)
        streams = info.get("streams", [])
        return any(stream.get("codec_type") == "audio" for stream in streams)

    def extract(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Extract the audio stream from a video into a 16kHz mono WAV file.

        Args:
            video_path: Path to the source video file.
            output_path: Target path for the extracted audio. If None, saves as
                         '<video_stem>_extracted.wav' in the same directory.
            overwrite: Whether to overwrite target file if it already exists.

        Returns:
            Path object pointing to the extracted WAV file.

        Raises:
            FileNotFoundError: If source video does not exist.
            AudioExtractionError: If video lacks audio or ffmpeg fails.
        """
        src = Path(video_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Source video file not found: {src}")

        if not self.has_audio_stream(src):
            raise AudioExtractionError(f"Source file contains no audio stream: {src}")

        if output_path is None:
            dst = src.parent / f"{src.stem}_extracted.wav"
        else:
            dst = Path(output_path).resolve()

        dst.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting audio from '%s' to '%s' (SR=%d, Channels=%d)...",
                    src.name, dst.name, self.sample_rate, self.channels)

        try:
            stream = ffmpeg.input(str(src))
            stream = ffmpeg.output(
                stream.audio,
                str(dst),
                acodec="pcm_s16le",
                ac=self.channels,
                ar=self.sample_rate,
                loglevel="error",
            )
            ffmpeg.run(stream, overwrite_output=overwrite, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as err:
            stderr = err.stderr.decode("utf-8", errors="replace") if err.stderr else str(err)
            raise AudioExtractionError(f"FFmpeg extraction failed: {stderr}") from err

        if not dst.is_file() or dst.stat().st_size == 0:
            raise AudioExtractionError(f"Extracted audio file is missing or empty: {dst}")

        logger.info("Audio extraction complete: %s (%.2f KB)", dst.name, dst.stat().st_size / 1024)
        return dst
