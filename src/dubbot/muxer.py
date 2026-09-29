"""
Audio Assembly & Video Muxing Module for Dubbot.

Assembles aligned audio segments on a timeline canvas matching total video
duration and muxes the new dubbed audio track back into the video using ffmpeg.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import ffmpeg
from pydub import AudioSegment

logger = logging.getLogger(__name__)


class MuxingError(Exception):
    """Raised when audio assembly or video muxing fails."""
    pass


class VideoMuxer:
    """
    Handles final audio track assembly from timestamped segments and video muxing.
    """

    def __init__(self, sample_rate: int = 16000, audio_bitrate: str = "192k"):
        """
        Initialize the VideoMuxer.

        Args:
            sample_rate: Audio sample rate in Hz.
            audio_bitrate: Output AAC audio bitrate for the video file.
        """
        self.sample_rate = sample_rate
        self.audio_bitrate = audio_bitrate

    def assemble_audio(
        self,
        segments: List[Dict[str, Any]],
        total_duration_sec: float,
        output_audio_path: str | Path,
    ) -> Path:
        """
        Create a silent audio canvas matching the total video duration and overlay
        all aligned segments at their respective start timestamps.

        Args:
            segments: List of segment dicts with 'aligned_audio_path' and 'start'.
            total_duration_sec: Total duration of the video in seconds.
            output_audio_path: Path where the assembled WAV file will be saved.

        Returns:
            Path to the assembled WAV file.
        """
        dst = Path(output_audio_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        total_ms = int(total_duration_sec * 1000)
        logger.info("Creating blank audio canvas: %.2fs (%d ms)...", total_duration_sec, total_ms)

        # Create silent canvas
        canvas = AudioSegment.silent(duration=total_ms, frame_rate=self.sample_rate)

        # Overlay each segment
        for i, seg in enumerate(segments):
            clip_path = seg.get("aligned_audio_path")
            if not clip_path or not Path(clip_path).is_file():
                logger.warning("Segment %d has no valid aligned audio. Skipping overlay.", i)
                continue

            start_ms = max(0, int(seg["start"] * 1000))
            clip = AudioSegment.from_file(clip_path)

            logger.debug("Overlaying segment %d at %d ms (length %d ms)", i, start_ms, len(clip))
            canvas = canvas.overlay(clip, position=start_ms)

        # Export assembled master track
        canvas.export(str(dst), format="wav")
        logger.info("Assembled audio track saved to: %s (%.2f KB)", dst.name, dst.stat().st_size / 1024)
        return dst

    def mux(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_video_path: str | Path,
        overwrite: bool = True,
    ) -> Path:
        """
        Strip the original audio from the video and merge the new dubbed audio track,
        preserving original video codec and quality without re-encoding.

        Args:
            video_path: Path to the original input video.
            audio_path: Path to the assembled dubbed audio file.
            output_video_path: Path to the final output dubbed video.
            overwrite: Whether to overwrite existing destination video.

        Returns:
            Path to the generated video file.
        """
        src_video = Path(video_path).resolve()
        src_audio = Path(audio_path).resolve()
        dst_video = Path(output_video_path).resolve()

        if not src_video.is_file():
            raise FileNotFoundError(f"Input video not found: {src_video}")
        if not src_audio.is_file():
            raise FileNotFoundError(f"Dubbed audio not found: {src_audio}")

        dst_video.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Muxing video '%s' with dubbed audio '%s' -> '%s'...",
                    src_video.name, src_audio.name, dst_video.name)

        try:
            video_in = ffmpeg.input(str(src_video))
            audio_in = ffmpeg.input(str(src_audio))

            out = ffmpeg.output(
                video_in["v"],
                audio_in["a"],
                str(dst_video),
                vcodec="copy",          # Preserve original video streams without quality loss
                acodec="aac",           # Encode audio track to high-compatibility AAC
                audio_bitrate=self.audio_bitrate,
                loglevel="error",
            )

            ffmpeg.run(out, overwrite_output=overwrite, capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as err:
            stderr = err.stderr.decode("utf-8", errors="replace") if err.stderr else str(err)
            raise MuxingError(f"FFmpeg muxing failed: {stderr}") from err

        if not dst_video.is_file() or dst_video.stat().st_size == 0:
            raise MuxingError(f"Output video was not generated or is empty: {dst_video}")

        logger.info("Successfully generated dubbed video: %s (%.2f MB)",
                    dst_video.name, dst_video.stat().st_size / (1024 * 1024))
        return dst_video
