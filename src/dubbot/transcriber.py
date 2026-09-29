"""
Speech-to-Text Transcription Module for Dubbot.

Transcribes audio with precise sentence-level and word-level timestamps
using faster-whisper (CTranslate2 backend).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class Transcriber:
    """
    Transcribes audio files into timestamped text segments using faster-whisper.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 4,
    ):
        """
        Initialize the Transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3').
            device: Computing device ('auto', 'cuda', 'cpu').
            compute_type: Quantization/precision ('auto', 'float16', 'int8', 'float32').
            cpu_threads: Number of CPU threads to use if running on CPU.
        """
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self.device, self.compute_type = self._resolve_device_and_compute(device, compute_type)
        self._model = None
        self.last_detected_language = None

    @staticmethod
    def _resolve_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
        """Resolve device and compute type based on hardware availability."""
        resolved_device = device.lower()

        if resolved_device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    resolved_device = "cuda"
                else:
                    resolved_device = "cpu"
            except ImportError:
                resolved_device = "cpu"

        resolved_compute = compute_type.lower()
        if resolved_compute == "auto":
            if resolved_device == "cuda":
                resolved_compute = "float16"
            else:
                resolved_compute = "int8"

        return resolved_device, resolved_compute

    @property
    def model(self):
        """Lazy loader for the WhisperModel to avoid heavy startup penalty."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as err:
                raise ImportError(
                    "faster-whisper is not installed. Please install it using 'pip install faster-whisper'."
                ) from err

            logger.info(
                "Loading faster-whisper model '%s' on %s (%s)...",
                self.model_size, self.device, self.compute_type
            )
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )
            except Exception as e:
                # If CUDA failed (e.g. cuDNN missing), fallback to CPU
                if self.device == "cuda":
                    logger.warning("Failed to initialize on CUDA (%s). Falling back to CPU...", e)
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=self.cpu_threads,
                    )
                else:
                    raise TranscriptionError(f"Could not load faster-whisper model: {e}") from e

        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe an audio file and extract timestamped speech segments.

        Args:
            audio_path: Path to the 16kHz WAV audio file.
            language: Optional 2-letter language code (e.g., 'en', 'es', 'tr').
                      If None, language is auto-detected.
            vad_filter: Whether to apply Voice Activity Detection to filter out silence.
            word_timestamps: Whether to extract word-level timestamps.

        Returns:
            List of dictionaries with the structure:
            [
                {
                    "id": int,
                    "start": float,
                    "end": float,
                    "text": str,
                    "words": [{"word": str, "start": float, "end": float, "probability": float}, ...]
                },
                ...
            ]

        Raises:
            FileNotFoundError: If the audio file does not exist.
            TranscriptionError: If transcription fails.
        """
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Transcribing audio: %s (language=%s, VAD=%s)", path.name, language or "auto", vad_filter)

        try:
            segments_generator, info = self.model.transcribe(
                str(path),
                language=language,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                beam_size=5,
            )

            self.last_detected_language = info.language

            logger.info(
                "Detected language: '%s' (probability: %.2f%%), duration: %.2fs",
                info.language,
                info.language_probability * 100,
                info.duration,
            )

            results: List[Dict[str, Any]] = []
            for seg in segments_generator:
                text = seg.text.strip()
                if not text:
                    continue

                segment_data: Dict[str, Any] = {
                    "id": seg.id,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": text,
                }

                if word_timestamps and seg.words:
                    segment_data["words"] = [
                        {
                            "word": w.word.strip(),
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 3),
                        }
                        for w in seg.words
                        if w.word.strip()
                    ]

                results.append(segment_data)

            logger.info("Transcription complete: extracted %d segments", len(results))
            return results

        except Exception as err:
            raise TranscriptionError(f"Transcription failed for '{path}': {err}") from err
