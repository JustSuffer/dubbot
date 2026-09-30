"""
Voice Cloning & Text-To-Speech Module for Dubbot.

Performs zero-shot cross-lingual voice cloning and speech synthesis
using Coqui XTTSv2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Automatically accept Coqui license for non-interactive execution
os.environ["COQUI_TOS_AGREED"] = "1"

# Supported XTTSv2 languages
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"
}

LANG_MAP_XTTS = {
    # Chinese aliases
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-hans": "zh-cn",
    "zh-hant": "zh-cn",
    "zh-tw": "zh-cn",
    "çince": "zh-cn",
    "cince": "zh-cn",
    "chinese": "zh-cn",
    "mandarin": "zh-cn",
    "mandarince": "zh-cn",

    # Japanese aliases
    "ja": "ja",
    "japonca": "ja",
    "japanese": "ja",

    # Arabic aliases
    "ar": "ar",
    "arapça": "ar",
    "arapca": "ar",
    "arabic": "ar",

    # English aliases
    "en": "en",
    "ingilizce": "en",
    "english": "en",

    # Turkish aliases
    "tr": "tr",
    "türkçe": "tr",
    "turkce": "tr",
    "turkish": "tr",

    # Spanish aliases
    "es": "es",
    "ispanyolca": "es",
    "spanish": "es",

    # German aliases
    "de": "de",
    "almanca": "de",
    "german": "de",

    # French aliases
    "fr": "fr",
    "fransızca": "fr",
    "fransizca": "fr",
    "french": "fr",

    # Russian aliases
    "ru": "ru",
    "rusça": "ru",
    "rusca": "ru",
    "russian": "ru",

    # Italian aliases
    "it": "it",
    "italyanca": "it",
    "italian": "it",

    # Korean aliases
    "ko": "ko",
    "korece": "ko",
    "korean": "ko",

    # Portuguese aliases
    "pt": "pt",
    "portekizce": "pt",
    "portuguese": "pt",

    # Hindi aliases
    "hi": "hi",
    "hintçe": "hi",
    "hintce": "hi",
    "hindi": "hi",

    # Dutch aliases
    "nl": "nl",
    "flemenkçe": "nl",
    "dutch": "nl",

    # Polish aliases
    "pl": "pl",
    "lehçe": "pl",
    "polish": "pl",

    # Czech aliases
    "cs": "cs",
    "çekçe": "cs",
    "czech": "cs",

    # Hungarian aliases
    "hu": "hu",
    "macarca": "hu",
    "hungarian": "hu",
}



class SynthesisError(Exception):
    """Raised when voice synthesis fails."""
    pass


class VoiceSynthesizer:
    """
    Synthesizes speech in target languages while preserving the speaker's original voice
    via zero-shot voice cloning (Coqui XTTSv2).
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
    ):
        """
        Initialize the VoiceSynthesizer.

        Args:
            model_name: Coqui model identifier (default: XTTSv2).
            device: 'auto', 'cuda', or 'cpu'.
        """
        self.model_name = model_name
        self.use_gpu = self._resolve_gpu(device)
        self._tts = None

    @staticmethod
    def _resolve_gpu(device: str) -> bool:
        if device == "auto":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return device.lower() == "cuda"

    @staticmethod
    def resolve_language(lang: str) -> str:
        """Normalize language code for XTTSv2."""
        cleaned = lang.strip().lower()
        if cleaned in LANG_MAP_XTTS:
            return LANG_MAP_XTTS[cleaned]
        if cleaned in XTTS_LANGUAGES:
            return cleaned
        # Try two-letter prefix
        prefix = cleaned[:2]
        if prefix in XTTS_LANGUAGES:
            return prefix
        raise ValueError(
            f"Unsupported target language for XTTSv2: '{lang}'. "
            f"Supported languages: {', '.join(sorted(XTTS_LANGUAGES))}"
        )

    @property
    def tts(self):
        """Lazy load the Coqui TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as err:
                raise ImportError(
                    "TTS is not installed or accessible. Please ensure coqui-tts is installed."
                ) from err

            logger.info("Loading XTTSv2 model (use_gpu=%s)...", self.use_gpu)
            try:
                self._tts = TTS(model_name=self.model_name, gpu=self.use_gpu)
            except Exception as err:
                raise SynthesisError(f"Failed to load TTS model '{self.model_name}': {err}") from err

        return self._tts

    def synthesize(
        self,
        text: str,
        speaker_wav: str | Path,
        language: str,
        output_path: str | Path,
    ) -> Path:
        """
        Synthesize speech from text cloned from the reference speaker.

        Args:
            text: Text to synthesize.
            speaker_wav: Path to clean reference audio WAV for cloning.
            language: Target language code (e.g. 'tr', 'en', 'es').
            output_path: Path to save the synthesized WAV.

        Returns:
            Path to the saved WAV file.
        """
        ref_path = Path(speaker_wav).resolve()
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference speaker WAV not found: {ref_path}")

        dst = Path(output_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        xtts_lang = self.resolve_language(language)

        logger.info("Synthesizing speech in '%s' (%d chars) -> %s", xtts_lang, len(text), dst.name)

        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(ref_path),
                language=xtts_lang,
                file_path=str(dst),
            )
        except Exception as err:
            raise SynthesisError(f"Speech synthesis failed for '{text[:30]}...': {err}") from err

        if not dst.is_file() or dst.stat().st_size == 0:
            raise SynthesisError(f"Generated audio file missing or empty: {dst}")

        return dst

    def synthesize_segments(
        self,
        segments: List[Dict[str, Any]],
        speaker_wav: str | Path,
        language: str,
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Synthesize speech for a sequence of translated segments.

        Args:
            segments: List of segment dicts with 'text', 'start', 'end'.
            speaker_wav: Path to reference speaker audio file.
            language: Target language code.
            output_dir: Directory where segment WAV files will be saved.

        Returns:
            List of updated segment dicts containing 'audio_path' to generated speech.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            seg_wav_path = out_dir / f"tts_segment_{i:04d}.wav"
            self.synthesize(
                text=seg_text,
                speaker_wav=speaker_wav,
                language=language,
                output_path=seg_wav_path,
            )

            updated_seg = dict(seg)
            updated_seg["tts_audio_path"] = str(seg_wav_path)
            results.append(updated_seg)

        return results
