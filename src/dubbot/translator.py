"""
Neural Translation Module for Dubbot.

Translates text between multiple languages using HuggingFace's
facebook/nllb-200-distilled-600M model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


# Mapping from standard 2-letter ISO 639-1 language codes to NLLB-200 language codes
ISO_TO_NLLB: Dict[str, str] = {
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "cs": "ces_Latn",
    "uk": "ukr_Cyrl",
    "el": "ell_Grek",
    "sv": "swe_Latn",
    "fa": "pes_Arab",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "he": "heb_Hebr",
}


class Translator:
    """
    Translates transcript segments from a source language to a target language.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the Translator.

        Args:
            model_name: HuggingFace model repo ID (default: facebook/nllb-200-distilled-600M).
            device: 'auto', 'cuda', or 'cpu'.
            torch_dtype: Optional torch dtype (e.g. torch.float16 for GPU).
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self._tokenizer = None
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device.lower()

    @staticmethod
    def resolve_language_code(lang: str) -> str:
        """
        Convert short language codes (e.g. 'en', 'tr') to NLLB format (e.g. 'eng_Latn').
        If already in NLLB format or unmapped, returns the input as is.
        """
        cleaned = lang.strip().lower()
        if cleaned in ISO_TO_NLLB:
            return ISO_TO_NLLB[cleaned]
        # Check case-insensitive match for values
        for val in ISO_TO_NLLB.values():
            if val.lower() == cleaned:
                return val
        return lang

    def _load_model(self):
        """Lazy load tokenizer and model."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading translation model '%s' on %s (%s)...",
                        self.model_name, self.device, self.torch_dtype)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                ).to(self.device)
                self._model.eval()
            except Exception as err:
                raise TranslationError(f"Failed to load translation model '{self.model_name}': {err}") from err

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 400,
    ) -> List[str]:
        """
        Translate a list of strings from source_lang to target_lang.

        Args:
            texts: List of text strings to translate.
            source_lang: Source language code (e.g. 'en' or 'eng_Latn').
            target_lang: Target language code (e.g. 'tr' or 'tur_Latn').
            max_length: Max generation tokens.

        Returns:
            List of translated strings.
        """
        if not texts:
            return []

        self._load_model()

        src_nllb = self.resolve_language_code(source_lang)
        tgt_nllb = self.resolve_language_code(target_lang)

        logger.info("Translating %d texts from %s to %s...", len(texts), src_nllb, tgt_nllb)

        try:
            self._tokenizer.src_lang = src_nllb
            encoded = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_nllb)

            with torch.no_grad():
                generated_tokens = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                )

            translated_texts = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return [t.strip() for t in translated_texts]
        except Exception as err:
            raise TranslationError(f"Batch translation failed: {err}") from err

    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> List[Dict[str, Any]]:
        """
        Translate timestamped segments and preserve timing metadata.

        Args:
            segments: List of segment dictionaries containing 'start', 'end', and 'text'.
            source_lang: Source language code (e.g. 'en').
            target_lang: Target language code (e.g. 'tr').

        Returns:
            List of segment dictionaries with added 'source_text' and updated 'text'.
        """
        if not segments:
            return []

        texts_to_translate = [seg["text"] for seg in segments]
        translations = self.translate_batch(texts_to_translate, source_lang, target_lang)

        translated_segments: List[Dict[str, Any]] = []
        for seg, translated in zip(segments, translations):
            new_seg = dict(seg)
            new_seg["source_text"] = seg["text"]
            new_seg["text"] = translated
            translated_segments.append(new_seg)

        return translated_segments
