"""
Dubbing Pipeline Orchestrator for Dubbot.

Connects and executes the complete 6-stage AI Dubbing Bot pipeline.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dubbot.aligner import AudioAligner
from dubbot.audio_extractor import AudioExtractor
from dubbot.muxer import VideoMuxer
from dubbot.synthesizer import VoiceSynthesizer
from dubbot.transcriber import Transcriber
from dubbot.translator import Translator

logger = logging.getLogger("dubbot")


class DubbingPipeline:
    """
    End-to-End AI Dubbing Bot Pipeline orchestrating:
    1. Audio Extraction
    2. Transcription & Timestamping
    3. Neural Translation
    4. Zero-shot Voice Cloning & TTS
    5. Duration Alignment & Time-Stretching
    6. Audio Assembly & Video Muxing
    """

    def __init__(
        self,
        whisper_model: str = "base",
        translation_model: str = "facebook/nllb-200-distilled-600M",
        tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "auto",
        work_dir: Optional[str | Path] = None,
        keep_temp: bool = False,
        audio_codec: str = "aac",
    ):
        """
        Initialize the DubbingPipeline with all modular components.
        """
        self.device = device
        self.keep_temp = keep_temp
        self.custom_work_dir = Path(work_dir).resolve() if work_dir else None

        logger.info("Initializing DubbingPipeline components...")
        self.extractor = AudioExtractor()
        self.transcriber = Transcriber(model_size=whisper_model, device=device)
        self.translator = Translator(model_name=translation_model, device=device)
        self.synthesizer = VoiceSynthesizer(model_name=tts_model, device=device)
        self.aligner = AudioAligner()
        self.muxer = VideoMuxer(audio_codec=audio_codec)

    def run(
        self,
        video_path: str | Path,
        target_lang: str,
        source_lang: Optional[str] = None,
        output_video_path: Optional[str | Path] = None,
        speaker_wav: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full dubbing pipeline on the input video.

        Args:
            video_path: Path to the input video.
            target_lang: Target language code (e.g. 'tr', 'es', 'fr', 'en').
            source_lang: Source language code (None for auto-detection).
            output_video_path: Optional destination video path.
            speaker_wav: Optional custom speaker reference audio. If None,
                         the extracted audio from the video is used.

        Returns:
            Dictionary containing execution metadata, segment details, and output file paths.
        """
        start_time = time.time()
        video_file = Path(video_path).resolve()
        if not video_file.is_file():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        # Output video destination
        if output_video_path is None:
            clean_tgt = target_lang.replace("-", "_")
            out_video = video_file.parent / f"{video_file.stem}_dubbed_{clean_tgt}.mp4"
        else:
            out_video = Path(output_video_path).resolve()

        # Workspace directory for intermediate files
        if self.custom_work_dir:
            work_dir = self.custom_work_dir
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup_work_dir = False
        else:
            temp_dir = tempfile.mkdtemp(prefix="dubbot_work_")
            work_dir = Path(temp_dir)
            cleanup_work_dir = not self.keep_temp

        logger.info("==================================================")
        logger.info("STARTING AI DUBBING PIPELINE")
        logger.info("Input Video : %s", video_file.name)
        logger.info("Target Lang : %s", target_lang)
        logger.info("Source Lang : %s", source_lang or "Auto-detect")
        logger.info("Output Video: %s", out_video.name)
        logger.info("Working Dir : %s", work_dir)
        logger.info("==================================================")

        try:
            # ---------------------------------------------------------
            # Stage 1: Audio Extraction
            # ---------------------------------------------------------
            logger.info("[Stage 1/6] Extracting 16kHz audio from video...")
            extracted_wav = work_dir / "original_audio.wav"
            self.extractor.extract(video_file, extracted_wav, overwrite=True)
            video_duration = self.extractor.get_duration(video_file)
            logger.info("Audio extracted. Total video duration: %.2fs", video_duration)

            # ---------------------------------------------------------
            # Stage 2: Transcription & Timestamps
            # ---------------------------------------------------------
            logger.info("[Stage 2/6] Transcribing speech with timestamps...")
            segments = self.transcriber.transcribe(
                extracted_wav,
                language=source_lang if source_lang and source_lang.lower() != "auto" else None,
                word_timestamps=False,
            )
            if not segments:
                raise ValueError("No speech segments detected in the video audio track.")

            logger.info("Transcribed %d speech segments.", len(segments))
            for seg in segments[:3]:
                logger.info("  [%0.2fs -> %0.2fs]: %s", seg["start"], seg["end"], seg["text"])
            if len(segments) > 3:
                logger.info("  ... and %d more segments", len(segments) - 3)

            # ---------------------------------------------------------
            # Stage 3: Neural Translation
            # ---------------------------------------------------------
            # Infer source language from Whisper if not explicitly provided
            if source_lang and source_lang.lower() != "auto":
                detected_src = source_lang
            else:
                detected_src = self.transcriber.last_detected_language or "en"

            logger.info("Translating %d segments from '%s' to '%s'...", len(segments), detected_src, target_lang)
            translated_segments = self.translator.translate_segments(
                segments,
                source_lang=detected_src,
                target_lang=target_lang,
            )

            logger.info("Translation completed:")
            for seg in translated_segments[:3]:
                logger.info("  [%0.2fs -> %0.2fs]: %s", seg["start"], seg["end"], seg["text"])

            # ---------------------------------------------------------
            # Stage 4: Voice Synthesis (TTS & Voice Cloning)
            # ---------------------------------------------------------
            logger.info("[Stage 4/6] Synthesizing cloned speech via Coqui XTTSv2...")
            ref_speaker = Path(speaker_wav).resolve() if speaker_wav else extracted_wav
            tts_dir = work_dir / "tts_clips"
            tts_segments = self.synthesizer.synthesize_segments(
                translated_segments,
                speaker_wav=ref_speaker,
                language=target_lang,
                output_dir=tts_dir,
            )

            # ---------------------------------------------------------
            # Stage 5: Audio Synchronization (Time-stretching)
            # ---------------------------------------------------------
            logger.info("[Stage 5/6] Aligning and time-stretching audio segments...")
            aligned_dir = work_dir / "aligned_clips"
            aligned_segments = self.aligner.align_segments(
                tts_segments,
                output_dir=aligned_dir,
            )

            # ---------------------------------------------------------
            # Stage 6: Audio Assembly & Video Muxing
            # ---------------------------------------------------------
            logger.info("[Stage 6/6] Assembling full audio track and muxing into video...")
            master_dubbed_wav = work_dir / "master_dubbed_audio.wav"
            self.muxer.assemble_audio(
                aligned_segments,
                total_duration_sec=video_duration,
                output_audio_path=master_dubbed_wav,
            )

            final_video = self.muxer.mux(
                video_path=video_file,
                audio_path=master_dubbed_wav,
                output_video_path=out_video,
                overwrite=True,
            )

            elapsed = time.time() - start_time
            logger.info("==================================================")
            logger.info("AI DUBLAJ ISLEMI BASARIYLA TAMAMLANDI!")
            logger.info("Cikti Videosu : %s", final_video)
            logger.info("Gecen Sure    : %.2f saniye", elapsed)
            logger.info("==================================================")

            # Save pipeline report metadata
            report = {
                "input_video": str(video_file),
                "output_video": str(final_video),
                "target_language": target_lang,
                "source_language": source_lang,
                "total_duration_sec": video_duration,
                "total_segments": len(aligned_segments),
                "elapsed_time_sec": round(elapsed, 2),
                "segments": aligned_segments,
            }

            report_file = out_video.parent / f"{out_video.stem}_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            return report

        finally:
            if cleanup_work_dir and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
