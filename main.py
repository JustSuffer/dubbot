#!/usr/bin/env python3
"""
Dubbot - 100% Local AI Dubbing Bot CLI

Takes a video in a source language and generates a dubbed version in a target
language with cloned voice, pitch-preserving timestamp alignment, and zero cloud APIs.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dubbot.pipeline import DubbingPipeline


def setup_logging(verbose: bool = False):
    """Configure console logging format and level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dubbot: 100% Local AI Video Dubbing with Voice Cloning & Timestamp Alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Dubbot v0.1.0",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported target languages for zero-shot voice cloning",
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Path to the input video file (mp4, mkv, mov, etc.)",
    )
    parser.add_argument(
        "-t", "--target_lang",
        type=str,
        default=None,
        help="Target language code (e.g. 'tr', 'es', 'fr', 'de', 'en', 'it', 'pt', 'ru', 'ja', 'zh')",
    )
    parser.add_argument(
        "-s", "--source_lang",
        type=str,
        default="auto",
        help="Source language code (default: 'auto' for automatic speech language detection)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the dubbed output video",
    )
    parser.add_argument(
        "--speaker_wav",
        type=str,
        default=None,
        help="Optional custom reference WAV for voice cloning (defaults to extracted video audio)",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Faster-Whisper model size to use for transcription",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware accelerator to use",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Custom directory for intermediate audio clips and working files",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep intermediate audio slices and working directory after completion",
    )
    parser.add_argument(
        "--audio_codec",
        type=str,
        default="aac",
        choices=["aac", "mp3"],
        help="Audio codec for output video ('aac' for universal standard, 'mp3' for VS Code preview)",
    )
    parser.add_argument(
        "-p", "--play",
        action="store_true",
        help="Automatically open dubbed video in default media player when finished",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logs",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.list_languages:
        from dubbot.synthesizer import XTTS_LANGUAGES
        print("\nDubbot - Supported Target Languages (XTTSv2 Zero-Shot Voice Cloning):")
        print("-------------------------------------------------------------------")
        lang_names = {
            "ar": "Arapça / Arabic",
            "cs": "Çekçe / Czech",
            "de": "Almanca / German",
            "en": "İngilizce / English",
            "es": "İspanyolca / Spanish",
            "fr": "Fransızca / French",
            "hi": "Hintçe / Hindi",
            "hu": "Macarca / Hungarian",
            "it": "İtalyanca / Italian",
            "ja": "Japonca / Japanese",
            "ko": "Korece / Korean",
            "nl": "Flemenkçe / Dutch",
            "pl": "Lehçe / Polish",
            "pt": "Portekizce / Portuguese",
            "ru": "Rusça / Russian",
            "tr": "Türkçe / Turkish",
            "zh-cn": "Çince / Chinese (Simplified)",
        }
        for code in sorted(XTTS_LANGUAGES):
            name = lang_names.get(code, code.upper())
            print(f"  {code:<8} : {name}")
        print("\nKullanim / Examples:")
        print("  Tek dil   : python main.py -v video.mp4 -t japonca")
        print("  Coklu dil : python main.py -v video.mp4 -t 'japonca, ingilizce, arapca, cince'\n")
        sys.exit(0)


    if not args.video:
        logging.error("Missing required argument: --video / -v")
        sys.exit(1)
    if not args.target_lang:
        logging.error("Missing required argument: --target_lang / -t")
        sys.exit(1)

    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        logging.error("Video file not found: %s", video_path)
        sys.exit(1)
        
    try:
        pipeline = DubbingPipeline(
            whisper_model=args.whisper_model,
            device=args.device,
            work_dir=args.work_dir,
            keep_temp=args.keep_temp,
            audio_codec=args.audio_codec,
        )
        
        target_langs = [t.strip() for t in args.target_lang.split(",") if t.strip()]
        results = []

        for tgt in target_langs:
            out_path = args.output
            if len(target_langs) > 1 and out_path:
                p = Path(out_path)
                out_path = p.parent / f"{p.stem}_{tgt}{p.suffix}"

            res = pipeline.run(
                video_path=video_path,
                target_lang=tgt,
                source_lang=None if args.source_lang == "auto" else args.source_lang,
                output_video_path=out_path,
                speaker_wav=args.speaker_wav,
            )
            results.append(res)

        print("\n==========================================")
        print("ALL DUBBING TASKS FINISHED SUCCESSFULLY!")
        print("==========================================")
        for r in results:
            print(f"[{r['target_language'].upper()}] -> {r['output_video']} ({r['elapsed_time_sec']}s)")

        if args.play and results:
            out_file = str(results[0]["output_video"])
            import os
            import subprocess
            if sys.platform.startswith("win"):
                os.startfile(out_file)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", out_file])
            else:
                subprocess.Popen(["xdg-open", out_file])

        
    except KeyboardInterrupt:
        logging.warning("Dubbing process interrupted by user.")
        sys.exit(130)
    except Exception as err:
        logging.exception("Dubbing pipeline failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
