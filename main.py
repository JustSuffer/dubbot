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
        "-v", "--video",
        type=str,
        required=True,
        help="Path to the input video file (mp4, mkv, mov, etc.)",
    )
    parser.add_argument(
        "-t", "--target_lang",
        type=str,
        required=True,
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
        "--verbose",
        action="store_true",
        help="Enable detailed debug logs",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    
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
        )
        
        result = pipeline.run(
            video_path=video_path,
            target_lang=args.target_lang,
            source_lang=None if args.source_lang == "auto" else args.source_lang,
            output_video_path=args.output,
            speaker_wav=args.speaker_wav,
        )
        
        print("\nDubbing finished successfully!")
        print(f"Output Video : {result['output_video']}")
        print(f"Elapsed Time : {result['elapsed_time_sec']}s")
        print(f"Segments     : {result['total_segments']}")
        
    except KeyboardInterrupt:
        logging.warning("Dubbing process interrupted by user.")
        sys.exit(130)
    except Exception as err:
        logging.exception("Dubbing pipeline failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
