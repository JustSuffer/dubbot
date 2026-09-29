"""
Quick Start Example for Dubbot Python API.

Demonstrates how to run the end-to-end local dubbing pipeline programmatically.
"""

from pathlib import Path
import sys

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dubbot import DubbingPipeline


def main():
    # 1. Initialize pipeline (runs 100% locally on GPU/CUDA)
    pipeline = DubbingPipeline(
        whisper_model="base",  # Options: 'tiny', 'base', 'small', 'medium', 'large-v3'
        device="auto",          # Auto-selects CUDA if available
        keep_temp=False,        # Automatically cleans up intermediate audio clips
    )

    # 2. Define input and dubbing settings
    input_video = "sample.mp4"
    target_language = "tr"      # e.g., 'tr' (Turkish), 'es' (Spanish), 'de' (German), etc.

    if not Path(input_video).is_file():
        print(f"Please provide a valid video file at '{input_video}' to run this example.")
        return

    # 3. Execute dubbing
    print(f"Starting dubbing for '{input_video}' into '{target_language}'...")
    result = pipeline.run(
        video_path=input_video,
        target_lang=target_language,
        source_lang="auto",     # Auto-detects spoken language
    )

    # 4. View results
    print("\nDubbing completed successfully!")
    print(f"  Output Video: {result['output_video']}")
    print(f"  Duration    : {result['total_duration_sec']}s")
    print(f"  Segments    : {result['total_segments']}")
    print(f"  Elapsed Time: {result['elapsed_time_sec']}s")


if __name__ == "__main__":
    main()
