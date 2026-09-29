"""
Dubbot - 100% Local AI Dubbing Bot Pipeline
"""

__version__ = "0.1.0"

from dubbot.audio_extractor import AudioExtractor, AudioExtractionError
from dubbot.transcriber import Transcriber, TranscriptionError
from dubbot.translator import Translator, TranslationError
from dubbot.synthesizer import VoiceSynthesizer, SynthesisError
from dubbot.aligner import AudioAligner, AlignmentError
from dubbot.muxer import VideoMuxer, MuxingError
from dubbot.pipeline import DubbingPipeline

__all__ = [
    "AudioExtractor",
    "AudioExtractionError",
    "Transcriber",
    "TranscriptionError",
    "Translator",
    "TranslationError",
    "VoiceSynthesizer",
    "SynthesisError",
    "AudioAligner",
    "AlignmentError",
    "VideoMuxer",
    "MuxingError",
    "DubbingPipeline",
]
