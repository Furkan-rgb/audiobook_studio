"""Text-to-speech synthesis backends."""

from .qwen import (
    generate_chunk,
    load_qwen_model,
    verify_supported_voice,
    verify_tts_dependencies,
)

__all__ = [
    "generate_chunk",
    "load_qwen_model",
    "verify_supported_voice",
    "verify_tts_dependencies",
]
