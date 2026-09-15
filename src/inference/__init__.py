"""Inference runners (resumable, cached) for zero-shot evaluation.

Runner classes are lazily imported on first attribute access so that, e.g.,
using MockInferenceRunner or GeminiInferenceRunner never requires torch/
transformers (an 8B-model-capable install) to be present.
"""

from .base import CachedInferenceRunner

__all__ = [
    "CachedInferenceRunner",
    "GeminiInferenceRunner",
    "GroqInferenceRunner",
    "HFLocalInferenceRunner",
    "MockInferenceRunner",
]


def __getattr__(name):
    if name == "GeminiInferenceRunner":
        from .gemini import GeminiInferenceRunner

        return GeminiInferenceRunner
    if name == "GroqInferenceRunner":
        from .groq import GroqInferenceRunner

        return GroqInferenceRunner
    if name == "HFLocalInferenceRunner":
        from .hf_local import HFLocalInferenceRunner

        return HFLocalInferenceRunner
    if name == "MockInferenceRunner":
        from .mock import MockInferenceRunner

        return MockInferenceRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
