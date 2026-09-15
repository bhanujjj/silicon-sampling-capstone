"""Gemini inference runner (using free tier via google-generativeai).

Default model is gemini-3.5-flash-lite. gemini-2.5-flash, this project's
original default, was retired for new API keys in early 2026 -- Google's own
error message on that model now points to gemini-3.6-flash, but as of this
writing gemini-3.6-flash returns empty candidates under the short
max_output_tokens this project uses (likely consumed by internal "thinking"
tokens), so gemini-3.5-flash-lite is used instead: verified working,
verified parseable single-token answers, and lite variants typically carry a
more generous free-tier quota than the full model. Override with
GeminiInferenceRunner(model_name=...) if your key has different access.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import CachedInferenceRunner
from .prompting import build_answer_instruction, parse_answer_from_text

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google.generativeai not installed. Install via: pip install google-generativeai")


DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiInferenceRunner(CachedInferenceRunner):
    """Inference using Gemini via AI Studio free tier."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        requests_per_minute: float = 12,
    ):
        """
        requests_per_minute: client-side pacing to stay under the free-tier
        quota. Verified empirically at 15 RPM for gemini-3.5-flash-lite on a
        fresh key (Google doesn't publish this per-model; check
        https://aistudio.google.com/rate-limit for your own key/model before
        raising this). Default of 12 leaves headroom rather than pacing to
        the exact limit and tripping it on timing jitter.
        """
        super().__init__(f"gemini-{model_name}" if not model_name.startswith("gemini") else model_name)
        self._min_interval = 60.0 / requests_per_minute
        self._last_call_time = 0.0

        if not GENAI_AVAILABLE:
            raise ImportError("google.generativeai required. Install with: pip install google-generativeai")

        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Get free key from https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=self.api_key)
        self.model_id = model_name
        self.model = genai.GenerativeModel(model_name)
        # None = not yet probed, True/False = known from the first live call.
        # Free-tier quota is scarce (as low as 15 requests/minute) -- probing
        # every single call by trying logprobs then retrying without it would
        # burn 2x the quota on models that never support it. Detect once,
        # remember for the lifetime of this runner.
        self._logprobs_supported: Optional[bool] = None
        logger.info(f"✓ Initialized Gemini ({model_name})")

    def _generate(self, prompt: str, config: dict):
        """generate_content with client-side pacing and one 429 retry.

        Paces to requests_per_minute regardless of outcome. On a 429, Google
        returns a suggested retry_delay in the error body -- honoring it
        (plus a 1s buffer) turns a wasted, silently-failed prediction into a
        slightly slower but successful one, which matters when quota is as
        tight as 15 requests/minute.
        """
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        try:
            response = self.model.generate_content(prompt, generation_config=config)
        except Exception as e:
            if "429" not in str(e) and "quota" not in str(e).lower():
                self._last_call_time = time.monotonic()
                raise
            match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
            wait = int(match.group(1)) + 1 if match else 10
            logger.warning(f"Gemini rate limit hit, waiting {wait}s and retrying once...")
            time.sleep(wait)
            response = self.model.generate_content(prompt, generation_config=config)

        self._last_call_time = time.monotonic()
        return response

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
        """
        Run inference and extract real per-token logprobs when the API
        returns them. Whether this model/key combination supports
        response_logprobs is detected on the first call and cached -- after
        that, every call spends exactly one request, not two.
        Falls back to a one-hot vector on the parsed answer -- never a flat
        uniform placeholder, since that would silently zero out NLL/JSD
        signal instead of failing loudly.
        """
        full_prompt = prompt + build_answer_instruction(answer_options)
        # 100 not 10: some models (gemini-3.5-flash confirmed) spend output
        # tokens on internal "thinking" before the visible answer -- at a
        # 10-token budget they exhaust it before ever emitting the digit,
        # which surfaces as a hard failure (finish_reason=MAX_TOKENS), not a
        # graceful refusal. 100 tokens is still cheap and covers this.
        base_config = {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 100}
        logprobs_config = {
            **base_config,
            "response_logprobs": True,
            "logprobs": min(20, max(5, len(answer_options) * 2)),
        }

        response = None
        used_real_logprobs = False
        try_logprobs = self._logprobs_supported is not False  # True or None (unknown)

        if try_logprobs:
            try:
                response = self._generate(full_prompt, logprobs_config)
                used_real_logprobs = True
                self._logprobs_supported = True
            except Exception as e:
                if "logprob" not in str(e).lower():
                    logger.error(f"Gemini inference error: {e}")
                    metadata = {"refusal": "blocked" in str(e).lower(), "error": str(e), "real_logprobs": False}
                    return None, self._one_hot_fallback(None, answer_options), metadata
                self._logprobs_supported = False
                logger.info(f"{self.model_id}: response_logprobs not supported by this model/tier; "
                            "using single-call fallback confidence for the rest of this run.")

        if response is None:
            try:
                response = self._generate(full_prompt, base_config)
            except Exception as e2:
                logger.error(f"Gemini inference error: {e2}")
                metadata = {"refusal": "blocked" in str(e2).lower(), "error": str(e2), "real_logprobs": False}
                return None, self._one_hot_fallback(None, answer_options), metadata

        try:
            answer_text = (response.text or "").strip()
        except Exception as e:
            # A finished-but-empty candidate (e.g. output entirely consumed by
            # internal reasoning tokens) raises here rather than returning "".
            logger.error(f"Gemini returned no usable text: {e}")
            metadata = {"refusal": False, "error": str(e), "real_logprobs": False}
            return None, self._one_hot_fallback(None, answer_options), metadata

        pred_answer = parse_answer_from_text(answer_text, answer_options)

        logprobs = self._extract_option_probs(response, answer_options) if used_real_logprobs else None
        real_logprobs_found = logprobs is not None
        if logprobs is None:
            logprobs = self._one_hot_fallback(pred_answer, answer_options)

        metadata = {
            "refusal": False,
            "error": None,
            "raw_text": answer_text,
            "real_logprobs": real_logprobs_found,
        }
        return pred_answer, logprobs, metadata

    def _extract_option_probs(self, response, answer_options: List[str]) -> Optional[np.ndarray]:
        """Pull top-candidate logprobs for the first output token, if the API
        returned them, and map them onto answer_options by exact text match."""
        try:
            candidate = response.candidates[0]
            top = candidate.logprobs_result.top_candidates[0].candidates
        except (AttributeError, IndexError, TypeError):
            return None

        token_logprob = {c.token.strip(): c.log_probability for c in top}
        logprobs = np.full(len(answer_options), -50.0)  # ~0 probability for unseen options
        found_any = False
        for idx, opt in enumerate(answer_options):
            if opt in token_logprob:
                logprobs[idx] = token_logprob[opt]
                found_any = True
        if not found_any:
            return None

        logprobs = logprobs - logprobs.max()
        probs = np.exp(logprobs)
        return probs / probs.sum()

    @staticmethod
    def _one_hot_fallback(pred_answer: Optional[str], answer_options: List[str]) -> np.ndarray:
        """A near-one-hot vector on the parsed text answer. Used only when the
        API didn't return logprobs -- keeps downstream NLL finite (unlike a
        true one-hot, which gives -log(0)) while still reflecting that the
        model committed to one answer, unlike a flat uniform guess."""
        n = len(answer_options)
        probs = np.full(n, 0.02 / max(n - 1, 1))
        if pred_answer in answer_options:
            probs[answer_options.index(pred_answer)] = 0.98
        else:
            probs[:] = 1.0 / n
        return probs / probs.sum()
