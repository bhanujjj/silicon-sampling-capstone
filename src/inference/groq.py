"""Groq inference runner (free tier, OpenAI-compatible chat completions API).

Groq hosts genuinely larger open-weight models on its free tier than any
Gemini free-tier model tested in this project -- llama-3.3-70b-versatile is a
70B-parameter model, versus the "Flash-Lite" tier Gemini was restricted to.
Free tier verified (2026): 30 requests/minute, 1,000 requests/day for this
model (https://tokenmix.ai/blog/groq-free-tier-limits-2026) -- both limits
are looser than every Gemini free-tier bucket this project hit.

No official Python SDK dependency: Groq's endpoint is OpenAI-compatible, so
plain `requests` against https://api.groq.com/openai/v1/chat/completions is
enough and avoids adding a new dependency for one provider.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from .base import CachedInferenceRunner
from .prompting import build_answer_instruction, parse_answer_from_text

logger = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile (this module's original target) is no longer on
# Groq's catalog as of this writing -- model availability there changes
# often. openai/gpt-oss-120b is OpenAI's own open-weight 120B model, hosted
# free on Groq: larger than the retired Llama option, and literally
# OpenAI-built architecture without a paid OpenAI API key. Verify current
# availability with GET https://api.groq.com/openai/v1/models before relying
# on this default long-term.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqInferenceRunner(CachedInferenceRunner):
    """Inference using Groq's free tier. No real logprobs available on chat
    completions here either, so this mirrors Gemini's one-hot-fallback
    confidence rather than pretending to have a real probability
    distribution."""

    # Empirically measured reasoning-token cost per effort level (see
    # infer_single's max_tokens comment) drives how many requests/minute keep
    # total token throughput under the observed 8000 TPM cap.
    _DEFAULT_RPM_BY_EFFORT = {"low": 18, "medium": 10, "high": 3}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        requests_per_minute: Optional[float] = None,
        reasoning_effort: str = "low",
    ):
        """
        requests_per_minute: defaults per reasoning_effort level (see
        _DEFAULT_RPM_BY_EFFORT) if not given explicitly. Verified via live
        response headers for openai/gpt-oss-120b on this key: 1000
        requests/(reset window), 8000 tokens/minute -- the token cap is the
        binding constraint for this project's longer P2 prompts, and "high"
        reasoning burns ~35x the tokens "low" does per call, so it needs a
        much slower pace to stay under the same TPM ceiling.

        reasoning_effort: "low"/"medium"/"high". Cache filename includes this
        so a "low" vs "high" comparison never collides in results/cache/.
        """
        self.reasoning_effort = reasoning_effort
        if requests_per_minute is None:
            requests_per_minute = self._DEFAULT_RPM_BY_EFFORT[reasoning_effort]
        # Groq model ids can contain "/" (e.g. "openai/gpt-oss-120b"); the base
        # class uses model_name as a literal cache filename component, so a
        # raw "/" would try to write into a nonexistent subdirectory.
        safe_name = model_name.replace("/", "_")
        super().__init__(f"groq-{safe_name}-{reasoning_effort}")
        self._min_interval = 60.0 / requests_per_minute
        self._last_call_time = 0.0

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get a free key from https://console.groq.com/keys"
            )

        self.model_id = model_name
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )
        logger.info(f"✓ Initialized Groq ({model_name})")

    def _generate(self, payload: dict) -> dict:
        """POST with client-side pacing and one retry on 429, honoring
        Groq's Retry-After header when present."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        response = self.session.post(API_URL, json=payload, timeout=60)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) + 1 if retry_after else 10
            logger.warning(f"Groq rate limit hit, waiting {wait:.0f}s and retrying once...")
            time.sleep(wait)
            response = self.session.post(API_URL, json=payload, timeout=60)

        self._last_call_time = time.monotonic()
        response.raise_for_status()
        return response.json()

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
        full_prompt = prompt + build_answer_instruction(answer_options)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
            # gpt-oss models reason internally before answering (visible via a
            # separate "reasoning" field in the response) -- at too low a
            # token budget the response is truncated mid-thought with empty
            # visible content (finish_reason="length"), never reaching the
            # answer. Measured reasoning-token usage on this task: "low" ~11,
            # "high" ~382 -- so the budget must scale with the effort level,
            # not use one fixed number for all three.
            "max_tokens": {"low": 150, "medium": 700, "high": 3000}[self.reasoning_effort],
            "reasoning_effort": self.reasoning_effort,
        }

        try:
            data = self._generate(payload)
        except Exception as e:
            logger.error(f"Groq inference error: {e}")
            metadata = {"refusal": False, "error": str(e), "real_logprobs": False}
            return None, self._one_hot_fallback(None, answer_options), metadata

        try:
            answer_text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Groq returned no usable text: {e} -- raw: {data}")
            metadata = {"refusal": False, "error": f"unparseable response: {e}", "real_logprobs": False}
            return None, self._one_hot_fallback(None, answer_options), metadata

        pred_answer = parse_answer_from_text(answer_text, answer_options)
        metadata = {
            "refusal": False,
            "error": None,
            "raw_text": answer_text,
            "real_logprobs": False,  # chat completions here never returns logprobs
        }
        return pred_answer, self._one_hot_fallback(pred_answer, answer_options), metadata

    @staticmethod
    def _one_hot_fallback(pred_answer: Optional[str], answer_options: List[str]) -> np.ndarray:
        """Same fallback-confidence convention as GeminiInferenceRunner --
        keeps NLL finite while still reflecting the model committed to one
        answer, unlike a flat uniform guess. See gemini.py for the rationale."""
        n = len(answer_options)
        probs = np.full(n, 0.02 / max(n - 1, 1))
        if pred_answer in answer_options:
            probs[answer_options.index(pred_answer)] = 0.98
        else:
            probs[:] = 1.0 / n
        return probs / probs.sum()
