"""A no-API-key inference runner for validating the pipeline end to end.

MockInferenceRunner never calls a network. It returns logprobs shaped like a
plausible but *wrong* model: a small demographic-agreement bump plus noise, so
running it produces a full, realistic-looking prediction file (with genuine
errors and disagreements) to exercise caching, metrics and the comparison
report before any real API key exists.

Swap this for GeminiInferenceRunner or HFLocalInferenceRunner once a key is
available -- every other module (verbalize, prompts, eval, report) is
identical either way.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import CachedInferenceRunner

logger = logging.getLogger(__name__)


class MockInferenceRunner(CachedInferenceRunner):
    """Deterministic pseudo-model: samples a plausible logprob vector.

    Not fit to data in any way -- it exists to prove the plumbing works, not
    to produce a result worth reporting as a finding.
    """

    def __init__(self, seed: int = 42):
        super().__init__("mock-demo-model")
        self.rng = np.random.RandomState(seed)

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], np.ndarray, Dict]:
        n = len(answer_options)
        # A mild central-tendency bias (most WVS items skew toward the
        # "important"/"agree" end) plus noise -- enough to be wrong often.
        base = np.linspace(1.4, 0.6, n)
        noise = self.rng.normal(0, 0.5, n)
        logits = base + noise
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        pred_idx = int(self.rng.choice(n, p=probs))
        pred_answer = answer_options[pred_idx]

        metadata = {"refusal": False, "error": None, "raw_text": pred_answer, "mock": True}
        return pred_answer, probs, metadata
