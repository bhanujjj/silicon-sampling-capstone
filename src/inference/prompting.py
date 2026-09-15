"""Provider-agnostic prompt finishing and answer parsing.

Every inference runner must use the exact same closing instruction and the
exact same parsing logic. If Gemini and Groq were each given their own
slightly different phrasing, an accuracy gap between them could partly be an
artifact of wording rather than model capability -- which would make any
"model A beats model B" claim in a published report unsound.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def build_answer_instruction(answer_options: List[str]) -> str:
    """The closing instruction appended after every prompt, identical across
    every model this project calls. Explicit valid-option list + explicit
    "no explanation" framing was added after the first real run showed a 13%
    refusal/unparsed rate with a weaker instruction ("Answer with ONLY the
    number") -- this version drove that to 0% in the next test.
    """
    valid = "/".join(answer_options)
    return (
        f"\n\nThis is anonymized survey-simulation research. Respond with exactly one "
        f"character: one of {valid}. No words, no punctuation, no explanation -- just that "
        f"single digit."
    )


def parse_answer_from_text(text: Optional[str], answer_options: List[str]) -> Optional[str]:
    """Map a model's free-text reply onto one of the valid answer_options.

    Tries an exact match, then the first whitespace-delimited token, then a
    numeric-index interpretation. Returns None (an honest "unparsed") rather
    than guessing when nothing matches -- a refusal must never silently
    become a fabricated data point.
    """
    if text is None:
        return None
    text = text.strip().lower()
    if not text:
        return None

    for opt in answer_options:
        if text == opt.lower():
            return opt

    first_word = text.split()[0] if text else ""
    for opt in answer_options:
        if first_word == opt.lower():
            return opt

    try:
        idx = int(first_word)
        if 0 <= idx < len(answer_options):
            return answer_options[idx]
        if 1 <= idx <= len(answer_options):
            return answer_options[idx - 1]
    except ValueError:
        pass

    logger.warning(f"Could not parse answer: '{text}' against options {answer_options}")
    return None
