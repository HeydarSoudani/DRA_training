"""Prompt templates for reranking models."""

from .base import PromptTemplate
from .qwen3 import QWEN3_TEMPLATE
from .rank1 import RANK1_INCOMPLETE_SCORE, RANK1_RELEVANCE_PROMPT
from .rank_r1 import RANK_R1_ANSWER_PATTERN, RANK_R1_SYSTEM_PROMPT, RANK_R1_USER_PROMPT
from .rankgpt import RANKGPT_TEMPLATE
from .rankzephyr import RANKZEPHYR_TEMPLATE

TEMPLATES = {
    "rankgpt": RANKGPT_TEMPLATE,
    "rankzephyr": RANKZEPHYR_TEMPLATE,
    "qwen3": QWEN3_TEMPLATE,
}


def get_template(name: str) -> PromptTemplate:
    """Look up a prompt template by name.

    Raises ``ValueError`` if the name is not registered.
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {name!r}. Available: {sorted(TEMPLATES)}"
        )
    return TEMPLATES[name]
