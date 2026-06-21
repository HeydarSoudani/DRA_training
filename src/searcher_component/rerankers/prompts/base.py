"""Base prompt template for listwise reranking."""

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PromptTemplate:
    """Prompt template for listwise LLM-based reranking.

    Supports both single-turn (RankZephyr, Qwen3) and multi-turn (RankGPT)
    conversation formats via the ``multi_turn`` flag.

    Format variables available in *prefix* and *suffix*: ``{num}``, ``{query}``.
    Format variables available in *body*: ``{rank}``, ``{candidate}``.
    Multi-turn acknowledgements use ``{rank}`` in *body_ack*.
    """

    name: str
    system_message: str
    prefix: str
    body: str
    suffix: str

    # Multi-turn support (RankGPT style)
    multi_turn: bool = False
    prefix_ack: str = ""
    body_ack: str = ""  # format var: {rank}

    # Output parsing
    output_validation_regex: str = r"\[\d+\]( > \[\d+\])*"
    output_extraction_regex: str = r"\[(\d+)\]"

    def build_messages( self, query: str, candidate_texts: List[str], ) -> List[Dict[str, str]]:
        """Build chat messages for the reranking prompt.

        Returns a list of ``{"role": ..., "content": ...}`` dicts ready for
        an OpenAI-compatible chat completions API.
        """
        num = len(candidate_texts)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_message},
        ]

        if self.multi_turn:
            # Multi-turn: each passage gets its own user/assistant turn pair
            messages.append(
                {"role": "user", "content": self.prefix.format(num=num, query=query)}
            )
            messages.append({"role": "assistant", "content": self.prefix_ack})
            for i, text in enumerate(candidate_texts, start=1):
                messages.append(
                    {"role": "user", "content": self.body.format(rank=i, candidate=text)}
                )
                messages.append(
                    {"role": "assistant", "content": self.body_ack.format(rank=i)}
                )
            messages.append(
                {"role": "user", "content": self.suffix.format(num=num, query=query)}
            )
        else:
            # Single-turn: concatenate everything into one user message
            prefix = self.prefix.format(num=num, query=query)
            body_parts = []
            for i, text in enumerate(candidate_texts, start=1):
                body_parts.append(self.body.format(rank=i, candidate=text))
            suffix = self.suffix.format(num=num, query=query)
            user_content = prefix + "".join(body_parts) + suffix
            messages.append({"role": "user", "content": user_content})

        return messages

    def parse_ranking(self, output: str, num_docs: int) -> List[int]:
        """Parse model output into a 0-based index permutation.

        Extracts bracket-number patterns (e.g. ``[2] > [1] > [3]``),
        converts to 0-based indices, deduplicates, and appends any missing
        indices at the end to guarantee a complete permutation.
        """
        matches = re.findall(self.output_extraction_regex, output)
        seen: set[int] = set()
        ranked: List[int] = []
        for m in matches:
            idx = int(m) - 1  # convert 1-based to 0-based
            if 0 <= idx < num_docs and idx not in seen:
                ranked.append(idx)
                seen.add(idx)
        # Append any missing indices (preserves original order for unranked docs)
        for i in range(num_docs):
            if i not in seen:
                ranked.append(i)
        return ranked
