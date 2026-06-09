"""Reusable OpenAI-compatible LLM client with retry and exponential backoff.

Consolidates the _call_server() pattern duplicated across 8+ agents into a
single VLLMClient class that all vLLM-backed agents can share.
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)


class VLLMClient:
    """Thin wrapper around OpenAI Chat Completions API with retry logic.

    Consolidates the identical _call_server() implementations found in:
    react.py, research.py, searcho1.py, searchr1.py, selfask.py,
    webweaver_agent.py, drtulu_agent.py, tongyi_agent.py, cpm_report.py.

    Usage::

        client = VLLMClient(
            model_url="http://127.0.0.1:6008/v1",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        )
        text = client.call(messages, stop=["</search>"], temperature=0.6)
    """

    def __init__( self, model_url: str, model_name: str, api_key: str = "EMPTY", timeout: float = 600.0, ) -> None:
        self.model_url = model_url
        self.model_name = model_name
        self._api_key = api_key
        self._timeout = timeout

    def call( self, messages: List[Dict[str, str]], *, stop: Optional[List[str]] = None, temperature: float = 0.6, top_p: float = 0.95, presence_penalty: float = 0.0, max_tokens: int = 10000, max_tries: int = 10, error_return: str = "", prepend_reasoning: bool = False, ) -> str:
        """Call the model via OpenAI Chat Completions API with retry + backoff.

        Args:
            messages:          Chat messages in OpenAI format.
            stop:              Stop sequences (None → no server-side stop).
            temperature:       Sampling temperature.
            top_p:             Nucleus sampling threshold.
            presence_penalty:  Presence penalty.
            max_tokens:        Max tokens to generate.
            max_tries:         Number of retry attempts.
            error_return:      Value returned when all retries are exhausted.
            prepend_reasoning: If True, check for a ``reasoning`` / ``reasoning_content``
                               attribute on the response and prepend it as ``<think>...</think>``.
                               Used by Tongyi, WebWeaver, DrTulu, and CPM agents.

        Returns:
            Stripped model response text, or *error_return* on total failure.
        """
        client = OpenAI(
            api_key=self._api_key,
            base_url=self.model_url,
            timeout=self._timeout,
        )

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
        }
        if stop is not None:
            create_kwargs["stop"] = stop

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(**create_kwargs)
                content = chat_response.choices[0].message.content

                if prepend_reasoning:
                    reasoning = (
                        getattr(chat_response.choices[0].message, "reasoning_content", None)
                        or getattr(chat_response.choices[0].message, "reasoning", None)
                    )
                    if reasoning:
                        content = "<think>\n" + reasoning.strip() + "\n</think>\n" + (content or "")

                if content and content.strip():
                    return content.strip()
                else:
                    logger.warning(f"Attempt {attempt + 1} received an empty response.")
            except APIError as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                # If the error is a context-length overflow (400), reduce
                # max_tokens so the request fits within the model's window
                # instead of retrying the same impossible request.
                if getattr(e, "status_code", None) == 400 and "context length" in str(e).lower():
                    current_max = create_kwargs["max_tokens"]
                    reduced = max(current_max // 2, 512)
                    if reduced < current_max:
                        logger.warning(
                            f"Reducing max_tokens from {current_max} to {reduced} "
                            "to fit within model context window."
                        )
                        create_kwargs["max_tokens"] = reduced
                    else:
                        logger.error("max_tokens already at minimum; cannot reduce further.")
                        return error_return
            except (APIConnectionError, APITimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep_time)

        return error_return
