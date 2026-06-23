"""Self-hosted vLLM generator and low-level transport.

Two ways to talk to an OpenAI-compatible vLLM server (``vllm serve``):

  * ``VLLMGenerator`` — a full ``BaseGenerator`` (structured output, tools, stop
    patterns) backed by a ``LiteLLMClient``. Use this for local/finetuned models
    in the generator pipeline.
  * ``VLLMClient`` — a thin OpenAI Chat Completions wrapper with retry/backoff,
    for the low-level single-call pattern used by some self-managed agents.
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, AzureOpenAI, APIError, APIConnectionError, APITimeoutError

from .base import LiteLLMBackedGenerator
from .litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)

DEFAULT_VLLM_API_BASE = "http://127.0.0.1:6008/v1"


def get_openai_client(
    api_url: str,
    api_key: str = "EMPTY",
    *,
    api_version: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Construct a raw OpenAI-compatible client for a self-hosted/vLLM endpoint.

    Returns the OpenAI SDK client directly — ``AzureOpenAI`` when *api_url* is an
    Azure endpoint, otherwise ``OpenAI`` — so callers that need low-level features
    deliberately not exposed by ``VLLMClient`` / ``VLLMGenerator`` (token
    ``logprobs``, the text ``completions`` endpoint, Azure routing) can use them.
    Keeps raw OpenAI client construction inside ``reasoner_component`` (the single
    LLM home) instead of scattering ``from openai import OpenAI`` across callers.
    """
    extra: Dict[str, Any] = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if ".openai.azure.com" in api_url:
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=api_url,
            api_version=api_version or "2025-01-01-preview",
            **extra,
        )
    return OpenAI(api_key=api_key, base_url=api_url, **extra)


class VLLMGenerator(LiteLLMBackedGenerator):
    """LLM generator backed by an OpenAI-compatible vLLM server.

    Args:
        model_name: The model served by vLLM (``hf/`` / ``vllm/`` prefixes are stripped).
        api_base:   Server URL. Defaults to ``$VLLM_API_BASE`` or ``DEFAULT_VLLM_API_BASE``.
        api_key:    Usually ignored by vLLM; defaults to ``"EMPTY"``.
        litellm_prefix: LiteLLM provider prefix for OpenAI-compatible servers
                        (``"hosted_vllm"`` or ``"openai"``).
    """

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        metadata: Optional[Dict[str, str]] = None,
        litellm_prefix: str = "hosted_vllm",
        **kwargs,
    ):
        bare = model_name.removeprefix("vllm/").removeprefix("hf/")
        api_base = api_base or os.getenv("VLLM_API_BASE", DEFAULT_VLLM_API_BASE)
        config: Dict[str, Any] = {
            "model": f"{litellm_prefix}/{bare}",
            "api_base": api_base,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        self._client = LiteLLMClient(config, metadata=metadata)


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
        # Build the client once; reusing it keeps the underlying HTTP connection
        # pool warm instead of reconnecting on every call().
        self._client = OpenAI(api_key=api_key, base_url=model_url, timeout=timeout)
        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

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
        client = self._client

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
                self.token_meter.record_usage(getattr(chat_response, "usage", None))
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
