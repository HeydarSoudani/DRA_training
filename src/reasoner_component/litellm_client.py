"""LiteLLM transport client (vendor-agnostic LLM API integration).

Low-level transport only. Provider/model routing lives in ``providers.py``.
Moved here from ``utils.llm_client`` as part of consolidating all LLM/generation
logic under ``reasoner_component``.
"""

import asyncio
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Type, TypeVar, Callable

import litellm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


_thread_local = threading.local()
def _get_or_create_thread_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the current thread."""
    if not hasattr(_thread_local, "loop") or _thread_local.loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    return _thread_local.loop


def _close_loop_with_litellm_cleanup(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel pending tasks, close litellm's async HTTP clients, then close the loop.

    Must be called from the same thread that owns the loop, before the loop
    is abandoned.  Skipping this causes aiohttp connectors to be bound to a
    stale loop, which raises "Future attached to a different loop" on the next
    call and breaks generation.
    """
    # Close litellm's async HTTP clients first (aiohttp connectors, etc.)
    try:
        from litellm import close_litellm_async_clients
        loop.run_until_complete(close_litellm_async_clients())
    except Exception:
        pass
    # Cancel remaining background tasks (e.g. litellm's LoggingWorker) so the
    # loop can be closed cleanly without "Task was destroyed but it is pending!"
    pending = asyncio.all_tasks(loop)
    if pending:
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
    try:
        loop.close()
    except Exception:
        pass
    asyncio.set_event_loop(None)


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)



class LiteLLMClient:
    """LiteLLM client for integration with multiple providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the LiteLLM client with configuration.

        Args:
            config: Optional dictionary containing LiteLLM parameters. If provided,
                   additional kwargs will be merged with this config.
            **kwargs: LiteLLM parameters as named arguments. These will be merged
                     with the config dict if both are provided. Parameters include:
                - model: str (required)
                - api_key: str
                - api_base: str
                - api_version: str
                - temperature: float
                - top_p: float
                - max_tokens: int
                - timeout: int
                - dimensions: int (for embeddings only)
                - Any other LiteLLM parameters

        Examples:
            # Using dict only
            client = LiteLLMClient({"model": "gpt-4", "api_key": "sk-..."})

            # Using named params only
            client = LiteLLMClient(model="gpt-4", api_key="sk-...")

            # Using both (kwargs override config)
            client = LiteLLMClient({"model": "gpt-3.5"}, model="gpt-4")
        """
        if config is None:
            config = {}
        self.config = {**config, **kwargs}
        self.metadata = metadata or {}
        # Cumulative token usage across all completion calls on this client.
        from utils.token_meter import TokenMeter
        self.token_meter = TokenMeter()

    def _get_completion_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for completion calls based on configuration."""
        # Start with a copy of the entire config dict
        kwargs = self.config.copy()

        # Ensure model is present (required)
        if "model" not in kwargs:
            raise ValueError("Model must be specified in config")

        # Rename timeout to match LiteLLM parameter name if needed
        if "request_timeout" in kwargs:
            kwargs["timeout"] = kwargs.pop("request_timeout")

        # Handle max_tokens vs max_completion_tokens conflict
        # If both are present, prefer max_completion_tokens and remove max_tokens
        if "max_completion_tokens" in kwargs and "max_tokens" in kwargs:
            del kwargs["max_tokens"]

        # Handle temperature/top_p conflict for Anthropic/Claude models
        # Claude API rejects requests that specify both parameters simultaneously
        model = kwargs.get("model", "")
        if isinstance(model, str) and ("claude" in model.lower() or "anthropic" in model.lower()):
            if "temperature" in kwargs and "top_p" in kwargs:
                del kwargs["top_p"]

        # Let litellm silently drop params unsupported by the provider
        # (e.g. presence_penalty for Anthropic/Bedrock models)
        kwargs["drop_params"] = True

        return kwargs

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a fresh metadata dict for a single call.

        Copies the caller's metadata (never mutates it), layers in the client's
        static ``self.metadata``, and prefixes ``trace_id`` with ``session_id``
        once. Working on a copy keeps repeated calls that share the same caller
        dict from re-prefixing ``trace_id`` on every call.
        """
        merged = dict(metadata or {})
        merged.update(self.metadata)
        if "trace_id" in merged and "session_id" in merged:
            merged["trace_id"] = f"{merged['session_id']}_{merged['trace_id']}"
        return merged

    async def _retry_with_exponential_backoff(self, func: Callable, max_total_wait: float = 60.0, initial_wait: float = 1.0, max_wait: float = 32.0, backoff_multiplier: float = 2.0) -> Any:
        """Retry a function with exponential backoff until success or max wait time exceeded.
        
        Args:
            func: Async function to retry
            max_total_wait: Maximum total wait time in seconds (default: 60)
            initial_wait: Initial wait time in seconds (default: 1.0)
            max_wait: Maximum wait time between retries in seconds (default: 32.0)
            backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
            
        Returns:
            Result from the function call
            
        Raises:
            Exception: If max wait time is exceeded, raises an error indicating query is skipped
        """
        total_wait_time = 0.0
        wait_time = initial_wait
        attempt = 0
        
        while True:
            try:
                result = await func()
                if attempt > 0:
                    print(f"{Colors.GREEN}✓ Query succeeded after {attempt} retry attempt(s) and {total_wait_time:.1f}s total wait time{Colors.END}")
                return result
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed: {str(e)}")

                # ContextWindowExceededError is deterministic — retrying won't help
                if "ContextWindowExceededError" in type(e).__name__ or "context length" in str(e).lower():
                    logger.error(f"Context window exceeded (attempt {attempt}), not retrying: {e}")
                    raise

                # Check if we've exceeded max total wait time
                if total_wait_time >= max_total_wait:
                    error_msg = f"Query skipped after {attempt} attempts and {total_wait_time:.1f}s total wait time. Last error: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                
                # Calculate next wait time with exponential backoff
                next_wait = min(wait_time, max_wait)
                
                # Check if adding next wait would exceed max total wait
                if total_wait_time + next_wait > max_total_wait:
                    next_wait = max_total_wait - total_wait_time
                
                print(f"{Colors.YELLOW}⚠ Retrying in {next_wait:.1f}s (total wait so far: {total_wait_time:.1f}s)...{Colors.END}")
                await asyncio.sleep(next_wait)
                total_wait_time += next_wait
                
                # Increase wait time for next iteration (exponential backoff)
                wait_time = min(wait_time * backoff_multiplier, max_wait)

    def complete(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None, strip_think: bool = True, return_reasoning_fallback: bool = False, **kwargs) -> str:
        """Perform text completion using LiteLLM with retry logic and exponential backoff.

        This is a synchronous wrapper that calls the async implementation.

        Args:
            messages: List of message dictionaries for the conversation
            metadata: Optional metadata for cost tracking
            strip_think: If True (default), strip <think>...</think> tags from the
                response. Set to False to preserve thinking content for extraction.
            return_reasoning_fallback: If True and the response content is empty,
                return reasoning_content prefixed with ``[reasoning_fallback]``.
            **kwargs: Runtime parameters to override config (e.g., temperature, top_p, max_tokens)

        Returns:
            The completion content as a string

        Raises:
            RuntimeError: If query is skipped after exceeding max wait time
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.acomplete(messages, metadata, strip_think=strip_think, return_reasoning_fallback=return_reasoning_fallback, **kwargs))
                    finally:
                        _close_loop_with_litellm_cleanup(new_loop)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            loop = _get_or_create_thread_event_loop()
            return loop.run_until_complete(self.acomplete(messages, metadata, strip_think=strip_think, return_reasoning_fallback=return_reasoning_fallback, **kwargs))

    async def acomplete(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None, strip_think: bool = True, return_reasoning_fallback: bool = False, **kwargs) -> str:
        """Async version of complete(). Perform text completion using LiteLLM with retry logic and exponential backoff.

        Args:
            messages: List of message dictionaries for the conversation
            metadata: Optional metadata for cost tracking
            strip_think: If True (default), strip <think>...</think> tags from the
                response. Set to False to preserve thinking content for extraction.
            return_reasoning_fallback: If True and the response content is empty,
                return reasoning_content prefixed with ``[reasoning_fallback]``.
            **kwargs: Runtime parameters to override config (e.g., temperature, top_p, max_tokens)

        Returns:
            The completion content as a string

        Raises:
            RuntimeError: If query is skipped after exceeding max wait time
        """
        metadata = self._prepare_metadata(metadata)

        async def _complete_call():
            # Get completion kwargs and merge with runtime overrides
            completion_kwargs = self._get_completion_kwargs()
            completion_kwargs.update(kwargs)
            completion_kwargs["messages"] = messages
            completion_kwargs["metadata"] = metadata

            # Handle max_tokens vs max_completion_tokens conflict after merge
            # If both are present, remove max_tokens (prefer max_completion_tokens)
            if "max_completion_tokens" in completion_kwargs and "max_tokens" in completion_kwargs:
                del completion_kwargs["max_tokens"]

            # Call acompletion with kwargs
            response = await litellm.acompletion(**completion_kwargs)
            self.token_meter.record_usage(getattr(response, "usage", None))

            msg = response.choices[0].message
            content = msg.content or ""
            if strip_think:
                content = content.split("</think>")[-1].strip()

            if not content.strip() and return_reasoning_fallback:
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if reasoning.strip():
                    return "[reasoning_fallback]" + reasoning

            return content

        return await self._retry_with_exponential_backoff(_complete_call)

    def complete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T], metadata: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """Perform structured completion using LiteLLM with retry logic and exponential backoff.

        This is a synchronous wrapper that calls the async implementation.

        Args:
            messages: List of message dictionaries for the conversation
            response_format: Pydantic model class for structured output
            metadata: Optional metadata for cost tracking
            **kwargs: Runtime parameters to override config (e.g., temperature, top_p, max_tokens)

        Returns:
            The parsed structured response

        Raises:
            RuntimeError: If query is skipped after exceeding max wait time
            ValueError: If structured output parsing fails
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.acomplete_with_structured_output(messages, response_format, metadata, **kwargs))
                    finally:
                        _close_loop_with_litellm_cleanup(new_loop)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            loop = _get_or_create_thread_event_loop()
            return loop.run_until_complete(self.acomplete_with_structured_output(messages, response_format, metadata, **kwargs))

    async def acomplete_with_structured_output(self, messages: List[Dict[str, Any]], response_format: Type[T], metadata: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """Async version of complete_with_structured_output(). Perform structured completion using LiteLLM with retry logic and exponential backoff.

        Args:
            messages: List of message dictionaries for the conversation
            response_format: Pydantic model class for structured output
            metadata: Optional metadata for cost tracking
            **kwargs: Runtime parameters to override config (e.g., temperature, top_p, max_tokens)

        Returns:
            The parsed structured response

        Raises:
            RuntimeError: If query is skipped after exceeding max wait time
            ValueError: If structured output parsing fails
        """
        metadata = self._prepare_metadata(metadata)

        async def _complete_call():
            # Get completion kwargs and merge with runtime overrides
            completion_kwargs = self._get_completion_kwargs()
            completion_kwargs.update(kwargs)
            completion_kwargs["messages"] = messages
            completion_kwargs["response_format"] = response_format
            completion_kwargs["metadata"] = metadata

            # Handle max_tokens vs max_completion_tokens conflict after merge
            # If both are present, remove max_tokens (prefer max_completion_tokens)
            if "max_completion_tokens" in completion_kwargs and "max_tokens" in completion_kwargs:
                del completion_kwargs["max_tokens"]

            # Call acompletion with kwargs
            response = await litellm.acompletion(**completion_kwargs)
            self.token_meter.record_usage(getattr(response, "usage", None))

            # Parse the structured response
            if (
                hasattr(response.choices[0].message, "parsed")
                and response.choices[0].message.parsed
            ):
                parsed_response = response.choices[0].message.parsed
            else:
                # Fallback: parse JSON content manually
                content = response.choices[0].message.content or ""
                content = content.split("</think>")[-1].strip()

                # Strip markdown code blocks if present
                if content.startswith("```"):
                    # Remove opening ```json or ``` and closing ```
                    lines = content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # Remove first line with ```json or ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]  # Remove last line with ```
                    content = '\n'.join(lines).strip()

                try:
                    parsed_json = json.loads(content)
                    parsed_response = response_format.model_validate(parsed_json)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    logger.error(f"Raw content: {content}")
                    raise ValueError(f"Failed to parse structured output: {e}")

            return parsed_response

        return await self._retry_with_exponential_backoff(_complete_call)

    async def acomplete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str], metadata: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[str, Optional[str]]:
        r"""Async streaming completion that stops when a pattern is detected.

        Args:
            messages: List of message dictionaries for the conversation
            stop_patterns: List of regex patterns to stop on (e.g., [r"Action \d+:"])
            metadata: Optional metadata for cost tracking
            **kwargs: Runtime parameters to override config

        Returns:
            Tuple of (generated_text, matched_pattern)
            - generated_text: Text generated up to (but not including) the matched pattern
            - matched_pattern: The pattern that was matched, or None if generation completed naturally
        """
        import re

        metadata = self._prepare_metadata(metadata)

        async def _streaming_complete_call():
            # Get completion kwargs and merge with runtime overrides
            completion_kwargs = self._get_completion_kwargs()
            completion_kwargs.update(kwargs)
            completion_kwargs["messages"] = messages
            completion_kwargs["metadata"] = metadata
            completion_kwargs["stream"] = True  # Enable streaming

            # Handle max_tokens vs max_completion_tokens conflict after merge
            if "max_completion_tokens" in completion_kwargs and "max_tokens" in completion_kwargs:
                del completion_kwargs["max_tokens"]

            # Call streaming completion
            response_stream = await litellm.acompletion(**completion_kwargs)

            generated_text = ""
            matched_pattern = None

            # Compile regex patterns
            compiled_patterns = [re.compile(pattern) for pattern in stop_patterns]

            async for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    new_content = chunk.choices[0].delta.content
                    generated_text += new_content

                    # Check if any stop pattern is matched in the accumulated text
                    for pattern in compiled_patterns:
                        match = pattern.search(generated_text)
                        if match:
                            # Extract text before the match
                            matched_pattern = match.group(0)
                            generated_text = generated_text[:match.start()]
                            # Stop streaming
                            return generated_text.strip(), matched_pattern

            # No pattern matched, return full generation
            return generated_text.strip(), None

        return await self._retry_with_exponential_backoff(_streaming_complete_call)

    def complete_with_stop_on_pattern(self, messages: List[Dict[str, Any]], stop_patterns: List[str], metadata: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[str, Optional[str]]:
        r"""Synchronous wrapper for streaming completion with pattern-based stopping.

        Args:
            messages: List of message dictionaries for the conversation
            stop_patterns: List of regex patterns to stop on (e.g., [r"Action \d+:"])
            metadata: Optional metadata for cost tracking
            **kwargs: Runtime parameters to override config

        Returns:
            Tuple of (generated_text, matched_pattern)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.acomplete_with_stop_on_pattern(messages, stop_patterns, metadata, **kwargs))
                    finally:
                        _close_loop_with_litellm_cleanup(new_loop)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            loop = _get_or_create_thread_event_loop()
            return loop.run_until_complete(self.acomplete_with_stop_on_pattern(messages, stop_patterns, metadata, **kwargs))

    async def acomplete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], tool_choice: Any = "auto", metadata: Optional[Dict[str, Any]] = None, strip_think: bool = True, **kwargs) -> tuple[str, List[Dict[str, Any]]]:
        """Async completion with tool/function calling.

        Args:
            messages: List of message dictionaries
            tools: Tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
            metadata: Optional metadata for cost tracking
            strip_think: If True, strip <think>...</think> from content
            **kwargs: Runtime parameters to override config

        Returns:
            Tuple of (content, tool_calls)
            - content: text content from the response (may be empty)
            - tool_calls: list of dicts [{"name": str, "arguments": dict}]
        """
        metadata = self._prepare_metadata(metadata)

        async def _complete_call():
            completion_kwargs = self._get_completion_kwargs()
            completion_kwargs.update(kwargs)
            completion_kwargs["messages"] = messages
            completion_kwargs["metadata"] = metadata
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = tool_choice

            if "max_completion_tokens" in completion_kwargs and "max_tokens" in completion_kwargs:
                del completion_kwargs["max_tokens"]

            response = await litellm.acompletion(**completion_kwargs)
            self.token_meter.record_usage(getattr(response, "usage", None))

            msg = response.choices[0].message
            content = msg.content or ""
            if strip_think:
                content = content.split("</think>")[-1].strip()

            parsed_tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        args = {}
                    parsed_tool_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            return content, parsed_tool_calls

        return await self._retry_with_exponential_backoff(_complete_call)

    def complete_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], tool_choice: Any = "auto", metadata: Optional[Dict[str, Any]] = None, strip_think: bool = True, **kwargs) -> tuple[str, List[Dict[str, Any]]]:
        """Synchronous wrapper for completion with tool/function calling.

        Args:
            messages: List of message dictionaries
            tools: Tool definitions in OpenAI format
            tool_choice: "auto", "none", "required", or specific tool
            metadata: Optional metadata for cost tracking
            strip_think: If True, strip <think>...</think> from content
            **kwargs: Runtime parameters to override config

        Returns:
            Tuple of (content, tool_calls)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.acomplete_with_tools(messages, tools, tool_choice, metadata, strip_think, **kwargs))
                    finally:
                        _close_loop_with_litellm_cleanup(new_loop)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result
        else:
            loop = _get_or_create_thread_event_loop()
            return loop.run_until_complete(self.acomplete_with_tools(messages, tools, tool_choice, metadata, strip_think, **kwargs))

    def cleanup(self):
        """Close this thread's persistent LiteLLM event loop, if any.

        The synchronous ``complete*`` wrappers reuse a per-thread event loop
        (see ``_get_or_create_thread_event_loop``). Call this when the client is
        no longer needed so LiteLLM's async HTTP clients and that loop are torn
        down cleanly. Safe to call repeatedly — the loop is recreated on demand
        by the next call.
        """
        loop = getattr(_thread_local, "loop", None)
        if loop is not None and not loop.is_closed():
            _close_loop_with_litellm_cleanup(loop)
        if hasattr(_thread_local, "loop"):
            del _thread_local.loop

    def __del__(self):
        """Cleanup on deletion."""
        try:
            import warnings
            # Suppress all warnings during shutdown cleanup
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.cleanup()
        except Exception:
            # Ignore all cleanup errors in destructor
            pass
