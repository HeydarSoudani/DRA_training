"""Unified listwise reranker using an OpenAI-compatible API (vLLM server).

Supports RankGPT, RankZephyr, Qwen3, and any future listwise model by
swapping prompt templates.  Replaces the need for separate backend-specific
reranker classes with a single vLLM-backed implementation.

Public API (preserved from existing rerankers)::

    rerank_batch(requests, rank_start, rank_end, ...)  -> List[Result]
    rerank(request, rank_start, rank_end, ...)         -> Result
    rerank(query_str, documents, top_k)                -> List[Dict]
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from openai import AzureOpenAI, OpenAI
from rank_llm.data import Candidate, Query, Request, Result

from .prompts import get_template
from .prompts.base import PromptTemplate

logger = logging.getLogger(__name__)

# Regex to strip <think>…</think> blocks produced by reasoning models
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ListwiseVLLMReranker:
    """Unified listwise reranker backed by an OpenAI-compatible API.

    Parameters
    ----------
    model_name : str
        Model identifier sent to the API
        (e.g. ``"castorini/rank_zephyr_7b_v1_full"``).
    api_url : str
        Base URL of the vLLM OpenAI-compatible server.
    api_key : str
        API key (``"EMPTY"`` for a local vLLM server).
    template : str or PromptTemplate
        Prompt template name (``"rankgpt"``, ``"rankzephyr"``, ``"qwen3"``)
        or a :class:`PromptTemplate` instance.
    window_size : int
        Number of candidates per sliding window.
    stride : int
        Step size when sliding the window upward.
    use_sliding_window : bool
        If *False*, rank all candidates in a single pass.
    max_passage_words : int
        Maximum words per passage (truncated to this limit).
    top_k : int
        Default top-k for the convenience dict API.
    max_tokens : int
        Max output tokens for the API response.
    temperature : float
        Sampling temperature (0.0 for deterministic).
    enable_thinking : bool
        Enable thinking mode for models that support it (e.g. Qwen3).
        When *True*, ``<think>…</think>`` blocks are stripped before parsing.
    max_retries : int
        Number of API call retries on failure.
    retry_delay : float
        Seconds to wait between retries.
    """

    def __init__( self, model_name: str, api_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", template: Union[str, PromptTemplate] = "rankzephyr", window_size: int = 20, stride: int = 10, use_sliding_window: bool = True, max_passage_words: int = 300, top_k: int = 100, max_tokens: int = 200, temperature: float = 0.0, enable_thinking: bool = False, max_retries: int = 3, retry_delay: float = 1.0, api_version: Optional[str] = None, ) -> None:
        self.model_name = model_name
        self.window_size = window_size
        self.stride = stride
        self.use_sliding_window = use_sliding_window
        self.max_passage_words = max_passage_words
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Resolve template
        if isinstance(template, str):
            self.template: PromptTemplate = get_template(template)
        else:
            self.template = template

        # Use AzureOpenAI when the endpoint is an Azure URL
        if ".openai.azure.com" in api_url:
            # Azure deployments use the model name without the "azure/" prefix
            self.model_name = model_name.removeprefix("azure/")
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=api_version or "2025-01-01-preview",
            )
            logger.info(
                "ListwiseVLLMReranker (Azure): model=%s, endpoint=%s, api_version=%s, template=%s",
                self.model_name, api_url, api_version, self.template.name,
            )
        else:
            self._client = OpenAI(api_key=api_key, base_url=api_url)
            logger.info(
                "ListwiseVLLMReranker: model=%s, api_url=%s, template=%s",
                model_name, api_url, self.template.name,
            )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _extract_text(self, candidate: Candidate) -> str:
        """Extract and truncate passage text from a Candidate."""
        text = candidate.doc.get("text") or candidate.doc.get("contents") or ""
        title = candidate.doc.get("title", "")
        content = f"{title} {text}".strip() if title else text
        words = content.split()
        if len(words) > self.max_passage_words:
            content = " ".join(words[: self.max_passage_words])
        return content

    # ------------------------------------------------------------------
    # Low-level inference
    # ------------------------------------------------------------------

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Call the OpenAI-compatible API and return the response text."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                api_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.enable_thinking:
                    api_params["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                response = self._client.chat.completions.create(**api_params)
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if (
                    "maximum context length" in err_str
                    or "context_length_exceeded" in err_str
                ):
                    logger.warning(
                        "Context length exceeded; returning empty ranking"
                    )
                    return ""
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise RuntimeError(
            f"API call failed after {self.max_retries} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Window-level reranking
    # ------------------------------------------------------------------

    def _rerank_window( self, query: str, candidates: List[Candidate] ) -> List[Candidate]:
        """Rerank a single window of candidates via one API call."""
        if not candidates:
            return candidates

        candidate_texts = [self._extract_text(c) for c in candidates]
        messages = self.template.build_messages(query, candidate_texts)
        output = self._generate(messages)
        logger.debug("Model output: %s", output)

        # Strip thinking blocks before parsing (Qwen3 reasoning mode)
        if self.enable_thinking and output:
            output = _THINK_BLOCK_RE.sub("", output).strip()

        ranked_indices = self.template.parse_ranking(output, len(candidates))
        return [candidates[i] for i in ranked_indices]

    # ------------------------------------------------------------------
    # Public API (preserved interface)
    # ------------------------------------------------------------------

    def rerank_batch( self, requests: List[Request], rank_start: int = 0, rank_end: int = 100, shuffle_candidates: bool = False, logging: bool = False, **kwargs: Any, ) -> List[Result]:
        """Rerank a batch of requests sequentially."""
        results: List[Result] = []
        for request in requests:
            result = self.rerank(
                request,
                rank_start=rank_start,
                rank_end=rank_end,
                shuffle_candidates=shuffle_candidates,
                logging=logging,
                **kwargs,
            )
            results.append(result)
        return results

    def rerank( self, request_or_query: Union[Request, str], documents: Optional[List[Dict[str, Any]]] = None, top_k: Optional[int] = None, rank_start: int = 0, rank_end: int = 100, shuffle_candidates: bool = False, logging: bool = False, **kwargs: Any, ) -> Union[Result, List[Dict[str, Any]]]:
        """Rerank candidates.

        Supports two calling conventions:

        1. **rank_llm compatible** – pass a :class:`Request` object::

               result = reranker.rerank(request, rank_start=0, rank_end=100)

        2. **Convenience dict API** – pass a query string and documents::

               docs = reranker.rerank("my query", documents=[...], top_k=10)
        """
        # Convenience wrapper path
        if isinstance(request_or_query, str):
            return self._rerank_dicts(
                query=request_or_query,
                documents=documents or [],
                top_k=top_k if top_k is not None else self.top_k,
            )

        # rank_llm.data.Request path
        request: Request = request_or_query
        candidates = list(request.candidates[rank_start:rank_end])
        tail = list(request.candidates[rank_end:])

        if self.use_sliding_window:
            end = len(candidates)
            while end > 0:
                start = max(0, end - self.window_size)
                candidates[start:end] = self._rerank_window(
                    request.query.text, candidates[start:end]
                )
                if start == 0:
                    break
                end -= self.stride
        else:
            candidates = self._rerank_window(request.query.text, candidates)

        reranked_candidates = candidates + tail
        total = len(reranked_candidates)
        for i, cand in enumerate(reranked_candidates):
            cand.score = float(total - i)

        return Result(query=request.query, candidates=reranked_candidates)

    def _rerank_dicts( self, query: str, documents: List[Dict[str, Any]], top_k: int, ) -> List[Dict[str, Any]]:
        """Convert ``List[Dict]`` → ``Request``, rerank, then convert back."""
        if not documents:
            return []

        rank_end = min(top_k, len(documents))
        candidates: List[Candidate] = []
        for i, doc in enumerate(documents):
            doc_id = str(
                doc.get("id") or doc.get("doc_id") or doc.get("_id") or i
            )
            text = doc.get("text") or doc.get("contents") or ""
            title = doc.get("title", "")
            candidates.append(
                Candidate(
                    docid=doc_id,
                    score=doc.get("score", 0.0),
                    doc={"text": text, "title": title},
                )
            )

        request = Request(
            query=Query(text=query, qid="0"), candidates=candidates
        )
        result = self.rerank(request, rank_start=0, rank_end=rank_end)

        # Map back to original dicts with rank_score
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        for i, doc in enumerate(documents):
            doc_id = str(
                doc.get("id") or doc.get("doc_id") or doc.get("_id") or i
            )
            id_to_doc[doc_id] = doc

        reranked_docs: List[Dict[str, Any]] = []
        for cand in result.candidates:
            orig = id_to_doc.get(str(cand.docid), {})
            doc_out = orig.copy()
            doc_out["rank_score"] = cand.score
            reranked_docs.append(doc_out)
        return reranked_docs


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def setup_listwise_vllm_reranker( model_name: str, api_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", template: str = "rankzephyr", top_k: int = 100, window_size: int = 20, stride: int = 10, use_sliding_window: bool = True, max_passage_words: int = 300, max_tokens: int = 200, temperature: float = 0.0, enable_thinking: bool = False, max_retries: int = 3, api_version: Optional[str] = None, ) -> ListwiseVLLMReranker:
    """Create a :class:`ListwiseVLLMReranker` with sensible defaults."""
    return ListwiseVLLMReranker(
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        template=template,
        window_size=window_size,
        stride=stride,
        use_sliding_window=use_sliding_window,
        max_passage_words=max_passage_words,
        top_k=top_k,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        max_retries=max_retries,
        api_version=api_version,
    )
