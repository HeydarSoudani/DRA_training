"""Rank-R1 (ielab) setwise document reranking via reasoning LLM.

Implements setwise reranking using the ``ielabgroup/Rank-R1-7B-v0.1`` model
(or any compatible variant).  The model takes a query and a **set** of
candidate documents, then selects the most relevant one.  A heap-sort
algorithm produces a full ranking from repeated set comparisons.

The model uses ``<think>…</think>`` reasoning followed by
``<answer>[N]</answer>`` to indicate its choice.

Reference: Zhuang et al., "Rank-R1: Enhancing Reasoning in LLM-based
Document Rerankers via Reinforcement Learning", arXiv:2503.06034.

Original implementation: https://github.com/ielab/llm-rankers

Public API
----------
rerank_batch(requests, rank_start, rank_end, ...)  ->  List[Result]
rerank(request, rank_start, rank_end, ...)         ->  Result
rerank(query_str, documents, top_k)               ->  List[Dict]   (convenience)
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Union

from rank_llm.data import Candidate, Query, Request, Result

from .prompts import RANK_R1_ANSWER_PATTERN, RANK_R1_SYSTEM_PROMPT, RANK_R1_USER_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RankR1SetwiseReranker:
    """Setwise reranker using the ielab Rank-R1 model.

    Uses an n-ary heap-sort algorithm where each comparison asks the LLM to
    pick the most relevant document from a set of candidates.

    Parameters
    ----------
    api_url:
        Base URL for the vLLM-compatible chat completions API.
    api_model_name:
        Model name sent to the API.  When serving LoRA adapters via vLLM's
        ``--lora-modules`` flag, this should be the adapter alias
        (e.g. ``"rank-r1"``).  When serving a merged model, this is the
        HuggingFace model ID (e.g. ``"ielabgroup/Rank-R1-7B-v0.1"``).
    api_key:
        API key (default ``"EMPTY"`` for local vLLM servers).
    num_child:
        Branching factor for the heap sort — how many candidate documents are
        compared per LLM call.  Each call presents ``num_child + 1`` docs
        (the root node + its children).  Default 19 (matches the original
        paper's training setup with 20 documents per comparison).
    k:
        Number of top documents to fully sort via heap sort.  Documents
        beyond position ``k`` retain their original order.
    max_tokens:
        Maximum tokens the model may generate per comparison (budget for
        the ``<think>…</think><answer>…</answer>`` output).
    top_k:
        Default rerank depth for the convenience ``rerank(query, docs)``
        wrapper.
    context_size:
        Maximum characters per document text shown to the model.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8001/v1",
        api_model_name: str = "rank-r1",
        api_key: str = "EMPTY",
        num_child: int = 19,
        k: int = 10,
        max_tokens: int = 2048,
        top_k: int = 100,
        context_size: int = 450,
    ) -> None:
        self.api_url = api_url
        self.api_model_name = api_model_name
        self.num_child = num_child
        self.k = k
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.context_size = context_size

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "RankR1SetwiseReranker requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(base_url=api_url, api_key=api_key)
        logger.info(
            "Rank-R1 setwise reranker: api=%s model=%s num_child=%d k=%d",
            api_url, api_model_name, num_child, k,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _truncate_text(self, text: str) -> str:
        """Truncate document text to ``context_size`` characters."""
        if len(text) > self.context_size:
            return text[: self.context_size]
        return text

    def _build_messages(self, query: str, passages: List[str]) -> List[Dict[str, str]]:
        """Build chat messages for a setwise comparison."""
        docs_text = "\n".join(
            f"[{i + 1}]: {self._truncate_text(p)}" for i, p in enumerate(passages)
        )
        return [
            {"role": "system", "content": RANK_R1_SYSTEM_PROMPT},
            {"role": "user", "content": RANK_R1_USER_PROMPT.format(query=query, docs=docs_text)},
        ]

    # ------------------------------------------------------------------
    # LLM comparison
    # ------------------------------------------------------------------

    def _compare(self, query: str, passages: List[str]) -> int:
        """Ask the LLM which passage is most relevant.

        Returns the 0-based index of the chosen passage within *passages*.
        Falls back to index 0 (the root) on parse failure.
        """
        messages = self._build_messages(query, passages)

        try:
            response = self._client.chat.completions.create(
                model=self.api_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Rank-R1 API call failed: %s — defaulting to index 0", exc)
            return 0

        match = RANK_R1_ANSWER_PATTERN.search(text)
        if not match:
            logger.debug("Rank-R1: no answer tag found in response, defaulting to 0")
            return 0

        answer = match.group(1).strip()
        # Extract number from e.g. "[3]"
        num_match = re.search(r"\[(\d+)\]", answer)
        if not num_match:
            logger.debug("Rank-R1: could not parse label from '%s', defaulting to 0", answer)
            return 0

        idx = int(num_match.group(1)) - 1  # 1-based → 0-based
        if 0 <= idx < len(passages):
            return idx

        logger.debug("Rank-R1: label %d out of range [1, %d], defaulting to 0", idx + 1, len(passages))
        return 0

    # ------------------------------------------------------------------
    # Heap-sort algorithm  (from ielab/llm-rankers SetwiseLlmRanker)
    # ------------------------------------------------------------------

    def _heapify(self, arr: List[Dict], n: int, i: int, query: str) -> None:
        """Sift down node *i* in an n-ary max-heap of size *n*."""
        largest = i
        children = list(range(self.num_child * i + 1, min(self.num_child * (i + 1) + 1, n)))
        if not children:
            return

        # Collect passages for the root + its children
        candidate_indices = [largest] + children
        passages = [
            arr[idx].get("text") or arr[idx].get("contents") or ""
            for idx in candidate_indices
        ]

        best_local = self._compare(query, passages)
        best_global = candidate_indices[best_local]

        if best_global != largest:
            arr[largest], arr[best_global] = arr[best_global], arr[largest]
            self._heapify(arr, n, best_global, query)

    def _heap_sort(self, arr: List[Dict], query: str, k: int) -> None:
        """In-place n-ary heap sort.  Best elements end up at the tail."""
        n = len(arr)
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self._heapify(arr, n, i, query)
        # Extract k max elements
        ranked = 0
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked >= k:
                break
            self._heapify(arr, i, 0, query)

    # ------------------------------------------------------------------
    # Core reranking
    # ------------------------------------------------------------------

    def _rerank_docs(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Rerank documents using setwise heap sort."""
        if not documents:
            return []

        k = min(self.k, top_k, len(documents))
        # Work on copies to avoid mutating the input
        ranking = [doc.copy() for doc in documents[:top_k]]
        original_tail = documents[top_k:]

        if len(ranking) <= 1:
            return ranking + [d.copy() for d in original_tail]

        logger.info(
            "Rank-R1 setwise reranking %d documents (k=%d, num_child=%d)",
            len(ranking), k, self.num_child,
        )

        self._heap_sort(ranking, query, k)

        # After heap sort, the top-k elements are at the tail in sorted order
        # (best last).  Reverse to get best-first.
        ranking = list(reversed(ranking))

        # Assign rank-derived scores
        n = len(ranking)
        reranked: List[Dict] = []
        for i, doc in enumerate(ranking):
            doc["rank_score"] = (n - i) / n
            reranked.append(doc)

        # Append tail documents in their original order
        for doc in original_tail:
            doc_out = doc.copy()
            doc_out["rank_score"] = 0.0
            reranked.append(doc_out)

        return reranked

    # ------------------------------------------------------------------
    # Public API  (matches Rank1Reranker / RankLLaMAReranker interface)
    # ------------------------------------------------------------------

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """Rerank a batch of requests."""
        return [
            self.rerank(req, rank_start=rank_start, rank_end=rank_end,
                        shuffle_candidates=shuffle_candidates, logging=logging, **kwargs)
            for req in requests
        ]

    def rerank(
        self,
        request_or_query: Union[Request, str],
        documents: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> Union[Result, List[Dict]]:
        """Rerank a single request.

        Accepts two calling conventions:

        **rank_llm-compatible**::

            result: Result = reranker.rerank(request)

        **Convenience wrapper** (used internally by the agent)::

            docs: List[Dict] = reranker.rerank(query_str, documents, top_k=10)
        """
        if isinstance(request_or_query, str):
            return self._rerank_docs(
                query=request_or_query,
                documents=documents or [],
                top_k=top_k if top_k is not None else self.top_k,
            )

        request: Request = request_or_query
        candidates_to_rank = list(request.candidates[rank_start:rank_end])
        tail = list(request.candidates[rank_end:])

        if not candidates_to_rank:
            return Result(query=request.query, candidates=tail)

        # Convert candidates to dicts for the heap sort
        doc_dicts = [
            {
                "text": cand.doc.get("text") or cand.doc.get("contents") or "",
                "_candidate": cand,
            }
            for cand in candidates_to_rank
        ]

        k = min(self.k, len(doc_dicts))
        self._heap_sort(doc_dicts, request.query.text, k)
        doc_dicts = list(reversed(doc_dicts))

        # Assign position-based integer scores
        total = len(doc_dicts)
        ranked_candidates = []
        for i, dd in enumerate(doc_dicts):
            cand = dd["_candidate"]
            cand.score = float(total - i)
            ranked_candidates.append(cand)

        return Result(query=request.query, candidates=ranked_candidates + tail)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def setup_rank_r1_reranker(
    api_url: str = "http://localhost:8001/v1",
    api_model_name: str = "rank-r1",
    api_key: str = "EMPTY",
    num_child: int = 19,
    k: int = 10,
    max_tokens: int = 2048,
    top_k: int = 100,
    context_size: int = 450,
) -> RankR1SetwiseReranker:
    """Create and return a :class:`RankR1SetwiseReranker`.

    Parameters
    ----------
    api_url:
        Base URL for the vLLM-compatible API (default: ``http://localhost:8001/v1``).
    api_model_name:
        Model name or LoRA alias sent to the API.
    num_child:
        Branching factor for heap sort (default 19 → 20 docs per comparison).
    k:
        Number of top positions to fully rank via heap sort.
    """
    return RankR1SetwiseReranker(
        api_url=api_url,
        api_model_name=api_model_name,
        api_key=api_key,
        num_child=num_child,
        k=k,
        max_tokens=max_tokens,
        top_k=top_k,
        context_size=context_size,
    )
