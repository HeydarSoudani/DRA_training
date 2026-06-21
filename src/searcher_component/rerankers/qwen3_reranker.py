"""Qwen3 pointwise reranker using a vLLM OpenAI-compatible API.

Implements pointwise reranking using the Qwen3-Reranker model family.
The model frames relevance as a binary yes/no classification: given a
query-document pair, it produces logits for "yes" and "no" tokens, and
the relevance score is::

    score = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))

The prompt follows the official Qwen3-Reranker format with the chat
template and empty ``<think>`` tags.

Reference: https://huggingface.co/Qwen/Qwen3-Reranker-4B

Public API
----------
score_pairs(pairs)                                  -> List[float]
rerank_batch(requests, rank_start, rank_end, ...)   -> List[Result]
rerank(request, rank_start, rank_end, ...)          -> Result
rerank(query_str, documents, top_k)                 -> List[Dict]   (convenience)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from rank_llm.data import Candidate, Query, Request, Result

logger = logging.getLogger(__name__)

# Default task instruction (same as the official HuggingFace example)
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

QWEN3_RERANKER_SIZES = {
    "0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "4B": "Qwen/Qwen3-Reranker-4B",
    "8B": "Qwen/Qwen3-Reranker-8B",
}


def get_qwen3_reranker_path(size: str = "4B") -> str:
    """Return the HuggingFace model path for a Qwen3-Reranker size variant."""
    if size not in QWEN3_RERANKER_SIZES:
        raise ValueError(f"Invalid Qwen3 reranker size '{size}'. Choose from: {list(QWEN3_RERANKER_SIZES.keys())}")
    return QWEN3_RERANKER_SIZES[size]

_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def _format_user_content(instruction: str, query: str, document: str) -> str:
    """Build the user-turn content for a single query-document pair."""
    return (
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {document}"
    )


class Qwen3Reranker:
    """Pointwise reranker backed by a Qwen3-Reranker model via vLLM API.

    The model is served by a vLLM server and accessed through the
    OpenAI-compatible completions endpoint.  Each (query, document) pair
    is scored independently; documents are then sorted by descending score.

    Parameters
    ----------
    model_name : str
        Model identifier sent to the API.
    api_url : str
        Base URL of the vLLM OpenAI-compatible server.
    api_key : str
        API key (``"EMPTY"`` for a local vLLM server).
    instruction : str
        Task instruction prepended to every pair.
    batch_size : int
        Number of pairs per API call batch.
    max_tokens : int
        Max tokens to generate (1 is sufficient for yes/no).
    top_logprobs : int
        Number of top log-probabilities to request.
    top_k : int
        Default number of top documents for the convenience API.
    enable_thinking : bool
        Whether to enable the Qwen3 thinking/reasoning mode.
    max_retries : int
        Retries on transient API errors.
    retry_delay : float
        Seconds between retries.
    """

    def __init__(
        self,
        model_name: str,
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        instruction: str = _DEFAULT_INSTRUCTION,
        batch_size: int = 32,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        top_k: int = 100,
        enable_thinking: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = OpenAI(base_url=api_url, api_key=api_key)
        logger.info(
            "Qwen3Reranker initialised: model=%s, api_url=%s",
            model_name, api_url,
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_single(self, query: str, document: str) -> float:
        """Score a single (query, document) pair via the chat completions API."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _format_user_content(
                self.instruction, query, document,
            )},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
                )
                return self._extract_score(response)
            except Exception as exc:
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All retries exhausted, returning score 0.0")
                    return 0.0

    @staticmethod
    def _extract_score(response) -> float:
        """Extract the yes/no relevance score from a chat completion response.

        Matches the official Qwen3-Reranker scoring: only the exact
        ``"yes"`` and ``"no"`` tokens are used (the specific token IDs the
        model was trained on).  Case variants like ``"Yes"`` / ``"NO"``
        are different tokens and are ignored, following the reference
        implementation.
        """
        # Navigate: response.choices[0].logprobs.content[0].top_logprobs
        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            logger.warning("No logprobs in response; returning 0.0")
            return 0.0

        top_lps = choice.logprobs.content[0].top_logprobs
        logprob_yes = -10.0
        logprob_no = -10.0
        for lp in top_lps:
            token = lp.token.strip()
            if token == "yes":
                logprob_yes = lp.logprob
            elif token == "no":
                logprob_no = lp.logprob

        score_yes = math.exp(logprob_yes)
        score_no = math.exp(logprob_no)
        return score_yes / (score_yes + score_no)

    def score_pairs(
        self, query: str, texts: List[str],
    ) -> List[float]:
        """Score a query against multiple documents.

        Returns a list of relevance scores in [0, 1].
        """
        if not texts:
            return []
        return [self._score_single(query, text) for text in texts]

    # ------------------------------------------------------------------
    # rank_llm-compatible core API
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
        results = []
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

        **Convenience wrapper**::

            docs: List[Dict] = reranker.rerank(query_str, documents, top_k=100)
        """
        if isinstance(request_or_query, str):
            return self._rerank_dicts(
                query=request_or_query,
                documents=documents or [],
                top_k=top_k if top_k is not None else self.top_k,
            )

        request: Request = request_or_query
        candidates_to_rank = list(request.candidates[rank_start:rank_end])
        tail = list(request.candidates[rank_end:])

        if candidates_to_rank:
            texts = [
                cand.doc.get("text") or cand.doc.get("contents") or ""
                for cand in candidates_to_rank
            ]
            raw_scores = self.score_pairs(request.query.text, texts)

            scored = sorted(
                zip(raw_scores, candidates_to_rank),
                key=lambda x: x[0],
                reverse=True,
            )
            total = len(scored)
            ranked_candidates = []
            for i, (_, cand) in enumerate(scored):
                cand.score = float(total - i)
                ranked_candidates.append(cand)
        else:
            ranked_candidates = candidates_to_rank

        return Result(query=request.query, candidates=ranked_candidates + tail)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def _rerank_dicts(
        self, query: str, documents: List[Dict], top_k: int,
    ) -> List[Dict]:
        if not documents:
            return []

        docs_to_rank = documents[: min(top_k, len(documents))]
        tail = documents[len(docs_to_rank):]

        texts = [
            doc.get("text") or doc.get("contents") or ""
            for doc in docs_to_rank
        ]
        scores = self.score_pairs(query, texts)

        reranked: List[Dict] = []
        for score, doc in sorted(
            zip(scores, docs_to_rank), key=lambda x: x[0], reverse=True,
        ):
            doc_out = doc.copy()
            doc_out["rank_score"] = score
            reranked.append(doc_out)

        reranked.extend(tail)
        return reranked


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def setup_qwen3_reranker(
    model_name: Optional[str] = None,
    size: str = "4B",
    api_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    batch_size: int = 32,
    top_k: int = 100,
    enable_thinking: bool = False,
) -> Qwen3Reranker:
    """Create and return a :class:`Qwen3Reranker`.

    Parameters
    ----------
    model_name : str, optional
        Explicit model identifier. If provided, ``size`` is ignored.
    size : str
        Qwen3-Reranker size variant: ``"0.6B"``, ``"4B"`` (default), or ``"8B"``.
        Only used when ``model_name`` is not provided.

    Expects a vLLM server running the Qwen3-Reranker model::

        vllm serve Qwen/Qwen3-Reranker-4B --max-model-len 8192
    """
    if model_name is None:
        model_name = get_qwen3_reranker_path(size)
    return Qwen3Reranker(
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        batch_size=batch_size,
        top_k=top_k,
        enable_thinking=enable_thinking,
    )
