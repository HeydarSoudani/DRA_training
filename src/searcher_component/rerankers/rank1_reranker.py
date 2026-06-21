"""Rank1 (JHU CLSP) pointwise document reranking via reasoning LLM.

Implements pointwise reranking using the ``jhu-clsp/rank1-7b`` model (or any
compatible model) that reasons about document relevance inside a ``<think>``
block and then outputs a ``true`` / ``false`` decision.  The relevance score
is P(true) extracted from the model's token logits.

Three inference backends are supported:

* ``"hf"``   – HuggingFace ``transformers`` (no extra dependencies; default).
* ``"vllm"`` – vLLM for faster batched throughput.
* ``"api"``  – OpenAI-compatible completions endpoint (e.g., a local vLLM
               server or any hosted service).

Unlike the listwise RankZephyr / RankGPT rerankers, this is a **pointwise**
reranker: each (query, document) pair is scored independently, and documents
are then sorted by descending score.

Reference: Weller et al., "Rank1: Test-Time Compute for Reranking in
Information Retrieval", arXiv:2502.18418.

Public API
----------
score_query(query, texts)                          ->  List[float]
rerank_batch(requests, rank_start, rank_end, ...)  ->  List[Result]
rerank(request, rank_start, rank_end, ...)         ->  Result
rerank(query_str, documents, top_k)               ->  List[Dict]   (convenience)

Data types ``Request``, ``Result``, ``Query``, ``Candidate`` are imported
directly from ``rank_llm.data`` so they stay fully compatible with the rest
of the rank_llm ecosystem.
"""

import logging
import math
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from rank_llm.data import Candidate, Query, Request, Result

from .prompts import RANK1_INCOMPLETE_SCORE, RANK1_RELEVANCE_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Rank1Reranker:
    """Pointwise reranker using the JHU CLSP Rank1 reasoning model.

    Scores each (query, document) pair independently by prompting the model
    to reason inside ``<think>…</think>`` and then output ``true`` or
    ``false``.  The relevance score is P(true) normalised over
    P(true) + P(false).

    Adapted from the ``Rank1`` class in orionw/rank1, wrapped in the same
    dual-API interface as :class:`RankZephyrReranker` and :class:`RankGPTReranker`.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or local path (default: ``"jhu-clsp/rank1-7b"``).
    backend:
        Inference backend: ``"hf"`` (HuggingFace transformers), ``"vllm"``,
        or ``"api"`` (OpenAI-compatible completions endpoint).
    api_url:
        Base URL for the API endpoint. Required when ``backend="api"``.
    api_model_name:
        Model name sent to the API (default: ``"jhu-clsp/rank1-7b"``).
    context_size:
        Maximum tokens per (query, passage) pair; longer inputs are truncated.
    batch_size:
        Number of query-document pairs processed per forward pass.
    max_output_tokens:
        Maximum tokens to generate per pair (budget for the reasoning chain).
    num_gpus:
        Number of GPUs for vLLM tensor parallelism.
    precision:
        Model weight precision: ``"float16"`` or ``"float32"``.
    device:
        Target device for the HuggingFace backend (``"cuda"`` or ``"cpu"``).
        Falls back to CPU if CUDA is unavailable.
    top_k:
        Default number of top documents returned by the convenience ``rerank``
        wrapper when called with ``(query_str, documents, top_k=…)``.
    """

    # vLLM is not thread-safe; serialise all engine calls
    _vllm_lock = threading.Lock()

    def __init__( self, model_path: str = "jhu-clsp/rank1-7b", backend: Literal["hf", "vllm", "api"] = "hf", api_url: Optional[str] = None, api_model_name: str = "jhu-clsp/rank1-7b", context_size: int = 450, batch_size: int = 8, max_output_tokens: int = 2000, num_gpus: int = 1, precision: str = "float16", device: str = "cuda", top_k: int = 100, ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.api_url = api_url
        self.api_model_name = api_model_name
        self.context_size = context_size
        self.batch_size = batch_size
        self.max_output_tokens = max_output_tokens
        self.precision = precision
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k

        logger.info("Loading Rank1 tokenizer: %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Token IDs for the relevance labels (single leading-space variants)
        self.true_token = self._tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self._tokenizer(" false", add_special_tokens=False).input_ids[0]

        self._model = None
        self._vllm_engine = None
        self._openai_client = None

        if backend == "api":
            if api_url is None:
                raise ValueError("api_url must be provided when backend='api'")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Rank1Reranker with backend='api' requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from exc
            self._openai_client = OpenAI(base_url=api_url, api_key="EMPTY")
            logger.info("Rank1: using API backend at %s", api_url)

        elif backend == "vllm":
            try:
                from vllm import LLM
            except ImportError as exc:
                raise ImportError(
                    "Rank1Reranker with backend='vllm' requires the 'vllm' package. "
                    "Install it with: pip install vllm"
                ) from exc
            logger.info("Loading Rank1 vLLM engine: %s", model_path)
            self._vllm_engine = LLM(
                model=model_path,
                tensor_parallel_size=num_gpus,
                dtype=precision,
                trust_remote_code=True,
                max_model_len=4000,
                gpu_memory_utilization=0.9,
                enforce_eager=True,
                compilation_config={"level": 1},
            )
            logger.info("Rank1 vLLM engine loaded")

        else:  # "hf"
            logger.info(
                "Loading Rank1 HF model: %s on device: %s", model_path, self.device
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if precision == "float16" else torch.float32,
                device_map=self.device,
            )
            self._model.eval()
            logger.info("Rank1 HF model loaded")

        self._generation_config = GenerationConfig(
            temperature=0,
            max_new_tokens=max_output_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, passage: str) -> str:
        return RANK1_RELEVANCE_PROMPT.format(query=query, passage=passage)

    def _truncate(self, prompt: str) -> str:
        """Truncate a prompt to at most ``context_size`` tokens."""
        encoded = self._tokenizer.encode(prompt)
        if len(encoded) > self.context_size:
            prompt = self._tokenizer.decode(encoded[: self.context_size - 10])
        return prompt

    # ------------------------------------------------------------------
    # Backend: HuggingFace transformers
    # ------------------------------------------------------------------

    def _run_batch_hf(self, prompts: List[str]) -> List[float]:
        """Run a batch of prompts through the HuggingFace model."""
        inputs = self._tokenizer(
            prompts, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                generation_config=self._generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        scores: List[float] = []
        for i, sequence in enumerate(outputs.sequences):
            text = self._tokenizer.decode(sequence, skip_special_tokens=False)

            # The model should naturally end its sequence with " true" or " false"
            # after the </think> tag. If </think> is absent, the reasoning was
            # cut short; fall back to a neutral score.
            if "</think>" not in text or not outputs.scores:
                logger.debug("Rank1 HF: incomplete response at index %d", i)
                scores.append(RANK1_INCOMPLETE_SCORE)
                continue

            # outputs.scores[-1] contains the logits for the last generated token,
            # which should be the true/false decision token.
            final_logits = outputs.scores[-1][i]
            true_false_logits = final_logits[[self.true_token, self.false_token]]
            probs = torch.softmax(true_false_logits, dim=-1)
            scores.append(probs[0].item())  # index 0 → true_token

        return scores

    # ------------------------------------------------------------------
    # Backend: vLLM
    # ------------------------------------------------------------------

    def _run_batch_vllm(self, prompts: List[str]) -> List[float]:
        """Run a batch of prompts through the vLLM engine."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_output_tokens,
            logprobs=20,
            stop=["</think> true", "</think> false"],
            skip_special_tokens=False,
        )
        with self._vllm_lock:
            outputs = self._vllm_engine.generate(
                prompts, sampling_params, use_tqdm=False
            )

        scores: List[Optional[float]] = [None] * len(prompts)
        incomplete_indices: List[int] = []
        incomplete_prompts: List[str] = []
        incomplete_texts: List[str] = []

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            try:
                final_logits = output.outputs[0].logprobs[-1]
            except Exception:
                incomplete_indices.append(i)
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                continue

            if self.true_token not in final_logits or self.false_token not in final_logits:
                incomplete_indices.append(i)
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                continue

            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            scores[i] = _p_true(true_logit, false_logit)

        if incomplete_indices:
            fixed = self._fix_incomplete_vllm(incomplete_prompts, incomplete_texts)
            for idx, score in zip(incomplete_indices, fixed):
                scores[idx] = score

        return [s if s is not None else RANK1_INCOMPLETE_SCORE for s in scores]

    def _fix_incomplete_vllm( self, prompts: List[str], texts: List[str] ) -> List[float]:
        """Force a true/false decision token for incomplete vLLM outputs."""
        from vllm import SamplingParams

        cleaned = []
        for text in texts:
            text = text.rstrip()
            last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if not text.endswith((".", "!", "?")) and last_punct != -1:
                text = text[: last_punct + 1]
            cleaned.append(text.strip())

        forced_prompts = [
            f"{p}\n{c}\n</think>" for p, c in zip(prompts, cleaned)
        ]
        fix_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False,
        )
        with self._vllm_lock:
            outputs = self._vllm_engine.generate(
                forced_prompts, fix_params, use_tqdm=False
            )

        result: List[float] = []
        for output in outputs:
            try:
                final_logits = output.outputs[0].logprobs[-1]
                assert (
                    self.true_token in final_logits
                    and self.false_token in final_logits
                )
                true_logit = final_logits[self.true_token].logprob
                false_logit = final_logits[self.false_token].logprob
                result.append(_p_true(true_logit, false_logit))
            except Exception:
                result.append(RANK1_INCOMPLETE_SCORE)

        return result

    # ------------------------------------------------------------------
    # Backend: OpenAI-compatible API
    # ------------------------------------------------------------------

    def _run_batch_api(self, prompts: List[str]) -> List[float]:
        """Run a batch of prompts through an OpenAI-compatible completions API."""
        try:
            response = self._openai_client.completions.create(
                model=self.api_model_name,
                prompt=prompts,
                temperature=0,
                max_tokens=self.max_output_tokens,
                logprobs=20,
                stop=["</think> true", "</think> false"],
                extra_body={"skip_special_tokens": False},
            )
            result = response.model_dump()
        except Exception as exc:
            logger.error("Rank1 API call failed: %s", exc)
            return [RANK1_INCOMPLETE_SCORE] * len(prompts)

        true_token_str = self._tokenizer.decode([self.true_token])
        false_token_str = self._tokenizer.decode([self.false_token])

        scores: List[float] = [RANK1_INCOMPLETE_SCORE] * len(prompts)

        for choice in result.get("choices", []):
            idx = choice.get("index", 0)
            logprobs_data = choice.get("logprobs") or {}
            top_logprobs = logprobs_data.get("top_logprobs") or []
            if not top_logprobs:
                continue

            final_top = top_logprobs[-1]
            true_logit = final_top.get(true_token_str)
            false_logit = final_top.get(false_token_str)
            if true_logit is None or false_logit is None:
                continue

            scores[idx] = _p_true(true_logit, false_logit)

        return scores

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_query(self, query: str, texts: List[str]) -> List[float]:
        """Score a single query against a list of passages.

        Parameters
        ----------
        query:
            The search query.
        texts:
            List of passage texts to score.

        Returns
        -------
        List[float]
            P(true) relevance score per passage in [0, 1]
            (higher = more relevant).
        """
        if not texts:
            return []

        all_scores: List[float] = []
        total = len(texts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        for batch_idx, start in enumerate(range(0, total, self.batch_size), 1):
            batch_texts = texts[start : start + self.batch_size]
            prompts = [
                self._truncate(self._build_prompt(query, text))
                for text in batch_texts
            ]

            logger.info(
                "Rank1 scoring batch %d/%d (docs %d-%d of %d)",
                batch_idx, num_batches, start + 1, start + len(batch_texts), total,
            )

            if self.backend == "vllm":
                batch_scores = self._run_batch_vllm(prompts)
            elif self.backend == "api":
                batch_scores = self._run_batch_api(prompts)
            else:  # "hf"
                batch_scores = self._run_batch_hf(prompts)

            all_scores.extend(batch_scores)

        return all_scores

    # ------------------------------------------------------------------
    # GitHub-compatible core API  (matches ZephyrReranker / SafeOpenai)
    # ------------------------------------------------------------------

    def rerank_batch( self, requests: List[Request], rank_start: int = 0, rank_end: int = 100, shuffle_candidates: bool = False, logging: bool = False, **kwargs: Any, ) -> List[Result]:
        """Rerank a batch of requests.

        Matches the signature of ``ZephyrReranker.rerank_batch`` from the
        castorini/rank_llm repository.
        """
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

    def rerank( self, request_or_query: Union[Request, str], documents: Optional[List[Dict]] = None, top_k: Optional[int] = None, rank_start: int = 0, rank_end: int = 100, shuffle_candidates: bool = False, logging: bool = False, **kwargs: Any, ) -> Union[Result, List[Dict]]:
        """Rerank a single request.

        Accepts two calling conventions:

        **GitHub-compatible** (matches ``ZephyrReranker.rerank``)::

            result: Result = reranker.rerank(request)
            result: Result = reranker.rerank(request, rank_end=50)

        **Convenience wrapper** (used internally by the agent)::

            docs: List[Dict] = reranker.rerank(query_str, documents, top_k=100)

        Returns
        -------
        ``Result`` when called with a ``Request``; ``List[Dict]`` otherwise.
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
            raw_scores = self.score_query(request.query.text, texts)

            # Sort by descending score, assign position-based integer scores
            # (consistent with the rank_llm ecosystem convention used by
            # RankZephyrReranker and RankGPTReranker)
            scored = sorted(
                zip(raw_scores, candidates_to_rank), key=lambda x: x[0], reverse=True
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
    # Convenience wrapper helpers
    # ------------------------------------------------------------------

    def _rerank_dicts( self, query: str, documents: List[Dict], top_k: int ) -> List[Dict]:
        """Score documents, sort by P(true), return as List[Dict] with rank_score."""
        if not documents:
            return []

        docs_to_rank = documents[: min(top_k, len(documents))]
        tail = documents[len(docs_to_rank):]

        texts = [doc.get("text") or doc.get("contents") or "" for doc in docs_to_rank]
        scores = self.score_query(query, texts)

        reranked_docs: List[Dict] = []
        for score, doc in sorted(
            zip(scores, docs_to_rank), key=lambda x: x[0], reverse=True
        ):
            doc_out = doc.copy()
            doc_out["rank_score"] = score   # P(true) in [0, 1]
            reranked_docs.append(doc_out)

        reranked_docs.extend(tail)
        return reranked_docs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _p_true(true_logit: float, false_logit: float) -> float:
    """Compute P(true) from log-probabilities of true and false tokens."""
    true_score = math.exp(true_logit)
    false_score = math.exp(false_logit)
    return true_score / (true_score + false_score)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def setup_rank1_reranker( model_name: str = "jhu-clsp/rank1-7b", backend: Literal["hf", "vllm", "api"] = "hf", api_url: Optional[str] = None, api_model_name: str = "jhu-clsp/rank1-7b", device: str = "cuda", top_k: int = 100, batch_size: int = 8, context_size: int = 450, max_output_tokens: int = 2000, num_gpus: int = 1, precision: str = "float16", ) -> Rank1Reranker:
    """Create and return a :class:`Rank1Reranker`.

    Parameter names intentionally match those used throughout the pipeline.

    Parameters
    ----------
    backend:
        ``"hf"`` (default, no extra dependencies), ``"vllm"`` (requires
        ``pip install vllm``), or ``"api"`` (requires ``pip install openai``
        and a running OpenAI-compatible completions server at ``api_url``).
    """
    return Rank1Reranker(
        model_path=model_name,
        backend=backend,
        api_url=api_url,
        api_model_name=api_model_name,
        context_size=context_size,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        num_gpus=num_gpus,
        precision=precision,
        device=device,
        top_k=top_k,
    )
