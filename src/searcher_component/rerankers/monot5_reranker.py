"""MonoT5 document reranking using a T5 sequence-to-sequence model.

Implements pointwise reranking using the ``castorini/monot5-*`` family of
models (or any compatible T5 / mT5 checkpoint).  The model is prompted with
``"Query: {q} Document: {d} Relevant:"`` and the relevance score is
P("true") derived from the decoder logits over the ``"true"`` / ``"false"``
tokens.

Two model families are supported:

* ``T5ForConditionalGeneration``  – standard English monoT5 models.
* ``MT5ForConditionalGeneration`` – multilingual mT5 models (set
  ``use_mt5=True``).

Unlike the listwise RankZephyr / RankGPT rerankers, this is a **pointwise**
reranker: each (query, document) pair is scored independently, and documents
are then sorted by descending score.

Public API
----------
score_query(query, texts)                          ->  List[float]
rerank_batch(requests, rank_start, rank_end, ...)  ->  List[Result]
rerank(request, rank_start, rank_end, ...)         ->  Result
rerank(query_str, documents, top_k)               ->  List[Dict]   (convenience)

Data types ``Request``, ``Result``, ``Query``, ``Candidate`` are imported
directly from ``rank_llm.data`` so they stay fully compatible with the rest
of the rank_llm ecosystem.

Dependencies
------------
Only ``transformers`` and ``torch`` are required (both already present in
this project).

Adapted from the ``MonoT5`` class in
ChuanMeng/text-ranking-in-deep-research (searcher/rerankers/monot5.py).
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

from rank_llm.data import Candidate, Query, Request, Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MonoT5Reranker:
    """Pointwise reranker using a MonoT5 (or mT5) sequence-to-sequence model.

    The model receives the prompt
    ``"Query: {query} Document: {passage} Relevant:"``
    and the relevance score is P("true") normalised over
    P("true") + P("false") from the decoder's first output token logits.

    Adapted from the ``MonoT5`` class in
    ChuanMeng/text-ranking-in-deep-research, wrapped in the same dual-API
    interface as :class:`RankLLaMAReranker` and :class:`Rank1Reranker`.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or local path.
        Default: ``"castorini/monot5-base-msmarco"``.
    tokenizer_name:
        Override for the tokenizer path (defaults to ``model_path``).
    use_mt5:
        Set to ``True`` to load an ``MT5ForConditionalGeneration`` model
        instead of ``T5ForConditionalGeneration``.
    batch_size:
        Inference batch size.
    device:
        ``"cuda"`` or ``"cpu"``. Falls back to CPU if CUDA is unavailable.
    top_k:
        Default number of top documents returned by the convenience
        ``rerank`` wrapper when called with ``(query_str, documents)``.
    """

    def __init__( self, model_path: str = "castorini/monot5-base-msmarco", tokenizer_name: Optional[str] = None, use_mt5: bool = False, batch_size: int = 32, device: str = "cuda", top_k: int = 100, ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k

        tok_name = tokenizer_name or model_path
        logger.info("Loading MonoT5 tokenizer: %s", tok_name)
        self._tokenizer = T5Tokenizer.from_pretrained(tok_name)

        if use_mt5:
            from transformers import MT5ForConditionalGeneration
            model_cls = MT5ForConditionalGeneration
        else:
            model_cls = T5ForConditionalGeneration

        logger.info("Loading MonoT5 model: %s on device: %s", model_path, self.device)
        self._model = model_cls.from_pretrained(model_path)
        self._model.to(self.device)
        self._model.eval()
        logger.info("MonoT5 model loaded")

        # Token IDs for the relevance decision tokens
        self._REL = self._tokenizer.encode("true")[0]
        self._NREL = self._tokenizer.encode("false")[0]

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
            P("true") relevance score per passage in [0, 1]
            (higher = more relevant).
        """
        if not texts:
            return []

        scores: List[float] = []

        # Pre-encode the decoder suffix ("Relevant:") once so we can append
        # it to every encoder input — this is the upstream monoT5 trick.
        suffix_batch = self._tokenizer.batch_encode_plus(
            ["Relevant:" for _ in range(self.batch_size)],
            return_tensors="pt",
            padding="longest",
        )
        max_vlen = (
            self._model.config.n_positions - suffix_batch["input_ids"].shape[1]
        )

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            actual_bs = len(batch_texts)

            enc = self._tokenizer.batch_encode_plus(
                [f"Query: {query} Document: {d}" for d in batch_texts],
                return_tensors="pt",
                padding="longest",
            )

            # Trim to fit within the model's position budget, then append suffix
            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1]          # drop trailing EOS
                enc_value = enc_value[:, :max_vlen]    # clip to budget
                enc[key] = torch.cat(
                    [enc_value, suffix_batch[key][:actual_bs]], dim=1
                )

            enc["decoder_input_ids"] = torch.full(
                (actual_bs, 1),
                self._model.config.decoder_start_token_id,
                dtype=torch.long,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self._model(**enc).logits  # (B, 1, vocab)

            rel_logits = logits[:, 0, [self._REL, self._NREL]]  # (B, 2)
            probs = F.softmax(rel_logits, dim=1)                # normalise
            batch_scores = probs[:, 0].cpu().tolist()           # P("true")
            scores.extend(batch_scores)

        return scores

    # ------------------------------------------------------------------
    # rank_llm-compatible core API
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

        **rank_llm-compatible**::

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
            doc_out["rank_score"] = score  # P("true") in [0, 1]
            reranked_docs.append(doc_out)

        reranked_docs.extend(tail)
        return reranked_docs


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def setup_monot5_reranker( model_name: str = "castorini/monot5-base-msmarco", tokenizer_name: Optional[str] = None, use_mt5: bool = False, batch_size: int = 32, device: str = "cuda", top_k: int = 100, ) -> MonoT5Reranker:
    """Create and return a :class:`MonoT5Reranker`.

    Parameter names intentionally match those used throughout the pipeline.

    Common ``model_name`` values
    ----------------------------
    * ``"castorini/monot5-base-msmarco"``    – base (250 M params, English)
    * ``"castorini/monot5-large-msmarco"``   – large (800 M params, English)
    * ``"castorini/monot5-3b-msmarco"``      – 3B params, English
    * ``"castorini/mt5-base-msmarco"``       – multilingual base (use_mt5=True)
    """
    return MonoT5Reranker(
        model_path=model_name,
        tokenizer_name=tokenizer_name,
        use_mt5=use_mt5,
        batch_size=batch_size,
        device=device,
        top_k=top_k,
    )
