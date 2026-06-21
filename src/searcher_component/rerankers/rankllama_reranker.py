"""RankLLaMA-based document reranking using the Tevatron inference library.

Uses the ``tevatron`` reranker modeling components (``RerankerModel``,
``RerankerInferenceCollator``) to perform pointwise relevance scoring of
query-document pairs.

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
Requires the ``tevatron`` package. Install it from source::

    pip install git+https://github.com/texttron/tevatron.git
"""

import logging
import threading
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from rank_llm.data import Candidate, Query, Request, Result

logger = logging.getLogger(__name__)

# Serialise model loading across threads: PyTorch CUDA device initialisation is
# not thread-safe when multiple threads target different devices simultaneously.
_MODEL_LOAD_LOCK = threading.Lock()

# Suppress noisy tokenizer/autocast warnings from tevatron internals.
warnings.filterwarnings(
    "ignore",
    message=".*with a fast tokenizer, using the `__call__` method is faster.*",
)


# ---------------------------------------------------------------------------
# Lazy tevatron import
# ---------------------------------------------------------------------------

def _import_tevatron():
    """Import tevatron reranker components, raising a clear error if missing."""
    try:
        from tevatron.reranker.arguments import DataArguments
        from tevatron.reranker.collator import RerankerInferenceCollator
        from tevatron.reranker.dataset import format_pair
        from tevatron.reranker.modeling import RerankerModel
        return DataArguments, RerankerInferenceCollator, format_pair, RerankerModel
    except ImportError as exc:
        raise ImportError(
            "RankLLaMAReranker requires the 'tevatron' package. "
            "Install it from source with:\n"
            "    pip install git+https://github.com/texttron/tevatron.git"
        ) from exc


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RankLLaMAReranker:
    """Pointwise reranker using RankLLaMA via the Tevatron inference library.

    Wraps the ``RankLlama`` implementation from the
    ChuanMeng/text-ranking-in-deep-research repository in the same dual-API
    interface as :class:`RankZephyrReranker` and :class:`RankGPTReranker`.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or local path.
        Default: ``castorini/rankllama-v1-7b-lora-passage``.
    tokenizer_name:
        Override for the tokenizer path (defaults to ``model_path``).
    lora_name_or_path:
        Optional LoRA adapter path (passed to ``RerankerModel.load``).
    batch_size:
        Inference batch size.
    rerank_max_len:
        Maximum token length for a query-document pair.
    query_prefix:
        String prepended to the query (default: ``"query: "``).
    passage_prefix:
        String prepended to each passage (default: ``"document: "``).
    append_eos_token:
        Whether to append EOS to each sequence (Tevatron ``DataArguments``).
    pad_to_multiple_of:
        Pad token lengths to a multiple of this value (Tevatron collator).
    fp16:
        Use float16 autocast during inference.
    device:
        ``"cuda"`` or ``"cpu"``. Falls back to CPU if CUDA is unavailable.
    top_k:
        Default number of top documents returned by the convenience ``rerank``
        wrapper when called with ``(query_str, documents, top_k=…)``.
    """

    def __init__( self, model_path: str = "castorini/rankllama-v1-7b-lora-passage", tokenizer_name: Optional[str] = None, lora_name_or_path: Optional[str] = None, batch_size: int = 8, rerank_max_len: int = 512, query_prefix: str = "query: ", passage_prefix: str = "document: ", append_eos_token: bool = False, pad_to_multiple_of: int = 16, fp16: bool = True, device: str = "cuda", top_k: int = 100, ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.fp16 = fp16
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k

        DataArguments, RerankerInferenceCollator, format_pair, RerankerModel = _import_tevatron()
        self._format_pair = format_pair
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

        tokenizer_source = tokenizer_name or model_path
        logger.info("Loading RankLLaMA tokenizer: %s", tokenizer_source)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except (TypeError, OSError, ValueError):
            # LoRA-only repos (e.g. castorini/rankllama-v1-7b-lora-passage) have
            # no tokenizer files. Use a public LLaMA tokenizer with identical vocabulary.
            fallback = "huggyllama/llama-7b"
            logger.warning(
                "No tokenizer files found in %s; using public tokenizer mirror %s.",
                tokenizer_source,
                fallback,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(fallback)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = 0
        self._tokenizer.padding_side = "right"

        # Auto-detect LoRA adapters: if model_path has an adapter_config.json
        # and no explicit lora_name_or_path was given, use model_path as the
        # LoRA adapter and load the declared base model instead.
        actual_model_path = model_path
        actual_lora_path = lora_name_or_path
        if lora_name_or_path is None:
            try:
                import json as _json
                from huggingface_hub import hf_hub_download
                _adapter_cfg_file = hf_hub_download(model_path, "adapter_config.json")
                with open(_adapter_cfg_file) as _f:
                    _adapter_cfg = _json.load(_f)
                _base = _adapter_cfg.get("base_model_name_or_path")
                if _base:
                    actual_model_path = _base
                    actual_lora_path = model_path
                    logger.info(
                        "Detected LoRA adapter at %s; loading base model %s + LoRA weights",
                        model_path, _base,
                    )
            except Exception:
                pass  # not a LoRA adapter repo, or hub unavailable

        logger.info("Loading RankLLaMA model: %s on device: %s", actual_model_path, self.device)
        with _MODEL_LOAD_LOCK:
            self._model = RerankerModel.load(
                actual_model_path,
                lora_name_or_path=actual_lora_path,
            )
            # castorini/rankllama-v1-7b-lora-passage saves score.weight inside
            # adapter_model.bin but has modules_to_save=null in the adapter
            # config, so PEFT never restores it.  Load it manually.
            if actual_lora_path is not None:
                self._patch_score_weight_from_adapter(actual_lora_path)
            self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("RankLLaMA model loaded")

        data_args = DataArguments(
            rerank_max_len=rerank_max_len,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            append_eos_token=append_eos_token,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self._collator = RerankerInferenceCollator(
            data_args=data_args, tokenizer=self._tokenizer
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_score_weight_from_adapter(self, lora_path: str) -> None:
        """Load score.weight from the adapter checkpoint if PEFT skipped it.

        Some LoRA checkpoints (e.g. castorini/rankllama-v1-7b-lora-passage)
        include ``score.weight`` in the adapter bin but set
        ``modules_to_save: null`` in ``adapter_config.json``, so PEFT never
        restores it and it stays randomly initialised.
        """
        hf_model = self._model.hf_model
        if not hasattr(hf_model, "score"):
            return

        from huggingface_hub import hf_hub_download
        adapter_weights = None
        for fname in ("adapter_model.safetensors", "adapter_model.bin"):
            try:
                path = hf_hub_download(lora_path, fname)
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    adapter_weights = load_file(path)
                else:
                    adapter_weights = torch.load(path, map_location="cpu", weights_only=True)
                break
            except Exception:
                continue

        if adapter_weights is None:
            return

        score_key = "base_model.model.score.weight"
        if score_key in adapter_weights:
            logger.info("Restoring score.weight from LoRA adapter checkpoint")
            hf_model.score.weight.data = adapter_weights[score_key].to(
                dtype=hf_model.score.weight.dtype,
                device=hf_model.score.weight.device,
            )

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
            A relevance score per passage (higher = more relevant).
        """
        if not texts:
            return []

        features = [
            (
                "q0",
                str(i),
                self._format_pair(query, text, "", self._query_prefix, self._passage_prefix),
            )
            for i, text in enumerate(texts)
        ]

        scores: List[float] = []
        for start in range(0, len(features), self.batch_size):
            batch_features = features[start : start + self.batch_size]
            _, _, batch = self._collator(batch_features)

            use_autocast = self.fp16 and self.device.startswith("cuda")
            ctx = torch.amp.autocast("cuda") if use_autocast else nullcontext()
            with ctx:
                with torch.no_grad():
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    model_output = self._model(batch)
                    batch_scores = model_output.scores.cpu().detach().float().numpy()

            for score_row in batch_scores:
                scores.append(float(score_row[0]))

        return scores

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
            # (to stay consistent with the rank_llm ecosystem convention)
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
        """Score documents, sort by score, return as List[Dict] with rank_score."""
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
            doc_out["rank_score"] = score   # raw reranker score (higher = more relevant)
            reranked_docs.append(doc_out)

        reranked_docs.extend(tail)
        return reranked_docs


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def setup_rankllama_reranker( model_name: str = "castorini/rankllama-v1-7b-lora-passage", device: str = "cuda", top_k: int = 100, batch_size: int = 8, rerank_max_len: int = 512, fp16: bool = True, lora_name_or_path: Optional[str] = None, ) -> RankLLaMAReranker:
    """Create and return a :class:`RankLLaMAReranker`.

    Parameter names intentionally match those used throughout the pipeline.

    Requires the ``tevatron`` package::

        pip install git+https://github.com/texttron/tevatron.git
    """
    return RankLLaMAReranker(
        model_path=model_name,
        lora_name_or_path=lora_name_or_path,
        batch_size=batch_size,
        rerank_max_len=rerank_max_len,
        fp16=fp16,
        device=device,
        top_k=top_k,
    )
