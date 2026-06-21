"""LLM-based document reranking with batched processing and self-consistency.

This module provides functionality for reranking retrieved documents using
LLM-based relevance scoring with batching support and self-consistency for
improved reliability. It supports shuffling documents before and within batches,
and running multiple repetitions for self-consistency.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from utils.llm_client import LiteLLMClient, get_litellm_client
from searcher_component.rerankers.prompts.relevance_scoring import (
    RERANK_RELEVANCE_SCORING_SYSTEM_PROMPT,
    get_rerank_relevance_scoring_user_prompt,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================


class ResultScore(BaseModel):
    """Model representing the relevance score of a document."""

    document_id: str = Field(description="The ID of the document being scored.")
    score: int = Field(
        description="The relevance score assigned to the document based on the query.",
        ge=0,
        le=5,
    )


class ResultScoreWithExplanation(BaseModel):
    """Model representing the relevance score of a document with explanation."""

    document_id: str = Field(description="The ID of the document being scored.")
    explanation: str = Field(
        description="A brief explanation for the assigned score in 1-2 sentences. "
        "Show factual information about the content only. "
        "Do not include the relevance label or score you give."
    )
    score: int = Field(
        description="The relevance score assigned to the document based on the query.",
        ge=0,
        le=5,
    )


class ResultScores(BaseModel):
    """Model representing a list of relevance scores for documents."""

    results: List[ResultScore] = Field(
        description="List of relevance scores for the documents."
    )


class ResultScoresWithExplanation(BaseModel):
    """Model representing a list of relevance scores for documents."""

    results: List[ResultScoreWithExplanation] = Field(
        description="List of relevance scores for the documents."
    )


class Result(BaseModel):
    """Relevance score of a document with optional explanation."""

    document_id: str
    explanation: str
    score: float


# ============================================================================
# LLM EVALUATOR CLASS WITH BATCHED PROCESSING
# ============================================================================


class BatchedLLMEvaluator:
    """A class to evaluate the relevance of documents using batched LLM scoring.

    This evaluator supports:
    - Batched processing of documents for better control over context size
    - Self-consistency through multiple runs with shuffling
    - Optional explanations for scores
    - Jinja2 templates for flexible prompt formatting
    """

    def __init__( self, model: LiteLLMClient, thinking: str = "", reasoning: bool = False, template_path: Path = Path("./"), template_name: str = "relevance_scoring.jinja", max_chars_per_document: int = 4096, ):
        """Initialize the BatchedLLMEvaluator with a model.

        Args:
            model: LiteLLM client for scoring
            thinking: Additional thinking instructions to append to system prompt
            reasoning: Whether to request explanations for scores
            template_path: Path to Jinja2 template directory
            template_name: Name of the Jinja2 template file
            max_chars_per_document: Maximum characters per document to send to LLM
        """
        self.model = model
        self.max_chars = max_chars_per_document
        self.template_path = template_path
        self.template_name = template_name
        self.thinking = thinking
        self.reasoning = reasoning

    def _get_prompt( self, query: str, data: List[Tuple[str, str]] ) -> Tuple[str, Dict[str, str]]:
        """Generate a prompt for the LLM based on the query and search results.

        Args:
            query: Query text
            data: List of (doc_id, content) tuples

        Returns:
            Tuple of (prompt, id_mapping) where id_mapping maps numeric IDs to doc_ids
        """
        id_pairs = {}
        documents = []

        for i, (doc_id, content) in enumerate(data):
            id_pairs[str(i)] = doc_id
            documents.append({"id": str(i), "content": content[: self.max_chars]})

        # Use Python prompts from rerank_relevance_scoring_prompts.py
        prompt = get_rerank_relevance_scoring_user_prompt(query, documents)

        return prompt, id_pairs

    def _render_template( self, params: Dict[str, Any], prompt_template_string: Optional[str] = None ) -> str:
        """Render a Jinja template with the given parameters.

        Args:
            params: Parameters to pass to the template
            prompt_template_string: Optional template string (if None, loads from file)

        Returns:
            Rendered template string
        """
        if prompt_template_string is None:
            environment = Environment(
                loader=FileSystemLoader(searchpath=self.template_path, encoding="utf-8"),
                undefined=StrictUndefined,
                keep_trailing_newline=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = environment.get_template(self.template_name)
            return template.render(params)
        else:
            template = Environment().from_string(prompt_template_string)
            return template.render(params)

    async def _get_llm_scores_single_batch( self, query: str, data: List[Tuple[str, str]], metadata: Optional[Dict[str, Any]] = None, ) -> Tuple[Dict[str, ResultScore | ResultScoreWithExplanation], bool]:
        """Get LLM scores for a single batch of documents.

        Args:
            query: Query text
            data: List of (doc_id, content) tuples
            metadata: Optional metadata for tracking

        Returns:
            Tuple of (scores_dict, is_skipped) where is_skipped is True if query was skipped
        """
        prompt, ids = self._get_prompt(query, data)
        messages = [
            {
                "role": "system",
                "content": RERANK_RELEVANCE_SCORING_SYSTEM_PROMPT + (self.thinking or ""),
            },
            {"role": "user", "content": prompt},
        ]

        if metadata is None:
            metadata = {}
        metadata.update({"doc_id": ids, "query": query[:100]})

        is_skipped = False
        try:
            scores = await self.model.acomplete_with_structured_output(
                messages=messages,
                response_format=(
                    ResultScores if not self.reasoning else ResultScoresWithExplanation
                ),
                metadata=metadata,
            )
        except RuntimeError as e:
            # Check if this is a skipped query error
            if "Query skipped" in str(e):
                is_skipped = True
                logger.error(f"Query skipped after retries: {query[:100]}: {e}")
            else:
                logger.error(f"LLM scoring failed for query {query[:100]}: {e}")
            scores = ResultScores(results=[])
        except Exception as e:
            # Log error during structured output parsing
            logger.error(f"LLM scoring failed for query {query[:100]}: {e}")
            scores = ResultScores(results=[])

        return (
            {
                ids[result.document_id]: result
                for result in scores.results
                if result.document_id in ids
            },
            is_skipped,
        )

    async def _get_llm_scores_in_batches( self, query: str, batched_data: List[Tuple[List[Tuple[str, str]], Optional[Dict[str, Any]]]], ) -> Tuple[Dict[str, List[ResultScore | ResultScoreWithExplanation]], bool]:
        """Get LLM scores for multiple batches in parallel.

        Args:
            query: Query text
            batched_data: List of (batch_data, metadata) tuples

        Returns:
            Tuple of (results_dict, is_skipped) where is_skipped is True if any batch was skipped
        """
        results: Dict[str, List[ResultScore | ResultScoreWithExplanation]] = {}
        is_skipped = False

        # Execute all batches in parallel
        batched_results = await asyncio.gather(
            *[
                self._get_llm_scores_single_batch(query, data, metadata)
                for data, metadata in batched_data
            ]
        )

        for batch_result, batch_skipped in batched_results:
            if batch_skipped:
                is_skipped = True
            for doc_id, result in batch_result.items():
                if doc_id not in results:
                    results[doc_id] = []
                results[doc_id].append(result)

        return results, is_skipped

    async def get_llm_scores( self, query: str, data: List[Tuple[str, str]], number_of_runs: int = 2, number_of_batches: int = 1, shuffle: bool = False, shuffle_seed_before_batch: Optional[int] = None, shuffle_seed_within_batch: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None, ) -> Tuple[Dict[str, Result], bool]:
        """Get LLM scores for all documents with batching and self-consistency.

        Args:
            query: Query text
            data: List of (doc_id, content) tuples
            number_of_runs: Number of repetitions for self-consistency
            number_of_batches: Number of batches to split documents into
            shuffle: Enable shuffling for self-consistency (default: False)
            shuffle_seed_before_batch: Seed for shuffling documents before batching (only used if shuffle=True, None = random)
            shuffle_seed_within_batch: Seed for shuffling documents within each batch (only used if shuffle=True, None = random)
            metadata: Optional metadata for tracking

        Returns:
            Tuple of (scores_dict, is_skipped) where is_skipped is True if any query was skipped
        """
        doc_ids = [doc_id for doc_id, _ in data]
        doc2txt = {doc_id: prompt for doc_id, prompt in data}

        tasks = []

        rand_before_batch = random.Random(shuffle_seed_before_batch or 0)
        rand_within_batch = random.Random(shuffle_seed_within_batch or 0)

        # Create batches and runs
        for run_index in range(number_of_runs):
            # Shuffle all documents before creating batches if shuffle enabled
            if shuffle:
                if shuffle_seed_before_batch is not None:
                    rand_before_batch.shuffle(doc_ids)
                else:
                    random.shuffle(doc_ids)

            # Split into batches
            batch_size = len(doc_ids) // number_of_batches
            for batch_index in range(number_of_batches):
                start_idx = batch_index * batch_size
                end_idx = (batch_index + 1) * batch_size if batch_index < number_of_batches - 1 else len(doc_ids)
                batch_doc_ids = doc_ids[start_idx:end_idx]

                # Shuffle within batch if shuffle enabled
                if shuffle:
                    if shuffle_seed_within_batch is not None:
                        rand_within_batch.shuffle(batch_doc_ids)
                    else:
                        random.shuffle(batch_doc_ids)

                metadata_copy = metadata.copy() if metadata else {}
                metadata_copy.update(
                    {
                        "run_index": run_index,
                        "batch_index": batch_index,
                        "generation_name": f"Run {run_index + 1}/{number_of_runs} - Batch {batch_index + 1}/{number_of_batches}",
                    }
                )

                batched_data = [(doc_id, doc2txt[doc_id]) for doc_id in batch_doc_ids]
                tasks.append((batched_data, metadata_copy))

        # Execute all batches across all runs
        results, is_skipped = await self._get_llm_scores_in_batches(query, tasks)

        # Calculate the average score for each document
        final_scores: Dict[str, Result] = {}
        for doc_id, result_list in results.items():
            score_list = [result.score for result in result_list]
            final_score = sum(score_list) / len(score_list)

            explanation = ""
            for result in result_list:
                if isinstance(result, ResultScoreWithExplanation):
                    explanation = result.explanation
                    break

            final_scores[doc_id] = Result(
                document_id=doc_id,
                explanation=explanation,
                score=final_score,
            )

        return final_scores, is_skipped

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank documents by LLM relevance scoring.

        Synchronous wrapper that matches the interface of other rerankers
        (e.g. RankLLaMAReranker, ListwiseVLLMReranker).

        Args:
            query: Query text
            documents: List of document dicts with at least ``doc_id``/``id``
                       and ``text``/``relevant_text`` keys.
            top_k: Optional limit on number of results to return.

        Returns:
            Documents sorted by descending relevance score, each annotated
            with a ``rank_score`` key.
        """
        if not documents:
            return []

        # Convert dicts to (doc_id, content) tuples
        data: List[Tuple[str, str]] = []
        for doc in documents:
            doc_id = doc.get("doc_id") or doc.get("id") or ""
            content = doc.get("text") or doc.get("relevant_text") or doc.get("contents") or ""
            data.append((str(doc_id), str(content)))

        # Run the async scoring in a sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                scores_dict, _ = pool.submit(
                    asyncio.run, self.get_llm_scores(query, data, number_of_runs=1)
                ).result()
        else:
            scores_dict, _ = asyncio.run(self.get_llm_scores(query, data, number_of_runs=1))

        # Build doc_id -> score mapping
        id_to_score: Dict[str, float] = {}
        for doc_id, result in scores_dict.items():
            id_to_score[doc_id] = result.score

        # Annotate and sort
        for doc in documents:
            doc_id = str(doc.get("doc_id") or doc.get("id") or "")
            doc["rank_score"] = id_to_score.get(doc_id, 0.0)

        ranked = sorted(documents, key=lambda d: d.get("rank_score", 0.0), reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    async def get_scores( self, query: str, data: List[Tuple[str, str]], *, number_of_runs: int = 2, number_of_batches: int = 1, shuffle: bool = False, shuffle_seed_before_batch: Optional[int] = None, shuffle_seed_within_batch: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None, **_, ) -> Dict[str, float | bool]:
        """Get LLM scores for all documents (wrapper method for pipeline compatibility).

        Args:
            query: Query text
            data: List of (doc_id, content) tuples
            number_of_runs: Number of repetitions for self-consistency
            number_of_batches: Number of batches to split documents into
            shuffle: Enable shuffling for self-consistency
            shuffle_seed_before_batch: Seed for shuffling documents before batching
            shuffle_seed_within_batch: Seed for shuffling documents within each batch
            metadata: Optional metadata for tracking
            **_: Additional kwargs (ignored)

        Returns:
            Dictionary with doc_id -> score mappings, plus a special key '__is_skipped'
            indicating if query was skipped
        """
        results, is_skipped = await self.get_llm_scores(
            query,
            data,
            number_of_runs,
            number_of_batches,
            shuffle,
            shuffle_seed_before_batch,
            shuffle_seed_within_batch,
            metadata,
        )
        scores = {doc_id: result.score for doc_id, result in results.items()}
        scores["__is_skipped"] = is_skipped
        return scores


# ============================================================================
# SETUP FUNCTION
# ============================================================================


def setup_batched_reranker( reranker_model: str, use_reranker: bool = True, temperature: float = 0.0, max_chars_per_document: int = 4096, thinking: str = "", reasoning: bool = False, template_path: Path = Path("./"), template_name: str = "relevance_scoring.jinja", metadata: Optional[Dict] = None, ) -> Optional[BatchedLLMEvaluator]:
    """Setup batched LLM-based reranker if enabled.

    Args:
        reranker_model: Name of the LLM model to use for reranking
        use_reranker: Whether to enable reranking (default: True)
        temperature: Temperature for LLM (default: 0.0 for consistent scoring)
        max_chars_per_document: Maximum characters per document (default: 4096)
        thinking: Additional thinking instructions to append to system prompt
        reasoning: Whether to request explanations for scores
        template_path: Path to Jinja2 template directory
        template_name: Name of the Jinja2 template file
        metadata: Optional metadata to attach to LLM calls

    Returns:
        BatchedLLMEvaluator instance if reranking is enabled, None otherwise
    """
    if not use_reranker:
        return None

    logger.info(f"Setting up batched reranker: {reranker_model}...")

    if metadata is None:
        metadata = {"model": reranker_model, "task": "reranking"}

    reranker_client = get_litellm_client(
        model_name=reranker_model,
        temperature=temperature,
        metadata=metadata,
    )

    reranker = BatchedLLMEvaluator(
        model=reranker_client,
        thinking=thinking,
        reasoning=reasoning,
        template_path=template_path,
        template_name=template_name,
        max_chars_per_document=max_chars_per_document,
    )
    logger.info(f"Batched reranker configured: {reranker_model}")
    return reranker
