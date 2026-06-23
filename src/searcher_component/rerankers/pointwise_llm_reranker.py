"""LLM-based document reranking with structured output and self-consistency.

This module provides functionality for reranking retrieved documents using
LLM-based relevance scoring with self-consistency for improved reliability.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from reasoner_component import get_litellm_client, LiteLLMClient

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


class ResultScores(BaseModel):
    """Model representing a list of relevance scores for documents."""
    results: List[ResultScore] = Field(description="List of relevance scores for the documents.")


# ============================================================================
# LLM EVALUATOR CLASS
# ============================================================================

class LLMEvaluator:
    """A class to evaluate the relevance of documents based on LLM-generated scores.

    This evaluator uses structured output with Pydantic models and self-consistency
    (multiple runs with shuffling) to improve scoring reliability.
    """

    def __init__(self, model: LiteLLMClient, max_chars_per_document: int = 4096):
        """Initialize the LLMEvaluator with a model.

        Args:
            model: LiteLLM client for scoring
            max_chars_per_document: Maximum characters per document to send to LLM
        """
        self.model = model
        self.max_chars = max_chars_per_document

    def _get_prompt(self, query: str, data: List[Tuple[str, str]]) -> Tuple[str, Dict[str, str]]:
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
            documents.append({"id": i, "content": content[: self.max_chars]})

        # Create prompt
        docs_text = "\n\n".join([f"Document {d['id']}:\n{d['content']}" for d in documents])
        prompt = f"""Query: {query}

Documents to score:
{docs_text}

Score each document from 0-5 based on relevance to the query."""

        return prompt, id_pairs

    async def get_llm_scores( self, query: str, data: List[Tuple[str, str]], repetitions: int = 2, shuffle: bool = False, shuffle_seed: Optional[int] = None, ) -> Dict[str, float]:
        """Get LLM scores for all the results for a single query with self-consistency.

        Args:
            query: Query text
            data: List of (doc_id, content) tuples
            repetitions: Number of repetitions for self-consistency (default: 2)
            shuffle: Whether to shuffle documents for each repetition (default: False)
            shuffle_seed: Random seed for deterministic shuffling (only used if shuffle=True, None = random) (default: None)

        Returns:
            Dictionary mapping doc_id to averaged score
        """
        # Create random generator for deterministic shuffling if seed provided
        rng = random.Random(shuffle_seed) if shuffle_seed is not None else None

        async def single_run() -> Dict[str, ResultScore]:
            """Run a single evaluation of the LLM scores."""
            data_copy = data.copy()
            if shuffle:
                if rng is not None:
                    rng.shuffle(data_copy)
                else:
                    random.shuffle(data_copy)

            prompt, ids = self._get_prompt(query, data_copy)

            messages = [
                {"role": "system", "content": "You are an expert at scoring document relevance. Score each document from 0 (not relevant) to 5 (highly relevant) based on the query."},
                {"role": "user", "content": prompt},
            ]

            try:
                scores = await self.model.acomplete_with_structured_output(
                    messages=messages,
                    response_format=ResultScores,
                    metadata={"doc_id": ids, "query": query[:100]},
                )
            except Exception as e:
                logger.error(f"LLM scoring failed for query {query[:100]}: {e}")
                scores = ResultScores(results=[])

            return {ids[result.document_id]: result for result in scores.results if result.document_id in ids}

        # Execute all repetitions in parallel
        import asyncio
        all_scores = await asyncio.gather(*[single_run() for _ in range(repetitions)])

        # Aggregate results
        results: Dict[str, List[ResultScore]] = {}
        for scores in all_scores:
            for doc_id, result in scores.items():
                if doc_id not in results:
                    results[doc_id] = []
                results[doc_id].append(result)

        # Calculate the average score for each document
        final_scores: Dict[str, float] = {}
        for doc_id, result_list in results.items():
            score_list = [result.score for result in result_list]
            final_scores[doc_id] = sum(score_list) / len(score_list) if score_list else 0.0

        return final_scores


# ============================================================================
# SETUP FUNCTION
# ============================================================================

def setup_reranker( reranker_model: str, use_reranker: bool = True, temperature: float = 0.0, max_chars_per_document: int = 4096, metadata: Optional[Dict] = None, ) -> Optional[LLMEvaluator]:
    """Setup LLM-based reranker if enabled.

    Args:
        reranker_model: Name of the LLM model to use for reranking
        use_reranker: Whether to enable reranking (default: True)
        temperature: Temperature for LLM (default: 0.0 for consistent scoring)
        max_chars_per_document: Maximum characters per document (default: 4096)
        metadata: Optional metadata to attach to LLM calls

    Returns:
        LLMEvaluator instance if reranking is enabled, None otherwise
    """
    if not use_reranker:
        return None

    logger.info(f"Setting up reranker: {reranker_model}...")

    if metadata is None:
        metadata = {"model": reranker_model, "task": "reranking"}

    reranker_client = get_litellm_client(
        model_name=reranker_model,
        temperature=temperature,
        metadata=metadata,
    )

    reranker = LLMEvaluator(model=reranker_client, max_chars_per_document=max_chars_per_document)
    logger.info(f"Reranker configured: {reranker_model}")
    return reranker
