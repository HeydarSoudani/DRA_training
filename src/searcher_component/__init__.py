"""Searcher component: local retrievers, fusion, and the unified search tool."""

from .retriever import (
    BaseRetriever,
    BM25Retriever,
    SPLADERetriever,
    RerankRetriever,
    DenseRetriever,
    Encoder,
)
from .fusion import (
    interleaving_fusion,
    nested_interleaving_fusion,
    reciprocal_rank_fusion,
    simple_concatenation,
    fuse_results,
    fuse_retrieval_results,
    FUSION_METHODS,
)
from .normalize import normalize_retrieval_response

from .searcher import RetrievalSearchTool


__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "SPLADERetriever",
    "RerankRetriever",
    "DenseRetriever",
    "Encoder",
    "interleaving_fusion",
    "nested_interleaving_fusion",
    "reciprocal_rank_fusion",
    "simple_concatenation",
    "fuse_results",
    "fuse_retrieval_results",
    "FUSION_METHODS",
    "normalize_retrieval_response",
    "RetrievalSearchTool",
]
