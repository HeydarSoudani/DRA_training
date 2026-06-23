"""Multi-method fusion evaluation for retrieval.

Builds a fused ranking from per-section ranked lists (:mod:`.builders`) and
scores every aggregation fusion method against qrels (:mod:`.evaluate`).

Imported explicitly (``from evaluation.retrieval.fusion import ...``) rather than
re-exported from :mod:`evaluation` / :mod:`evaluation.retrieval`, so that the
base ``import evaluation`` stays free of the heavy ``searcher_component`` and
``indexing_corpus_dataset`` dependencies this subpackage pulls in.
"""

from .builders import (
    ALL_AGGREGATION_FUSION_METHODS,
    DIVERSITY_FUSION_METHODS,
    build_embeddings_dictionary,
    build_fusion_obj,
    build_ranking_with_fusion,
)
from .evaluate import (
    evaluate_all_fusions_and_save,
    run_fusion_eval,
)

__all__ = [
    "ALL_AGGREGATION_FUSION_METHODS",
    "DIVERSITY_FUSION_METHODS",
    "build_embeddings_dictionary",
    "build_fusion_obj",
    "build_ranking_with_fusion",
    "evaluate_all_fusions_and_save",
    "run_fusion_eval",
]
