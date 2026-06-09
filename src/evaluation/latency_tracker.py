"""Per-component latency tracking for retrieval pipelines.

Provides a lightweight context-manager–based API for measuring wall-clock
time of individual pipeline steps on a per-query basis, then aggregating
the results across all queries.

Typical usage inside a pipeline runner::

    tracker = LatencyTracker()

    for qid, query_text in queries.items():
        with tracker.track(qid, "outline_generation"):
            outline = outline_gen.generate(qid, query_text)

        for i, sq in enumerate(outline):
            with tracker.track(qid, "retrieval"):
                docs = retriever.retrieve(sq)
            with tracker.track(qid, "post_retrieval_rerank"):
                docs = reranker.rerank(sq, docs)

    with tracker.track_global("fusion"):
        fused = fuse(results)

    summary = tracker.summary()
"""

import time
import statistics
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Dict, List, Optional


class LatencyTracker:
    """Track per-component, per-query wall-clock latency.

    Each ``track(qid, component)`` call records a single timing sample for
    *component* under *qid*.  ``track_global(component)`` records a timing
    that is not tied to a specific query (e.g. fusion across all queries).

    Samples are stored so that both per-query breakdowns and corpus-level
    aggregates (mean / median / std / total) can be reported.
    """

    def __init__(self) -> None:
        # {qid: {component: [elapsed_seconds, ...]}}
        self._per_query: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # {component: [elapsed_seconds, ...]}  — query-independent measurements
        self._global: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    @contextmanager
    def track(self, query_id: str, component: str):
        """Context manager that times a block and records it for *query_id*."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._per_query[query_id][component].append(elapsed)

    @contextmanager
    def track_global(self, component: str):
        """Context manager that times a block not tied to a specific query."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._global[component].append(elapsed)

    def record(self, query_id: str, component: str, elapsed: float) -> None:
        """Manually record an elapsed time (seconds) for *query_id*."""
        self._per_query[query_id][component].append(elapsed)

    def record_global(self, component: str, elapsed: float) -> None:
        """Manually record an elapsed time (seconds) not tied to a query."""
        self._global[component].append(elapsed)

    # ------------------------------------------------------------------
    # Query-level helpers
    # ------------------------------------------------------------------

    def query_total(self, query_id: str) -> float:
        """Total latency recorded for a single query (all components)."""
        return sum(
            sum(times) for times in self._per_query[query_id].values()
        )

    def query_breakdown(self, query_id: str) -> Dict[str, float]:
        """Per-component total latency for a single query."""
        return {
            comp: round(sum(times), 6)
            for comp, times in self._per_query[query_id].items()
        }

    # ------------------------------------------------------------------
    # Corpus-level aggregation
    # ------------------------------------------------------------------

    @property
    def components(self) -> List[str]:
        """All component names that were tracked (per-query + global)."""
        comps: set = set()
        for qid_data in self._per_query.values():
            comps.update(qid_data.keys())
        comps.update(self._global.keys())
        return sorted(comps)

    @property
    def query_ids(self) -> List[str]:
        return sorted(self._per_query.keys())

    @property
    def num_queries(self) -> int:
        return len(self._per_query)

    def _aggregate(self, values: List[float]) -> Dict[str, float]:
        """Compute summary statistics for a list of timing values."""
        if not values:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "total": 0.0, "count": 0}
        return {
            "mean": round(statistics.mean(values), 6),
            "median": round(statistics.median(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "total": round(sum(values), 6),
            "count": len(values),
        }

    def summary(self) -> Dict[str, Any]:
        """Return aggregate latency statistics (for summary.json).

        Structure::

            {
                "num_queries": int,
                "per_component": {
                    "<component>": {
                        "per_query": {"mean": ..., "median": ..., "std": ..., "total": ..., "count": ...},
                        "global": {"mean": ..., ...}   # only if track_global was used
                    },
                    ...
                },
                "per_query_total": {"mean": ..., "median": ..., "std": ..., "total": ..., "count": ...},
            }
        """
        result: Dict[str, Any] = {"num_queries": self.num_queries}
        per_component: Dict[str, Any] = {}

        # Per-query components: for each component, collect the per-query
        # totals (sum of all samples within a single query) across queries.
        all_per_query_components: set = set()
        for qid_data in self._per_query.values():
            all_per_query_components.update(qid_data.keys())

        for comp in sorted(all_per_query_components):
            # Per-query total for this component (one value per query)
            per_query_totals = [
                sum(self._per_query[qid].get(comp, []))
                for qid in self._per_query
                if comp in self._per_query[qid]
            ]
            entry: Dict[str, Any] = {"per_query": self._aggregate(per_query_totals)}
            if comp in self._global:
                entry["global"] = self._aggregate(self._global[comp])
            per_component[comp] = entry

        # Global-only components (not tracked per-query)
        for comp in sorted(self._global.keys()):
            if comp not in per_component:
                per_component[comp] = {"global": self._aggregate(self._global[comp])}

        result["per_component"] = per_component

        # Per-query total (sum across all components for each query)
        query_totals = [self.query_total(qid) for qid in self._per_query]
        result["per_query_total"] = self._aggregate(query_totals)

        return result

    def per_query_details(self) -> Dict[str, Dict[str, float]]:
        """Return raw per-query, per-component latencies (for a separate file).

        Structure::

            {
                "<query_id>": {
                    "<component>": <total_seconds>,
                    ...
                    "_total": <sum_of_all_components>
                },
                ...
            }
        """
        details: Dict[str, Dict[str, float]] = {}
        for qid in sorted(self._per_query.keys()):
            breakdown = self.query_breakdown(qid)
            breakdown["_total"] = round(sum(breakdown.values()), 6)
            details[qid] = breakdown
        return details
