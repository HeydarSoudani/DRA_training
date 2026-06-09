"""Per-search-step retrieval analysis.

Reads per-query TREC files from a ``retrieval/`` directory, groups documents
by search step using the 6th column (``iter_N``), and computes:

1. **Recall per search step** — each step evaluated independently against qrels.
2. **Marginal recall per search step** — sequential evaluation where each step
   is evaluated only on gold docs not yet found by earlier steps.  Produces
   MarginalCoverage@K and CumulativeRecall@K.

Results are saved as JSON and visualised as PNG/PDF plots in a
``per_search_analysis/`` output directory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .retrieval_evaluation import compute_trec_metrics

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# TREC file parsing
# ────────────────────────────────────────────────────────────────────────────

def _parse_trec_file(path: Path) -> List[Dict[str, Any]]:
    """Parse a single TREC file into a list of record dicts.

    Each record has keys: qid, doc_id, rank, score, iter_tag.
    """
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            records.append({
                "qid": parts[0],
                "doc_id": parts[2],
                "rank": int(parts[3]),
                "score": float(parts[4]),
                "iter_tag": parts[5],
            })
    return records


def _iter_index(tag: str) -> int:
    """Extract the numeric step index from an iter tag like ``iter_3``.

    Falls back to 0 when the tag cannot be parsed.
    """
    try:
        return int(tag.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return 0


def load_retrieval_dir(retrieval_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load all per-query TREC files and group docs by (query_id, step).

    Returns:
        Nested dict ``{query_id: {step_idx: {doc_id: score}}}``.
        The query_id is derived from the file stem (e.g. ``PL-500``).
    """
    retrieval_dir = Path(retrieval_dir)
    data: Dict[str, Dict[int, Dict[str, float]]] = {}
    for trec_path in sorted(retrieval_dir.glob("*.trec")):
        qid = trec_path.stem
        records = _parse_trec_file(trec_path)
        steps: Dict[int, Dict[str, float]] = {}
        for rec in records:
            step = _iter_index(rec["iter_tag"])
            step_docs = steps.setdefault(step, {})
            # keep first occurrence (highest rank) per doc per step
            if rec["doc_id"] not in step_docs:
                step_docs[rec["doc_id"]] = rec["score"]
        if steps:
            data[qid] = steps
    return data


# ────────────────────────────────────────────────────────────────────────────
# Analysis class
# ────────────────────────────────────────────────────────────────────────────

class PerSearchAnalysis:
    """Compute and plot per-search-step retrieval metrics.

    Parameters
    ----------
    qrels : dict
        Ground-truth relevance judgements ``{query_id: {doc_id: score}}``.
    k_values : list[int], optional
        Recall cutoffs (default ``[1, 3, 5, 10, 25, 100, 500, 1000]``).
    """

    DEFAULT_K_VALUES = [1, 3, 5, 10, 25, 100, 500, 1000]

    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        k_values: Optional[List[int]] = None,
    ) -> None:
        self.qrels = qrels
        self.k_values = k_values or self.DEFAULT_K_VALUES

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, retrieval_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Run the full analysis: load, compute, plot, save.

        Parameters
        ----------
        retrieval_dir : Path
            Directory containing per-query ``.trec`` files.
        output_dir : Path
            Directory where ``per_search_analysis/`` will be created.

        Returns
        -------
        dict
            ``{"retrieval_by_step": …, "retrieval_marginal_per_step": …}``
        """
        retrieval_dir = Path(retrieval_dir)
        output_dir = Path(output_dir)
        data = load_retrieval_dir(retrieval_dir)
        if not data:
            logger.warning("No TREC files found in %s — skipping per-search analysis", retrieval_dir)
            return {}

        by_step = self._recall_per_step(data)
        marginal = self._marginal_per_step(data)

        out = output_dir / "per_search_analysis"
        out.mkdir(parents=True, exist_ok=True)

        # Save JSON
        payload = {
            "retrieval_by_step": by_step,
            "retrieval_marginal_per_step": marginal,
        }
        with open(out / "per_search_analysis.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Plots
        self._plot(by_step, out, plot_type="retrieval_by_step")
        self._plot(marginal, out, plot_type="retrieval_marginal_per_step")

        logger.info("Per-search analysis saved to %s", out)
        return payload

    # ------------------------------------------------------------------ #
    #  Recall per step (independent)                                       #
    # ------------------------------------------------------------------ #

    def _recall_per_step(
        self, data: Dict[str, Dict[int, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate each search step independently against qrels."""
        # Collect all step indices across queries
        all_steps: set = set()
        for steps in data.values():
            all_steps.update(steps.keys())

        result: Dict[str, Dict[str, Any]] = {}
        for step_idx in sorted(all_steps):
            # Build search_results & qrels subsets for this step
            search_results: Dict[str, Dict[str, float]] = {}
            step_qrels: Dict[str, Dict[str, int]] = {}
            for qid, steps in data.items():
                if step_idx not in steps:
                    continue
                search_results[qid] = steps[step_idx]
                if qid in self.qrels:
                    step_qrels[qid] = self.qrels[qid]

            if not search_results or not step_qrels:
                continue

            ndcg, _map, recall, precision, f1, success = compute_trec_metrics(
                qrels=step_qrels,
                results=search_results,
                k_values=self.k_values,
                ignore_identical_ids=False,
                include_all_metrics=False,
            )
            result[f"step_{step_idx}"] = {
                "NDCG": ndcg, "MAP": _map, "Recall": recall,
                "Precision": precision, "F1": f1, "Success": success,
            }
        return result

    # ------------------------------------------------------------------ #
    #  Marginal recall per step (sequential)                               #
    # ------------------------------------------------------------------ #

    def _marginal_per_step(
        self, data: Dict[str, Dict[int, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Sequential evaluation — each step only on gold docs not yet found."""
        all_steps: set = set()
        for steps in data.values():
            all_steps.update(steps.keys())
        sorted_steps = sorted(all_steps)

        total_gold_per_qid: Dict[str, int] = {
            qid: len(qrel) for qid, qrel in self.qrels.items()
        }

        # Track already-found gold docs per k independently
        already_found_per_k: Dict[int, Dict[str, set]] = {k: {} for k in self.k_values}

        result: Dict[str, Dict[str, Any]] = {}
        for step_idx in sorted_steps:
            step_label = f"step_{step_idx}"
            step_entry: Dict[str, Any] = {}

            # Collect docs for this step per query
            step_docs_per_qid: Dict[str, Dict[str, float]] = {}
            for qid, steps in data.items():
                if step_idx in steps:
                    step_docs_per_qid[qid] = steps[step_idx]

            if not step_docs_per_qid:
                continue

            for k in self.k_values:
                already_found = already_found_per_k[k]

                # Build modified qrels: remove gold docs already found at this k
                modified_qrels: Dict[str, Dict[str, int]] = {}
                search_results: Dict[str, Dict[str, float]] = {}
                for qid, docs in step_docs_per_qid.items():
                    if qid not in self.qrels:
                        continue
                    found_so_far = already_found.get(qid, set())
                    modified_qrels[qid] = {
                        doc_id: score
                        for doc_id, score in self.qrels[qid].items()
                        if doc_id not in found_so_far
                    }
                    search_results[qid] = docs

                if not modified_qrels:
                    continue

                ndcg, _map, recall, precision, f1, success = compute_trec_metrics(
                    qrels=modified_qrels,
                    results=search_results,
                    k_values=[k],
                    ignore_identical_ids=False,
                    include_all_metrics=False,
                )
                for metric_name, metric_value in {
                    "NDCG": ndcg, "MAP": _map, "Recall": recall,
                    "Precision": precision, "F1": f1, "Success": success,
                }.items():
                    if isinstance(metric_value, dict):
                        step_entry.setdefault(metric_name, {}).update(metric_value)

                # Coverage metrics
                marginal_scores: List[float] = []
                cumulative_scores: List[float] = []
                for qid, docs in step_docs_per_qid.items():
                    if qid not in self.qrels:
                        continue
                    gold_set = set(self.qrels[qid].keys())
                    total_gold = total_gold_per_qid.get(qid, 0)
                    if total_gold == 0:
                        continue
                    found_before = already_found.get(qid, set())
                    remaining_gold = gold_set - found_before
                    # top-k docs by score
                    sorted_doc_ids = sorted(docs, key=docs.get, reverse=True)[:k]
                    top_k_set = set(sorted_doc_ids)
                    new_gold = remaining_gold & top_k_set
                    if remaining_gold:
                        marginal_scores.append(len(new_gold) / len(remaining_gold))
                    cumulative_after = found_before | new_gold
                    cumulative_scores.append(len(cumulative_after) / total_gold)

                step_entry.setdefault("MarginalCoverage", {})[f"MarginalCoverage@{k}"] = round(
                    sum(marginal_scores) / len(marginal_scores) if marginal_scores else 0.0, 5,
                )
                step_entry.setdefault("CumulativeRecall", {})[f"CumulativeRecall@{k}"] = round(
                    sum(cumulative_scores) / len(cumulative_scores) if cumulative_scores else 0.0, 5,
                )

            # Update already-found for every k
            for k in self.k_values:
                already_found = already_found_per_k[k]
                for qid, docs in step_docs_per_qid.items():
                    if qid not in self.qrels:
                        continue
                    gold_set = set(self.qrels[qid].keys())
                    sorted_doc_ids = sorted(docs, key=docs.get, reverse=True)[:k]
                    top_k_set = set(sorted_doc_ids)
                    already_found.setdefault(qid, set()).update(gold_set & top_k_set)

            if step_entry:
                result[step_label] = step_entry

        return result

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _plot(
        metrics: Dict[str, Dict[str, Any]],
        out_dir: Path,
        plot_type: str = "retrieval_by_step",
    ) -> None:
        """Create a Recall-vs-cutoff line plot, one line per step + average."""
        if not metrics:
            return

        recall_cutoffs = [1, 3, 5, 10, 25, 100, 500, 1000]
        x_labels = [f"R@{k}" for k in recall_cutoffs]

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = plt.cm.tab10.colors

        all_y: List[List[float]] = []
        for i, (step_name, step_data) in enumerate(metrics.items()):
            recall = step_data.get("Recall", {})
            y_vals = [recall.get(f"Recall@{k}", float("nan")) * 100 for k in recall_cutoffs]
            all_y.append(y_vals)
            ax.plot(
                x_labels, y_vals,
                marker="o",
                color=colors[i % len(colors)],
                label=step_name,
                linewidth=1.8,
                markersize=5,
            )

        # Average line
        avg_y = np.nanmean(all_y, axis=0).tolist()
        ax.plot(
            x_labels, avg_y,
            marker="D", color="black", label="Average",
            linewidth=2.5, markersize=7, linestyle="--", zorder=10,
        )
        for x, y in zip(x_labels, avg_y):
            ax.annotate(
                f"{y:.1f}", xy=(x, y), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="black", zorder=11,
            )

        ax.set_xlabel("Retrieval Cutoff", fontsize=12)
        ax.set_ylabel("Recall (%)", fontsize=12)
        ax.set_ylim(0, 100)

        title = (
            "Recall by Search Step"
            if plot_type == "retrieval_by_step"
            else "Marginal Recall per Search Step"
        )
        ax.set_title(f"{title}  ({len(metrics)} steps)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout()

        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{plot_type}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
