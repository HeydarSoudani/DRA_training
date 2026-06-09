"""
Universal ranking results format for dataset search results.

This module defines a standardized format for ranking results that includes:
- query_id: Identifier for the query
- doc_id: Identifier for the document
- relevant_text: The relevant text content from the document
- summary_text: Summary or snippet of the document
- rank: Position in the ranking (1-based)
- rank_score: Numerical score for the ranking

The format supports multiple export/import formats and maintains backward
compatibility with existing search result formats.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import csv
import pandas as pd
from datetime import datetime, timezone


@dataclass
class RankingResult:
    """Individual ranking result with query, document, and relevance information."""

    query_id: str
    doc_id: str
    rank: int
    rank_score: float
    relevant_text: Optional[str] = None
    summary_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the ranking result after initialization."""
        if self.rank < 0:
            raise ValueError(f"Rank must be >= 0, got {self.rank}")
        if not isinstance(self.rank_score, (int, float)):
            raise ValueError(f"Rank score must be numeric, got {type(self.rank_score)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "rank": self.rank,
            "rank_score": self.rank_score,
            "relevant_text": self.relevant_text,
            "summary_text": self.summary_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankingResult":
        """Create RankingResult from dictionary."""
        # Handle backward compatibility with existing formats
        return cls(
            query_id=str(data.get("query_id", data.get("qid", ""))),
            doc_id=str(data.get("doc_id", data.get("docid", data.get("pid", "")))),
            rank=int(data.get("rank", data.get("pos", 1))),
            rank_score=float(data.get("rank_score", data.get("score", 0.0))),
            relevant_text=data.get("relevant_text"),
            summary_text=data.get("summary_text"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RankingResults:
    """Collection of ranking results with experiment metadata."""

    results: List[RankingResult] = field(default_factory=list)
    experiment_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Add timestamp if not present in experiment_info."""
        if "created_at" not in self.experiment_info:
            self.experiment_info["created_at"] = datetime.now(timezone.utc).isoformat()

    def add_result(self, result: RankingResult):
        """Add a single ranking result."""
        self.results.append(result)

    def get_results_for_query(
        self, query_id: str, top_k: int = -1
    ) -> List[RankingResult]:
        """Get all results for a specific query."""
        results = sorted(
            [r for r in self.results if r.query_id == query_id],
            key=lambda r: (r.rank, -r.rank_score),
        )
        return results[:top_k] if top_k >= 0 else results

    def get_unique_queries(self) -> List[str]:
        """Get list of unique query IDs."""
        return list(set(r.query_id for r in self.results))

    def get_unique_documents(self) -> List[str]:
        """Get list of unique document IDs."""
        return list(set(r.doc_id for r in self.results))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        data = []
        for result in self.results:
            row = result.to_dict()
            # Flatten metadata into columns
            for key, value in result.metadata.items():
                row[f"metadata_{key}"] = value
            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def from_ranking_results(
        results: "List[RankingResult]",
    ) -> "RankingResults":
        """Create RankingResults from another RankingResults instance."""
        return RankingResults(
            results=results.copy(),
            experiment_info={},
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, experiment_info: Optional[Dict[str, Any]] = None
    ) -> "RankingResults":
        """Create RankingResults from pandas DataFrame."""
        results = []

        for _, row in df.iterrows():
            # Extract metadata columns
            metadata = {}
            row_dict = row.to_dict()

            # Separate metadata columns from main columns
            main_data = {}
            for key, value in row_dict.items():
                if key.startswith("metadata_"):
                    metadata_key = key.replace("metadata_", "")
                    metadata[metadata_key] = value
                else:
                    main_data[key] = value

            main_data["metadata"] = metadata
            results.append(RankingResult.from_dict(main_data))

        return cls(
            results=results,
            experiment_info=experiment_info or {},
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        data = {
            "experiment_info": self.experiment_info,
            "results": [result.to_dict() for result in self.results],
        }
        return json.dumps(data, indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "RankingResults":
        """Create RankingResults from JSON string."""
        data = json.loads(json_str)

        # Handle different JSON formats for backward compatibility
        if "results" in data:
            # New format
            results = [RankingResult.from_dict(r) for r in data["results"]]
            experiment_info = data.get("experiment_info", {})
            # Ignore metrics field for backward compatibility - can be stored in experiment_info if needed
        else:
            # Legacy format - assume it's a list of results or dict with query_id as keys
            results = []
            if isinstance(data, list):
                results = [RankingResult.from_dict(r) for r in data]
            elif isinstance(data, dict):
                for query_id, query_results in data.items():
                    if isinstance(query_results, list):
                        for result_data in query_results:
                            result_data["query_id"] = query_id
                            results.append(RankingResult.from_dict(result_data))

            experiment_info = {}

        return cls(
            results=results,
            experiment_info=experiment_info,
        )

    def to_csv(self) -> str:
        """Convert to CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)

    @classmethod
    def from_csv(
        cls, csv_str: str, experiment_info: Optional[Dict[str, Any]] = None
    ) -> "RankingResults":
        """Create RankingResults from CSV string."""
        import io

        df = pd.read_csv(io.StringIO(csv_str))
        return cls.from_dataframe(df, experiment_info)

    def normalize_scores_by_rank(self) -> "RankingResults":
        """Reassign ``rank_score`` so that scores decrease monotonically with rank.

        pytrec_eval (and trec_eval) ignores the rank column and re-sorts by
        score.  After fusion or reranking the original retrieval scores may no
        longer reflect the intended rank order.  This method overwrites each
        result's ``rank_score`` with ``(n - rank + 1) / n`` **per query** so
        that score-based sorting reproduces the exact rank ordering.

        Returns ``self`` for convenient chaining.
        """
        from collections import defaultdict

        by_query: dict[str, list[RankingResult]] = defaultdict(list)
        for r in self.results:
            by_query[r.query_id].append(r)

        for _qid, qresults in by_query.items():
            qresults.sort(key=lambda r: r.rank)
            n = len(qresults)
            for rank_idx, r in enumerate(qresults, 1):
                r.rank_score = (n - rank_idx + 1) / n

        return self

    def to_trec(self) -> str:
        """Convert to TREC format string.

        Scores are always normalized to match rank order so that
        trec_eval / pytrec_eval (which re-sorts by score) reproduces the
        exact intended ranking.

        When a ``reranker_score`` is present in the result metadata, it is
        appended to the run_tag in the 6th column as
        ``run_tag_rs=<score>`` so that the raw reranker score is
        preserved in the TREC file for analysis.
        """
        self.normalize_scores_by_rank()
        lines = []
        global_run_name = self.experiment_info.get("run_name", "default")
        for result in self.results:
            # TREC format: qid Q0 docid rank score run_tag
            # Per-result run_tag (e.g. iter_1) takes precedence over global run_name
            run_tag = result.metadata.get("run_tag", global_run_name)
            reranker_score = result.metadata.get("reranker_score")
            if reranker_score is not None:
                run_tag = f"rerank_score={reranker_score}"
            line = f"{result.query_id} Q0 {result.doc_id} {result.rank} {result.rank_score} {run_tag}"
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def from_trec(
        cls, trec_str: str, experiment_info: Optional[Dict[str, Any]] = None
    ) -> "RankingResults":
        """Create RankingResults from TREC format string."""
        results = []
        for line in trec_str.strip().split("\n"):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, rank, score = parts[:5]
                    run_name = " ".join(parts[5:]) if len(parts) > 5 else "default"

                    result = RankingResult(
                        query_id=query_id,
                        doc_id=doc_id,
                        rank=int(rank),
                        rank_score=float(score),
                        metadata={"run_name": run_name},
                    )
                    results.append(result)

        if experiment_info is None:
            experiment_info = {}

        return cls(results=results, experiment_info=experiment_info)


def load_ranking_results(file_path: Union[str, Path]) -> RankingResults:
    """Load ranking results from file, auto-detecting format.

    Supports both local paths and S3 URIs.
    """
    from agentic_retrieval_research.utils.s3_utils import is_s3_path, s3_read_text, s3_exists

    file_path_str = str(file_path)

    if is_s3_path(file_path_str):
        if not s3_exists(file_path_str):
            raise FileNotFoundError(f"File not found: {file_path_str}")
        content = s3_read_text(file_path_str)
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Auto-detect format based on file extension
    suffix = file_path_str.rsplit(".", 1)[-1].lower() if "." in file_path_str else ""

    if suffix == "json":
        return RankingResults.from_json(content)
    elif suffix == "csv":
        return RankingResults.from_csv(content)
    elif suffix in ["trec", "txt"]:
        return RankingResults.from_trec(content)
    else:
        # Try to auto-detect from content
        content_stripped = content.strip()
        if content_stripped.startswith("{") or content_stripped.startswith("["):
            return RankingResults.from_json(content)
        elif "," in content_stripped.split("\n")[0]:  # Likely CSV
            return RankingResults.from_csv(content)
        else:  # Assume TREC format
            return RankingResults.from_trec(content)


def save_ranking_results(
    ranking_results: RankingResults,
    file_path: Union[str, Path],
    format_type: Optional[str] = None,
) -> None:
    """Save ranking results to file, auto-detecting format if not specified.

    Supports both local paths and S3 URIs.
    """
    from agentic_retrieval_research.utils.s3_utils import is_s3_path, s3_write_text

    file_path_str = str(file_path)

    # Create parent directories if local
    if not is_s3_path(file_path_str):
        Path(file_path_str).parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format_type is None:
        suffix = Path(file_path_str).suffix.lower()
        if suffix == ".json":
            format_type = "json"
        elif suffix == ".csv":
            format_type = "csv"
        elif suffix in [".trec", ".txt"]:
            format_type = "trec"
        else:
            format_type = "json"  # Default

    # Generate content based on format
    if format_type.lower() == "json":
        content = ranking_results.to_json()
    elif format_type.lower() == "csv":
        content = ranking_results.to_csv()
    elif format_type.lower() == "trec":
        content = ranking_results.to_trec()
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    # Write to file (S3 or local)
    if is_s3_path(file_path_str):
        s3_write_text(file_path_str, content)
    else:
        with open(file_path_str, "w", encoding="utf-8") as f:
            f.write(content)
