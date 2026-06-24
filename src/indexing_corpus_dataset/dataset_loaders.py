"""Read dataset queries, qrels, and answers from the canonical layout.

The on-disk layout, path builders, split-id rule, and record schemas live in
:mod:`indexing_corpus_dataset.layout` (the single source of truth, shared with
``download_datasets.py``).  This module is the *reader* half: given a
``data_path`` and a split, it parses queries / qrels / answers.

All functions take an explicit ``data_path`` so they work regardless of the
current working directory.
"""

from pathlib import Path
import csv
import json

# Layout vocabulary (single source of truth).  Re-exported below so existing
# ``from ...dataset_loaders import queries_base`` call sites keep working.
from .layout import (
    DATA_ROOT,
    queries_base,
    qrels_base,
    corpus_path,
    corpus_name,
    resolve_split_id,
    resolve_data_path,
    _resolve_split_file,
)


# ===========================================================================
# Queries + answers (single-pass)
# ===========================================================================

def _load_queries_and_answers(
    data_path: Path | str, split: str, query_key: str = "text"
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse the queries file once, returning (queries, answers).

    ``answers`` is populated only from JSONL records that carry an ``"answer"``
    field (e.g. BrowseComp-Plus); it is empty for TSV files or datasets without
    answers.
    """
    file_path = _resolve_split_file(queries_base(data_path, split), (".tsv", ".jsonl"))
    if file_path is None:
        print(f"Error reading queries file: No valid file found for "
              f"{data_path}/queries/queries_{split}")
        return {}, {}

    queries: dict[str, str] = {}
    answers: dict[str, str] = {}
    try:
        if file_path.suffix == ".jsonl":
            # Supports standard {"id","text"} and NeuCLIR {"topic_id","topic_title",...}.
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = obj.get("id") or obj.get("topic_id")
                    if not qid:
                        continue
                    text = obj.get(query_key)
                    if text:
                        queries[qid] = text
                    answer = obj.get("answer")
                    if answer:
                        answers[qid] = answer
        else:  # .tsv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for rows in reader:
                    if len(rows) >= 2:
                        queries[rows[0]] = rows[1]
    except Exception as e:
        print(f"Error reading queries file {file_path}: {e}")

    return queries, answers


def load_queries(
    data_path: Path | str, data_set: str = "set1", query_key: str = "text"
) -> dict[str, str]:
    """Load queries from a dataset directory.

    Args:
        data_path:  Path to dataset directory.
        data_set:   Split identifier (e.g. "test", "2024_news").
        query_key:  Key in each JSONL record to use as the query text.
                    Defaults to "text", which is correct for every dataset:
                    the downloaders flatten each source's best query text
                    (including NeuCLIR's topic_* fields) into "text".
    """
    return _load_queries_and_answers(data_path, data_set, query_key)[0]


def load_query_answers(
    data_path: Path | str, data_set: str = "set1"
) -> dict[str, str]:
    """Load ground-truth answers from the JSONL queries file.

    Returns ``{query_id: answer_text}`` for records that have an ``"answer"``
    field (e.g. BrowseComp-Plus); empty for datasets without answers.
    """
    return _load_queries_and_answers(data_path, data_set)[1]


# ===========================================================================
# Qrels
# ===========================================================================

def load_qrels(
    data_path: Path | str, data_set: str = "set1", min_relevance_score: int | None = None
) -> dict[str, dict[str, int]]:
    """Load qrels from a dataset directory.

    Args:
        data_path: Path to dataset directory.
        data_set: Split identifier (default: "set1").
        min_relevance_score: If set, only passages with relevance
            >= min_relevance_score are kept.

    Returns:
        ``{query_id: {doc_id: relevance_score}}``
    """
    file_path = _resolve_split_file(qrels_base(data_path, data_set), (".tsv", ".txt"))
    if file_path is None:
        print(f"Error reading qrels file: No valid file found for "
              f"{data_path}/qrels/qrels_{data_set}")
        return {}

    qrels: dict[str, dict[str, int]] = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".txt":
                # TREC qrels: "query_id 0 doc_id relevance" (space-separated)
                rows = (line.split() for line in f)
                parsed = ((r[0], r[2], int(r[3])) for r in rows if len(r) >= 4)
            else:  # .tsv: "query_id \t doc_id \t relevance"
                reader = csv.reader(f, delimiter="\t")
                parsed = ((r[0], r[1], int(r[2])) for r in reader if len(r) >= 3)

            for qid, docid, rel in parsed:
                if min_relevance_score is not None and rel < min_relevance_score:
                    continue
                qrels.setdefault(qid, {})[docid] = rel
    except Exception as e:
        print(f"Error reading qrels file {file_path}: {e}")
        return {}

    return qrels


# ===========================================================================
# One-pass convenience loader
# ===========================================================================

def load_split(
    data_path: Path | str,
    data_set: str = "set1",
    *,
    query_key: str = "text",
    min_relevance_score: int | None = None,
    only_with_qrels: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """Load queries, qrels, and answers for a split in a single call.

    The queries file is read only once (answers come from the same records).

    Args:
        data_path:           Path to dataset directory.
        data_set:            Split identifier (e.g. "test", "2024_news").
        query_key:           Key in each JSONL record to use as the query text.
        min_relevance_score: Minimum qrel score to keep (None = keep all).
        only_with_qrels:     If True, drop queries that have no qrels entry.

    Returns:
        ``(queries, qrels, answers)``.
    """
    queries, answers = _load_queries_and_answers(data_path, data_set, query_key)
    qrels = load_qrels(data_path, data_set, min_relevance_score)

    if only_with_qrels:
        queries = {qid: q for qid, q in queries.items() if qid in qrels}

    return queries, qrels, answers
