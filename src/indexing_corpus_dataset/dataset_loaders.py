"""Read dataset queries and qrels from a dataset directory.

All loaders take an explicit ``data_path``, so the module works regardless of
where Python is invoked from.
"""

from pathlib import Path
import csv
import json


def load_queries(data_path: Path | str, data_set: str = "set1", query_key: str = "text") -> dict[str, str]:
    """Load queries from a dataset directory.

    Args:
        data_path:  Path to dataset directory.
        data_set:   Dataset split identifier used to locate the file
                    (e.g. "set1", "2024", "2024_technical").
        query_key:  Key in each JSONL record to use as the query text.
                    Defaults to "text" for standard datasets.
                    For NeuCLIR use "topic_title", "topic_description",
                    or "topic_narrative".
    """
    base_path = Path(data_path) / "queries" / f"queries_{data_set}"
    possible_files = [
        str(base_path.with_suffix(".tsv")),
        str(base_path.with_suffix(".jsonl")),
    ]

    for file_str in possible_files:
        if Path(file_str).exists():
            try:
                # Determine file type from extension
                is_jsonl = file_str.endswith(".jsonl")

                if is_jsonl:
                    # Parse JSONL format.
                    # Supports standard {"id": ..., "text": ...} and NeuCLIR
                    # {"topic_id": ..., "topic_title": ..., ...} schemas.
                    queries = {}
                    with open(file_str, "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line.strip())
                            qid = obj.get("id") or obj.get("topic_id")
                            text = obj.get(query_key)
                            if qid and text:
                                queries[qid] = text
                    return queries
                else:
                    # Parse TSV format
                    with open(file_str, "r", encoding="utf-8") as f:
                        reader = csv.reader(f, delimiter="\t")
                        return {rows[0]: rows[1] for rows in reader}
            except Exception as e:
                print(f"Error reading queries file {file_str}: {e}")
                continue

    print(f"Error reading queries file: No valid file found for {data_path}/queries/queries_{data_set}")
    return {}


def load_query_answers(data_path: Path | str, data_set: str = "set1") -> dict[str, str]:
    """Load ground-truth answers from a JSONL queries file.

    Reads the ``"answer"`` field from each record in the queries file.
    Returns an empty dict when the file has no ``"answer"`` field
    (e.g. datasets other than BrowseComp-Plus).

    Args:
        data_path: Path to dataset directory.
        data_set:  Dataset split identifier (e.g. "test").

    Returns:
        ``{query_id: answer_text}`` for records that have an answer.
    """
    jsonl_path = str(Path(data_path) / "queries" / f"queries_{data_set}.jsonl")

    answers: dict[str, str] = {}
    if not Path(jsonl_path).exists():
        return answers

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                qid = obj.get("id") or obj.get("topic_id")
                answer = obj.get("answer")
                if qid and answer:
                    answers[qid] = answer
    except Exception as e:
        print(f"Error reading answers from {jsonl_path}: {e}")

    return answers


def load_qrels(
    data_path: Path | str, data_set: str = "set1", min_relevance_score: int | None = None
) -> dict[str, dict[str, int]]:
    """Load qrels from a dataset directory.

    Args:
        data_path: Path to dataset directory (can be local or S3 path)
        data_set: Dataset split to use (default: "set1")
        min_relevance_score: Minimum relevance score to consider (default: None, includes all).
                           If specified, only passages with relevance >= min_relevance_score are included.
                           For example, if a dataset has scores 1 and 3, setting min_relevance_score=3
                           will only include passages with score 3 as relevant.

    Returns:
        Dictionary mapping query_id -> {doc_id: relevance_score}
    """
    base_path = Path(data_path) / "qrels" / f"qrels_{data_set}"
    possible_files = [
        str(base_path.with_suffix(".tsv")),
        str(base_path.with_suffix(".txt")),
    ]

    qrels: dict[str, dict[str, int]] = {}

    for file_str in possible_files:
        if Path(file_str).exists():
            try:
                # Determine file type from extension
                is_txt = file_str.endswith(".txt")

                with open(file_str, "r", encoding="utf-8") as f:
                    if is_txt:
                        # Parse TREC qrels format: query_id 0 doc_id relevance (space-separated)
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                qid, docid, rel = parts[0], parts[2], int(parts[3])
                                # Filter by minimum relevance score if specified
                                if min_relevance_score is not None and rel < min_relevance_score:
                                    continue
                                if qid not in qrels:
                                    qrels[qid] = {}
                                qrels[qid][docid] = rel
                    else:
                        # Parse TSV format: query_id \t doc_id \t relevance
                        reader = csv.reader(f, delimiter="\t")
                        for rows in reader:
                            qid, docid, rel = rows[0], rows[1], int(rows[2])
                            # Filter by minimum relevance score if specified
                            if min_relevance_score is not None and rel < min_relevance_score:
                                continue
                            if qid not in qrels:
                                qrels[qid] = {}
                            qrels[qid][docid] = rel
                return qrels
            except Exception as e:
                print(f"Error reading qrels file {file_str}: {e}")
                continue

    print(f"Error reading qrels file: No valid file found for {data_path}/qrels/qrels_{data_set}")
    return {}
