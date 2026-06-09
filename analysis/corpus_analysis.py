"""
Corpus analysis utilities for inspecting and checking the retrieval corpus.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    class _NoOpBar:
        def update(self, n=1): pass
        def close(self): pass
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NoOpBar()

# Default path to the MS MARCO v2.1 segmented document corpus (JSONL)
DEFAULT_CORPUS_PATH = "/Users/c302748/Desktop/msmarco_v2.1_doc_segmented.jsonl"
CORPUS_PATH_NEW_FORMAT = "data/neuclir/corpus/corpus_en.jsonl"

def load_corpus(corpus_path: str):
    """Load the corpus from a JSONL file as a HuggingFace Dataset."""
    import datasets
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train"
    )
    return corpus

def get_corpus_entry(
    index: int,
    corpus_path: Optional[str] = None,
    corpus: Optional[Any] = None,
) -> dict:
    """
    Return the corpus entry at the given index (0-based row index in the JSONL).

    Either pass a pre-loaded corpus (e.g. from load_corpus) or a path to the JSONL.
    If both are None, DEFAULT_CORPUS_PATH is used.

    Args:
        index: 0-based integer index of the document (line number in the JSONL).
        corpus_path: Path to the corpus JSONL file. Used only if corpus is None.
        corpus: Pre-loaded corpus (HuggingFace Dataset). If provided, corpus_path is ignored.

    Returns:
        The document entry as a dict (e.g. with keys like 'id', 'title', 'contents').
    """
    if corpus is not None:
        entry = corpus[int(index)]
        # HuggingFace Dataset row can be a dict-like; ensure we return a plain dict
        if hasattr(entry, "items"):
            return dict(entry)
        return entry

    path = corpus_path if corpus_path is not None else DEFAULT_CORPUS_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    # For a single lookup without loading the full dataset, read only that line
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
            if i > index:
                break
    raise IndexError(f"Corpus index {index} out of range (file has fewer lines)")

def get_corpus_entry_from_file(index: int, corpus_path: Optional[str] = None) -> dict:
    """
    Return the corpus entry at the given index by reading only that line from the file.
    Does not load the full corpus into memory; suitable for large files and spot checks.

    Args:
        index: 0-based integer index (line number) in the JSONL.
        corpus_path: Path to the corpus JSONL. If None, DEFAULT_CORPUS_PATH is used.

    Returns:
        The document entry as a dict.
    """
    return get_corpus_entry(index, corpus_path=corpus_path or DEFAULT_CORPUS_PATH, corpus=None)

def entry_to_retriever_format(entry: dict) -> dict:
    """
    Convert a corpus entry to retriever-readable format with "id", "title", "contents".

    - docid -> id
    - segment -> contents
    - All other fields are kept as-is. "title" is set to "" if missing.

    Args:
        entry: A single document dict (e.g. from the segmented corpus with docid/segment).

    Returns:
        A new dict with keys "id", "title", "contents" and any other original keys
        (excluding docid and segment, which are renamed).
    """
    result = {}
    for k, v in entry.items():
        if k == "docid":
            result["id"] = v
        elif k == "segment":
            result["contents"] = v
        else:
            result[k] = v
    result.setdefault("title", "")
    return result

def convert_corpus_to_retriever_format(input_path: str, output_path: str) -> int:
    """
    Convert a full corpus JSONL to retriever format (id, title, contents).
    Reads and writes line-by-line so large files stay in memory safely.
    If the process is "Terminated", that is usually the OS killing it due to
    RAM (out-of-memory), not disk space; request more memory if running under a job scheduler.

    Args:
        input_path: Path to the current corpus JSONL (e.g. docid/segment keys).
        output_path: Path to write the new JSONL with id/title/contents keys.

    Returns:
        Number of documents written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    count = 0
    gc_interval = 1_000_000  # run gc every N docs to limit RAM growth (avoids OOM "Terminated")
    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        pbar = tqdm(desc="Converting", unit=" docs", unit_scale=True)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            converted = entry_to_retriever_format(entry)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            del entry, converted
            count += 1
            pbar.update(1)
            if count % gc_interval == 0:
                gc.collect()
        pbar.close()
    return count

def extract_title_and_contents(input_path: str, output_path: str) -> int:
    """
    Extract title and contents from corpus JSONL where the first newline separates them.

    Reads from input_path (e.g., data/neuclir/corpus/corpus_en.jsonl) and writes to
    output_path (e.g., data/neuclir/corpus/corpus_en_with_title.jsonl).

    For each entry, splits the "contents" field at the first newline:
    - Text before first \n becomes "title"
    - Text after first \n becomes "contents"

    Args:
        input_path: Path to the input corpus JSONL file.
        output_path: Path to write the output JSONL with separated title and contents.

    Returns:
        Number of documents processed.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    gc_interval = 1_000_000  # run gc every N docs to limit RAM growth

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        pbar = tqdm(desc="Extracting titles", unit=" docs", unit_scale=True)
        for line in fin:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            # Extract title and contents from the "contents" field
            contents = entry.get("contents", "")
            if "\n" in contents:
                title, remaining_contents = contents.split("\n", 1)
            else:
                # If no newline, use the whole content as title and empty contents
                title = contents
                remaining_contents = ""

            # Update entry with extracted title and contents
            entry["title"] = title.strip()
            entry["contents"] = remaining_contents.strip()

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            del entry
            count += 1
            pbar.update(1)

            if count % gc_interval == 0:
                gc.collect()

        pbar.close()

    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corpus analysis: get entry by index or convert corpus to retriever format.")
    parser.add_argument("--corpus_path", type=str, default=CORPUS_PATH_NEW_FORMAT, help="Path to the corpus JSONL file (for get-entry).")
    parser.add_argument("--input", type=str, default=DEFAULT_CORPUS_PATH, help="Input JSONL corpus to convert (for convert).")
    parser.add_argument("--output", type=str, default=CORPUS_PATH_NEW_FORMAT, help="Output JSONL path for converted corpus (enables convert).")
    parser.add_argument("--extract_titles", action="store_true", help="Extract titles from contents (split by first newline).")
    args = parser.parse_args()

    # 1) Extract titles and contents
    # input_path = "data/neuclir/corpus/corpus_en.jsonl"
    # output_path = "data/neuclir/corpus/corpus_en_with_title.jsonl"
    # n = extract_title_and_contents(input_path, output_path)
    # print(f"Extracted titles for {n} documents to {output_path}")

    # 2) Convert the corpus to retriever format
    # input_path = args.input if args.input is not None else args.corpus_path
    # n = convert_corpus_to_retriever_format(input_path, args.output)
    # print(f"Converted {n} documents to {args.output}")

    # 3) Get the entry at the given index
    index = 12
    entry = get_corpus_entry(index, corpus_path=args.corpus_path)
    print(f"Entry at index {index}:")
    for key, value in entry.items():
        print(f"  {key}: {value}")


# uv run python src/agentic_retrieval_research/analysis/corpus_analysis.py
