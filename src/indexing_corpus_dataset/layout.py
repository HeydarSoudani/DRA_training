"""Canonical on-disk dataset layout — the single source of truth for where
dataset files live and how their paths, split ids, and corpus names are built.

Both the readers (``dataset_loaders.py``) and the writers/downloaders
(``download_datasets.py``) import this module, as do external consumers (the
inference CLI, the index builder, and its test).  Centralising the layout here
means the data root, the split-id rule, and the corpus-naming rule each have
exactly one definition.

Canonical layout::

    {data_path}/queries/queries_{split}.jsonl   (or .tsv)
    {data_path}/qrels/qrels_{split}.txt         (or .tsv)
    {data_path}/corpus/{name}.jsonl

Record schemas:
    queries : {"id": str, "text": str, "answer"?: str}   (NeuCLIR uses topic_*)
    qrels   : TREC -> "qid 0 docid rel"
    corpus  : {"id": str, "contents": str}

Deliberately import-light (only :mod:`pathlib`) so any module — including the
low-level ``utils.config`` — can import it without pulling heavy dependencies.
"""

import os
from pathlib import Path

# The one canonical dataset root.  Defaults to the cluster project location but
# can be overridden with the ``DRA_DATA_ROOT`` env var (no code edit needed).
# Hardcoded default (not derived from ``__file__`` depth) so it is stable
# regardless of where the package is imported from.
DATA_ROOT = Path(os.environ.get(
    "DRA_DATA_ROOT", "/projects/0/prjs0834/heydars/DRA_training/data"
))

# Per-subset NeuCLIR English-MT corpus file stems (no ``.jsonl`` suffix).
_NEUCLIR_CORPUS_NAMES = {
    "news":      "corpus_en_news",
    "technical": "corpus_en_technical",
}

# Per-subset TRQA corpus file stems (no ``.jsonl`` suffix).  wiki1 and wiki2
# share the single Wikipedia corpus; ecommerce has its own.  The partial
# Wikipedia corpus is selected via the ``partial`` flag of ``trqa_corpus_name``.
_TRQA_CORPUS_NAMES = {
    "wiki1":     "wiki",
    "wiki2":     "wiki",
    "ecommerce": "ecommerce",
}


# ===========================================================================
# Path builders (suffix-less bases + concrete corpus path)
# ===========================================================================

def queries_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less queries path, e.g. ``.../queries/queries_test``."""
    return Path(data_path) / "queries" / f"queries_{split}"


def qrels_base(data_path: Path | str, split: str) -> Path:
    """Return the suffix-less qrels path, e.g. ``.../qrels/qrels_test``."""
    return Path(data_path) / "qrels" / f"qrels_{split}"


def corpus_path(data_path: Path | str, name: str = "corpus") -> Path:
    """Return the corpus file path, e.g. ``.../corpus/corpus.jsonl``."""
    return Path(data_path) / "corpus" / f"{name}.jsonl"


def _resolve_split_file(base: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Return the first existing ``base + suffix`` file, or None."""
    for suffix in suffixes:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


# ===========================================================================
# Naming conventions: corpus file stem, split id, data-path resolution
# ===========================================================================

def corpus_name(subset: str) -> str:
    """NeuCLIR corpus file stem for a subset, e.g. ``news`` -> ``corpus_en_news``.

    Falls back to ``corpus_en_{subset}`` for subsets not in the explicit map.
    """
    return _NEUCLIR_CORPUS_NAMES.get(subset, f"corpus_en_{subset}")


def trqa_corpus_name(subset: str, partial: bool = True) -> str:
    """TRQA corpus file stem for a subset (no ``.jsonl`` suffix).

    ``wiki1``/``wiki2`` -> ``wiki_partial`` (the canonical default; ``wiki``
    only when ``partial=False``), ``ecommerce`` -> ``ecommerce``.  Unknown
    subsets fall back to themselves.
    """
    name = _TRQA_CORPUS_NAMES.get(subset, subset)
    if partial and name == "wiki":
        return "wiki_partial"
    return name


def resolve_split_id(
    dataset: str,
    dataset_year: str | None = None,
    subset: str | None = None,
) -> str:
    """Build the on-disk split identifier from (dataset, year, subset).

    Mirrors the conventions used across the pipeline:
      neuclir         -> "{year}_{subset}"  (falls back to subset/year)
      trqa            -> "{subset}_{eval}"  (eval split carried in dataset_year)
      browsecomp_plus -> subset or "test"
      other           -> subset or "set1"
    """
    if dataset == "neuclir":
        if dataset_year and subset:
            return f"{dataset_year}_{subset}"
        return subset or dataset_year or "2024_news"
    if dataset == "trqa":
        # TRQA splits combine the collection (subset: wiki1/wiki2/ecommerce)
        # with the evaluation split (test/validation), carried in dataset_year
        # to reuse the existing two-dimensional arg plumbing.  This matches the
        # HuggingFace split names, e.g. "wiki1_test".
        eval_split = dataset_year or "test"
        return f"{subset}_{eval_split}" if subset else eval_split
    if dataset == "browsecomp_plus":
        return subset or "test"
    return subset or "set1"


def resolve_data_path(dataset: str, project_root: Path | str | None = None) -> str:
    """Resolve the dataset root directory.

    Uses the canonical ``DATA_ROOT/{dataset}`` location, falling back to the
    repo-local ``data/{dataset}/`` for legacy setups.
    """
    candidate = DATA_ROOT / dataset
    if candidate.exists():
        return str(candidate)
    base = Path(project_root) if project_root is not None else Path.cwd()
    return str((base / "data" / dataset).resolve())
