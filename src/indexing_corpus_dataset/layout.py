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


def _load_dra_env_defaults() -> None:
    """Populate ``DRA_*`` env vars from the project-root ``.env`` if unset.

    Every entry point imports this module (inference CLI, downloaders, index
    builder), but the standalone scripts never call ``python-dotenv``, and
    ~/.bashrc is not sourced in non-interactive SLURM jobs.  This tiny,
    dependency-free reader makes ``DRA_DATA_ROOT`` / ``DRA_OUTPUT_ROOT`` set in
    ``.env`` take effect everywhere.  A real environment value always wins, so
    exporting the var (shell or sbatch) still overrides ``.env``.
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("DRA_") and key not in os.environ:
            value = value.split("#", 1)[0].strip().strip('"').strip("'")
            if value:
                os.environ[key] = value


_load_dra_env_defaults()

# The one canonical dataset root.  Defaults to the cluster project location but
# can be overridden with the ``DRA_DATA_ROOT`` env var (no code edit needed) —
# set it in the shell, in an sbatch script, or in the project-root ``.env``.
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


# ===========================================================================
# Index-build path derivation (corpus + index dir from dataset/subset)
# ===========================================================================

def default_corpus_path(
    dataset: str, subset: str | None = None, *, partial: bool = True
) -> Path:
    """Canonical full-corpus JSONL path for a dataset/subset under ``DATA_ROOT``.

    The corpus file stem encodes the subset (NeuCLIR ``corpus_en_news``, TRQA
    ``wiki_partial``/``ecommerce``), so an index built from this path is already
    named per dataset/subset by ``Index_Builder``.  Used by both the production
    index builder and its test to avoid passing ``--corpus-path`` by hand.
    """
    if dataset == "neuclir":
        return DATA_ROOT / "neuclir" / "corpus" / f"{corpus_name(subset)}.jsonl"
    if dataset == "trqa":
        return DATA_ROOT / "trqa" / "corpus" / f"{trqa_corpus_name(subset, partial)}.jsonl"
    if dataset == "browsecomp_plus":
        return DATA_ROOT / "browsecomp_plus" / "corpus" / "corpus.jsonl"
    # Unknown dataset: best-effort canonical location.
    return DATA_ROOT / dataset / "corpus" / "corpus.jsonl"


def default_index_dir(dataset: str) -> Path:
    """Canonical index output directory for a dataset, ``DATA_ROOT/{dataset}/indices``.

    Sibling of the ``corpus/`` directory, matching the production layout.
    """
    return DATA_ROOT / dataset / "indices"
