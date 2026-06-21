"""Shared experiment-result I/O helpers (local filesystem).

Functions for reading/writing experiment run outputs (JSON, TREC, generation
markdown) and for resuming interrupted pipeline runs.
"""

import json as _json
from pathlib import Path
from typing import Dict, List, Union

try:
    from orjson import loads as _json_loads
except ImportError:
    _json_loads = _json.loads


def load_json_from_path(path: Union[str, Path], default=None):
    """Load a JSON file, returning *default* if missing.

    Args:
        path: Local path to a ``.json`` file.
        default: Value to return when the file does not exist.
                 Defaults to ``None``; pass ``{}`` or ``[]`` as needed.

    Returns:
        Parsed JSON object, or *default* on missing/error.
    """
    if default is None:
        default = {}
    try:
        return _json_loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return default
    except Exception:
        return default


def save_json_to_path(path: Union[str, Path], data, *, indent: int = 2) -> None:
    """Serialize *data* as JSON and write to a local path.

    Args:
        path: Destination local path.
        data: JSON-serializable object.
        indent: Pretty-print indent (default 2).
    """
    Path(path).write_text(_json.dumps(data, indent=indent), encoding="utf-8")


def setup_output_dirs(
    run_dir: Union[str, Path],
    subdirs: List[str],
) -> Dict[str, str]:
    """Create output sub-directories under *run_dir* and return their paths.

    Args:
        run_dir: Root run directory.
        subdirs: List of sub-directory names,
                 e.g. ``["retrieval", "generation", "trajectory"]``.

    Returns:
        Dict mapping each sub-directory name to its full path string.
    """
    rd = Path(str(run_dir).rstrip("/"))
    rd.mkdir(parents=True, exist_ok=True)
    result: Dict[str, str] = {}
    for name in subdirs:
        d = rd / name
        d.mkdir(parents=True, exist_ok=True)
        result[name] = str(d)
    return result


def get_processed_queries(run_dir: Union[str, Path]) -> set:
    """Return query IDs that already have a saved retrieval TREC file.

    Enables resuming an interrupted pipeline run by skipping previously
    completed queries.

    Args:
        run_dir: Root directory of the current run.
            Results are expected under ``{run_dir}/retrieval/*.trec``.

    Returns:
        Set of query-ID strings (filename stems of existing ``.trec`` files),
        or an empty set if the retrieval directory does not exist.
    """
    retrieval_dir = Path(run_dir) / "retrieval"
    if not retrieval_dir.exists():
        return set()
    try:
        return {f.stem for f in retrieval_dir.glob("*.trec")}
    except Exception as e:
        print(f"Warning: could not read existing results from {retrieval_dir}: {e}")
        return set()


def load_result_from_trec(trec_file: Union[str, Path], query_id: str) -> dict:
    """Reconstruct a minimal result dict from a saved per-query TREC file.

    The TREC file format written by ``RetrievalEvaluator.save_item`` is::

        qid Q0 doc_id rank score iter_N

    Args:
        trec_file: Path to the ``.trec`` file.
        query_id:  Query identifier (used only for warning messages).

    Returns:
        Dict ``{"trajectory": [{"docs": [...]}, ...]}`` or ``{}`` on failure.
    """
    iterations: Dict[int, list] = {}
    try:
        with open(trec_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                doc_id, rank, score, run_tag = parts[2], int(parts[3]), float(parts[4]), parts[5]
                try:
                    iter_idx = int(run_tag.split("_")[1])
                except (IndexError, ValueError):
                    iter_idx = 1
                iterations.setdefault(iter_idx, []).append(
                    {"doc_id": doc_id, "rank_score": score}
                )
    except Exception as e:
        print(f"Warning: could not load {trec_file}: {e}")
        return {}

    if not iterations:
        return {}

    trajectory = [{"docs": iterations[i]} for i in sorted(iterations)]
    return {"trajectory": trajectory}


def load_result_from_saved_files(
    run_dir: Union[str, Path], query_id: str, *, lightweight: bool = False,
) -> dict:
    """Reconstruct a full result dict from all saved per-query files.

    Tries to load from the trajectory JSON first (which contains trajectory,
    generation, num_steps, num_searches, and agent-specific metadata).  Falls
    back to TREC-only reconstruction when the trajectory JSON is unavailable.

    Also loads the generation markdown and cited-docs TREC when available and
    the trajectory JSON didn't already provide them.

    Args:
        run_dir:  Root directory of the run.  Contains retrieval/,
                  trajectory/, generation/, cited_docs_retrieval/
                  subdirectories.
        query_id: Query identifier.
        lightweight: When True, skip the (potentially large) trajectory JSON
                     and reconstruct from TREC + generation + cited-docs files
                     only.  Evaluators need only doc_ids and scores, not the
                     full document text stored in trajectory JSONs.

    Returns:
        Reconstructed result dict, or ``{}`` on failure.
    """
    run_dir_str = str(run_dir).rstrip("/")

    def _file_path(subdir: str, filename: str) -> str:
        return f"{run_dir_str}/{subdir}/{filename}"

    def _read_text(subdir: str, filename: str) -> str:
        return Path(_file_path(subdir, filename)).read_text(encoding="utf-8")

    def _read_bytes(subdir: str, filename: str) -> bytes:
        return Path(_file_path(subdir, filename)).read_bytes()

    result: dict = {}

    # ── 1. Try trajectory JSON (richest source) ─────────────────────────────
    if not lightweight:
        try:
            content = _read_bytes("trajectory", f"{query_id}.json")
            saved = _json_loads(content)
            result = {
                "trajectory":   saved.get("trajectory", []),
                "generation":   saved.get("generation", ""),
                "num_steps":    saved.get("num_steps", 0),
                "num_searches": saved.get("num_searches", 0),
            }
            for key in ("citation_to_doc_id", "cited_docs_ranked_list",
                        "memory_bank", "query_outputs",
                        "tracker_score_history", "tracker_unique_doc_ids",
                        "tracker_unique_doc_count"):
                if key in saved and saved[key]:
                    result[key] = saved[key]
        except (FileNotFoundError, OSError):
            pass
        except Exception as e:
            print(f"Warning: could not load trajectory JSON for {query_id}: {e}")

    # ── 2. Fall back to TREC if no trajectory loaded ─────────────────────────
    if not result:
        try:
            trec_path = _file_path("retrieval", f"{query_id}.trec")
            result = load_result_from_trec(trec_path, query_id)
        except (FileNotFoundError, OSError):
            pass

    if not result:
        return {}

    # ── 3. Load generation from MD if not in trajectory JSON ─────────────────
    if not result.get("generation"):
        try:
            result["generation"] = _read_text("generation", f"{query_id}.md")
        except Exception:
            pass

    # ── 4. Load cited docs from TREC if not already present ──────────────────
    if not result.get("cited_docs_ranked_list"):
        try:
            content = _read_text("cited_docs_retrieval", f"{query_id}.trec")
            doc_ids = []
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2]:
                    doc_ids.append({"doc_id": parts[2]})
            if doc_ids:
                result["cited_docs_ranked_list"] = doc_ids
        except Exception:
            pass

    # ── 5. Load seen-doc iterations from TREC if not in trajectory ──────────
    if not result.get("seen_docs_iterations"):
        try:
            seen_trec_path = _file_path("seen_docs_retrieval", f"{query_id}.trec")
            seen_parsed = load_result_from_trec(seen_trec_path, query_id)
            if seen_parsed:
                seen_iters = []
                for step in seen_parsed.get("trajectory", []):
                    docs = step.get("docs", [])
                    seen_iters.append([{"doc_id": d["doc_id"]} for d in docs if d.get("doc_id")])
                if seen_iters:
                    result["seen_docs_iterations"] = seen_iters
        except Exception:
            pass

    return result
