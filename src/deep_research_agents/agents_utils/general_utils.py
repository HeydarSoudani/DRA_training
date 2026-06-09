"""General utility functions for deep research agents pipeline.

Heavy functions have been split into focused modules:
    utils.tracker_factory  — build_trajectory_tracker()
    utils.cited_docs       — _build_cited_docs_ranked_list()
    utils.worker           — _build_components_from_config(), _init_worker(), _gpu_worker()

This module re-exports them for backwards compatibility.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_retrieval_research.utils.s3_utils import (
    is_s3_path, s3_exists, s3_glob, s3_open, s3_read_text, s3_write_text,
    get_processed_queries,
    load_result_from_trec,
    load_result_from_saved_files,
    setup_output_dirs,
    load_json_from_path,
    save_json_to_path,
)

# Re-exports from split modules (preserves existing import paths)
from .constants import (  # noqa: F401
    FINETUNED_MODEL_MAP,
    SELF_MANAGED_LLM_AGENTS,
    _RetrieverConfig,
    _IR_ROOT,
    get_reranker_configs,
    is_local_finetuned,
)
from .tracker_factory import build_trajectory_tracker  # noqa: F401
from .cited_docs import _build_cited_docs_ranked_list  # noqa: F401
from .worker import _build_components_from_config, _init_worker, _gpu_worker  # noqa: F401




# ---------------------------------------------------------------------------
# CLI setup helpers
# ---------------------------------------------------------------------------


def _resolve_dataset_defaults(args) -> None:
    """Auto-select dataset-related CLI arguments that were left as None.

    Mutates *args* in-place; prints each auto-selected value.
    """
    if args.data_path is None:
        data_paths = {
            "trec_rag":        _IR_ROOT / "trec_rag",
            "neuclir":         _IR_ROOT / "neuclir",
            "browsecomp_plus": _IR_ROOT / "browsecomp_plus",
        }
        args.data_path = str(data_paths[args.dataset])
        print(f"Auto-selected data path: {args.data_path}")

    if args.dataset_year is None and args.dataset in ["trec_rag", "neuclir"]:
        args.dataset_year = "2024"
        print(f"Auto-selected dataset year: {args.dataset_year}")

    if args.subset is None:
        if args.dataset == "neuclir":
            args.subset = "news"
            print(f"Auto-selected subset: {args.subset}")
        elif args.dataset == "browsecomp_plus":
            args.subset = "test"
            print(f"Auto-selected subset: {args.subset}")
        # trec_rag: subset stays None (year alone identifies the split)

    if args.min_relevance_score is None and args.dataset in ["trec_rag", "neuclir"]:
        args.min_relevance_score = 3

    if args.query_key is None:
        args.query_key = "topic_description" if args.dataset == "neuclir" else "text"
        print(f"Auto-selected query key: {args.query_key}")

    if args.dataset in ["neuclir", "trec_rag", "browsecomp_plus"]:
        if args.index_dir is None:
            args.index_dir = str(_IR_ROOT / args.dataset / "indices")
            print(f"Auto-selected index directory: {args.index_dir}")
        if args.corpus_path is None:
            if args.dataset == "neuclir":
                _neuclir_corpus_map = {
                    "news":      "corpus_en_news.jsonl",
                    "technical": "corpus_en_technical.jsonl",
                }
                _corpus_file = _neuclir_corpus_map.get(args.subset, f"corpus_en_{args.subset}.jsonl")
                args.corpus_path = str(_IR_ROOT / "neuclir" / "corpus" / _corpus_file)
            elif args.dataset == "browsecomp_plus":
                args.corpus_path = str(_IR_ROOT / "browsecomp_plus" / "corpus" / "corpus.jsonl")
            else:
                args.corpus_path = str(_IR_ROOT / args.dataset / "corpus" / "msmarco_v2.1_doc_segmented_pyserini_format.jsonl")
            print(f"Auto-selected corpus path: {args.corpus_path}")

    if getattr(args, "qrels_data_path", None) is None:
        args.qrels_data_path = args.data_path


def _detect_num_gpus(args_num_gpus: int) -> int:
    """Resolve the number of GPU workers.

    When *args_num_gpus* is 0, auto-detects via ``torch.cuda.device_count()``.
    Falls back to 1 if PyTorch is not installed.
    """
    if args_num_gpus != 0:
        return args_num_gpus
    try:
        import torch
        detected = torch.cuda.device_count()
        num_gpus = max(1, detected)
        if detected > 1:
            print(f"Auto-detected {detected} GPUs — enabling parallel mode ({num_gpus} workers)")
        else:
            print(f"Auto-detected {detected} GPU(s) — running single-GPU / sequential mode")
        return num_gpus
    except ImportError:
        return 1


def _setup_llm(args, num_gpus: int):
    """Instantiate the LLM client from parsed CLI args.

    Returns:
        llm_client — may be None for self-managed or multi-GPU workers
        that load HF models locally on their assigned GPU.
    """
    llm_client = None

    # REASONING_AGENTS (including cpm_report which now uses the unified path)
    if args.agentic_model in SELF_MANAGED_LLM_AGENTS:
        pass  # These agents create their own client in __init__
    elif is_local_finetuned(args.agentic_model, args.llm_model):
        hf_model = args.llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
        from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
        llm_client = LiteLLMClient(
            model=f"openai/{hf_model}",
            api_base=api_base,
            api_key="EMPTY",
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens_per_call,
            request_timeout=args.request_timeout,
        )
        print(f"Using vLLM-served finetuned model: {hf_model} at {api_base}")
    else:
        from agentic_retrieval_research.utils.setup_utils import setup_llm_client
        llm_client = setup_llm_client(
            model_name=args.llm_model,
            temperature=args.llm_temperature,
            top_p=args.llm_top_p,
            max_completion_tokens=args.llm_max_tokens_per_call,
            metadata={"model": args.llm_model},
            request_timeout=args.request_timeout,
        )

    return llm_client


def _setup_retriever(args):
    """Instantiate the retriever from parsed CLI args."""
    # Public datasets — local retrievers
    from agentic_retrieval_research.searcher_component.retriever import (
        BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever,
    )

    config = _RetrieverConfig(
        retriever_name=args.retriever,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        topk=args.top_k,
    )
    if args.retriever == "qwen3_emb":
        config.qwen3_size = getattr(args, 'qwen3_size', '4B')
    if args.dataset == "browsecomp_plus" and args.retriever == "qwen3_emb":
        config.retrieval_query_max_length = 8196
    if args.dataset == "browsecomp_plus" and args.retriever == "agentir_4b":
        config.retrieval_query_max_length = 8196
    if args.retriever == "bm25":
        return BM25Retriever(config)
    elif args.retriever in ("spladepp", "spladev3"):
        return SPLADERetriever(config)
    elif args.retriever in ["rerank_l6", "rerank_l12"]:
        return RerankRetriever(config)
    else:
        return DenseRetriever(config)


def _assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus: int, verbose: bool, gpu_ids=None) -> dict:
    """Build the kwargs dict to pass to ``run_pipeline()`` (includes worker_config).

    Also embeds the serialisable ``worker_config`` sub-dict that spawned GPU
    workers use to reconstruct the full agent stack from scratch.
    """
    pipeline_kwargs = {
        "retriever":                  retriever,
        "llm_client":                 llm_client,
        "llm_model":                  args.llm_model,
        "top_k":                      args.top_k,
        "seen_top_k":                 args.seen_top_k,
        "dataset":                    args.dataset,
        "min_relevance_score":        args.min_relevance_score,
        "k_values":                   args.k_values,
        "interleaving_window":        args.interleaving_window,
        "rrf_k":                      args.rrf_k,
        # Universal iteration / retry limits
        "max_iteration":              args.max_iteration,
        "max_retries":                args.max_retries,
    }

    if args.dataset in ["trec_rag", "neuclir", "browsecomp_plus"]:
        pipeline_kwargs["retriever_name"] = args.retriever
        pipeline_kwargs["qwen3_size"] = getattr(args, "qwen3_size", "4B")

    pipeline_kwargs["qrels_data_path"] = getattr(args, "qrels_data_path", None)

    pipeline_kwargs["temperature"] = args.llm_temperature
    pipeline_kwargs["max_output_tokens_total"] = args.max_output_tokens_total
    pipeline_kwargs["use_plan"] = getattr(args, "use_plan", False)
    pipeline_kwargs["max_extend_steps"] = args.max_extend_steps
    pipeline_kwargs["hard_mode"] = args.hard_mode
    pipeline_kwargs["oracle_outline_path"] = getattr(args, "with_oracle_outline", None)

    # Trajectory tracker (applies to all agent types)
    pipeline_kwargs["trajectory_tracker"] = getattr(args, "trajectory_tracker", "monitor")
    pipeline_kwargs["llm_intervene"] = getattr(args, "llm_intervene", None)
    pipeline_kwargs["llm_decision_maker"] = getattr(args, "llm_decision_maker", None)
    pipeline_kwargs["decision_maker_history_window"] = getattr(args, "decision_maker_history_window", None)
    pipeline_kwargs["dm_prompt_variant"] = getattr(args, "dm_prompt_variant", "nov_cov_sim")
    pipeline_kwargs["aspect_coverage_mode"] = getattr(args, "aspect_coverage_mode", "dynamic")
    pipeline_kwargs["aspect_coverage_max_aspects"] = getattr(args, "aspect_coverage_max_aspects", 8)
    pipeline_kwargs["llm_aspect_coverage"] = getattr(args, "llm_aspect_coverage", None)

    # Serialisable config for spawned GPU workers (no live objects)
    worker_config = {
        "agentic_model":           args.agentic_model,
        "llm_model":               args.llm_model,
        "llm_temperature":         args.llm_temperature,
        "llm_max_tokens_per_call":          args.llm_max_tokens_per_call,
        "llm_top_p":               args.llm_top_p,
        "request_timeout":         args.request_timeout,
        "dataset":                 args.dataset,
        "retriever_type":          args.retriever,
        "index_dir":               getattr(args, "index_dir", None),
        "corpus_path":             getattr(args, "corpus_path", None),
        "top_k":                   args.top_k,
        "seen_top_k":              args.seen_top_k,
        "max_iteration":           args.max_iteration,
        "max_extend_steps":        args.max_extend_steps,
        "max_retries":             args.max_retries,
        "hard_mode":               args.hard_mode,
        "temperature":             args.llm_temperature,
        "verbose":                 verbose,
        "gpu_ids":                 gpu_ids if gpu_ids is not None else list(range(num_gpus)),
        # Reranker settings (serialisable strings/dicts, not live objects)
        "post_retrieval_reranker_type": args.post_retrieval_reranker,
        "post_fusion_reranker_type":    args.post_fusion_reranker,
        "rerank_top_k":                 args.rerank_top_k,
        "retrieval_input":              args.retrieval_input,
        "post_fusion_reranker_input":   args.post_fusion_reranker_input,
        "qwen3_size":                   getattr(args, "qwen3_size", "4B"),
        "max_output_tokens_total":       getattr(args, "max_output_tokens_total", 40000),
        "use_plan":                      getattr(args, "use_plan", False),
        "oracle_outline_path":           getattr(args, "with_oracle_outline", None),
        # Trajectory tracker
        "trajectory_tracker":            getattr(args, "trajectory_tracker", "monitor"),
        "llm_intervene":        getattr(args, "llm_intervene", None),
        "llm_decision_maker":           getattr(args, "llm_decision_maker", None),
        "decision_maker_history_window": getattr(args, "decision_maker_history_window", None),
        "dm_prompt_variant":             getattr(args, "dm_prompt_variant", "nov_cov_sim"),
        "aspect_coverage_mode":          getattr(args, "aspect_coverage_mode", "dynamic"),
        "aspect_coverage_max_aspects":   getattr(args, "aspect_coverage_max_aspects", 8),
        "llm_aspect_coverage":           getattr(args, "llm_aspect_coverage", None),
        # Searcher novelty filter
        "ensure_novel_seen_docs":        getattr(args, "ensure_novel_seen_docs", False),
        "quiet":                         getattr(args, "quiet", False),
    }

    pipeline_kwargs["num_gpus"]      = num_gpus
    pipeline_kwargs["worker_config"] = worker_config
    return pipeline_kwargs


def _cleanup_event_loop() -> None:
    """Cancel any pending async tasks and close the current event loop."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if not loop.is_closed():
                loop.close()
    except Exception:
        pass
