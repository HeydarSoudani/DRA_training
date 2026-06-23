"""CLI argument resolution and pipeline-kwargs assembly.

Leaf-layer helpers that turn parsed argparse args into the dataset defaults,
worker config, and pipeline kwargs consumed by the orchestration layer. Depends
only on stdlib and ``utils`` — no heavy component imports.
"""

import asyncio
import logging

from utils.config import _IR_ROOT, _DATA_ROOT

logger = logging.getLogger(__name__)


def resolve_dataset_defaults(args) -> None:
    """Auto-select dataset-related CLI arguments that were left as None."""
    if args.data_path is None:
        data_paths = {
            "trec_rag":        _IR_ROOT / "trec_rag",
            "neuclir":         _IR_ROOT / "neuclir",
            "neuclir_oracle_criteria_augmented": _DATA_ROOT / "neuclir_oracle_criteria_augmented",
            "neuclir_prompt_criteria_augmented": _DATA_ROOT / "neuclir_prompt_criteria_augmented",
            "browsecomp_plus": _IR_ROOT / "browsecomp_plus",
        }
        args.data_path = str(data_paths[args.dataset])
        print(f"Auto-selected data path: {args.data_path}")

    if args.dataset_year is None and args.dataset in ["trec_rag", "neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented"]:
        args.dataset_year = "2024"
        print(f"Auto-selected dataset year: {args.dataset_year}")

    if args.subset is None:
        if args.dataset in ["neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented"]:
            args.subset = "news"
            print(f"Auto-selected subset: {args.subset}")
        elif args.dataset == "browsecomp_plus":
            args.subset = "test"
            print(f"Auto-selected subset: {args.subset}")

    if args.min_relevance_score is None and args.dataset in ["trec_rag", "neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented"]:
        args.min_relevance_score = 3

    if args.query_key is None:
        args.query_key = "topic_description" if args.dataset == "neuclir" else "text"
        print(f"Auto-selected query key: {args.query_key}")

    _NEUCLIR_FAMILY = {"neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented"}
    if args.dataset in _NEUCLIR_FAMILY | {"trec_rag", "browsecomp_plus"}:
        if args.index_dir is None:
            _index_dataset = "neuclir" if args.dataset in (_NEUCLIR_FAMILY - {"neuclir"}) else args.dataset
            args.index_dir = str(_IR_ROOT / _index_dataset / "indices")
            print(f"Auto-selected index directory: {args.index_dir}")
        if args.corpus_path is None:
            if args.dataset in _NEUCLIR_FAMILY:
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

    _QRELS_PARENT = {
        "neuclir_oracle_criteria_augmented": _IR_ROOT / "neuclir",
        "neuclir_prompt_criteria_augmented": _IR_ROOT / "neuclir",
    }
    if getattr(args, "qrels_data_path", None) is None:
        _parent_path = _QRELS_PARENT.get(args.dataset)
        if _parent_path is not None:
            args.qrels_data_path = str(_parent_path)
            print(f"Auto-selected qrels data path: {args.qrels_data_path}")
        else:
            args.qrels_data_path = args.data_path


def detect_num_gpus(args_num_gpus: int) -> int:
    """Resolve the number of GPU workers."""
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


def assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus: int, verbose: bool, gpu_ids=None) -> dict:
    """Build the kwargs dict to pass to run_pipeline() (includes worker_config)."""
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
        "max_iteration":              args.max_iteration,
        "max_retries":                args.max_retries,
    }

    if args.dataset in ["trec_rag", "neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented", "browsecomp_plus"]:
        pipeline_kwargs["retriever_name"] = args.retriever
        pipeline_kwargs["qwen3_size"] = getattr(args, "qwen3_size", "4B")

    pipeline_kwargs["qrels_data_path"] = getattr(args, "qrels_data_path", None)
    pipeline_kwargs["temperature"] = args.llm_temperature
    pipeline_kwargs["max_output_tokens_total"] = args.max_output_tokens_total
    pipeline_kwargs["use_plan"] = getattr(args, "use_plan", False)
    pipeline_kwargs["max_extend_steps"] = args.max_extend_steps
    pipeline_kwargs["hard_mode"] = args.hard_mode
    pipeline_kwargs["oracle_outline_path"] = getattr(args, "with_oracle_outline", None)

    pipeline_kwargs["controller"] = getattr(args, "controller", "monitor")
    pipeline_kwargs["llm_intervene"] = getattr(args, "llm_intervene", None)
    pipeline_kwargs["llm_controller"] = getattr(args, "llm_controller", None)
    pipeline_kwargs["controller_history_window"] = getattr(args, "controller_history_window", None)
    pipeline_kwargs["controller_prompt_variant"] = getattr(args, "controller_prompt_variant", "nov_cov_sim")
    pipeline_kwargs["criteria_coverage_mode"] = getattr(args, "criteria_coverage_mode", "dynamic")
    pipeline_kwargs["criteria_coverage_max_criteria"] = getattr(args, "criteria_coverage_max_criteria", 8)
    pipeline_kwargs["llm_criteria_coverage"] = getattr(args, "llm_criteria_coverage", None)

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
        "post_retrieval_reranker_type": args.post_retrieval_reranker,
        "post_fusion_reranker_type":    args.post_fusion_reranker,
        "rerank_top_k":                 args.rerank_top_k,
        "retrieval_input":              args.retrieval_input,
        "post_fusion_reranker_input":   args.post_fusion_reranker_input,
        "qwen3_size":                   getattr(args, "qwen3_size", "4B"),
        "max_output_tokens_total":       getattr(args, "max_output_tokens_total", 40000),
        "use_plan":                      getattr(args, "use_plan", False),
        "oracle_outline_path":           getattr(args, "with_oracle_outline", None),
        "controller":                    getattr(args, "controller", "monitor"),
        "llm_intervene":        getattr(args, "llm_intervene", None),
        "llm_controller":           getattr(args, "llm_controller", None),
        "controller_history_window": getattr(args, "controller_history_window", None),
        "controller_prompt_variant":             getattr(args, "controller_prompt_variant", "nov_cov_sim"),
        "criteria_coverage_mode":          getattr(args, "criteria_coverage_mode", "dynamic"),
        "criteria_coverage_max_criteria":   getattr(args, "criteria_coverage_max_criteria", 8),
        "llm_criteria_coverage":           getattr(args, "llm_criteria_coverage", None),
        "ensure_novel_seen_docs":        getattr(args, "ensure_novel_seen_docs", False),
        "quiet":                         getattr(args, "quiet", False),
    }

    pipeline_kwargs["num_gpus"]      = num_gpus
    pipeline_kwargs["worker_config"] = worker_config
    return pipeline_kwargs


def cleanup_event_loop() -> None:
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
