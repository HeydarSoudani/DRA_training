"""Multi-GPU worker helpers for the deep research pipeline.

Top-level functions so they are picklable by multiprocessing.spawn.
"""

import contextlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

from agentic_retrieval_research.utils.s3_utils import is_s3_path

from .constants import (
    FINETUNED_MODEL_MAP,
    SELF_MANAGED_LLM_AGENTS,
    _RetrieverConfig,
    get_reranker_configs,
    is_local_finetuned,
)
from .tracker_factory import build_trajectory_tracker
from .cited_docs import _build_cited_docs_ranked_list


def _build_components_from_config(worker_config: dict):
    """Rebuild LLM client and retriever from a serializable config dict.

    Called inside spawned worker processes to avoid pickling live objects.

    Returns:
        (llm_client, retriever)  — llm_client may be None for self-managed agents.
    """
    _script_dir = Path(__file__).resolve().parent
    _repo_root  = _script_dir.parents[2]
    _pipeline_dir = _script_dir.parent
    for _p in [str(_repo_root / "src"), str(_repo_root), str(_pipeline_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    agentic_model = worker_config["agentic_model"]
    llm_model     = worker_config["llm_model"]
    dataset       = worker_config["dataset"]

    llm_client = None

    if agentic_model in SELF_MANAGED_LLM_AGENTS:
        pass
    elif is_local_finetuned(agentic_model, llm_model):
        hf_model = llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
        from agentic_retrieval_research.llm_utils.litellm_client import LiteLLMClient
        llm_client = LiteLLMClient(
            model=f"openai/{hf_model}",
            api_base=api_base,
            api_key="EMPTY",
            temperature=worker_config["llm_temperature"],
            max_tokens=worker_config["llm_max_tokens_per_call"],
            request_timeout=worker_config.get("request_timeout", 300),
        )
    else:
        from agentic_retrieval_research.utils.setup_utils import setup_llm_client
        llm_client = setup_llm_client(
            model_name=llm_model,
            temperature=worker_config["llm_temperature"],
            top_p=worker_config["llm_top_p"],
            max_completion_tokens=worker_config["llm_max_tokens_per_call"],
            metadata={"model": llm_model},
            request_timeout=worker_config.get("request_timeout"),
        )

    from agentic_retrieval_research.searcher_component.retriever import (
        BM25Retriever, RerankRetriever, DenseRetriever, SPLADERetriever,
    )

    cfg = _RetrieverConfig(
        retriever_name=worker_config["retriever_type"],
        index_dir=worker_config.get("index_dir"),
        corpus_path=worker_config.get("corpus_path"),
        topk=worker_config["top_k"],
    )
    if worker_config["retriever_type"] == "qwen3_emb":
        cfg.qwen3_size = worker_config.get("qwen3_size", "4B")
    if dataset == "browsecomp_plus" and worker_config["retriever_type"] == "qwen3_emb":
        cfg.retrieval_query_max_length = 8196
    retriever_type = worker_config["retriever_type"]
    if retriever_type == "bm25":
        retriever = BM25Retriever(cfg)
    elif retriever_type in ("spladepp", "spladev3"):
        retriever = SPLADERetriever(cfg)
    elif retriever_type in ["rerank_l6", "rerank_l12"]:
        retriever = RerankRetriever(cfg)
    else:
        retriever = DenseRetriever(cfg)

    return llm_client, retriever


def _init_worker(worker_id: int, worker_config: dict):
    """Initialise a GPU worker: pin GPU, load models, build agent.

    Returns:
        (agent, search_tool, verbose)
    """
    gpu_ids = worker_config.get("gpu_ids", [])
    verbose  = worker_config.get("verbose", False)
    if worker_config.get("quiet", False):
        logging.getLogger("agents").setLevel(logging.ERROR)
        logging.getLogger("agent_tools").setLevel(logging.ERROR)
        logging.getLogger("utils").setLevel(logging.ERROR)
        logging.getLogger("prompts").setLevel(logging.ERROR)
    if gpu_ids and worker_id < len(gpu_ids):
        physical_gpu = gpu_ids[worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES={physical_gpu}", flush=True)
        import torch
        if torch.cuda.is_available():
            n_visible = torch.cuda.device_count()
            if n_visible == 1:
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(physical_gpu)
            print(f"[Worker {worker_id}] torch.cuda.current_device()={torch.cuda.current_device()}, "
                  f"device_count={n_visible}", flush=True)
    else:
        print(f"[Worker {worker_id}] No specific GPU assigned (gpu_ids={gpu_ids})", flush=True)

    _script_dir   = Path(__file__).resolve().parent
    _repo_root    = _script_dir.parents[2]
    _pipeline_dir = _script_dir.parent
    for _p in [str(_repo_root / "src"), str(_repo_root), str(_pipeline_dir)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import shutil
    _java_bin = shutil.which("java")
    if _java_bin:
        _java_real = os.path.realpath(_java_bin)
        _java_home = os.path.dirname(os.path.dirname(_java_real))
        _jvm_so = os.path.join(_java_home, "lib", "server", "libjvm.so")
        if os.path.isfile(_jvm_so):
            os.environ["JAVA_HOME"] = _java_home
            os.environ["JVM_PATH"] = _jvm_so

    try:
        import jnius_config
        jnius_config.add_options(
            '-Xmx2g',
            '-Xms512m',
            '-XX:ParallelGCThreads=4',
            '-XX:ConcGCThreads=2',
        )
    except ImportError:
        pass

    print(f"[Worker {worker_id}] Building components (LLM, retriever)...", flush=True)
    llm_client, retriever = _build_components_from_config(worker_config)
    print(f"[Worker {worker_id}] Components built", flush=True)

    agentic_model = worker_config["agentic_model"]
    dataset = worker_config["dataset"]

    from agentic_retrieval_research.searcher_component.searcher import RetrievalSearchTool
    from decision_making_agent.utils import build_reranker_from_config

    _reranker_configs = get_reranker_configs(worker_config.get("rerank_top_k", 100))

    _post_ret_type = worker_config.get("post_retrieval_reranker_type", "null")
    _post_fus_type = worker_config.get("post_fusion_reranker_type", "null")
    _post_ret_reranker = build_reranker_from_config(_post_ret_type, _reranker_configs) if _post_ret_type != "null" else None
    _post_fus_reranker = build_reranker_from_config(_post_fus_type, _reranker_configs) if _post_fus_type != "null" else None

    search_tool = None
    if retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=retriever,
            post_retrieval_reranker=_post_ret_reranker,
            post_fusion_reranker=_post_fus_reranker,
            top_k=worker_config.get("top_k", 100),
            rerank_top_k=worker_config.get("rerank_top_k", 100),
            retrieval_input=worker_config.get("retrieval_input", "subquery"),
            post_fusion_reranker_input=worker_config.get("post_fusion_reranker_input", "original_query"),
            ensure_novel_seen_docs=worker_config.get("ensure_novel_seen_docs", False),
            seen_top_k=worker_config.get("seen_top_k", 5),
        )

    print(f"[Worker {worker_id}] Creating agent ({agentic_model})...", flush=True)
    from agents import AGENT_MAP, OSS_BedrockAgent, BEDROCK_OSS_MODELS
    llm_model = worker_config["llm_model"]
    model_class = AGENT_MAP[agentic_model]
    _max_out = worker_config.get("max_output_tokens_total", 40000)
    _reasoning_extra: dict = {}
    if worker_config.get("use_plan", False) and agentic_model == "react":
        _reasoning_extra["use_plan"] = True
    if agentic_model == "glm":
        _reasoning_extra["max_output_tokens"] = min(_max_out, 20000)
    elif agentic_model == "oss":
        _reasoning_extra["max_output_tokens"] = _max_out
        _reasoning_extra["model_name"] = f"openai/{llm_model}"
    elif agentic_model == "tongyi":
        _reasoning_extra["max_tokens_per_step"] = min(_max_out, 20000)
    elif agentic_model == "cpm_explore":
        _reasoning_extra["max_output_tokens"] = min(_max_out, 16384)
        _reasoning_extra["temperature"] = worker_config.get("temperature") or 1.0
    elif agentic_model == "cpm_report":
        _reasoning_extra["max_extend_steps"] = worker_config.get("max_extend_steps", 5)
        _reasoning_extra["max_retries"] = worker_config.get("max_retries", 3)
        _reasoning_extra["hard_mode"] = worker_config.get("hard_mode", True)
        _reasoning_extra["oracle_outline_path"] = worker_config.get("oracle_outline_path")
        _reasoning_extra["max_passage_chars"] = worker_config.get("max_passage_chars", 4000)
        _reasoning_extra["model_name"] = llm_model
        _reasoning_extra["model_url"] = worker_config.get("model_url")

    if agentic_model == "oss" and llm_model in BEDROCK_OSS_MODELS:
        agent = OSS_BedrockAgent(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=worker_config.get("max_iteration", 20),
            seen_top_k=worker_config.get("seen_top_k", 5),
            model_name=BEDROCK_OSS_MODELS[llm_model],
            max_output_tokens=_max_out,
            verbose=verbose,
        )
    else:
        agent = model_class(
            llm_client=llm_client,
            retriever=retriever,
            max_iteration=worker_config.get("max_iteration", 20),
            seen_top_k=worker_config.get("seen_top_k", 5),
            verbose=verbose,
            **_reasoning_extra,
        )

    if agent is not None and search_tool is not None and hasattr(agent, "search_tool"):
        agent.search_tool = search_tool

    _tt_mode = worker_config.get("trajectory_tracker", "monitor")
    trajectory_tracker = build_trajectory_tracker(
        tt_mode=_tt_mode,
        retriever=retriever,
        qrels=worker_config.get("qrels"),
        seen_top_k=worker_config.get("seen_top_k", 5),
        llm_decision_maker=worker_config.get("llm_decision_maker"),
        llm_intervene=worker_config.get("llm_intervene"),
        agent=agent if hasattr(agent, "trajectory_tracker") else None,
        agentic_model=agentic_model,
        strict=False,
        decision_maker_history_window=worker_config.get("decision_maker_history_window"),
        ac_temperature=worker_config.get("temperature", 0.7),
        dm_prompt_variant=worker_config.get("dm_prompt_variant", "nov_cov_sim"),
        max_iteration=worker_config.get("max_iteration"),
        aspect_coverage_mode=worker_config.get("aspect_coverage_mode", "dynamic"),
        aspect_coverage_max_aspects=worker_config.get("aspect_coverage_max_aspects", 8),
        llm_aspect_coverage=worker_config.get("llm_aspect_coverage"),
        dataset=dataset,
    )
    if trajectory_tracker is not None and hasattr(agent, "trajectory_tracker"):
        agent.trajectory_tracker = trajectory_tracker

    print(f"[Worker {worker_id}] Initialisation complete", flush=True)
    return agent, search_tool, verbose


def _gpu_worker(worker_id: int, query_items: list, temp_dir_str: str, worker_config: dict, progress_queue=None, init_lock=None) -> dict:
    """Worker process: pin to one GPU, rebuild the agent, process queries, save files."""
    _lock_ctx = init_lock if init_lock is not None else contextlib.nullcontext()

    try:
        with _lock_ctx:
            agent, search_tool, verbose = _init_worker(
                worker_id, worker_config,
            )
    except Exception as exc:
        print(f"[Worker {worker_id}] INIT FAILED: {exc}", flush=True)
        traceback.print_exc()
        if progress_queue is not None:
            progress_queue.put(None)
        return {}

    _is_s3_out = is_s3_path(temp_dir_str)
    if _is_s3_out:
        retrieval_dir  = f"{temp_dir_str.rstrip('/')}/retrieval"
        generation_dir = f"{temp_dir_str.rstrip('/')}/generation"
        trajectory_dir = f"{temp_dir_str.rstrip('/')}/trajectory"
        cited_doc_dir  = f"{temp_dir_str.rstrip('/')}/cited_docs_retrieval"
        seen_doc_dir   = f"{temp_dir_str.rstrip('/')}/seen_docs_retrieval"
        tracker_dir    = f"{temp_dir_str.rstrip('/')}/tracker"
    else:
        temp_dir       = Path(temp_dir_str)
        retrieval_dir  = str(temp_dir / "retrieval")
        generation_dir = str(temp_dir / "generation")
        trajectory_dir = str(temp_dir / "trajectory")
        cited_doc_dir  = str(temp_dir / "cited_docs_retrieval")
        seen_doc_dir   = str(temp_dir / "seen_docs_retrieval")
        tracker_dir    = str(temp_dir / "tracker")
        for _d in [retrieval_dir, generation_dir, trajectory_dir, cited_doc_dir, seen_doc_dir, tracker_dir]:
            Path(_d).mkdir(parents=True, exist_ok=True)

    from agentic_retrieval_research.evaluation import RetrievalEvaluator, GenerationEvaluator, TrajectoryEvaluator, TrackerEvaluator, CitedDocRetrievalEvaluator, SeenDocRetrievalEvaluator
    _ret_eval     = RetrievalEvaluator(qrels={}, k_values=[])
    _gen_eval     = GenerationEvaluator()
    _traj_eval    = TrajectoryEvaluator()
    _tracker_eval = TrackerEvaluator()
    _cited_eval   = CitedDocRetrievalEvaluator(qrels={}, k_values=[])
    _seen_eval    = SeenDocRetrievalEvaluator(qrels={}, k_values=[])

    results     = {}
    temperature = worker_config.get("temperature", 0.7)
    total       = len(query_items)
    max_iteration = worker_config.get("max_iteration", "?")

    _cb_state = [None, 0]

    def _status_cb(stage: str, iteration: int):
        if progress_queue is not None:
            progress_queue.put(
                (worker_id, _cb_state[0], _cb_state[1], total, "update",
                 (stage, iteration, max_iteration))
            )

    for idx, (query_id, query_text) in enumerate(query_items, 1):
        _cb_state[0] = query_id
        _cb_state[1] = idx
        if search_tool is not None:
            search_tool.reset()
        if progress_queue is not None:
            progress_queue.put((worker_id, query_id, idx, total, "processing", None))
        if verbose:
            print(
                f"  [Worker {worker_id}] [{idx}/{len(query_items)}] Processing query: {query_id}\n    Query text: {query_text}",
                flush=True,
            )
        result = agent.run_single(
            query_id=query_id,
            query_text=query_text,
            temperature=temperature,
            status_callback=_status_cb if progress_queue is not None else None,
        )
        if result is None:
            if verbose:
                print(f"  [Worker {worker_id}] ✗ Skipping {query_id}", flush=True)
            if progress_queue is not None:
                progress_queue.put((worker_id, query_id, idx, total, "skipped", None))
            continue
        result["cited_docs_ranked_list"] = _build_cited_docs_ranked_list(result)
        from utils.doc_formatting import build_references_section
        references = build_references_section(result)
        if references:
            result["generation"] = result["generation"].rstrip() + references
        results[query_id] = result
        _ret_eval.save_item(query_id, result, retrieval_dir)
        _gen_eval.save_item(query_id, result, generation_dir)
        _traj_eval.save_item(query_id, query_text, result, trajectory_dir)
        _cited_eval.save_item(query_id, result, cited_doc_dir)
        _seen_eval.save_item(query_id, result, seen_doc_dir)
        _tracker_eval.save_item(query_id, query_text, result, tracker_dir)
        if progress_queue is not None:
            num_iters = result.get("num_iterations", "?")
            progress_queue.put((worker_id, query_id, idx, total, "done", f"{num_iters}/{max_iteration}"))
        if verbose:
            print(f"  [Worker {worker_id}] ✓ Saved: {query_id}", flush=True)

    agent.cleanup()
    from agentic_retrieval_research.utils.s3_utils import detach_s3fs_finalizers
    detach_s3fs_finalizers()
    if verbose:
        print(
            f"[Worker {worker_id}] Completed {len(results)}/{len(query_items)} queries",
            flush=True,
        )
    if progress_queue is not None:
        progress_queue.put(None)
    return results
