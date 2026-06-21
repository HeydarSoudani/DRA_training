"""Pipeline orchestration: controller factory, multi-GPU workers, CLI helpers.

Top-level worker functions (_build_components_from_config, _init_worker, _gpu_worker)
must stay at module level to be picklable by multiprocessing.spawn.
"""

import asyncio
import contextlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io_utils import (
    get_processed_queries,
    load_result_from_trec,
    load_result_from_saved_files,
    setup_output_dirs,
    load_json_from_path,
    save_json_to_path,
)

from .config import (
    FINETUNED_MODEL_MAP,
    SELF_MANAGED_LLM_AGENTS,
    _RetrieverConfig,
    _IR_ROOT,
    get_reranker_configs,
    is_local_finetuned,
)
from .text_utils import _build_cited_docs_ranked_list

logger = logging.getLogger(__name__)

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


# ===========================================================================
# LLM client / retriever factories
# ===========================================================================

def setup_llm_client(
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_completion_tokens: int = 2000,
    metadata: Optional[Dict[str, Any]] = None,
    request_timeout: Optional[int] = None,
):
    """Setup LLM client for generation tasks.

    Args:
        model_name: Name of the LLM model to use
        temperature: Sampling temperature for generation (default: 0.7)
        top_p: Nucleus sampling parameter (default: 1.0)
        max_completion_tokens: Maximum tokens to generate (default: 2000)
        metadata: Optional metadata to attach to LLM calls
        request_timeout: HTTP request timeout in seconds (default: None, uses LiteLLM default)

    Returns:
        Configured LiteLLMClient instance
    """
    from utils.llm_client import get_litellm_client

    logger.info(f"Setting up LLM client: {model_name}...")

    if metadata is None:
        metadata = {"model": model_name}

    extra_kwargs = {}
    if request_timeout is not None:
        extra_kwargs["request_timeout"] = request_timeout

    llm_client = get_litellm_client(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        metadata=metadata,
        **extra_kwargs,
    )

    logger.info(f"LLM client configured: model={model_name}, temp={temperature}")
    return llm_client


def setup_retriever(
    region: str,
    top_k: int = 10,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    llm_selector: bool = False,
    **kwargs,
):
    """Setup retriever endpoint client.

    Args:
        region: Region for retrieval ("us" or "uk")
        top_k: Number of passages to retrieve (default: 10)
        base_url: Optional base URL for retrieval endpoint
        endpoint: Optional endpoint path
        endpoint_name: Optional named endpoint configuration
        llm_selector: Whether to activate LLM selector (default: False)
        **kwargs: Additional arguments to pass to retriever

    Returns:
        Configured retriever endpoint
    """
    logger.info("Setting up retriever endpoint...")

    # Endpoint-based retrieval is an optional path; import lazily so this module
    # loads even when the endpoint_retrieval helper is unavailable.
    try:
        from agentic_retrieval_research.retrieval_utils.endpoint_retrieval import get_retriever_endpoint
    except ImportError as e:
        raise ImportError(
            "Endpoint-based retrieval (setup_retriever) requires "
            "retrieval_utils.endpoint_retrieval, which is not available in this "
            "deployment. Use the local retriever path instead."
        ) from e

    # Build retriever config
    config = {
        "region": region,
        "top_k": top_k,
        **kwargs,
    }

    # Add LLM selector if specified
    if llm_selector:
        config["llm_selector"] = llm_selector

    # Add base_url if provided
    if base_url:
        config["base_url"] = base_url

    # Determine endpoint path
    if endpoint:
        config["endpoint"] = endpoint
    elif not endpoint_name:
        # Use default endpoints if no endpoint or endpoint_name specified
        default_endpoints = {
            "us": "/v2/rag/pl-rag-v061-experimental",
            "uk": "/v2/rag/pl-uk-rag-v061-experimental",
        }
        config["endpoint"] = default_endpoints.get(region, default_endpoints["us"])

    # Create retriever
    if endpoint_name:
        retriever = get_retriever_endpoint(
            endpoint_name=endpoint_name,
            **config,
        )
    else:
        retriever = get_retriever_endpoint(**config)

    logger.info(f"Retriever configured: region={region}, top_k={top_k}")
    return retriever


# ===========================================================================
# Controller factory
# ===========================================================================

def build_controller(
    controller_mode: str,
    retriever=None,
    qrels=None,
    seen_top_k: int = 5,
    llm_controller: Optional[str] = None,
    llm_intervene: Optional[str] = None,
    agent=None,
    agentic_model: Optional[str] = None,
    strict: bool = True,
    controller_history_window: Optional[int] = None,
    controller_prompt_variant: str = "nov_cov_sim",
    max_iteration: Optional[int] = None,
    criteria_coverage_mode: str = "dynamic",
    criteria_coverage_max_criteria: int = 8,
    llm_criteria_coverage: Optional[str] = None,
    ac_temperature: float = 0.0,
    dataset: Optional[str] = None,
):
    """Build a Controller with controller and answer candidate generator.

    Shared by both the main process (run_pipeline) and spawned GPU workers.

    Args:
        controller_mode: "off", "monitor", or "action".
        retriever: Retriever instance (for encoding function).
        qrels: Ground-truth relevance judgements.
        seen_top_k: Number of docs considered "seen" per step.
        llm_controller: Model name for controller LLM.
        llm_intervene: Model name for critical thinking generator LLM.
        agent: Agent instance (to wire up answer candidate generator).
        agentic_model: Agent type name (for format selection).
        strict: If True, raise ValueError for missing required params.
        criteria_coverage_mode: "static" or "dynamic".
        criteria_coverage_max_criteria: Soft cap on the number of criteria.
        llm_criteria_coverage: Model name for criteria coverage LLM. Falls
            back to llm_controller or llm_intervene if not set.

    Returns:
        Controller instance, or None if controller_mode is "off".
    """
    if controller_mode == "off":
        return None

    from controller_component import (
        Controller, LLMCriticalThinkingGenerator, LLMControllerPolicy,
        encode_fn_from_retriever,
    )
    from controller_component.prompts.answer_prompts import get_candidate_format

    _tt_affect = "none" if controller_mode == "monitor" else "active"

    _tt_notice_gen = None
    if controller_mode == "action":
        if llm_intervene:
            from utils.llm_client import LiteLLMClient
            _tt_llm_client = LiteLLMClient(model=llm_intervene)
            _tt_notice_gen = LLMCriticalThinkingGenerator(llm_client=_tt_llm_client)
        elif strict:
            raise ValueError("--llm-intervene is required when using intervene action")

    _controller_policy = None
    _controller_llm_model = llm_controller or llm_intervene
    if _controller_llm_model:
        from utils.llm_client import LiteLLMClient
        _controller_llm_client = LiteLLMClient(model=_controller_llm_model)
        _controller_policy = LLMControllerPolicy(llm_client=_controller_llm_client, history_window=controller_history_window, controller_prompt_variant=controller_prompt_variant, max_iteration=max_iteration)
    elif controller_mode == "action" and strict:
        raise ValueError(
            "--llm-controller (or --llm-intervene) is required "
            "when --controller=action"
        )

    controller = Controller(
        affect_mode=_tt_affect,
        critical_thinking_generator=_tt_notice_gen,
        encode_fn=encode_fn_from_retriever(retriever) if retriever else None,
        qrels=qrels or {},
        seen_top_k=seen_top_k,
        controller_policy=_controller_policy,
    )

    _ac_llm_model = llm_criteria_coverage or llm_controller or llm_intervene
    if _ac_llm_model:
        from utils.llm_client import LiteLLMClient
        from controller_component import CriteriaCoverageSignal
        _ac_llm_client = LiteLLMClient(model=_ac_llm_model)
        controller._criteria_coverage = CriteriaCoverageSignal(
            llm_client=_ac_llm_client,
            mode=criteria_coverage_mode,
            max_criteria=criteria_coverage_max_criteria,
            temperature=ac_temperature,
        )
        print(f"Criteria coverage signal enabled: mode={criteria_coverage_mode}, model={_ac_llm_model}")
    else:
        logger.warning("No LLM model available for criteria coverage; signal disabled")

    if agent is not None and agentic_model is not None:
        _cfg = getattr(agent, "inference_config", None)
        if _cfg is not None:
            _cfg.format_instructions = get_candidate_format(agentic_model)

        if dataset == "browsecomp_plus" and agentic_model != "cpm_report":
            if hasattr(agent, "generate_answer_candidate"):
                controller._answer_candidate_fn = agent.generate_answer_candidate
                _model_id = _cfg.model_name if _cfg else "unknown"
                print(f"Answer candidate via agent.generate_answer_candidate: {_model_id}")

    return controller


# ===========================================================================
# Multi-GPU workers (must remain top-level for multiprocessing.spawn pickling)
# ===========================================================================

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
        from utils.llm_client import LiteLLMClient
        llm_client = LiteLLMClient(
            model=f"openai/{hf_model}",
            api_base=api_base,
            api_key="EMPTY",
            temperature=worker_config["llm_temperature"],
            max_tokens=worker_config["llm_max_tokens_per_call"],
            request_timeout=worker_config.get("request_timeout", 300),
        )
    else:
        llm_client = setup_llm_client(
            model_name=llm_model,
            temperature=worker_config["llm_temperature"],
            top_p=worker_config["llm_top_p"],
            max_completion_tokens=worker_config["llm_max_tokens_per_call"],
            metadata={"model": llm_model},
            request_timeout=worker_config.get("request_timeout"),
        )

    _UK_FAMILY = {"uk", "uk_oracle_criteria_augmented", "uk_prompt_criteria_augmented"}
    _US_FAMILY = {"us", "us_oracle_criteria_augmented", "us_prompt_criteria_augmented"}
    if dataset in _US_FAMILY | _UK_FAMILY:
        retriever = setup_retriever(
            region="uk" if dataset in _UK_FAMILY else "us",
            top_k=worker_config["top_k"],
            base_url=worker_config["base_url"],
            endpoint=worker_config.get("endpoint"),
            endpoint_name=worker_config.get("endpoint_name"),
            llm_selector=worker_config.get("llm_selector", False),
        )
    else:
        from searcher_component.retriever import (
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

    from agent_tools import RetrievalSearchTool
    from utils.eval_utils import build_reranker_from_config

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

    _controller_mode = worker_config.get("controller", "monitor")
    controller = build_controller(
        controller_mode=_controller_mode,
        retriever=retriever,
        qrels=worker_config.get("qrels"),
        seen_top_k=worker_config.get("seen_top_k", 5),
        llm_controller=worker_config.get("llm_controller"),
        llm_intervene=worker_config.get("llm_intervene"),
        agent=agent if hasattr(agent, "controller") else None,
        agentic_model=agentic_model,
        strict=False,
        controller_history_window=worker_config.get("controller_history_window"),
        ac_temperature=worker_config.get("temperature", 0.7),
        controller_prompt_variant=worker_config.get("controller_prompt_variant", "nov_cov_sim"),
        max_iteration=worker_config.get("max_iteration"),
        criteria_coverage_mode=worker_config.get("criteria_coverage_mode", "dynamic"),
        criteria_coverage_max_criteria=worker_config.get("criteria_coverage_max_criteria", 8),
        llm_criteria_coverage=worker_config.get("llm_criteria_coverage"),
        dataset=dataset,
    )
    if controller is not None and hasattr(agent, "controller"):
        agent.controller = controller

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

    temp_dir       = Path(temp_dir_str)
    retrieval_dir  = str(temp_dir / "retrieval")
    generation_dir = str(temp_dir / "generation")
    trajectory_dir = str(temp_dir / "trajectory")
    cited_doc_dir  = str(temp_dir / "cited_docs_retrieval")
    seen_doc_dir   = str(temp_dir / "seen_docs_retrieval")
    tracker_dir    = str(temp_dir / "tracker")
    for _d in [retrieval_dir, generation_dir, trajectory_dir, cited_doc_dir, seen_doc_dir, tracker_dir]:
        Path(_d).mkdir(parents=True, exist_ok=True)

    from evaluation import RetrievalEvaluator, GenerationEvaluator, TrajectoryEvaluator, TrackerEvaluator, CitedDocRetrievalEvaluator, SeenDocRetrievalEvaluator
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
        from utils.text_utils import build_references_section
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
    if verbose:
        print(
            f"[Worker {worker_id}] Completed {len(results)}/{len(query_items)} queries",
            flush=True,
        )
    if progress_queue is not None:
        progress_queue.put(None)
    return results


# ===========================================================================
# CLI setup helpers
# ===========================================================================

def _resolve_dataset_defaults(args) -> None:
    """Auto-select dataset-related CLI arguments that were left as None."""
    if args.data_path is None:
        data_paths = {
            "us":              _IR_ROOT / "pl_aalp_us",
            "uk":              _IR_ROOT / "pl_aalp_uk",
            "uk_oracle_criteria_augmented":  _DATA_ROOT / "pl_aalp_uk_oracle_criteria_augmented",
            "uk_prompt_criteria_augmented":  _DATA_ROOT / "pl_aalp_uk_prompt_criteria_augmented",
            "us_oracle_criteria_augmented":  _DATA_ROOT / "pl_aalp_us_oracle_criteria_augmented",
            "us_prompt_criteria_augmented":  _DATA_ROOT / "pl_aalp_us_prompt_criteria_augmented",
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
        if args.dataset in ["us", "uk", "uk_oracle_criteria_augmented", "uk_prompt_criteria_augmented", "us_oracle_criteria_augmented", "us_prompt_criteria_augmented"]:
            args.subset = "set1"
            print(f"Auto-selected subset: {args.subset}")
        elif args.dataset in ["neuclir", "neuclir_oracle_criteria_augmented", "neuclir_prompt_criteria_augmented"]:
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
        "uk_oracle_criteria_augmented": _IR_ROOT / "pl_aalp_uk",
        "uk_prompt_criteria_augmented": _IR_ROOT / "pl_aalp_uk",
        "us_oracle_criteria_augmented": _IR_ROOT / "pl_aalp_us",
        "us_prompt_criteria_augmented": _IR_ROOT / "pl_aalp_us",
    }
    if getattr(args, "qrels_data_path", None) is None:
        _parent_path = _QRELS_PARENT.get(args.dataset)
        if _parent_path is not None:
            args.qrels_data_path = str(_parent_path)
            print(f"Auto-selected qrels data path: {args.qrels_data_path}")
        else:
            args.qrels_data_path = args.data_path


def _detect_num_gpus(args_num_gpus: int) -> int:
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


def _setup_llm(args, num_gpus: int):
    """Instantiate the LLM client from parsed CLI args."""
    llm_client = None

    if args.agentic_model in SELF_MANAGED_LLM_AGENTS:
        pass
    elif is_local_finetuned(args.agentic_model, args.llm_model):
        hf_model = args.llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
        from utils.llm_client import LiteLLMClient
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
    _UK_FAMILY = {"uk", "uk_oracle_criteria_augmented", "uk_prompt_criteria_augmented"}
    _US_FAMILY = {"us", "us_oracle_criteria_augmented", "us_prompt_criteria_augmented"}
    if args.dataset in _US_FAMILY | _UK_FAMILY:
        _region = "uk" if args.dataset in _UK_FAMILY else "us"
        return setup_retriever(
            region=_region,
            top_k=args.top_k,
            base_url=args.base_url,
            endpoint=args.endpoint,
            endpoint_name=args.endpoint_name,
            llm_selector=args.llm_selector,
        )

    from searcher_component.retriever import (
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
        "base_url":                args.base_url,
        "endpoint":                getattr(args, "endpoint", None),
        "endpoint_name":           getattr(args, "endpoint_name", None),
        "llm_selector":            args.llm_selector,
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


# ===========================================================================
# Run / output naming helpers
# ===========================================================================

def _build_run_name_for_pipeline(agentic_model: str, llm_model: str, **kwargs) -> str:
    """Build a consistent output directory name for the current pipeline run.

    Template: {agent_name}_agent_{model}
    Dataset/retriever/query_key info lives in the parent dataset_dir.
    """
    name = agentic_model
    if agentic_model == "react":
        name = "react_w_plan" if kwargs.get("use_plan", False) else "react_wo_plan"
    return f"{name}_agent_{llm_model.replace('/', '--')}"


def _build_searcher_config_name(**kwargs) -> str:
    """Build an abbreviated searcher-config directory name.

    Encodes: seen_top_k, post_retrieval_reranker, post_fusion_reranker,
    rerank_top_k, retrieval_input, post_fusion_reranker_input, controller.

    Example: stk10_prr-null_pfr-bat_rrk100_ri-sq_pfri-oq_ctrl-mon
    """
    _reranker_alias = {
        "null":              "null",
        "batched_reranker":  "bat",
        "rankllama":         "rll",
        "rank1":             "rk1",
        "qwen3_reranker":    "qw3",
    }
    _ri_alias = {
        "subquery":                "sq",
        "original_query+subquery": "oq+sq",
        "reasoning+subquery":      "rs+sq",
    }
    _pfri_alias = {
        "original_query":            "oq",
        "original_query+subqueries": "oq+sqs",
        "original_query+reasoning":  "oq+rs",
        "reasoning+subqueries":      "rs+sqs",
    }
    _ctrl_alias = {
        "off":            "off",
        "monitor":        "mon",
        "action":         "act",
    }

    seen_top_k = kwargs.get("seen_top_k", 10)
    prr        = kwargs.get("post_retrieval_reranker_name", "null")
    pfr        = kwargs.get("post_fusion_reranker_name", "batched_reranker")
    rrk        = kwargs.get("rerank_top_k", 100)
    ri         = kwargs.get("retrieval_input", "subquery")
    pfri       = kwargs.get("post_fusion_reranker_input", "original_query")
    controller_mode = kwargs.get("controller", "monitor")
    controller_llm     = kwargs.get("llm_controller")

    name = (
        f"stk{seen_top_k}"
        f"_prr-{_reranker_alias.get(prr, prr)}"
        f"_pfr-{_reranker_alias.get(pfr, pfr)}"
        f"_rrk{rrk}"
        f"_ri-{_ri_alias.get(ri, ri)}"
        f"_pfri-{_pfri_alias.get(pfri, pfri)}"
        f"_ctrl-{_ctrl_alias.get(controller_mode, controller_mode)}"
    )
    if controller_mode == "action":
        _controller_label = controller_llm or kwargs.get("llm_intervene") or "default"
        _controller_variant = kwargs.get("controller_prompt_variant", "nov_cov_sim")
        name += f"_cllm-{_controller_label.replace('/', '--')}_cvar-{_controller_variant}"
    if kwargs.get("ensure_novel_seen_docs", False):
        name += "_novel"
    return name
