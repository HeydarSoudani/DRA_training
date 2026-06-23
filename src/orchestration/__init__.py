"""Pipeline orchestration: retriever/controller factories and multi-GPU workers.

This is the wiring layer that assembles heavy component packages
(``controller_component``, ``searcher_component``, ``deep_research_agents``,
``evaluation``) into a runnable agent. It is consumed by the ``run_pipeline``
entry point in ``experiments/run_dra_inference.py``.

Leaf-level helpers (LLM-client factory, CLI arg resolution, output naming) live
in ``utils`` so this module is the only place that depends on the heavy
component packages — keeping ``utils`` a true leaf layer.

The worker functions (_build_components_from_config, _init_worker, gpu_worker)
must stay at module level to be picklable by multiprocessing.spawn.
"""

import contextlib
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

from utils.config import (
    SELF_MANAGED_LLM_AGENTS,
    _RetrieverConfig,
    get_reranker_configs,
    is_local_finetuned,
)
from utils.llm_client import LiteLLMClient, setup_llm_client
from utils.text_utils import _build_cited_docs_ranked_list, build_references_section

logger = logging.getLogger(__name__)


# ===========================================================================
# Retriever factory
# ===========================================================================

def setup_retriever_from_args(args):
    """Instantiate the local retriever from parsed CLI args."""
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
            _tt_llm_client = LiteLLMClient(model=llm_intervene)
            _tt_notice_gen = LLMCriticalThinkingGenerator(llm_client=_tt_llm_client)
        elif strict:
            raise ValueError("--llm-intervene is required when using intervene action")

    _controller_policy = None
    _controller_llm_model = llm_controller or llm_intervene
    if _controller_llm_model:
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
    agentic_model = worker_config["agentic_model"]
    llm_model     = worker_config["llm_model"]
    dataset       = worker_config["dataset"]

    llm_client = None

    if agentic_model in SELF_MANAGED_LLM_AGENTS:
        pass
    elif is_local_finetuned(agentic_model, llm_model):
        hf_model = llm_model
        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:6008/v1")
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

    from searcher_component import RetrievalSearchTool
    from searcher_component.rerankers import build_reranker_from_config

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
    from deep_research_agents.agents import AGENT_MAP
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


def gpu_worker(worker_id: int, query_items: list, temp_dir_str: str, worker_config: dict, progress_queue=None, init_lock=None) -> dict:
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
    controller_dir = str(temp_dir / "controller")
    for _d in [retrieval_dir, generation_dir, trajectory_dir, cited_doc_dir, seen_doc_dir, controller_dir]:
        Path(_d).mkdir(parents=True, exist_ok=True)

    from evaluation import SurfacedDocEvaluator, GenerationEvaluator, TrajectoryEvaluator, ControllerEvaluator, CitedDocEvaluator, SeenDocEvaluator
    _ret_eval        = SurfacedDocEvaluator(qrels={}, k_values=[])
    _gen_eval        = GenerationEvaluator()
    _traj_eval       = TrajectoryEvaluator(
        save_doc_text=worker_config.get("save_doc_text", True),
        dedup_docs=worker_config.get("dedup_docs", True),
    )
    _controller_eval = ControllerEvaluator()
    _cited_eval      = CitedDocEvaluator(qrels={}, k_values=[])
    _seen_eval       = SeenDocEvaluator(qrels={}, k_values=[])

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
        references = build_references_section(result)
        if references:
            result["generation"] = result["generation"].rstrip() + references
        results[query_id] = result
        _ret_eval.save_item(query_id, result, retrieval_dir)
        _gen_eval.save_item(query_id, result, generation_dir)
        _traj_eval.save_item(query_id, query_text, result, trajectory_dir)
        _cited_eval.save_item(query_id, result, cited_doc_dir)
        _seen_eval.save_item(query_id, result, seen_doc_dir)
        _controller_eval.save_item(query_id, query_text, result, controller_dir)
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
