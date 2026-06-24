"""Run deep research agents pipeline.

Supported agents via --agentic-model (the LLM is selected automatically per agent):
    cpm_report      Writing-as-Reasoning agent (CPMReport)         → openbmb/AgentCPM-Report (vLLM)
    searchr1        SearchR1 reasoning agent                       → PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3 (vLLM)
    research        ReSearch reasoning agent                       → agentrl/ReSearch-Qwen-7B-Instruct (vLLM)
    stepsearch      StepSearch reasoning agent                     → Zill1/StepSearch-7B-Instruct (vLLM)
    react           ReAct reasoning agent (--use-plan optional)    → claude-sonnet-4-6 (API)
    selfask         SelfAsk reasoning agent                        → claude-sonnet-4-6 (API)
    searcho1        SearchO1 reasoning agent                       → claude-sonnet-4-6 (API)
    webweaver       WebWeaver outline agent                        → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    drtulu          DR-Tulu reasoning agent                        → rl-research/DR-Tulu-8B (vLLM)
    glm             GLM reasoning agent                            → zai-org/GLM-4.7-Flash (vLLM)
    oss_20b         GPT-OSS-20B reasoning agent                    → gpt-oss-20b (vLLM)
    oss_120b        GPT-OSS-120B reasoning agent                   → gpt-oss-120b (vLLM)
    tongyi          Tongyi-DeepResearch ReAct agent                → Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (vLLM)
    cpm_explore     AgentCPM-Explore deep search agent             → openbmb/AgentCPM-Explore (vLLM)

Agentic workflows:
    ReAct-style (react, selfask, searcho1, research, searchr1, stepsearch, drtulu, glm, oss_20b, oss_120b, tongyi, cpm_explore):
        Query → [Think → Search → Observe]* → Report → Evaluate
        Instruction-tuned : react, selfask, searcho1
        RL-trained        : research, searchr1, stepsearch, drtulu, tongyi, cpm_explore, glm, oss_20b, oss_120b

    Outline-style (webweaver):
        Query → [Think → Search → Write_outline]* → Outline → [Think → Retrieve → Write_section]* → Report → Evaluate

    Report-style (cpm_report):
        Query → Search → Init Plan → [Search → Write]* → [Extend Plan → [Search → Write]*]* → Report → Evaluate

Output structure:
    run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_agent_{model}/{searcher_config}/
    e.g. run_outputs/neuclir_2024_news_e5/oss_agent_gpt-oss-20b/stk10_prr-null_pfr-bat_rrk100_ri-sq_pfri-oq/
    ├── retrieval/
    │   └── {query_id}.trec          per-query TREC file (all iterations, col 6 = iter_N)
    ├── generation/
    │   └── {query_id}.md            per-query generation output
    ├── trajectory/
    │   └── {query_id}.json          per-query trajectory: {qid, question, trajectory}
    ├── controller/
    │   └── {query_id}.json          per-query controller signals: {qid, per_iteration: [...]}
    ├── cited_docs_retrieval/
    │   └── {query_id}.trec          per-query cited-doc TREC file (docs cited by the LLM)
    ├── seen_docs_retrieval/
    │   └── {query_id}.trec          per-query seen-doc TREC file (docs shown to the LLM)
    ├── ranking_results.trec
    └── summary.json                 includes "cited_doc_retrieval" section when available
"""

import argparse
import warnings
import logging
import time
import os
import concurrent.futures
import multiprocessing
import threading
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# Suppress async cleanup noise
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*")
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("asyncio.sslproto").setLevel(logging.CRITICAL)

from indexing_corpus_dataset.dataset_loaders import load_qrels, load_split, resolve_split_id

from deep_research_agents.agents import ALL_AGENTS
from utils.config import AGENTIC_MODEL_TO_LLM, AGENTIC_MODEL_ALIAS
from orchestration import (
    gpu_worker,
    setup_retriever_from_args,
)
from utils.llm_client import setup_llm
from utils.cli_setup import (
    resolve_dataset_defaults,
    detect_num_gpus,
    assemble_pipeline_kwargs,
    cleanup_event_loop,
    load_run_config,
    parse_cli_overrides,
    apply_config_to_args,
    _sm_bool,
)
from utils.text_utils import build_references_section, _build_cited_docs_ranked_list
from utils.io_utils import (
    get_processed_queries,
    setup_output_dirs,
    build_run_name_for_pipeline,
    build_searcher_config_name,
)
from evaluation.runner import evaluate_and_save, build_evaluators, load_processed_results
from evaluation.retrieval.fusion import run_fusion_eval

_OUTPUT_PREFIX = os.environ.get(
    "DRA_OUTPUT_ROOT", "/projects/0/prjs0834/heydars/DRA_training/run_outputs"
)
_CONFIG_DEFAULT = str(Path(__file__).resolve().parent / "configs" / "default.yaml")


# ============================================================================
# Pipeline
# ============================================================================
def run_pipeline(data_path: str, subset: Optional[str] = None, dataset_year: Optional[str] = None, query_key: Optional[str] = None, output_path: Optional[str] = None, agentic_model: str = "cpm_report", limit: Optional[int] = None, verbose: bool = True, num_gpus: int = 1, worker_config: Optional[dict] = None, **kwargs):
    """Execute the deep research agents pipeline on a dataset.

    Args:
        data_path:     Path to dataset directory.
        subset:        Dataset subset identifier.  "news" or "technical" for
                       neuclir; "test" for browsecomp_plus.
        dataset_year:  Year of the dataset version (e.g. "2023", "2024").
                       Used for neuclir.
        query_key:     Key in each JSONL query record to use as the query text.
                       Defaults to "text", which is correct for every dataset:
                       the downloaders flatten each source's best query text
                       (including NeuCLIR's topic_* fields) into "text".
        output_path:   Root path for saving results (optional).
        agentic_model: Agent to use; one of ALL_AGENTS.
        limit:         Cap the number of queries (for quick tests).
        verbose:       Whether to print detailed logs.
        num_gpus:      Number of GPU workers for query-level parallelism.
                       1 = sequential (default). >1 = spawn one process per GPU,
                       split queries across workers, merge temp files at the end.
        worker_config: Serialisable config dict passed to each GPU worker so it
                       can rebuild the agent stack.  Required when num_gpus > 1.
        **kwargs:      Agent-specific and evaluation parameters.
    """
    if agentic_model not in ALL_AGENTS:
        raise ValueError(
            f"Unknown agentic_model '{agentic_model}'. Choose from: {', '.join(ALL_AGENTS)}"
        )

    # ==================== Resolve file-level dataset identifier ====================
    # Build the split string used to locate queries/qrels files on disk
    # (e.g. "2023_news", "2024", "test"); see resolve_split_id.
    dataset = kwargs.pop("dataset", "trqa")
    qrels_data_path = kwargs.pop("qrels_data_path", None) or data_path
    file_data_set = resolve_split_id(dataset, dataset_year, subset)

    if query_key is None:
        query_key = "text"

    # ==================== Load Dataset ====================
    # Single-pass load of queries + qrels + answers; filter to queries with qrels.
    min_rel_score = kwargs.get("min_relevance_score")
    same_qrels_path = qrels_data_path == data_path
    queries, qrels, answers = load_split(
        data_path, file_data_set, query_key=query_key,
        min_relevance_score=min_rel_score, only_with_qrels=same_qrels_path,
    )
    if not same_qrels_path:
        # qrels live in a separate directory; reload and re-filter against them.
        qrels = load_qrels(qrels_data_path, file_data_set, min_relevance_score=min_rel_score)
        queries = {qid: q for qid, q in queries.items() if qid in qrels}
    if answers:
        print(f"Loaded {len(answers)} ground-truth answers (accuracy evaluation available)")

    # Keep the full set of questions for accuracy evaluation (before resume filtering)
    all_questions = dict(queries)

    # ==================== Resume: skip already-processed queries ====================
    llm_model = kwargs.pop("llm_model", "claude-sonnet-4-5")
    run_name = None
    run_dir  = None
    processed: set = set()

    if output_path:
        run_name = build_run_name_for_pipeline(agentic_model=agentic_model, llm_model=llm_model, **kwargs)
        retriever_label = kwargs.get("retriever_name", "e5")
        qk_part = f"_{query_key}" if query_key and query_key != "text" else ""
        dataset_dir = f"{dataset}_{file_data_set}{qk_part}_{retriever_label}"
        searcher_config_name = build_searcher_config_name(**kwargs)

        run_dir = str(Path(output_path) / dataset_dir / run_name / searcher_config_name)

        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] Loading/saving results from: {run_dir}")
        print(f"{'=' * 80}")

        processed = get_processed_queries(run_dir)
        if processed:
            print(f"Found {len(processed)} already processed queries — skipping them")
            original_count = len(queries)
            queries = {qid: q for qid, q in queries.items() if qid not in processed}
            print(f"Remaining: {len(queries)} queries (skipped {original_count - len(queries)})")

        if not queries:
            print("All queries already processed — loading results from disk for evaluation")

    # ==================== Limit ====================
    if limit is not None and limit > 0:
        queries = dict(list(queries.items())[:limit])
        print(f"Limited to {len(queries)} queries for testing")

    # ==================== Eval-only mode ====================
    eval_only = kwargs.get("eval_only", False)
    if eval_only:
        if not run_dir:
            print("--eval-only requires a valid --output directory with existing results.")
            return
        _run_dir_str = str(run_dir)
        if not Path(_run_dir_str).exists():
            print(f"--eval-only: run_dir does not exist: {run_dir}")
            return

        _retrieval_prefix = f"{_run_dir_str.rstrip('/')}/retrieval"
        processed_all = get_processed_queries(run_dir)
        if not processed_all:
            print(f"No retrieval data found in {_retrieval_prefix}. "
                  "Run the full pipeline first (without --eval-only).")
            return

        results: dict = {}
        load_processed_results(processed_all, _retrieval_prefix, results, lightweight=True)
        if not results:
            print(f"Could not load any results from {_run_dir_str}.")
            return
        print(f"Loaded {len(results)} queries for evaluation")

        retrieval_evaluator, generation_evaluator, trajectory_evaluator, \
            cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator, report_evaluator, \
            controller_evaluator = \
            build_evaluators(qrels, kwargs, answers=answers, questions=all_questions)

        _vllm_mgr = kwargs.get("vllm_manager")
        _total_gpus = kwargs.get("total_gpus_on_machine", 8)
        _judge_api_url = kwargs.get("judge_api_url")
        _judge_started = False
        _needs_judge = accuracy_evaluator is not None or report_evaluator is not None
        if _needs_judge and _judge_api_url is None and _vllm_mgr is not None:
            judge_urls = _vllm_mgr.start_judge_server(_total_gpus)
            for _je in (accuracy_evaluator, report_evaluator):
                if _je is not None:
                    _je.judge_api_bases = judge_urls
                    _je._clients = []  # reset so clients are re-created
            _judge_started = True

        try:
            evaluate_and_save(
                results, generation_evaluator, trajectory_evaluator, run_dir,
                cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator,
                controller_evaluator=controller_evaluator,
                report_evaluator=report_evaluator,
            )
        finally:
            if _judge_started:
                _vllm_mgr.shutdown_judge_server()

        run_fusion_eval(results, qrels, kwargs, run_dir, num_gpus)
        return

    # ==================== Inject qrels into worker_config for multi-GPU controller ==
    _controller_mode = kwargs.get("controller", "monitor")
    if worker_config is not None and _controller_mode != "off":
        worker_config["qrels"] = qrels

    # ==================== Build search tool ====================
    from searcher_component.searcher import RetrievalSearchTool

    _retriever = kwargs.get("retriever")
    search_tool = None
    if _retriever is not None:
        search_tool = RetrievalSearchTool(
            retriever=_retriever,
            post_retrieval_reranker=kwargs.get("post_retrieval_reranker"),
            post_fusion_reranker=kwargs.get("post_fusion_reranker"),
            top_k=kwargs.get("top_k", 100),
            rerank_top_k=kwargs.get("rerank_top_k", 100),
            retrieval_input=kwargs.get("retrieval_input", "subquery"),
            post_fusion_reranker_input=kwargs.get("post_fusion_reranker_input", "original_query"),
            ensure_novel_seen_docs=kwargs.get("ensure_novel_seen_docs", False),
            seen_top_k=kwargs.get("seen_top_k", 5),
        )

    # ==================== Instantiate Agent + Controller ====================
    # In multi-GPU mode neither the agent nor the controller is built in the main
    # process; each worker spawns its own instances on its assigned GPU (see
    # orchestration._init_worker).  Building them here would create unused LLM
    # clients that are immediately torn down.
    from orchestration import build_agent, build_controller

    agent = None
    controller = None
    if num_gpus <= 1 and queries:
        agent = build_agent(
            agentic_model=agentic_model,
            llm_model=llm_model,
            llm_client=kwargs.get("llm_client"),
            retriever=_retriever,
            max_iteration=kwargs.get("max_iteration", 100),
            seen_top_k=kwargs.get("seen_top_k", 5),
            verbose=verbose,
            search_tool=search_tool,
            use_plan=kwargs.get("use_plan", False),
            max_output_tokens_total=kwargs.get("max_output_tokens_total", 40000),
            temperature=kwargs.get("temperature", 0.0),
            max_extend_steps=kwargs.get("max_extend_steps", 5),
            max_retries=kwargs.get("max_retries", 3),
            hard_mode=kwargs.get("hard_mode", True),
            oracle_outline_path=kwargs.get("oracle_outline_path"),
            max_passage_chars=kwargs.get("max_passage_chars", 4000),
        )

        # Build the controller AFTER the agent so build_controller wires the
        # per-agent answer format + answer-candidate generator through its
        # constructor (no external private-attribute injection).
        controller = build_controller(
            controller_mode=_controller_mode,
            retriever=_retriever,
            qrels=qrels,
            seen_top_k=kwargs.get("seen_top_k", 5),
            llm_controller=kwargs.get("llm_controller"),
            llm_intervene=kwargs.get("llm_intervene"),
            agent=agent if hasattr(agent, "controller") else None,
            agentic_model=agentic_model,
            controller_history_window=kwargs.get("controller_history_window"),
            controller_prompt_variant=kwargs.get("controller_prompt_variant", "nov_cov_sim"),
            max_iteration=kwargs.get("max_iteration"),
            criteria_coverage_mode=kwargs.get("criteria_coverage_mode", "dynamic"),
            criteria_coverage_max_criteria=kwargs.get("criteria_coverage_max_criteria", 8),
            llm_criteria_coverage=kwargs.get("llm_criteria_coverage"),
            ac_temperature=kwargs.get("temperature", 0.0),
            dataset=dataset,
        )
        if controller is not None and hasattr(agent, "controller"):
            agent.controller = controller

    # ==================== Setup output dirs + evaluators ====================
    retrieval_evaluator, generation_evaluator, trajectory_evaluator, cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator, report_evaluator, controller_evaluator = build_evaluators(qrels, kwargs, answers=answers, questions=all_questions)

    retrieval_dir = generation_dir = trajectory_dir = cited_doc_dir = seen_doc_dir = controller_dir = None
    if output_path:
        _dirs = setup_output_dirs(run_dir, ["retrieval", "generation", "trajectory", "cited_docs_retrieval", "seen_docs_retrieval", "controller"])
        retrieval_dir  = _dirs["retrieval"]
        generation_dir = _dirs["generation"]
        trajectory_dir = _dirs["trajectory"]
        cited_doc_dir  = _dirs["cited_docs_retrieval"]
        seen_doc_dir   = _dirs["seen_docs_retrieval"]
        controller_dir = _dirs["controller"]
        print(f"\nProcessing {len(queries)} queries, saving results to {run_dir}/...")

    # ==================== Loop: run + save per query ====================
    results     = {}
    query_items = list(queries.items())

    if num_gpus > 1 and output_path and worker_config is not None:
        # ── Multi-GPU path ──────────────────────────────────────────────────
        # Split queries round-robin across GPU workers so load is balanced even
        # when queries vary in difficulty.
        chunks = [query_items[i::num_gpus] for i in range(num_gpus)]

        print(f"\nParallel mode: {num_gpus} workers")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i}: {len(chunk)} queries")

        bars = [
            tqdm(
                total=len(chunks[i]),
                position=i,
                leave=True,
                desc=f"[Worker {i}]",
                bar_format=(
                    "{desc} {percentage:3.0f}%|{bar}|"
                    " {n}/{total} [{elapsed}<{remaining}] {postfix}"
                ),
                dynamic_ncols=True,
            )
            for i in range(num_gpus)
        ]
        for i, bar in enumerate(bars):
            bar.set_postfix_str(f"(—, —/—, starting)", refresh=True)

        mp_ctx = multiprocessing.get_context("spawn")
        _manager = mp_ctx.Manager()
        progress_queue = _manager.Queue()

        def _drain_progress(stop_event):
            done_count = 0
            while not stop_event.is_set():
                try:
                    item = progress_queue.get(timeout=0.3)
                except Exception:
                    continue
                if item is None:
                    done_count += 1
                    if done_count >= num_gpus:
                        break
                    continue
                w_id, qid, idx, total, stage, iter_info = item
                bar = bars[w_id]
                if stage == "processing":
                    bar.set_postfix_str(f"({qid}, —/—, starting)", refresh=True)
                elif stage == "update":
                    stage_name, iteration, max_iter_val = iter_info
                    bar.set_postfix_str(
                        f"({qid}, {iteration}/{max_iter_val}, {stage_name})", refresh=True
                    )
                else:
                    bar.n = idx
                    iter_str = iter_info if iter_info else "—/—"
                    bar.set_postfix_str(f"({qid}, {iter_str}, {stage})", refresh=True)
                    bar.refresh()

        _drain_stop = threading.Event()
        drain_thread = threading.Thread(target=_drain_progress, args=(_drain_stop,), daemon=True)
        drain_thread.start()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_gpus, mp_context=mp_ctx
        ) as executor:
            futures = {
                executor.submit(gpu_worker, i, chunks[i], str(run_dir), worker_config, progress_queue, None): i
                for i in range(num_gpus)
            }
            for future in concurrent.futures.as_completed(futures):
                worker_id = futures[future]
                try:
                    worker_results = future.result()
                    results.update(worker_results)
                    tqdm.write(
                        f"[Worker {worker_id}] Completed {len(worker_results)}/{len(chunks[worker_id])} queries"
                    )
                except Exception as e:
                    tqdm.write(f"  Worker {worker_id} failed: {e}")

        _drain_stop.set()
        drain_thread.join(timeout=3)
        _manager.shutdown()

        for bar in bars:
            bar.close()

        print(f"\nCompleted {len(results)} queries (parallel)")

    else:
        # ── Sequential path (default / single GPU) ──────────────────────────
        for idx, (query_id, query_text) in enumerate(query_items, 1):
            if search_tool is not None:
                search_tool.reset()
            _answer_line = f"\n  Answer: {answers[query_id]}" if answers.get(query_id) else ""
            print(f"\n[{idx}/{len(query_items)}] Processing query: {query_id}\n  Query text: {query_text}{_answer_line}")
            result = agent.run_single(
                query_id=query_id,
                query_text=query_text,
                temperature=kwargs.get("temperature", 0.7),
            )
            if result is None:
                print(f"  ✗ Skipping {query_id} (error during processing)")
                continue
            result["cited_docs_ranked_list"] = _build_cited_docs_ranked_list(result)
            # Append a References section to the generation (skips CPMReport which already has one)
            references = build_references_section(result)
            if references:
                result["generation"] = result["generation"].rstrip() + references
            results[query_id] = result

            if output_path:
                retrieval_evaluator.save_item(query_id, result, retrieval_dir)
                generation_evaluator.save_item(query_id, result, generation_dir)
                trajectory_evaluator.save_item(query_id, query_text, result, trajectory_dir)
                cited_doc_evaluator.save_item(query_id, result, cited_doc_dir)
                seen_doc_evaluator.save_item(query_id, result, seen_doc_dir)
                controller_evaluator.save_item(query_id, query_text, result, controller_dir)
                print(f"  ✓ Saved: {query_id}")

        if agent:
            agent.cleanup()
        print(f"\nCompleted {len(results)} queries")

    # ==================== Load already-processed results from disk ====================
    # When resuming a run, `results` only contains newly-processed queries.
    # Load the saved TREC files for previously-skipped queries so evaluation
    # covers ALL processed queries, not just the current batch.
    load_processed_results(processed, retrieval_dir, results)

    # ==================== Evaluate + Save ====================
    _vllm_mgr = kwargs.get("vllm_manager")
    _total_gpus = kwargs.get("total_gpus_on_machine", 8)
    _judge_api_url = kwargs.get("judge_api_url")

    # ── Aggressively release ALL GPU memory ──────────────────────────
    # Break every reference chain to GPU-resident objects so gc can
    # collect them before we reclaim the GPUs.
    #
    # Reference chains that keep the encoder model alive:
    #   1. controller._answer_candidate_fn → bound method → agent → retriever.encoder
    #   2. controller._consec_query_sim._encode_fn → closure → encoder
    #   3. controller._orig_query_sim._encode_fn  → closure → encoder
    #   4. agent.search_tool.retriever → retriever.encoder
    #   5. agent.retriever → retriever.encoder
    #   6. kwargs["retriever"] → retriever.encoder

    # 1) Sever closure/bound-method refs inside controller
    if controller is not None:
        if hasattr(controller, "_answer_candidate_fn"):
            controller._answer_candidate_fn = None
        if hasattr(controller, "_consec_query_sim"):
            controller._consec_query_sim._encode_fn = None
        if hasattr(controller, "_orig_query_sim"):
            controller._orig_query_sim._encode_fn = None

    # 2) Move the encoder model off GPU *before* dropping references.
    #    Accelerate's device_map hooks can prevent gc from freeing GPU
    #    tensors even after all Python refs are gone; .cpu() forces the
    #    move and remove_hook_from_submodules detaches dispatch hooks.
    _enc = getattr(_retriever, "encoder", None) if _retriever is not None else None
    if _enc is not None and hasattr(_enc, "model"):
        import torch
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(_enc.model)
        except Exception:
            pass
        _enc.model.cpu()
        del _enc.model
    del _enc

    # 3) Drop all local + kwargs references
    del agent, search_tool, _retriever, controller
    kwargs.pop("retriever", None)
    kwargs.pop("post_retrieval_reranker", None)
    kwargs.pop("post_fusion_reranker", None)

    # 4) Force garbage collection and return GPU memory to CUDA
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) Kill vLLM servers, orphan processes, and verify GPUs are free
    if _vllm_mgr is not None:
        _vllm_mgr.shutdown_and_release_gpus(_total_gpus)

    # Start the vLLM judge server(s) for accuracy evaluation (Qwen3-32B).
    # Skipped when --judge-api-url points to an externally-managed server.
    _judge_started = False
    _needs_judge = accuracy_evaluator is not None or report_evaluator is not None
    if _needs_judge and _judge_api_url is None and _vllm_mgr is not None:
        judge_urls = _vllm_mgr.start_judge_server(_total_gpus)
        for _je in (accuracy_evaluator, report_evaluator):
            if _je is not None:
                _je.judge_api_bases = judge_urls
                _je._clients = []  # reset so clients are re-created
        _judge_started = True

    try:
        evaluate_and_save(
            results,
            generation_evaluator,
            trajectory_evaluator,
            run_dir,
            cited_doc_evaluator,
            seen_doc_evaluator,
            accuracy_evaluator,
            controller_evaluator=controller_evaluator,
            report_evaluator=report_evaluator,
        )
    finally:
        if _judge_started:
            _vllm_mgr.shutdown_judge_server()

    # ==================== Multi-fusion evaluation ====================
    if results:
        run_fusion_eval(results, qrels, kwargs, run_dir, num_gpus)

    # ==================== Final status ==========================================
    if output_path and run_dir:
        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] All results saved to: {run_dir}")
        print(f"{'=' * 80}")


# ============================================================================
# CLI
# ============================================================================
def _parse_args():
    """Build and parse the CLI argument parser.

    Only the frequently-varied knobs are declared as CLI arguments.  The
    mostly-fixed variables live in a YAML config file (``--config``, default
    experiments/configs/default.yaml) and are merged onto ``args`` afterwards;
    any of them can still be overridden by passing the matching ``--flag`` on
    the command line (handled via ``parse_known_args`` -> ``apply_config_to_args``).
    """
    parser = argparse.ArgumentParser(
        description="Run deep research agents pipeline (unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── File-backed config ─────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=_CONFIG_DEFAULT, help="Path to the YAML file holding the mostly-fixed pipeline variables. Any value in it can be overridden by passing the matching --flag on the CLI.")

    # ── Frequently-varied knobs (everything else lives in --config) ─────────
    parser.add_argument("--agentic-model", type=str, default="glm", choices=list(AGENTIC_MODEL_TO_LLM), help="Agent to run; the LLM is selected automatically from the agent. cpm_report = Writing-as-Reasoning (report generation); searchr1/research/stepsearch/react/selfask/searcho1 = Reasoning-augmented retrieval; glm/oss_20b/oss_120b/tongyi = vendor-specific ReAct agents.")
    parser.add_argument("--dataset", type=str, default="trqa", choices=["trqa", "browsecomp_plus", "neuclir"], help="Dataset. trqa/neuclir/browsecomp_plus use local indices.")
    parser.add_argument("--retriever", type=str, default="qwen3_emb_4b", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb_0.6b", "qwen3_emb_4b", "qwen3_emb_8b", "agentir_4b"], help="Retriever type for public datasets (neuclir only)")
    parser.add_argument("--controller", type=str, default="action", choices=["off", "monitor", "action"], help="Controller mode. 'off': disabled. 'monitor': compute and log scores only, no intervention. 'action': controller takes corrective actions (intervene/stop) via the controller policy.")
    parser.add_argument("--controller-prompt-variant", type=str, default="nov_cov_sim", choices=["nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"], help="Controller policy prompt variant controlling which signals the controller sees. 'nov': novelty only. 'nov_cov': novelty + criteria coverage. 'nov_sim': novelty + consec_query_sim + orig_query_sim. 'nov_cov_sim': novelty + criteria coverage + consec_query_sim + orig_query_sim. 'sim': consec_query_sim (primary) + orig_query_sim (guardrail). 'cov_sim': criteria coverage (primary) + consec_query_sim + orig_query_sim (no novelty). Default: 'nov_cov_sim'.")

    # ── Run-control flags ───────────────────────────────────────────────────
    parser.add_argument("--limit", type=int, default=None, help="Cap number of queries (for quick tests)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPU workers for query-level parallelism. 0 = auto-detect from torch.cuda.device_count(). Each worker loads its own model instance on its assigned GPU.")
    parser.add_argument("--eval-only", type=_sm_bool, nargs="?", const=True, default=False, help="Skip agent execution and run only evaluation on already-generated results. Runs all evaluators (generation, trajectory, controller, cited-doc, seen-doc, accuracy, fusion). Requires the run to have been completed at least once so that trajectory/ and retrieval/ files exist. When --judge-api-url is set, also runs accuracy evaluation.")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (overrides verbose)")

    args, extras = parser.parse_known_args()

    # ── Merge file-backed config (+ any CLI overrides) onto args ────────────
    config = load_run_config(args.config)
    overrides = parse_cli_overrides(extras)
    apply_config_to_args(args, config, overrides)

    # ── Derive --llm-model from --agentic-model ────────────────────────────
    # --llm-model is not a user input; the agent fully determines the LLM.
    # Resolve the model from the CLI key FIRST (oss_20b / oss_120b are distinct
    # keys), then collapse the CLI key to its internal agent name.
    args.llm_model = AGENTIC_MODEL_TO_LLM[args.agentic_model]
    args.agentic_model = AGENTIC_MODEL_ALIAS.get(args.agentic_model, args.agentic_model)

    if args.output is None:
        args.output = _OUTPUT_PREFIX

    return args

def main():
    """Main entry point."""
    args    = _parse_args()

    verbose = args.verbose and not args.quiet

    if args.quiet:
        logging.getLogger("agents").setLevel(logging.ERROR)
        logging.getLogger("agent_tools").setLevel(logging.ERROR)
        logging.getLogger("utils").setLevel(logging.ERROR)
        logging.getLogger("prompts").setLevel(logging.ERROR)

    resolve_dataset_defaults(args)

    # ── Default criteria-coverage mode per dataset (user can override via CLI) ──
    if args.criteria_coverage_mode is None:
        if args.dataset == "browsecomp_plus":
            args.criteria_coverage_mode = "static"
        else:
            args.criteria_coverage_mode = "dynamic"

    # ── Resolve --with-oracle-outline to an actual file path ─────────────
    if args.with_oracle_outline:
        _gold_root = Path(__file__).resolve().parent.parent / "gold_retriever_analysis" / "run_outputs"
        _oracle_dataset_dir = f"{args.dataset}_{args.dataset_year}_{args.subset}_{args.retriever}"
        _oracle_path = _gold_root / _oracle_dataset_dir / "gold_analysis_one_by_one" / "generation.json"
        if _oracle_path.exists():
            args.with_oracle_outline = str(_oracle_path)
            print(f"Auto-resolved oracle outline: {args.with_oracle_outline}")
        else:
            print(f"WARNING: Oracle outline not found at {_oracle_path} — disabling oracle outline")
            args.with_oracle_outline = None
    else:
        args.with_oracle_outline = None

    # ── Parse --gpu-ids and reconcile with --num-gpus ────────────────────
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        if args.num_gpus <= 1:
            args.num_gpus = len(gpu_ids)
    num_gpus            = detect_num_gpus(args.num_gpus)

    # ── Auto-start vLLM servers if needed ─────────────────────────────────
    from utils.vllm_manager import VLLMServerManager
    vllm_manager = VLLMServerManager()

    # Detect the *real* total GPU count on the machine (not the user's
    # --num-gpus which is the desired worker count).
    try:
        import torch
        total_gpus_on_machine = max(1, torch.cuda.device_count())
    except ImportError:
        total_gpus_on_machine = num_gpus

    if args.eval_only:
        # Eval-only: skip LLM/reranker vLLM servers — only the judge server
        # (managed inside run_pipeline) may be needed.
        if num_gpus > 1:
            gpu_ids = gpu_ids or list(range(min(num_gpus, total_gpus_on_machine)))
    elif gpu_ids is None:
        # Let the manager allocate GPUs: vLLM servers get leftmost GPUs,
        # remaining GPUs go to pipeline workers.
        worker_gpu_ids = vllm_manager.auto_start(args, total_gpus=total_gpus_on_machine)
        if worker_gpu_ids is not None:
            gpu_ids = worker_gpu_ids
            num_gpus = len(worker_gpu_ids)
            args.num_gpus = num_gpus
        elif num_gpus > 1:
            gpu_ids = list(range(min(num_gpus, total_gpus_on_machine)))
    else:
        # User specified --gpu-ids explicitly; still check whether vLLM
        # servers need to be started and adjust the worker set accordingly.
        worker_gpu_ids = vllm_manager.auto_start(args, total_gpus=total_gpus_on_machine)
        if worker_gpu_ids is not None:
            gpu_ids = worker_gpu_ids
            num_gpus = len(worker_gpu_ids)
            args.num_gpus = num_gpus

    # ── Restrict main process to pipeline GPUs before any CUDA init ───────
    # When --gpu-ids is specified the main process must not touch vLLM's
    # GPUs (typically 0..3).  Setting CUDA_VISIBLE_DEVICES early prevents
    # accidental CUDA context creation on those devices.
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    if args.eval_only:
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client=None, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    elif num_gpus > 1:
        # Multi-GPU: each worker loads its own retriever on its assigned GPU.
        # Skip loading in the main process to avoid OOM on shared GPUs.
        llm_client = setup_llm(args, num_gpus)
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    else:
        llm_client  = setup_llm(args, num_gpus)
        retriever   = setup_retriever_from_args(args)
        pipeline_kwargs = assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus, verbose, gpu_ids=gpu_ids)

    # ── Deep-research-pipeline-specific kwargs ────────────────────────
    _judge_url = args.judge_api_url
    pipeline_kwargs.update({
        "fusion_k": args.fusion_k,
        "fusion_methods": args.fusion_methods,
        "eval_only": args.eval_only,
        "report_eval": args.report_eval,
        "vllm_manager": vllm_manager,
        "total_gpus_on_machine": total_gpus_on_machine,
        "judge_api_url": _judge_url,
    })

    # ── Optional post-retrieval & post-fusion rerankers ─────────────────
    # In multi-GPU mode each worker builds its own reranker on its assigned
    # GPU (see _init_worker).  Loading one in the main process would waste
    # GPU memory on a device that a worker needs (e.g. rankllama = ~14 GB).
    if num_gpus <= 1 and not args.eval_only:
        from utils.config import get_reranker_configs
        _reranker_configs = get_reranker_configs(args.rerank_top_k)
        from searcher_component.rerankers import build_reranker_from_config
        if args.post_retrieval_reranker != "null":
            pipeline_kwargs["post_retrieval_reranker"] = build_reranker_from_config(
                args.post_retrieval_reranker, _reranker_configs,
            )
        if args.post_fusion_reranker != "null":
            pipeline_kwargs["post_fusion_reranker"] = build_reranker_from_config(
                args.post_fusion_reranker, _reranker_configs,
            )
    pipeline_kwargs["rerank_top_k"] = args.rerank_top_k
    pipeline_kwargs["retrieval_input"] = args.retrieval_input
    pipeline_kwargs["post_fusion_reranker_input"] = args.post_fusion_reranker_input
    pipeline_kwargs["post_retrieval_reranker_name"] = args.post_retrieval_reranker
    pipeline_kwargs["post_fusion_reranker_name"] = args.post_fusion_reranker
    pipeline_kwargs["ensure_novel_seen_docs"] = args.ensure_novel_seen_docs

    try:
        run_pipeline(
            data_path=args.data_path,
            subset=args.subset,
            dataset_year=args.dataset_year,
            query_key=args.query_key,
            output_path=args.output,
            agentic_model=args.agentic_model,
            limit=args.limit,
            verbose=verbose,
            **pipeline_kwargs,
        )
        print("Pipeline execution completed successfully")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise
    finally:
        time.sleep(0.5)
        cleanup_event_loop()
        vllm_manager.shutdown()

if __name__ == "__main__":
    main()


# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
#   run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_agent_{model}/{searcher_config}/
#   e.g. run_outputs/neuclir_2024_news_e5/oss_agent_gpt-oss-20b/stk10_prr-null_pfr-bat_rrk100_ri-sq_pfri-oq/
#     ├── retrieval/
#     │   └── {query_id}.trec   per-query TREC file (all iterations, col 6 = iter_N)
#     ├── generation/
#     │   └── {query_id}.md     per-query generation output
#     ├── trajectory/
#     │   └── {query_id}.json   per-query trajectory: {qid, question, trajectory}
#     ├── controller/
#     │   └── {query_id}.json   per-query controller signals: {qid, per_iteration: [...]}
#     ├── cited_docs_retrieval/
#     │   └── {query_id}.trec   per-query cited-doc TREC file (docs cited by the LLM)
#     ├── seen_docs_retrieval/
#     │   └── {query_id}.trec   per-query seen-doc TREC file (docs shown to the LLM)
#     ├── ranking_results.trec
#     └── summary.json
#
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
#   CUDA_VISIBLE_DEVICES=5,6 python experiments/run_dra_inference.py --dataset browsecomp_plus --limit 1
#   CUDA_VISIBLE_DEVICES=0,1,2 python experiments/run_dra_inference.py --dataset neuclir --limit 1
#   python experiments/run_dra_inference.py --dataset neuclir --num-gpus 6 --quiet --limit 6
#   python experiments/run_dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 8 --quiet --limit 12