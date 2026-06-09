"""Run deep research agents pipeline.

Supported agents via --agentic-model:
    cpm_report      Writing-as-Reasoning agent (CPMReport)
    searchr1        SearchR1 reasoning agent
    research        ReSearch reasoning agent
    stepsearch      StepSearch reasoning agent
    react           ReAct reasoning agent (use --use-plan for optional planning)
    selfask         SelfAsk reasoning agent
    searcho1        SearchO1 reasoning agent
    webweaver       WebWeaver reasoning agent (finetuned: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
    drtulu          DR-Tulu reasoning agent (finetuned: rl-research/DR-Tulu-8B)
    glm             GLM reasoning agent (ZhipuAI cloud API)
    oss             GPT-OSS reasoning agent (OpenAI Responses API / vLLM)
    tongyi          Tongyi-DeepResearch ReAct agent (vLLM)
    cpm_explore     AgentCPM-Explore deep search agent (vLLM, finetuned: openbmb/AgentCPM-Explore)

Agentic workflows:
    ReAct-style (react, selfask, searcho1, research, searchr1, stepsearch, drtulu, glm, oss, tongyi, cpm_explore):
        Query → [Think → Search → Observe]* → Report → Evaluate
        Instruction-tuned : react, selfask, searcho1
        RL-trained        : research, searchr1, stepsearch, drtulu, tongyi, cpm_explore, glm, oss

    Outline-style (webweaver):
        Query → [Think → Search → Write_outline]* → Outline → [Think → Retrieve → Write_section]* → Report → Evaluate

    Report-style (cpm_report):
        Query → Search → Init Plan → [Search → Write]* → [Extend Plan → [Search → Write]*]* → Report → Evaluate

Output structure:
    run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_agent_{model}/{searcher_config}/
    e.g. run_outputs/neuclir_2024_news_topic_description_e5/oss_agent_gpt-oss-20b/stk10_prr-null_pfr-bat_rrk100_ri-sq_pfri-oq/
    ├── retrieval/
    │   └── {query_id}.trec          per-query TREC file (all iterations, col 6 = iter_N)
    ├── generation/
    │   └── {query_id}.md            per-query generation output
    ├── trajectory/
    │   └── {query_id}.json          per-query trajectory: {qid, question, trajectory}
    ├── tracker/
    │   └── {query_id}.json          per-query tracker signals: {qid, per_iteration: [...]}
    ├── cited_docs_retrieval/
    │   └── {query_id}.trec          per-query cited-doc TREC file (docs seen by LLM)
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

import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "src"))
sys.path.append(str(repo_root))
# agents/ package lives next to this file
sys.path.append(str(Path(__file__).resolve().parent))
# decision_making_agent/ package lives under experiments/
sys.path.append(str(repo_root / "experiments"))

from agentic_retrieval_research.corpus_dataset.dataset import load_queries, load_qrels, load_query_answers

from agents import (
    AGENT_MAP, OSS_BedrockAgent, BEDROCK_OSS_MODELS,
    ALL_AGENTS,
)
from utils.general_utils import (
    _build_cited_docs_ranked_list,
    _gpu_worker,
    _resolve_dataset_defaults,
    _detect_num_gpus,
    _setup_llm,
    _setup_retriever,
    _assemble_pipeline_kwargs,
    _cleanup_event_loop,
)
from utils.doc_formatting import build_references_section
from agentic_retrieval_research.evaluation import RetrievalEvaluator, GenerationEvaluator, TrajectoryEvaluator, TrackerEvaluator, CitedDocRetrievalEvaluator, SeenDocRetrievalEvaluator
from agentic_retrieval_research.utils.s3_utils import (
    is_s3_path, s3_exists,
    get_processed_queries, load_result_from_trec, setup_output_dirs,
)
# Guard: decision_making_agent.utils transitively imports torch (via
# fusion_utils.py) which can pre-initialise CUDA in multiprocessing-spawn'd
# children before they pin CUDA_VISIBLE_DEVICES to their assigned GPU.
# Children only need _gpu_worker (from general_utils), not these symbols.
if __name__ != "__mp_main__":
    from decision_making_agent.utils import evaluate_and_save, ALL_AGGREGATION_FUSION_METHODS, run_fusion_eval, build_evaluators, load_processed_results

# ── S3 output defaults ─────────────────────────────────────────────────────
_S3_BUCKET = "a204383-ml-workspace-practicallawqw7t-use1"
_S3_OUTPUT_PREFIX = f"s3://{_S3_BUCKET}/agentic_retrieval/run_outputs/experiments_deepresearch_agents"
_LOCAL_OUTPUT_PREFIX = str(Path(__file__).resolve().parent / "run_outputs")

# Redirect HuggingFace caches (model weights, datasets Arrow files) and TMPDIR
# so they don't fill the small root filesystem on SageMaker instances.
_NVME_HF = "/mnt/sagemaker-nvme/huggingface"
if os.path.isdir("/mnt/sagemaker-nvme"):
    os.makedirs(_NVME_HF, exist_ok=True)
    os.environ.setdefault("HF_HOME", _NVME_HF)
elif os.path.isdir("/opt/ml"):
    _EBS_TMP = "/opt/ml/tmp"
    os.makedirs(_EBS_TMP, exist_ok=True)
    _EBS_HF = os.path.join(_EBS_TMP, "huggingface")
    os.makedirs(_EBS_HF, exist_ok=True)
    os.environ.setdefault("HF_HOME", _EBS_HF)
    os.environ.setdefault("TMPDIR", _EBS_TMP)

# Set API keys only if not already set (e.g. by .env or shell env)
os.environ.setdefault("TAVILY_API_KEY", "")  # https://app.tavily.com/
os.environ.setdefault("SERPER_API_KEY", "")  # https://serper.dev/
os.environ.setdefault("S2_API_KEY",     "")  # https://api.semanticscholar.org/
os.environ.setdefault("JINA_API_KEY",   "")  # https://jina.ai/reader/

# ============================================================================
# Pipeline
# ============================================================================
def run_pipeline(data_path: str, subset: Optional[str] = None, dataset_year: Optional[str] = None, query_key: Optional[str] = None, output_path: Optional[str] = None, agentic_model: str = "cpm_report", limit: Optional[int] = None, verbose: bool = True, num_gpus: int = 1, worker_config: Optional[dict] = None, **kwargs):
    """Execute the deep research agents pipeline on a dataset.

    Args:
        data_path:     Path to dataset directory.
        subset:        Dataset subset identifier.  None for trec_rag
                       (year-only); "news", "technical", or "report" for
                       neuclir; "test" for browsecomp_plus.
        dataset_year:  Year of the dataset version (e.g. "2023", "2024").
                       Used for trec_rag and neuclir.
        query_key:     Key in each JSONL query record to use as the query text.
                       Defaults to "text" for standard datasets.  For neuclir
                       use "topic_title", "topic_description", or
                       "topic_narrative".
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
    # Build the string used to locate queries/qrels files on disk.
    # neuclir:         queries_{year}_{subset}.jsonl  →  file_data_set = "{year}_{subset}"
    # trec_rag:        queries_{year}.tsv             →  file_data_set = "{year}"
    # browsecomp_plus: queries_{subset}.jsonl         →  file_data_set = "{subset}"
    dataset = kwargs.pop("dataset", "neuclir")
    qrels_data_path = kwargs.pop("qrels_data_path", None) or data_path
    if dataset == "neuclir":
        file_data_set = f"{dataset_year}_{subset}" if (dataset_year and subset) else (subset or dataset_year or "2024_technical")
    elif dataset == "trec_rag":
        file_data_set = dataset_year or subset or "2024"
    else:  # browsecomp_plus
        file_data_set = subset or "test"

    if query_key is None:
        query_key = "text"

    # ==================== Load Dataset ====================
    queries = load_queries(data_path, file_data_set, query_key=query_key)
    min_rel_score = kwargs.get("min_relevance_score")
    qrels = load_qrels(qrels_data_path, file_data_set, min_relevance_score=min_rel_score)
    answers = load_query_answers(data_path, file_data_set)
    if answers:
        print(f"Loaded {len(answers)} ground-truth answers (accuracy evaluation available)")

    # Filter to queries that have qrels
    queries_with_qrels = {qid: q for qid, q in queries.items() if qid in qrels}
    queries_without_qrels = len(queries) - len(queries_with_qrels)
    if queries_without_qrels > 0:
        queries = queries_with_qrels

    # Keep the full set of questions for accuracy evaluation (before resume filtering)
    all_questions = dict(queries)

    # ==================== Resume: skip already-processed queries ====================
    llm_model = kwargs.pop("llm_model", "claude-sonnet-4-5")
    run_name = None
    run_dir  = None
    processed: set = set()

    if output_path:
        run_name = _build_run_name_for_pipeline(agentic_model=agentic_model, llm_model=llm_model, **kwargs)
        retriever_name = kwargs.get("retriever_name", "e5")
        qwen3_size = kwargs.get("qwen3_size", "4B")
        retriever_label = f"qwen3_emb_{qwen3_size}" if retriever_name == "qwen3_emb" else retriever_name
        qk_part = f"_{query_key}" if query_key and query_key != "text" else ""
        dataset_dir = f"{dataset}_{file_data_set}{qk_part}_{retriever_label}"
        searcher_config_name = _build_searcher_config_name(**kwargs)

        # Build run_dir as S3 URI or local Path
        _is_s3_output = is_s3_path(str(output_path))
        if _is_s3_output:
            run_dir = f"{str(output_path).rstrip('/')}/{dataset_dir}/{run_name}/{searcher_config_name}"
        else:
            run_dir = str(Path(output_path) / dataset_dir / run_name / searcher_config_name)

        _storage_label = "S3" if _is_s3_output else "local"
        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] Loading/saving results from {_storage_label}: {run_dir}")
        print(f"{'=' * 80}")

        processed = get_processed_queries(run_dir)
        if processed:
            print(f"Found {len(processed)} already processed queries on {_storage_label} — skipping them")
            original_count = len(queries)
            queries = {qid: q for qid, q in queries.items() if qid not in processed}
            print(f"Remaining: {len(queries)} queries (skipped {original_count - len(queries)})")

        if not queries:
            print(f"All queries have already been processed — loading results from {_storage_label} for evaluation")

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
        _is_s3_run = is_s3_path(_run_dir_str)
        if not _is_s3_run and not Path(_run_dir_str).exists():
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
            cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator = \
            build_evaluators(qrels, kwargs, answers=answers, questions=all_questions)
        tracker_evaluator = TrackerEvaluator()

        _vllm_mgr = kwargs.get("vllm_manager")
        _total_gpus = kwargs.get("total_gpus_on_machine", 8)
        _judge_api_url = kwargs.get("judge_api_url")
        _judge_started = False
        if accuracy_evaluator is not None and _judge_api_url is None and _vllm_mgr is not None:
            judge_urls = _vllm_mgr.start_judge_server(_total_gpus)
            accuracy_evaluator.judge_api_bases = judge_urls
            accuracy_evaluator._clients = []  # reset so clients are re-created
            _judge_started = True

        try:
            evaluate_and_save(
                results, generation_evaluator, trajectory_evaluator, run_dir,
                cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator,
                tracker_evaluator=tracker_evaluator,
            )
        finally:
            if _judge_started:
                _vllm_mgr.shutdown_judge_server()

        run_fusion_eval(results, retrieval_evaluator.qrels, kwargs, run_dir, num_gpus)
        return

    # ==================== Inject qrels into worker_config for multi-GPU tracker ==
    _tt_mode = kwargs.get("trajectory_tracker", "monitor")
    if worker_config is not None and _tt_mode != "off":
        worker_config["qrels"] = qrels

    # ==================== Build trajectory tracker ====================
    from utils.general_utils import build_trajectory_tracker

    trajectory_tracker = build_trajectory_tracker(
        tt_mode=_tt_mode,
        retriever=kwargs.get("retriever"),
        qrels=qrels,
        seen_top_k=kwargs.get("seen_top_k", 5),
        llm_decision_maker=kwargs.get("llm_decision_maker"),
        llm_intervene=kwargs.get("llm_intervene"),
        decision_maker_history_window=kwargs.get("decision_maker_history_window"),
        dm_prompt_variant=kwargs.get("dm_prompt_variant", "nov_cov_sim"),
        max_iteration=kwargs.get("max_iteration"),
        aspect_coverage_mode=kwargs.get("aspect_coverage_mode", "dynamic"),
        aspect_coverage_max_aspects=kwargs.get("aspect_coverage_max_aspects", 8),
        llm_aspect_coverage=kwargs.get("llm_aspect_coverage"),
    )

    # ==================== Build search tool ====================
    from agentic_retrieval_research.searcher_component.searcher import RetrievalSearchTool

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

    # ==================== Instantiate Agent ====================
    # In multi-GPU mode the agent is NOT loaded in the main process; each worker
    # spawns its own instance on its assigned GPU.
    agent = None
    if num_gpus <= 1 and queries:
        model_class = AGENT_MAP[agentic_model]
        _max_out = kwargs.get("max_output_tokens_total", 40000)
        _reasoning_extra: dict = {}
        if agentic_model == "react":
            if kwargs.get("use_plan", False):
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
            _reasoning_extra["temperature"] = kwargs.get("temperature", 0.0) or 1.0
        elif agentic_model == "cpm_report":
            _reasoning_extra["max_extend_steps"] = kwargs.get("max_extend_steps", 5)
            _reasoning_extra["max_retries"] = kwargs.get("max_retries", 3)
            _reasoning_extra["hard_mode"] = kwargs.get("hard_mode", True)
            _reasoning_extra["oracle_outline_path"] = kwargs.get("oracle_outline_path")
            _reasoning_extra["max_passage_chars"] = kwargs.get("max_passage_chars", 4000)
            _reasoning_extra["model_name"] = llm_model
            _reasoning_extra["model_url"] = kwargs.get("model_url")

        # Route OSS models to Bedrock runner instead of vLLM
        if agentic_model == "oss" and llm_model in BEDROCK_OSS_MODELS:
            agent = OSS_BedrockAgent(
                llm_client=kwargs.get("llm_client"),
                retriever=kwargs.get("retriever"),
                max_iteration=kwargs.get("max_iteration", 100),
                seen_top_k=kwargs.get("seen_top_k", 5),
                model_name=BEDROCK_OSS_MODELS[llm_model],
                max_output_tokens=_max_out,
                verbose=verbose,
            )
        else:
            agent = model_class(
                llm_client=kwargs.get("llm_client"),
                retriever=kwargs.get("retriever"),
                max_iteration=kwargs.get("max_iteration", 100),
                seen_top_k=kwargs.get("seen_top_k", 5),
                verbose=verbose,
                **_reasoning_extra,
            )

        # Inject search tool into agents that inherit from BasicAgent
        # (avoids modifying every subclass constructor)
        if agent is not None and search_tool is not None and hasattr(agent, "search_tool"):
            agent.search_tool = search_tool

        # Inject trajectory tracker into agents that inherit from BasicAgent
        if agent is not None and trajectory_tracker is not None and hasattr(agent, "trajectory_tracker"):
            from prompts.trajectory_tracker.answer_prompts import get_candidate_format

            # Always set the canonical per-agent format on inference_config
            _cfg = getattr(agent, "inference_config", None)
            if _cfg is not None:
                _cfg.format_instructions = get_candidate_format(agentic_model)

            # Wire answer candidate generation through the agent (browsecomp_plus only)
            if dataset == "browsecomp_plus" and agentic_model != "cpm_report":
                if hasattr(agent, "generate_answer_candidate"):
                    trajectory_tracker._answer_candidate_fn = agent.generate_answer_candidate
                    _model_id = _cfg.model_name if _cfg else "unknown"
                    print(f"Answer candidate via agent.generate_answer_candidate: {_model_id}")
                else:
                    pass

            agent.trajectory_tracker = trajectory_tracker

    # (else: multi-GPU — agent loading deferred to worker processes)

    # ==================== Setup output dirs + evaluators ====================
    retrieval_evaluator, generation_evaluator, trajectory_evaluator, cited_doc_evaluator, seen_doc_evaluator, accuracy_evaluator = build_evaluators(qrels, kwargs, answers=answers, questions=all_questions)

    tracker_evaluator = TrackerEvaluator()

    retrieval_dir = generation_dir = trajectory_dir = cited_doc_dir = seen_doc_dir = tracker_dir = None
    if output_path:
        _dirs = setup_output_dirs(run_dir, ["retrieval", "generation", "trajectory", "cited_docs_retrieval", "seen_docs_retrieval", "tracker"])
        retrieval_dir  = _dirs["retrieval"]
        generation_dir = _dirs["generation"]
        trajectory_dir = _dirs["trajectory"]
        cited_doc_dir  = _dirs["cited_docs_retrieval"]
        seen_doc_dir   = _dirs["seen_docs_retrieval"]
        tracker_dir    = _dirs["tracker"]
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
                executor.submit(_gpu_worker, i, chunks[i], str(run_dir), worker_config, progress_queue, None): i
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
                tracker_evaluator.save_item(query_id, query_text, result, tracker_dir)
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
    #   1. trajectory_tracker._answer_candidate_fn → bound method → agent → retriever.encoder
    #   2. trajectory_tracker._consec_query_sim._encode_fn → closure → encoder
    #   3. trajectory_tracker._orig_query_sim._encode_fn  → closure → encoder
    #   4. agent.search_tool.retriever → retriever.encoder
    #   5. agent.retriever → retriever.encoder
    #   6. kwargs["retriever"] → retriever.encoder

    # 1) Sever closure/bound-method refs inside trajectory_tracker
    if trajectory_tracker is not None:
        if hasattr(trajectory_tracker, "_answer_candidate_fn"):
            trajectory_tracker._answer_candidate_fn = None
        if hasattr(trajectory_tracker, "_consec_query_sim"):
            trajectory_tracker._consec_query_sim._encode_fn = None
        if hasattr(trajectory_tracker, "_orig_query_sim"):
            trajectory_tracker._orig_query_sim._encode_fn = None

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
    del agent, search_tool, _retriever, trajectory_tracker
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
    if accuracy_evaluator is not None and _judge_api_url is None and _vllm_mgr is not None:
        judge_urls = _vllm_mgr.start_judge_server(_total_gpus)
        accuracy_evaluator.judge_api_bases = judge_urls
        accuracy_evaluator._clients = []  # reset so clients are re-created
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
            tracker_evaluator=tracker_evaluator,
        )
    finally:
        if _judge_started:
            _vllm_mgr.shutdown_judge_server()

    # ==================== Multi-fusion evaluation ====================
    if results:
        run_fusion_eval(results, retrieval_evaluator.qrels, kwargs, run_dir, num_gpus)

    # ==================== Final status ==========================================
    if output_path and run_dir:
        _storage = "S3" if is_s3_path(str(run_dir)) else "local"
        print(f"\n{'=' * 80}")
        print(f"[OUTPUT] All results saved to {_storage}: {run_dir}")
        print(f"{'=' * 80}")


# ============================================================================
# Run-name builder
# ============================================================================
def _build_searcher_config_name(**kwargs) -> str:
    """Build an abbreviated searcher-config directory name.

    Encodes: seen_top_k, post_retrieval_reranker, post_fusion_reranker,
    rerank_top_k, retrieval_input, post_fusion_reranker_input.

    Example: stk10_prr-null_pfr-bat_rrk100_ri-sq_pfri-oq
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

    _tt_alias = {
        "off":             "off",
        "monitor":         "mon",
        "decision_maker":  "dm",
    }

    seen_top_k   = kwargs.get("seen_top_k", 10)
    prr          = kwargs.get("post_retrieval_reranker_name", "null")
    pfr          = kwargs.get("post_fusion_reranker_name", "batched_reranker")
    rrk          = kwargs.get("rerank_top_k", 100)
    ri           = kwargs.get("retrieval_input", "subquery")
    pfri         = kwargs.get("post_fusion_reranker_input", "original_query")
    tt_mode      = kwargs.get("trajectory_tracker", "monitor")
    dm_llm       = kwargs.get("llm_decision_maker")

    prr_short  = _reranker_alias.get(prr, prr)
    pfr_short  = _reranker_alias.get(pfr, pfr)
    ri_short   = _ri_alias.get(ri, ri)
    pfri_short = _pfri_alias.get(pfri, pfri)
    tt_short   = _tt_alias.get(tt_mode, tt_mode)
    ensure_novel = kwargs.get("ensure_novel_seen_docs", False)

    name = f"stk{seen_top_k}_prr-{prr_short}_pfr-{pfr_short}_rrk{rrk}_ri-{ri_short}_pfri-{pfri_short}_tt-{tt_short}"
    if tt_mode == "decision_maker":
        _dm_label = dm_llm or kwargs.get("llm_intervene") or "default"
        _dmv = kwargs.get("dm_prompt_variant", "nov_cov_sim")
        name += f"_dm-{_dm_label.replace('/', '--')}_dmv-{_dmv}"
    if ensure_novel:
        name += "_novel"
    return name

def _build_run_name_for_pipeline(agentic_model, llm_model, **kwargs) -> str:
    """Build a consistent output directory name for the current pipeline run.

    Template: {agent_name}_{model}
    Dataset/retriever/query_key info now lives in the parent dataset_dir.
    """
    name = agentic_model
    if agentic_model == "react":
        name = "react_w_plan" if kwargs.get("use_plan", False) else "react_wo_plan"
    return f"{name}_agent_{llm_model.replace('/', '--')}"


# ============================================================================
# CLI
# ============================================================================
def _sm_bool(v):
    """SageMaker-compatible boolean argument type.

    Accepts bare flags (``--flag``) via ``nargs="?"`` / ``const=True``,
    as well as explicit string values (``--flag true``) that SageMaker sends
    when hyperparameters are forwarded as ``--key value`` pairs.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "t"):
        return True
    if v.lower() in ("false", "0", "no", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")

def _parse_args():
    """Build and parse the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run deep research agents pipeline (unified)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Agent ──────────────────────────────────────────────────────
    parser.add_argument("--agentic-model", type=str, default="glm", choices=ALL_AGENTS, help="Agent to run. cpm_report = Writing-as-Reasoning (report generation); searchr1/research/stepsearch/react/selfask/searcho1 = Reasoning-augmented retrieval; glm/oss/tongyi = vendor-specific ReAct agents.")
    # ── Dataset ────────────────────────────────────────────────────────────
    parser.add_argument("--dataset", type=str, default="neuclir", choices=["trec_rag", "neuclir", "browsecomp_plus"], help="Dataset. trec_rag/neuclir/browsecomp_plus use local indices.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to dataset directory (auto-selected if omitted)")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset: None for trec_rag; 'news', 'technical', or 'report' for neuclir; 'test' for browsecomp_plus. Auto-selected if omitted.")
    parser.add_argument("--dataset-year", type=str, default=None, help="Dataset year for trec_rag/neuclir (e.g. '2023', '2024'). Auto-selected if omitted.")
    parser.add_argument("--query-key", type=str, default=None, help="Key in each JSONL query record to use as the query text. Defaults to 'text' for standard datasets. For neuclir: 'topic_title', 'topic_description', or 'topic_narrative'. Auto-selected if omitted.")
    parser.add_argument("--qrels-data-path", type=str, default=None, help="Path to load qrels from (defaults to --data-path; auto-selected for criteria-augmented variants)")
    parser.add_argument("--min-relevance-score", type=int, default=None, help="Minimum relevance score to treat as relevant")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of queries (for quick tests)")
    # ── LLM ────────────────────────────────────────────────────────────────
    parser.add_argument("--llm-model", type=str, default="zai-org/GLM-4.7-Flash", choices=["gpt-oss-20b", "gpt-oss-120b", "zai-org/GLM-4.7-Flash", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", "openbmb/AgentCPM-Report", "openbmb/AgentCPM-Explore", "rl-research/DR-Tulu-8B", "claude-sonnet-4-5", "claude-sonnet-4-6", "gpt-4.1", "gpt-4.1-mini", "qwen3-max", "qwen3-235B-A22B", "gpt-5.1"], help="LLM to use. For vLLM-served finetuned models pass the HF model name directly (e.g. openbmb/AgentCPM-Report, rl-research/DR-Tulu-8B).")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--llm-max-tokens-per-call", type=int, default=10000, help="Max tokens generated per single LLM API call")
    parser.add_argument("--llm-top-p", type=float, default=1.0, help="Top-p sampling (reasoning agents only)")
    parser.add_argument("--max-output-tokens-total", type=int, default=20000, help="Cumulative output token budget for the entire agent run per query (oss/glm/tongyi only)")
    parser.add_argument("--request-timeout", type=int, default=300, help="HTTP request timeout in seconds for LLM API calls")
    # ── Retriever: public (trec_rag / neuclir / browsecomp_plus) ───────────
    parser.add_argument("--retriever", type=str, default="qwen3_emb", choices=["bm25", "spladepp", "spladev3", "rerank_l6", "rerank_l12", "contriever", "dpr", "e5", "bge", "qwen3_emb", "agentir_4b"], help="Retriever type for public datasets (trec_rag/neuclir only)")
    parser.add_argument("--qwen3-size", type=str, default="4B", choices=["0.6B", "4B", "8B"], help="Qwen3-Embedding model size variant (only used when --retriever=qwen3_emb)")
    parser.add_argument("--index-dir", type=str, default=None, help="Directory with retrieval indices (trec_rag/neuclir only)")
    parser.add_argument("--corpus-path", type=str, default=None, help="Glob path to corpus files (trec_rag/neuclir only)")
    parser.add_argument("--top-k", type=int, default=100, help="Documents to retrieve per search")
    parser.add_argument("--seen-top-k", type=int, default=5, help="Passages passed to planning/writing components. For cpm_report (=20): docs shown per search step. For webweaver(=10): docs added to memory bank per search (after interleaving multi-query results). For glm_agent and oss_agent (=5): docs shown per search step.")
    # ── Searcher Component ─────────────────────────────--------------------
    parser.add_argument("--post-retrieval-reranker", type=str, default="null", choices=["null", "batched_reranker", "rankllama", "rank1", "qwen3_reranker"], help="Post-retrieval reranking mode: reranks each sub-query's doc list individually before fusion")
    parser.add_argument("--post-fusion-reranker", type=str, default="null", choices=["null", "batched_reranker", "rankllama", "rank1", "qwen3_reranker"], help="Post-fusion reranking mode: reranks the fused list against the original query")
    parser.add_argument("--rerank-top-k", type=int, default=100, help="Number of top documents to rerank (used by both post-retrieval and post-fusion rerankers)")
    parser.add_argument("--retrieval-input", type=str, default="subquery", choices=["subquery", "original_query+subquery", "reasoning+subquery"], help="Controls what text is sent to the retriever for each sub-query. 'subquery' (default) uses the raw sub-query; 'original_query+subquery' prepends the original query; 'reasoning+subquery' prepends current trajectory reasoning.")
    parser.add_argument("--post-fusion-reranker-input", type=str, default="original_query", choices=["original_query", "original_query+subqueries", "original_query+reasoning", "reasoning+subqueries"], help="Controls what text is sent to the post-fusion reranker. 'original_query' (default) uses the original query; 'original_query+subqueries' concatenates the original query with all current sub-queries; 'original_query+reasoning' appends current trajectory reasoning; 'reasoning+subqueries' concatenates current trajectory reasoning with all current sub-queries.")
    parser.add_argument("--ensure-novel-seen-docs", type=_sm_bool, nargs="?", const=True, default=False, help="When set, the searcher filters out documents already seen in previous iterations before returning results. Guarantees the agent sees seen-top-k novel documents each step.")

    # == Agents =============================================================
    # ── ReAct-specific ───---
    parser.add_argument("--use-plan", type=_sm_bool, nargs="?", const=True, default=False, help="Enable planning in ReAct agent (create plan before loop, update after each search). Only used when --agentic-model=react.")
    # ── cpm_report-specific ──
    parser.add_argument("--max-extend-steps", type=int, default=5, help="Max outline extensions (cpm_report)")
    parser.add_argument("--with-oracle-outline", type=_sm_bool, nargs="?", const=True, default=False, help="When set, the initial search is skipped and the outline is generated from oracle aspects (gold_retriever_analysis). Path to generation.json is auto-resolved from the dataset config. (cpm_report only)")
    parser.add_argument("--hard-mode", type=_sm_bool, nargs="?", const=True, default=True, help="Enforce strict validation rules (cpm_report)")
    # ── Trajectory tracker ────────────────────────────────────────────────
    parser.add_argument("--trajectory-tracker", type=str, default="decision_maker", choices=["off", "monitor", "decision_maker"], help="Trajectory tracker mode. 'off': disabled. 'monitor': compute and log scores only, no intervention. 'decision_maker': trajectory-tracker-based decision maker.")
    parser.add_argument("--llm-decision-maker", type=str, default="claude-sonnet-4-6", help="LLM model for the decision maker (e.g. 'gpt-4.1-mini', 'claude-sonnet-4-6'). When set, overrides --llm-intervene for the decision_maker mode. If not set, falls back to --llm-intervene.")
    parser.add_argument("--llm-intervene", type=str, default="claude-sonnet-4-6", help="LLM model for generating critical_thinking interventions (e.g. 'gpt-4.1-mini', 'claude-sonnet-4-6'). Required when --trajectory-tracker=decision_maker. Uses a lightweight LiteLLMClient separate from the agent's main LLM.")
    parser.add_argument("--decision-maker-history-window", type=int, default=None, help="Number of recent signal/action history entries shown to the decision maker. None (default) means full history.")
    parser.add_argument("--dm-prompt-variant", type=str, default="sim", choices=["nov", "nov_cov", "nov_sim", "nov_cov_sim", "sim", "cov_sim"], help="Decision maker prompt variant controlling which signals the DM sees. 'nov': novelty only. 'nov_cov': novelty + aspect coverage. 'nov_sim': novelty + consec_query_sim + orig_query_sim. 'nov_cov_sim': novelty + aspect coverage + consec_query_sim + orig_query_sim. 'sim': consec_query_sim (primary) + orig_query_sim (guardrail). 'cov_sim': aspect coverage (primary) + consec_query_sim + orig_query_sim (no novelty). Default: 'nov_cov_sim'.")
    parser.add_argument("--aspect-coverage-mode", type=str, default=None, choices=["static", "dynamic"], help="Aspect coverage mode. 'static': aspects extracted from query text (e.g. BrowseCompPlus). 'dynamic': LLM decomposes query and updates aspects each iteration.")
    parser.add_argument("--aspect-coverage-max-aspects", type=int, default=8, help="Soft cap on the number of aspects tracked.")
    parser.add_argument("--llm-aspect-coverage", type=str, default="claude-sonnet-4-6", help="LLM model for aspect coverage evaluation. Falls back to --llm-decision-maker if not set.")

    # ── Evaluation ─────────────────────────────────────────────────────────
    parser.add_argument("--k-values", type=str, nargs="+", default=[1, 3, 5, 10, 25, 50, 75, 100, 500, 1000], help="K values for retrieval evaluation metrics")
    parser.add_argument("--interleaving-window", type=int, default=3, help="Block size for interleaving fusion – consecutive items taken from each list per round (default: 3)")
    parser.add_argument("--rrf-k", type=int, default=60, help="K parameter for reciprocal rank fusion")
    parser.add_argument("--fusion-k", type=int, default=100, help="If set, only the first K documents from each list are passed to the fusion method; documents beyond position K are appended via cheap interleaving.")
    parser.add_argument("--fusion-methods", type=str, nargs="+", default=["interleaving"], choices=ALL_AGGREGATION_FUSION_METHODS, metavar="METHOD", help=f"Subset of aggregation fusion methods to evaluate. If omitted, all methods are run. Choices: {ALL_AGGREGATION_FUSION_METHODS}")
    parser.add_argument("--judge-api-url", type=str, default=None, help="OpenAI-compatible API base URL for an externally-managed accuracy-evaluation judge server (e.g. http://localhost:6009/v1). When set, the pipeline uses this URL instead of auto-starting its own Qwen3-32B judge server. Start the server with: bash experiments/deep_research_agents/vllm_server_scripts/serve_judge_qwen3.sh")

    # ── Execution & Output ─────────────────────────────────────────────────
    parser.add_argument("--result-files-src", type=str, default="s3", choices=["s3", "local"], help="Source for saving/loading results: 's3' uses S3, 'local' uses experiments/deep_research_agents/run_outputs")
    parser.add_argument("--output", type=str, default=None, help="Root directory for results (overrides --result-files-src)")
    parser.add_argument("--max-iteration", type=int, default=100, help="Max iterations/steps per query (all agents)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries when output format fails (all agents)")
    parser.add_argument("--verbose", type=_sm_bool, nargs="?", const=True, default=True, help="Print detailed logs")
    parser.add_argument("--quiet", type=_sm_bool, nargs="?", const=True, default=False, help="Print minimal logs (overrides --verbose)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPU workers for query-level parallelism. 0 = auto-detect from torch.cuda.device_count(). Each worker loads its own model instance on its assigned GPU.")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated physical GPU IDs for workers (e.g. '4,5,6,7'). Overrides the default 0..num_gpus-1 assignment. Also sets num-gpus to the number of IDs if --num-gpus is 1.")
    parser.add_argument("--eval-only", type=_sm_bool, nargs="?", const=True, default=False, help="Skip agent execution and run only evaluation on already-generated results. Runs all evaluators (generation, trajectory, tracker, cited-doc, seen-doc, accuracy, fusion). Requires the run to have been completed at least once so that trajectory/ and retrieval/ files exist. When --judge-api-url is set, also runs accuracy evaluation.")
    parser.add_argument("--parse-only", action="store_true", default=False, help="Parse arguments, print them as JSON, and exit. Used by the submitter to validate hyperparameters before launching a SageMaker job.")

    args = parser.parse_args()

    # ── Post-parse: resolve --output from --result-files-src ──────────────
    if args.output is None:
        args.output = _S3_OUTPUT_PREFIX if args.result_files_src == "s3" else _LOCAL_OUTPUT_PREFIX

    # ── Post-parse: normalise SageMaker-serialised list args ───────────────
    # SageMaker sends nargs="+" values as a single space-separated string;
    # split and coerce them so downstream code always sees a proper list.
    if args.k_values and isinstance(args.k_values[0], str):
        args.k_values = [int(x) for v in args.k_values for x in v.split()]
    if args.fusion_methods and len(args.fusion_methods) == 1 and " " in args.fusion_methods[0]:
        args.fusion_methods = args.fusion_methods[0].split()

    return args

def main():
    """Main entry point."""
    args    = _parse_args()

    if args.parse_only:
        import json
        print(json.dumps({k: v for k, v in vars(args).items() if k != "parse_only"}, default=str))
        return

    verbose = args.verbose and not args.quiet

    if args.quiet:
        logging.getLogger("agents").setLevel(logging.ERROR)
        logging.getLogger("agent_tools").setLevel(logging.ERROR)
        logging.getLogger("utils").setLevel(logging.ERROR)
        logging.getLogger("prompts").setLevel(logging.ERROR)

    _resolve_dataset_defaults(args)

    # ── Default aspect-coverage mode per dataset (user can override via CLI) ──
    if args.aspect_coverage_mode is None:
        if args.dataset == "browsecomp_plus":
            args.aspect_coverage_mode = "static"
        else:
            args.aspect_coverage_mode = "dynamic"

    # ── Default DM prompt variant per dataset (user can override via CLI) ──
    if args.dm_prompt_variant is None:
        args.dm_prompt_variant = "nov_cov_sim"

    # ── Resolve --with-oracle-outline to an actual file path ─────────────
    if args.with_oracle_outline:
        _gold_root = Path(__file__).resolve().parent.parent / "gold_retriever_analysis" / "run_outputs"
        _oracle_retriever = f"qwen3_emb_{args.qwen3_size}" if args.retriever == "qwen3_emb" else args.retriever
        _oracle_dataset_dir = f"{args.dataset}_{args.dataset_year}_{args.subset}_{_oracle_retriever}"
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
    num_gpus            = _detect_num_gpus(args.num_gpus)

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
        pipeline_kwargs = _assemble_pipeline_kwargs(args, llm_client=None, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    elif num_gpus > 1:
        # Multi-GPU: each worker loads its own retriever on its assigned GPU.
        # Skip loading in the main process to avoid OOM on shared GPUs.
        llm_client = _setup_llm(args, num_gpus)
        pipeline_kwargs = _assemble_pipeline_kwargs(args, llm_client, retriever=None, num_gpus=num_gpus, verbose=verbose, gpu_ids=gpu_ids)
    else:
        llm_client  = _setup_llm(args, num_gpus)
        retriever   = _setup_retriever(args)
        pipeline_kwargs = _assemble_pipeline_kwargs(args, llm_client, retriever, num_gpus, verbose, gpu_ids=gpu_ids)

    # ── Deep-research-pipeline-specific kwargs ────────────────────────
    _judge_url = args.judge_api_url
    pipeline_kwargs.update({
        "fusion_k": args.fusion_k,
        "fusion_methods": args.fusion_methods,
        "eval_only": args.eval_only,
        "vllm_manager": vllm_manager,
        "total_gpus_on_machine": total_gpus_on_machine,
        "judge_api_url": _judge_url,
    })

    # ── Optional post-retrieval & post-fusion rerankers ─────────────────
    # In multi-GPU mode each worker builds its own reranker on its assigned
    # GPU (see _init_worker).  Loading one in the main process would waste
    # GPU memory on a device that a worker needs (e.g. rankllama = ~14 GB).
    if num_gpus <= 1 and not args.eval_only:
        from utils.general_utils import get_reranker_configs
        _reranker_configs = get_reranker_configs(args.rerank_top_k)
        from decision_making_agent.utils import build_reranker_from_config
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
        _cleanup_event_loop()
        vllm_manager.shutdown()

if __name__ == "__main__":
    main()
    from agentic_retrieval_research.utils.s3_utils import detach_s3fs_finalizers
    detach_s3fs_finalizers()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# Output structure:
#   run_outputs/{dataset}_{split}_{query_key}_{retriever}/{agent}_agent_{model}/
#   e.g. run_outputs/neuclir_2024_news_topic_description_e5/oss_agent_gpt-oss-20b/
#     ├── retrieval/
#     │   └── {query_id}.trec   per-query TREC file (all iterations, col 6 = iter_N)
#     ├── generation/
#     │   └── {query_id}.md     per-query generation output
#     ├── trajectory/
#     │   └── {query_id}.json   per-query trajectory: {qid, question, trajectory}
#     ├── ranking_results.trec
#     └── summary.json
#
# 
# CUDA_VISIBLE_DEVICES=5,6 python experiments/deep_research_agents/run_deepresearch_pipeline.py --dataset browsecomp_plus --limit 1
#
# CUDA_VISIBLE_DEVICES=0,1,2 python experiments/deep_research_agents/run_deepresearch_pipeline.py --dataset neuclir  --query-key topic_narrative --limit 1
# python experiments/deep_research_agents/run_deepresearch_pipeline.py --dataset neuclir --num-gpus 6 --quiet --limit 6
# python experiments/deep_research_agents/run_deepresearch_pipeline.py --dataset browsecomp_plus --eval-only --num-gpus 8 --quiet --eval-only --limit 12

# --dataset neuclir --query-key topic_narrative