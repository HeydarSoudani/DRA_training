# Evaluation Refactor Plan

**Aim:** make `src/evaluation/` a coherent subsystem with three clear concerns —
**generation** (short-answer correctness + report quality), **retrieval** (three
doc levels: surfaced / seen / cited), and **trajectory** (a single rich per-step
trace including controller values and token counts) — and delete the dead code
that has accumulated.

This is a planning doc only. Code lands in the phases below.

---

## Current state (investigation summary)

| File | Class / role | Status |
|---|---|---|
| `retrieval_evaluation.py` | TREC metric fns + `RetrievalEvaluator` (**surfaced**) + `SeenDocRetrievalEvaluator` (**seen**) | active |
| `citation_evaluator.py` | citation metric fns + `CitedDocRetrievalEvaluator` (**cited**) | active |
| `generation_evaluator.py` | `GenerationEvaluator` — length/word/citation counts + optional ROUGE | active, thin |
| `accuracy_evaluator.py` | `AccuracyEvaluator` — LLM-judge short-answer correctness | active |
| `trajectory_evaluator.py` | `TrajectoryEvaluator` — step stats + writes per-query trajectory JSON | active |
| `tracker_evaluator.py` | `TrackerEvaluator` — controller signals, **separate** per-iteration JSON | active |
| `efficiency_tracker.py` | `EfficiencyTracker` (LLM calls/tokens) | **dead** — only used by unused `save_results()` |
| `latency_tracker.py` | `LatencyTracker` | **dead** — no callers |
| `per_search_analysis.py` | `PerSearchAnalysis` | **dead** — no callers |

### Findings
1. **The three retrieval levels already exist but are misnamed and split across two files.**
   - `Doc_surfaced` → `RetrievalEvaluator` (reads `trajectory[*].docs`/`all_docs` = full top-k per step)
   - `Doc_seen` → `SeenDocRetrievalEvaluator` (reads `component_doc_ids` / `output.doc_ids`)
   - `Doc_cited` → `CitedDocRetrievalEvaluator` (reads `cited_docs_ranked_list` / `memory_bank`)

   `build_ranking_results`, the `Metrics@N` block, and `save_item` are duplicated
   ~3× across these classes. `Doc_cited` is coupled to the CPM `memory_bank`
   concept; general report-citation parsing lives separately in
   `evaluate_citation_quality`.
2. **Report generation eval is the real gap.** Short-answer correctness exists
   (`AccuracyEvaluator`, LLM judge). There is no long-form report evaluator.
3. **No token logging in the trajectory.** Steps log `think`, `search_query`,
   `docs`, `component_doc_ids` — no tokens. GLM tracks `_cumulative_output_tokens`
   transiently and discards it. `EfficiencyTracker` was the intended home but is
   unwired. Controller values are logged in a *separate* `tracker/` stream
   (`tracker_score_history`), not alongside the trajectory step they belong to.

### Wiring (callers to keep working)
- `src/pipeline/eval_utils.py::build_evaluators` constructs all evaluators.
- `src/pipeline/eval_utils.py::evaluate_and_save` runs generation/trajectory/
  cited/seen/accuracy/tracker and writes `summary.json`.
- `src/pipeline/eval_utils.py::run_fusion_eval` computes retrieval metrics per
  fusion method (so `RetrievalEvaluator.evaluate()` is effectively bypassed for
  the surfaced level — metrics come from fusion).
- `experiments/run_dra_inference.py` and `src/pipeline/orchestration.py` call
  `*.save_item(...)` per query and `evaluate_and_save` / `run_fusion_eval` at the end.

---

## Target layout

```
src/evaluation/
  __init__.py                 # re-exports + back-compat aliases
  retrieval/
    __init__.py
    metrics.py                # compute_trec_metrics, evaluate_results, evaluate_from_ranking_results, _metrics_at_n
    base.py                   # _BaseDocRetrievalEvaluator: fuse -> RankingResults -> TREC + Metrics@N + save_item
    surfaced.py               # SurfacedDocEvaluator   (was RetrievalEvaluator)
    seen.py                   # SeenDocEvaluator        (was SeenDocRetrievalEvaluator)
    cited.py                  # CitedDocEvaluator       (was CitedDocRetrievalEvaluator + report-citation parsing)
  generation/
    __init__.py
    basic_stats.py            # GenerationEvaluator (unchanged)
    short_answer.py           # ShortAnswerEvaluator (was AccuracyEvaluator)
    report.py                 # ReportEvaluator (NEW: LLM-judge rubric + citation faithfulness)
  trajectory/
    __init__.py
    trajectory_evaluator.py   # TrajectoryEvaluator + per-step token fields, controller values folded in
    token_tracker.py          # repurposed EfficiencyTracker -> per-step/per-query token aggregator
  citation_metrics.py         # compute_citation_metrics / compute_recall_at_n (pure fns; shared by cited.py & report.py)
```

**Deleted:** `latency_tracker.py`, `per_search_analysis.py`,
`retrieval_evaluation.save_results()`.

**Back-compat:** `__init__.py` keeps aliases (`RetrievalEvaluator = SurfacedDocEvaluator`,
`SeenDocRetrievalEvaluator = SeenDocEvaluator`, `CitedDocRetrievalEvaluator = CitedDocEvaluator`,
`AccuracyEvaluator = ShortAnswerEvaluator`) so existing imports in `eval_utils.py`,
`orchestration.py`, and `run_dra_inference.py` keep working until they're migrated.

---

## Phase 1 — Retrieval: full reorg + shared base
1. Move TREC helpers (`compute_trec_metrics`, `evaluate_results`,
   `evaluate_from_ranking_results`) into `retrieval/metrics.py`. Extract the
   triplicated `Metrics@N` computation into a `_metrics_at_n(qrels, ranking_results)` helper.
2. Write `_BaseDocRetrievalEvaluator` with one abstract hook
   `_doc_iterations(result) -> List[List[doc]]`; lift the shared
   `build_ranking_results`, fusion, `Metrics@N`, `evaluate`, `print_results`,
   `save_item` into it.
3. Implement the three subclasses by overriding only `_doc_iterations`:
   - `surfaced.py`: full per-step `docs`/`all_docs` (current `_extract_trajectory_iterations`).
   - `seen.py`: `component_doc_ids` / `output.doc_ids` (current `_extract_seen_iterations`).
   - `cited.py`: `cited_docs_ranked_list` → `memory_bank` → **NEW** general
     `extract_citations_from_text(report) + citation_to_doc_id` so "cited" works
     for any report agent, not just CPM.
4. Update `build_evaluators` return + `__init__` aliases.

**Risk:** low — confined to `src/evaluation/` + aliases.

## Phase 2 — Generation: add `ReportEvaluator`
1. Keep `GenerationEvaluator` (basic stats) as-is.
2. Rename `AccuracyEvaluator` → `ShortAnswerEvaluator` (alias kept).
3. New `ReportEvaluator` (LLM-as-judge):
   - Rubric scoring: coverage / relevance-to-question / organization, via existing
     `LiteLLMClient` + vLLM judge-server plumbing (already started at
     `run_dra_inference.py:237`).
   - Citation faithfulness: reuse `compute_citation_metrics` (cited ∩ relevant)
     from `citation_metrics.py`.
4. `build_evaluators` selects `ShortAnswerEvaluator` vs `ReportEvaluator` by task
   type (short `answers` present vs report task flag). `evaluate_and_save`
   prints/saves whichever is active.

**Risk:** low–medium — new judge calls; gate behind task type so default runs unaffected.

## Phase 3 — Trajectory: tokens + unified trace (widest blast radius — do last)
1. Add `input_tokens`/`output_tokens` to each agent's step-append dict from
   `response.usage`, across **all** agents: base_agent, react, oss, glm, drtulu,
   tongyi, searcho1, searchr1, selfask, stepsearch, research, webweaver, cpm_*.
2. `token_tracker.py` (repurposed `EfficiencyTracker`) aggregates per-step →
   per-query totals.
3. Fold controller values (`controller_action`, `controller_reasoning`, signals)
   from `tracker_score_history` into the matching trajectory step by `iter_num`
   (keep the `tracker/` file or cross-reference) so a single step record carries:
   reasoning + search_query + retrieved doc ids + controller values + tokens.
4. `TrajectoryEvaluator.evaluate` adds token stats (mean/total per query).

**Risk:** high — touches every agent. Do agent-by-agent, verifying one agent's
trajectory JSON before fanning out.

## Phase 4 — Cleanup
- Delete `latency_tracker.py`, `per_search_analysis.py`,
  `retrieval_evaluation.save_results()`.
- Prune `__init__.py` exports + module docstring; drop the `EfficiencyTracker`
  import in `retrieval_evaluation.py`.

---

## Decisions locked
- Retrieval: **full reorg + shared base class** (aliases for compatibility).
- Report eval: **LLM-judge rubric + citation faithfulness**.
- Token logging: **all agents, per-step + per-query total**.
