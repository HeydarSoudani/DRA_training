# `analysis/`

Offline analysis tooling for the DRA pipeline. Two independent tools live here:

1. **`reasoning_error_analysis.py`** — attributes *why* agent trajectories fail
   on TRQA, assigning each query a primary root-cause stage.
2. **`corpus_analysis.py`** — utilities for inspecting and reformatting the
   retrieval corpus.

Both are standalone entry points; nothing else in the pipeline imports them.

---

## 1. Reasoning-error attribution — `reasoning_error_analysis.py`

### The main question

> **When an agent's reasoning trajectory fails, *at which stage* did it break —
> and across a whole run, what is the distribution of those failure sources?**

Given a completed TRQA run (agent trajectories under `run_outputs/…/trajectory/`)
and the per-query **intermediate-info gold**, this tool assigns each query a
single primary root-cause stage in the agent's pipeline:

```
success  ←  planning  →  retrieval  →  reading / aggregation
                                            (+ gold_incomplete = data problem, not the agent)
```

The distribution of these labels over the run *is* the answer to the question.
Attribution is **earliest-failing**: a query is blamed on the *first* gate it
breaks, so the stages are mutually exclusive and every query lands in exactly
one bucket. The method is **rule-based**; two small LLM judges are optional
(`--judge`) and only fire on the slices where the rules are provably blind.

### The idea — recompute the answer under counterfactual coverage

The key enabler is that TRQA questions are multi-hop **aggregations** over a set
of gold entities `E*`: find a property value for each entity, then reduce them
with a *known* operator `f ∈ {COUNT, AVG, SUM, MAX, MIN, LATEST, EARLIEST}`.

Because `f` is known, we don't have to guess where the agent went wrong — we can
**replay the aggregation over different subsets of the gold entities** and see
which subset reproduces the agent's actual answer. If the agent's answer matches
`f` computed over only the entities it *planned*, its planning set is what
bottlenecked it; if it matches `f` over only what it *retrieved*, retrieval is
the ceiling; and so on. This turns a fuzzy "why did it fail?" into concrete
numeric comparisons — no LLM judgment needed for the common cases.

Concretely, per trajectory we build three coverage sets / values and three
counterfactual answers from them:

| Set / value | Meaning | Source |
|-------------|---------|--------|
| `E_plan` | entities the agent *named* in its think + query text | string match |
| `E_ret`  | entities appearing in the *content* of retrieved docs | needs `--corpus` |
| `a`      | the agent's final numeric answer | re-extracted from the generation |

`a_star = f(E*)`  ·  `a_plan = f(E_plan)`  ·  `a_ret = f(E_ret)`

### Metrics — and why each one exists

| Metric | Definition | Why we need it (what it isolates) |
|--------|------------|-----------------------------------|
| `a_star = f(E*)` | answer recomputed over all gold entities | Sanity anchor: should equal the stored `answer`. If it doesn't, the *gold* is incomplete (`gold_consistent = false`) and every counterfactual below is untrustworthy — so we must detect this before attributing anything. |
| `gold_consistent` | does `a_star` reproduce the stored `answer`? | Separates a **data problem** from an **agent problem**. Rows that fail this are shunted to `gold_incomplete` and *excluded*, so we never blame the agent for a broken gold row. |
| `plan_recall` = \|E_plan ∩ E*\| / \|E*\| | fraction of gold entities the agent ever *named* | Measures the **planning** stage in isolation: did the agent even enumerate all the entities it needed? `< 1.0` ⇒ an entity was never in play, which no amount of good retrieval/reading can fix. |
| `retrieval_recall` = \|E_ret ∩ E*\| / \|E*\| | fraction of gold entities surfaced in retrieved-doc *content* | Measures the **retrieval** stage in isolation, *conditioned on* good planning: the agent named the entity but the corpus never surfaced it. Needs `--corpus`; without one this stage is left unassessed (and folded into reading/aggregation). |
| `decisive_gap` | is the retrieval miss big enough to change the answer? | Guards against blaming retrieval for a **benign** miss. A missing entity that wouldn't move `f` (e.g. a non-max value under `MAX`) shouldn't count as a retrieval *failure* — only gaps that actually break the answer are decisive. |
| `a_plan`, `a_ret` matches | does the agent's answer equal `f` over its planned / retrieved set? | **Corroboration.** If the answer is exactly what a correct aggregation over the (incomplete) planned/retrieved set would give, that's positive evidence the blame really sits at that gate — not a coincidence downstream. |
| `downstream_subtype` | reading vs aggregation-logic (zero-LLM) | Splits the final bucket: is `a` reproducible as some *other* reducer over the *correct* gold values? If yes ⇒ **aggregation-logic** error (right facts, wrong operator/arithmetic); if no ⇒ **reading** error (used a value not in the gold). Tells us *whether the agent got the facts and fumbled the math, or got the facts wrong*. |
| `n_gold_entities` | size of `E*` (a proxy for hop count) | Difficulty axis: lets us ask whether the failure *stage* shifts as questions get harder (more entities to track). |
| `aggregation` | the operator `f` for the query | Operator axis: lets us ask which operators are hardest and *how* they fail (e.g. COUNT failing on polarity vs AVG failing on a missed entity). |
| `num_steps` | trajectory length | Recorded per query for cross-run/effort context (not yet a headline figure). |

### Attribution rule (earliest-failing gate)

```
not gold_consistent                         → gold_incomplete   (data issue, excluded)
a ≈ answer                                   → success
E_plan ⊊ E*                                  → planning          (never planned an entity)
E_ret  ⊊ E*  and the gap is decisive         → retrieval         (planned, not retrieved; needs corpus)
E_ret == E*  but a ≠ a_star                  → reading / aggregation
```

Reading the rule top-down mirrors the pipeline: we only blame retrieval once
planning is clean, and only blame reading/aggregation once the agent provably
had every entity in hand — so each stage's count reflects failures *originating*
there, not inherited from an earlier break.

### Gold quirks handled explicitly

- **`gold_consistent`** — `entity_values` is occasionally incomplete (recomputing
  `f` over it doesn't reach the stored `answer`); those rows are flagged and
  attributed conservatively as `gold_incomplete` rather than trusting the
  counterfactuals.
- **`COUNT`** — counts entities whose value matches a target *named in the
  question*; polarity (positive / negated) is chosen to reproduce `answer`.
- **No per-entity gold docs** — `E_ret` is assessed by matching entity names
  against retrieved-doc content, so it requires `--corpus`. Without one, the
  retrieval stage is left unassessed.

### Usage

Run-selecting knobs mirror `experiments/dra_inference.py`; the run dir, gold
file and corpus are **derived automatically** from them (see `utils/config.py`).
Mostly-fixed variables live in `experiments/configs/dra_analysis.yaml`
(`--config`) and any of them can be overridden with the matching `--flag`.

```bash
# analyse a run (controller off)
python analysis/reasoning_error_analysis.py --agentic-model glm --controller off

# a different subset, with the gated LLM judges enabled
python analysis/reasoning_error_analysis.py --agentic-model glm --subset wiki2 --judge
```

Only `--dataset trqa` is supported (attribution needs the intermediate-info gold).

### Pipeline

The tool runs as **compute → write jsonl → read jsonl → plot**, so the jsonl is
the single source of truth and plotting is decoupled from attribution:

1. `analyze_qid()` computes the metrics/attribution for each query,
2. `write_outputs()` writes them to `reasoning_error_per_query.jsonl`,
3. the plotter *reads that file back* and renders the figures + CSV.

Because step 3 depends only on the file, you can re-plot (or point at any past
run) without re-running attribution:

```bash
python analysis/reasoning_error_analysis.py --plot-only {run_dir}/analysis/
```

### Outputs

Written to `{run_dir}/analysis/` by default. Deliberately plain (matplotlib
defaults, no custom styling) — these are exploration figures, not paper figures:

- `reasoning_error_per_query.jsonl` — one root-cause record per query (+ meta header)
- `reasoning_error_summary.json` — stage distribution, mean recalls, by-aggregation breakdown
- `reasoning_error_per_query.csv` — tidy per-query rows (for cross-run comparison)
- `stage_distribution.png` — counts per failure stage (the headline)
- `stage_by_aggregation.png` — stage mix per aggregation operator
- `stage_by_complexity.png` — stage mix per #gold-entities bin (hop count)
- `coverage_scatter.png` — plan-recall vs retrieval-recall (only when a corpus was supplied)

A compact stage table is also printed to stdout (the full summary lives in
`reasoning_error_summary.json`).

### Layout

```
reasoning_error_analysis.py     CLI + per-query loop -> write jsonl -> plot (thin driver); --plot-only re-plots
utils/config.py                 CLI/YAML config + run-dir / gold / corpus path derivation
utils/reasoning_error_utils.py  everything else: loaders, gold resolution, attribution, judges,
                                jsonl I/O, and the matplotlib rendering (jsonl -> PNGs + CSV)
```

---

## 2. Corpus analysis — `corpus_analysis.py`

Utilities for inspecting and reformatting the retrieval corpus (JSONL). Provides:

- `get_corpus_entry(index, …)` — spot-check a document by row index (reads only
  that line for large files, or indexes a pre-loaded HuggingFace `Dataset`).
- `convert_corpus_to_retriever_format(in, out)` — rename `docid → id`,
  `segment → contents` and add a `title` field, streaming line-by-line (with
  periodic GC to avoid OOM on huge corpora).
- `extract_title_and_contents(in, out)` — split each doc's `contents` at the
  first newline into `title` + `contents`.

The `__main__` block toggles between these; edit it (or the commented recipes)
to pick an operation. Example spot-check:

```bash
python analysis/corpus_analysis.py --corpus_path data/neuclir/corpus/corpus_en.jsonl
```
