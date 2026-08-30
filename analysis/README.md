# `analysis/`

Offline analysis tooling for the DRA pipeline. Two independent tools live here:

1. **`reasoning_error_analysis.py`** — localises *where* agent trajectories break
   on TRQA, by comparing the gold decomposition against the trajectory.
2. **`corpus_analysis.py`** — utilities for inspecting and reformatting the
   retrieval corpus.

Both are standalone entry points; nothing else in the pipeline imports them.

---

## 1. Reasoning-error analysis — `reasoning_error_analysis.py`

### 1.1 The question

> **When an agent's trajectory fails, which part of the gold did it fail to
> reproduce — the entities, the documents, or the arithmetic?**

TRQA questions are multi-hop **aggregations**: find one property value for each of
a set of gold entities, then reduce those values with a known operator. The gold
ships *both* sides of that decomposition, so the method is a straight two-sided
comparison — no LLM judge and no manual annotation anywhere in the loop.

---

### 1.2 Formalisation and notation

#### 1.2.1 Gold side

For a query `q` the gold is the tuple

```
G(q) = ( Q, E*, V*, τ, f, a*, D* )
```

| Symbol | Meaning | Where it comes from |
|---|---|---|
| `Q` | question text | `queries_<split>.jsonl` → `text` |
| `E* = (e_1, …, e_n)` | gold entities, `n = \|E*\|` (typically 2–50) | `…_intermediate_info.jsonl` → `entity_values[].entity` |
| `V* = (v*_1, …, v*_n)` | the property value of each `e_i` | `entity_values[].value` |
| `τ` | the property datatype — **one per query**, shared by all `v*_i` | `property.datatype` |
| `f` | the aggregation operator (`AVG`, `COUNT`, `MAX`, `EARLIEST`, …) | `aggregation` |
| `a*` | the stored answer | `answer` |
| `D*` | gold documents, **query-level** (relevant qrels judgements) | `qrels_<split>.txt` |
| `D*_i ⊆ D*` | gold documents of entity `e_i` | derived — §1.2.3 |

`τ` fixes the shape of every `v*_i`:

| `τ` | shape of `v*_i` | example |
|---|---|---|
| `Time` | year, as a number | `1878` |
| `Quantity` | number, arbitrary unit/precision | `8270.79237550612` |
| `WikibaseItem` | list of labels (79% of wiki2) | `["Russian Orthodox Church", "Eastern Orthodoxy"]` |
| `GlobeCoordinate` | one `[lat, lon]` pair | `[55.75, 37.61]` |

Example gold row:

```json
{"qid": "13_p569",
 "entity_values": [{"entity": "Joseph Stalin", "value": 1878},
                   {"entity": "Nikita Khrushchev", "value": 1894}],
 "property": {"label": "date of birth", "datatype": "Time"},
 "aggregation": "AVG", "answer": 1886}
```

#### 1.2.2 Trajectory side

A run writes one trajectory per query: a sequence of `M` steps, plus the final
generation.

```
T(q) = ( (t_k , s_k , O_k) )_{k = 1 … M} ,  g
```

with, at step `k`: `t_k` the think text, `s_k` the search query issued, `O_k` the
doc ids returned; and `g` the agent's final free-text generation.

That is collapsed into exactly **two channels plus one scalar**:

```
A  =  strip_cite( t_1 ⊕ s_1 ⊕ … ⊕ t_M ⊕ s_M )     what the agent said
O  =  O_1 ∪ … ∪ O_M                                what retrieval delivered
â  =  extract(g)                                    what the agent committed to
```

- `A` merges thinking **and** querying on purpose: an entity the agent only ever
  typed into a search box was still enumerated, and that is a planning success.
- `strip_cite` removes inline citations (`[15641-0002]`, `[6]`) from `A`: they
  would otherwise normalise to bare tokens (`15641`, `0002`) that collide with
  year-shaped gold values.
- `g` is kept **verbatim** (citations are stripped from `A` only), so `â` is read
  exactly the way `accuracy.jsonl` reads it and the numbers stay commensurable.

#### 1.2.3 Per-entity gold documents `D*_i`

qrels are query-level, but TRQA doc ids are `pageid-passage`, so a query's qrels
can be grouped by page and each page assigned to the gold entity its **title**
names:

```
page(d)  :  "15641-0012"  ↦  "15641"
title(p) :  page id       ↦  Wikipedia page title      (cached lookup)

D*_i  =  { d ∈ D*  :  title(page(d)) ≡ name(e_i) }
```

`≡` is equality after normalisation, or a disambiguated prefix
(`"Georgia (U.S. state)" ≡ "Georgia"`). Pages matching no gold entity are the
roster/list documents (the ones that enumerate `E*` itself) and are left out.

Titles come from `qrels/page_titles.json`, a `pageid → title` cache built once by
a single pass over the corpus (first line of each doc's `contents`). After that
the analysis never touches the corpus.

Resolution rate: **93%** of wiki2 entities, **82%** of wiki1. When `D*_i = ∅` the
retrieval column is `null` (*unassessed*), **never** `false` — a title-lookup miss
must never be reported as a retrieval failure.

#### 1.2.4 Three-valued results

Every column below is `true` / `false` / `null`, and

> **`null` always means *not assessed* — never zero, never failure.**

Rates are means over the assessed entries only, and each one carries its own
denominator into the output so the two can never be silently conflated.

---

### 1.3 Error types

Three error types — two scored per gold entity, one per query.

| Error type | Column | Scope | Definition | `null` when |
|---|---|---|---|---|
| **Planning** — did the agent enumerate the entity? | `e_in_A(i)` | entity | `1[ some name variant of e_i occurs in A ]` | never |
| **Retrieval** — did evidence for it surface? | `retrieved(i)` | entity | `1[ D*_i ∩ O ≠ ∅ ]` | `D*_i = ∅` (title unresolved) |
| **Aggregation** — did the arithmetic hold? | `agg_ok(q)` | query | `f(V_found) ≈ â` | `f` unsupported, or `V_found = ∅` |

`V_found = { v*_i : v*_i occurs within VALUE_WINDOW = 400 characters of a mention
of e_i in A }` — the gold values the agent actually stated. It is the **input** to
the aggregation check, not an error type of its own: `agg_ok` asks whether the
answer follows from the facts the agent had, so it has to know which those were.
(The per-entity indicator behind it, `v_in_A`, is still written to the output
tables as a diagnostic — §1.6.)

`agg_ok` compares against **`â`, the agent's own answer** — not `a*`. It measures
*consistency* with the gathered facts, which is a different question from
correctness.

The two kinds of test are matched differently: **containment** (`e_in_A`,
`V_found`) searches normalised text with a per-datatype value matcher, while
**comparison** (`agg_ok`, `gold_ok`, `answer_correct`) reuses the evaluator's
`soft_exact_match` so the numbers stay commensurable with `accuracy.jsonl`.
Containment was audited on 109 (query, entity) pairs: **0 entity errors, 0 value
errors**. The mechanics are documented in the `reasoning_error_utils` docstrings.

**No cascade.** The three are marginals, not ordered stages, so the interesting
joint cases fall out of the signature rather than being encoded as rules:

- `retrieved ∧ ¬e_in_A` — the evidence was in hand before the agent had a plan.
- full `plan_cov` and `ret_recall`, `agg_ok = false` — had everything, fumbled the
  arithmetic.

Two columns support the three but are not errors themselves:

| Column | Definition | Role |
|---|---|---|
| `gold_ok(q)` | `f(V*) ≈ a*` | **validity filter** — is the gold itself recomputable? Holds on **91%** of wiki2; the rest cannot support `agg_ok`. |
| `answer_correct(q)` | `â ≈ a*` | end-to-end correctness, same matcher as `accuracy.jsonl` |

#### 1.3.1 Per-query (macro) scores

Pooling the entity columns over all entities of all queries is the **micro** view.
A 50-entity query then counts fifty times and a 3-entity query three times, which
answers "what fraction of gold entities were lost" but not "how badly does a
typical query break". So every query is *also* scored on its own — **macro** — and
it is that distribution the figures are drawn from.

For query `q` with `n = |E*(q)|` and `S = { i : D*_i ≠ ∅ }`:

| Score | Error type | Definition | Denominator | `null` when |
|---|---|---|---|---|
| `plan_cov` | planning | `#{ i : e_in_A(i) } / n` | all gold entities | never |
| `ret_recall` | retrieval | `#{ i ∈ S : retrieved(i) } / #S` | entities whose gold docs resolved | `S = ∅` |
| `agg_ok` | aggregation | one boolean per query | — | unsupported `f`, or `V_found = ∅` |
| `answer_correct` | — | `â ≈ a*` | — | never |

A query where no gold page title resolved has no retrieval recall at all; plotting
it as `0.0` would report a title-lookup miss as a total retrieval failure. The
figures drop those points and print the count in the caption.

Retrieval is scored **per entity, not per document**: "did every hop get evidence",
not "what fraction of the qrels was seen". The per-entity doc counts
(`n_gold_docs`, `n_gold_docs_seen`) ride along on the entity rows as diagnostics,
so a doc-level recall is a groupby away if it is ever wanted.

Micro and macro will not agree. That disagreement is a result — it says the error
mass is concentrated in the large-`n` queries — which is why the summary table
prints both blocks side by side.

---

### 1.4 Scope

Only `--dataset trqa` — the analysis needs the intermediate-info gold.

Planning and retrieval are scored on **every** query. Aggregation additionally
needs `f` to be recomputable, so it is `null` (*not assessed*, never `false`) when:

- the operator's parameter lives only in the question text (`SUM_TOP_K`,
  `NTH_LATEST`, `COUNT_LT_X`, … — wiki1 only). Recomputable operators cover all of
  wiki2 and ~30% of wiki1;
- `V_found = ∅` — otherwise a query where the agent surfaced nothing would score as
  an aggregation *error* under `COUNT` (80% of wiki2) and as *unassessed* under
  `AVG`, and the denominator would stop being comparable across operators.

`COUNT` counts the entities whose value matches a target named in the question
("Eastern Orthodoxy"), positive polarity only — that reproduces `answer` on 87% of
wiki2 `COUNT` rows, and the rest are caught by `gold_ok`.

---

### 1.5 Inputs — variables and files

#### 1.5.1 Run-selecting knobs (CLI)

These mirror `experiments/dra_inference.py` one-for-one; the run directory and
**every** gold-side path is derived from them, so no path is ever hand-typed.

| Flag | Default | Effect |
|---|---|---|
| `--agentic-model` | `glm` | agent whose run to analyse (run dir) |
| `--dataset` | `trqa` | only `trqa` is supported |
| `--subset` | from config (`wiki2`) | `wiki1` \| `wiki2` — picks the gold split |
| `--retriever` | `qwen3_emb_4b` | retriever of the run (run dir) |
| `--controller` | `off` | `off` \| `monitor` \| `action` (run dir) |
| `--controller-prompt-variant` | `nov_cov_sim` | only affects the dir name when `--controller action` |

Run-control flags: `--limit N`, `--qids …`, `--figures {none,core,all}`,
`-v/--verbose`, `--config PATH`.

#### 1.5.2 File-backed variables

Mostly-fixed variables live in `experiments/configs/dra_analysis.yaml`
(`--config`); any of them can be overridden by passing the matching `--flag`.

Precedence: **built-in fallback < YAML file < CLI flag.**

| Key | Default | Meaning |
|---|---|---|
| `subset` | `wiki2` | gold split (also a real CLI flag) |
| `dataset_year` | `null` → `test` | eval split |
| `query_key` | `null` → `text` | JSONL key holding the query text |
| `data_root` | `null` → `$DRA_DATA_ROOT` | holds `trqa/queries` and `trqa/qrels` |
| `output` | `null` → `$DRA_OUTPUT_ROOT` | run-outputs root |
| `use_plan` | `false` | react run-name variant — must match the run |
| `ensure_novel_seen_docs` | `false` | appends `_novel` to the controller-config dir |
| `llm_controller` | `claude-sonnet-4-6` | in the dir name only when `--controller action` |
| `run_dir`, `gold`, `queries`, `qrels`, `page_titles`, `out` | `null` | explicit path overrides; `null` = derive |
| `agg_tol_pct` | `1.0` | % tolerance for `f(V) ≈ answer` (`agg_ok`, `gold_ok`) |
| `value_tol_rel` | `0.005` | relative tolerance when finding a `Quantity` value in `A` |

#### 1.5.3 Derived paths

```
run_dir     {output}/{dataset}_{split}_{query_key?}_{retriever}/{agent}_{backend}_{model}/{ctrl}/
gold        {data_root}/trqa/queries/queries_{split}_intermediate_info.jsonl
queries     {data_root}/trqa/queries/queries_{split}.jsonl
qrels       {data_root}/trqa/qrels/qrels_{split}.txt
page_titles {data_root}/trqa/qrels/page_titles.json
out         {run_dir}/analysis
```

Each derived value is echoed on stdout at startup (`Auto-derived run dir: …`), so a
mismatch shows up before any work happens. `.env` is loaded first, exactly like
`dra_inference.py`, so the API/vLLM backend segment of the run dir resolves the
same way it did at inference time.

#### 1.5.4 Input file formats

| File | Format | Fields used |
|---|---|---|
| `queries_<split>_intermediate_info.jsonl` | one JSON per query | `qid`, `entity_values[].{entity,value}`, `property.datatype`, `aggregation`, `answer` |
| `queries_<split>.jsonl` | one JSON per query | `id`, `text` (needed by `COUNT`, which reads its target out of the question) |
| `qrels_<split>.txt` | TREC qrels | `qid 0 docid rel`, `rel ≠ 0` kept |
| `page_titles.json` | one JSON object | `{pageid: title}` |
| `{run_dir}/trajectory/{qid}.jsonl` | line 1 `{"record":"meta", …}`, then one line per step | meta: `generation`; steps: `think`, `search_query`, `seen_docs[]` |

The trajectory directory is the **only** run-side input. Nothing under
`generation/`, `retrieval/` or `controller/` is read, and the corpus is never
touched.

---

### 1.6 Outputs

Written to `{run_dir}/analysis/` — two tables and a figure directory.

#### 1.6.1 `reasoning_errors.jsonl` — per-entity table (micro)

Line 1 is a `{"record": "meta", …}` header carrying the run key and the whole
summary; each following line is one `(qid, entity)` row:

```json
{"qid":"13_p569","entity":"Joseph Stalin","aggregation":"AVG",
 "e_in_A":true,"v_in_A":true,"retrieved":true,"agg_ok":true,"gold_ok":true,
 "a_pred_structured":true,"n_gold_docs":4,"n_gold_docs_seen":2}
```

| Field | Meaning |
|---|---|
| `qid`, `entity`, `aggregation` | keys / query-level context repeated onto every row |
| `e_in_A`, `retrieved` | the two per-entity error columns — planning, retrieval (§1.3) |
| `v_in_A` | diagnostic: was this entity's gold value stated? (the membership test behind `V_found`) |
| `agg_ok`, `gold_ok`, `a_pred_structured` | query-level, repeated so the table groups by either key |
| `n_gold_docs`, `n_gold_docs_seen` | `\|D*_i\|` and `\|D*_i ∩ O\|` — diagnostics, not metrics |

`a_pred_structured` is `false` when the generation carried no `\boxed{}` /
`<answer>` / `Exact Answer:` marker, so the answer was read as the **first number
in the prose** — on *"…served before Brezhnev (1966 onward)… the earliest is
1922"* that reads 1966. Such rows are still scored (dropping every prose answer
would cost too much) but `agg_ok` on them wants an eyeball; the count is in the
summary and each one is warned about on stderr.

#### 1.6.2 `reasoning_errors_by_query.jsonl` — per-query table (macro)

Same meta header, then one row per query:

| Field | Meaning |
|---|---|
| `qid`, `aggregation`, `datatype` | keys and grouping variables |
| `n_entities`, `n_ret_assessed` | `n` and `#S` — the two denominators |
| `plan_cov`, `ret_recall` | the per-entity error types, per query (§1.3.1) |
| `read_cov`, `read_cov_given_named` | diagnostics: share of gold values stated, over all entities / over the named ones |
| `agg_ok`, `gold_ok`, `answer_correct` | the query-level booleans |
| `f_found`, `a_pred`, `answer` | `f(V_found)`, `â`, `a*` — the raw operands behind `agg_ok` |
| `a_pred_structured` | see above |
| `n_steps`, `n_seen_docs` | `M` and `\|O\|` — trajectory size, for correlating with effort |

This is the table every distribution figure reads, so re-plotting never re-runs the
matching.

#### 1.6.3 `figures/`

Controlled by `--figures {none,core,all}` (default `core`).

| File | Level | Reads | What it shows |
|---|---|---|---|
| `reasoning_errors_funnel.png` | core | entity rows | the micro view: four bars over the pooled gold entities; the drop between bars is the error mass at that column |
| `dist_overlay_ecdf.png` | core | query rows | **the primary figure** — per-query plan / retrieval / reading as three ECDFs on one axes; furthest left is the worst stage |
| `dist_plan.png`, `dist_retrieval.png`, `dist_reading.png` | all | query rows | per error type: distribution, then box+strip split by gold-entity count and by operator |
| `agg_vs_coverage.png` | all | query rows | `agg_ok` rate against the fraction of gold values the agent stated — the rightmost bar is arithmetic failing with every fact in hand |
| `joint_retrieval_vs_reading.png` | all | query rows | per-query scatter, coloured by end-to-end correctness; below the diagonal = surfaced but never stated |

ECDF rather than histograms for the primary figure because the scores are ratios
over 3–50 entities: any fixed binning invents structure a step function does not.
Where a histogram *is* used, `0` and `1` get their own bars — they are the modes,
and smearing them into a neighbouring interval hides them.

#### 1.6.4 stdout summary

```
========================================================================
REASONING-ERROR ANALYSIS  (4 queries, 9 gold entities)
========================================================================
  MICRO  (pooled over entities)
    named in A     (planning)     1.000   9 entities
    gold doc seen  (retrieval)    0.889   9 assessed
    value in A     (reading)      0.778   9 entities
    f(V_found)=ans (aggregation)  0.667   3 assessed
------------------------------------------------------------------------
  MACRO  (per query, then averaged — every query weighs the same)
    score                        mean   median      n   n/a
    plan coverage             1.000    1.000      4     0
    retrieval recall          0.875    0.917      4     0
    reading coverage          0.792    0.833      4     0
    reading | named           0.792    0.833      4     0
    end-to-end correct        0.500                4     0
------------------------------------------------------------------------
  gold recomputable             1.000   (0 queries: unsupported operator)
  unstructured final answer         1   of the assessed (agg_ok read the first number in prose)
========================================================================
```

The `n/a` column is the count of queries the score was **not assessed** on — it is
never folded into the mean.

---

### 1.7 Usage

```bash
python analysis/reasoning_error_analysis.py --agentic-model glm --subset wiki2
python analysis/reasoning_error_analysis.py --subset wiki1 --controller action
python analysis/reasoning_error_analysis.py --subset wiki2 --figures all
python analysis/reasoning_error_analysis.py --subset wiki2 --limit 20 -v
```

To rebuild the page-title cache (only needed if the qrels change), scan the corpus
for the page ids appearing in `qrels/` and keep the first line of each doc's
`contents`.

---

### 1.8 Analysis structure

Pipeline, end to end:

```
CLI knobs + YAML  ──▶  resolve_paths      run_dir / gold / queries / qrels / page_titles / out
                          │
      gold jsonl ─────────┤
      queries jsonl ──────┼─▶ load_gold          G(q) = (Q, E*, V*, τ, f, a*, D*_i)
      qrels + titles ─────┘
                          │
   trajectory/*.jsonl ──▶ load_trajectories      T(q) ↦ (A, O, g)
                          │
                          ├─▶ analyze_qid        per entity: e_in_A, retrieved, v_in_A
                          │                      per query : agg_ok, gold_ok, macro scores
                          │
                          ├─▶ summarize          micro rates + macro distributions
                          │
                          ├─▶ write_jsonl        reasoning_errors.jsonl
                          ├─▶ write_query_jsonl  reasoning_errors_by_query.jsonl
                          └─▶ make_figures       figures/*.png
```

Files:

```
reasoning_error_analysis.py     CLI + per-query loop -> two jsonl tables + figures (thin driver)
utils/config.py                 CLI/YAML config + run-dir / gold / qrels path derivation
utils/reasoning_error_utils.py  matching -> aggregators -> loading -> analysis -> jsonl
utils/reasoning_error_plots.py  the figures (the only module that imports matplotlib)
```

`reasoning_error_utils.py` is laid out in the order the pipeline uses it:
(1) matching, (2) aggregators, (3) loading, (4) analysis, (5) output. It imports
`trqa_match.py` straight from its file rather than through the `evaluation`
package, whose `__init__` pulls in the whole retriever stack (torch, pyserini/JVM)
and costs ~65s of startup this analysis has no use for.

---

### 1.9 Caveats

- **`V_found` is a lower bound.** A value the agent read, understood and
  paraphrased out of numeric recognisability is not counted, which makes `agg_ok`
  recompute `f` over a subset. The audit puts the false-*positive* rate at 0; the
  false-negative rate is not measured.
- **`retrieved` depends on title resolution**, which fails on 7% of wiki2 and 18%
  of wiki1 entities. Those are `null`, so retrieval recall is computed on a
  slightly different entity population than plan coverage.
- **`agg_ok` is consistency, not correctness.** An agent that mis-reads two values
  in compensating directions and lands on a number matching its own recomputation
  scores `agg_ok = true` while `answer_correct = false`.
- **`gold_ok` is a data-quality signal, not an agent metric.** Rows failing it
  (9% of wiki2) have gold that does not recompute; treat their `agg_ok` with
  suspicion.

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
