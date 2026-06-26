# DRA Training

Inference and training pipeline for **Deep Research Agents (DRA)** — agentic
retrieval-augmented models that iterate *think → search → observe → report* over
a document index.

**Supported agents** (the LLM is picked automatically per agent):

| Family | Agents |
|---|---|
| Instruction-tuned (API) | `react`, `selfask`, `searcho1` (claude-sonnet-4-6) |
| RL-trained (vLLM) | `searchr1`, `research`, `stepsearch`, `drtulu`, `glm`, `oss_20b`, `oss_120b`, `tongyi`, `cpm_explore` |
| Outline / report (vLLM) | `webweaver`, `cpm_report` |

**Datasets:** `trqa` (Wikipedia / e-commerce), `neuclir` (news + technical, 2022–2024),
`browsecomp_plus`.

**Retrievers:** `bm25`, `spladepp`, `spladev3` (sparse); `bge`, `e5`, `dpr`,
`contriever`, `reasonir`, `qwen3_emb_{0.6b,4b,8b}` (dense).

## Installation

```bash
pip install -e .
```

This registers all packages so imports resolve from any working directory.
Set `DRA_DATA_ROOT` (corpus + indices) and `DRA_OUTPUT_ROOT` (run outputs) to
control where data is read/written.

## 1. Download datasets

```bash
# trqa  (subset: wiki1 | wiki2 | ecommerce)
python src/indexing_corpus_dataset/download_datasets.py trqa --subset wiki1

# neuclir  (subset: news | technical)
python src/indexing_corpus_dataset/download_datasets.py neuclir --year 2023 --subset news

# browsecomp_plus  (corpus only)
python src/indexing_corpus_dataset/download_datasets.py browsecomp_plus --skip-queries-qrels
```

Canonical layout written under `$DRA_DATA_ROOT/<dataset>/`:

```
queries/queries_{split}.jsonl     {"id","text","answer"?}
qrels/qrels_{split}.txt           TREC: qid 0 docid rel
corpus/{name}.jsonl               {"id","contents"}
```

## 2. Build index

```bash
python -m indexing_corpus_dataset.index_builder \
    --retriever qwen3_emb_4b --dataset browsecomp_plus \
    --use_fp16 --max_length 4096 --batch_size 16 --faiss_type Flat --save_embedding
```

On Slurm: `sbatch scripts/run_index_builder.sh` (sets per-retriever args
automatically). Build config defaults live in
`src/indexing_corpus_dataset/configs/index_build.yaml`.

**Test the build** — small smoke index over gold + distractor passages, then
checks entity recall:

```bash
python src/indexing_corpus_dataset/index_builder_test.py \
    --retriever bge --dataset neuclir --query-limit 20
```

## 3. Inference

Run via the wrapper script (edit `DATASET` / `RETRIEVER` / `AGENT` /
`CONTROLLER` at the top):

```bash
sbatch scripts/run_dra_inference.sh
```

Underlying command:

```bash
python experiments/dra_inference.py \
    --dataset browsecomp_plus \
    --retriever qwen3_emb_4b \
    --agentic-model glm \
    --controller action \
    --controller-prompt-variant nov_cov_sim \
    --num-gpus 0
```

Quick checks:

```bash
# single-query smoke test
python experiments/dra_inference.py --dataset browsecomp_plus --limit 1 --num-gpus 0
# evaluate already-saved runs, no generation
python experiments/dra_inference.py --dataset browsecomp_plus --eval-only --num-gpus 0
```

Run defaults (top_k, rerankers, controller LLM, eval k-values, …) are in
`experiments/configs/default.yaml`.

### Output format

```
$DRA_OUTPUT_ROOT/{dataset}_{split}_{query_key}_{retriever}/{agent}_agent_{model}/{searcher_config}/
├── retrieval/{qid}.trec            per-query TREC (all iterations, col 6 = iter_N)
├── generation/{qid}.md             per-query report (markdown)
├── trajectory/{qid}.json           {qid, question, trajectory}
├── controller/{qid}.json           {qid, per_iteration: [...]}
├── cited_docs_retrieval/{qid}.trec docs cited by the LLM
├── seen_docs_retrieval/{qid}.trec  docs shown to the LLM
├── ranking_results.trec            aggregated retrieval
└── summary.json                    metrics
```

## Layout

Dependency direction is one-way: `experiments → pipeline → src → utils`.

```
utils/         Leaf helpers (config, io, llm_client, vllm_manager).
pipeline/      Orchestration, multi-GPU workers, evaluation/fusion.
src/           Components: deep_research_agents, searcher_component,
               reasoner_component, controller_component, evaluation,
               indexing_corpus_dataset.
experiments/   CLI entry points (dra_inference.py, dra_train.py).
```
