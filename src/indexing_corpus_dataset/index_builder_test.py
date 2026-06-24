#!/usr/bin/env python3
"""
Index Builder Test Script

Tests that the indexing pipeline works correctly by:
1. Creating a small corpus from a subset of queries, their qrels, and gold passages
2. Building an index on that small corpus
3. Running retrieval using local retrieval models
4. Evaluating entity recall

High recall indicates the indexing pipeline is working correctly (gold passages are in the corpus
and should be retrievable when the index is built properly).

Dataset loading follows run_sequential_pipeline.py conventions:
  - Queries and qrels are loaded from data/{dataset}/ via the dataset module.
  - Corpus is read from DATA_ROOT (auto-selected per dataset/subset).

Index creation follows index_builder.py conventions:
  - Index_Builder is constructed with all parameters including device_id.

Inputs: Same as index_builder.py plus:
  - --dataset / --dataset-year / --subset: identify the dataset split
  - --query-key: key in query JSONL records to use as query text
  - --min-relevance-score: minimum qrel score to treat as relevant
  - --query_limit: Optional limit on number of queries to use (default: 10)
  - --sub_corpus_max_size: Total sub corpus size (gold passages + distractors). If set, e.g. 1000,
    the sub corpus has 1000 passages total: all gold passages for the queries plus non-gold
    distractors to reach 1000. If not set, sub corpus = gold passages only.

Usage:
  python src/agentic_retrieval_research/indexing/index_builder_test.py \\
    --retrieval_method bge \\
    --dataset neuclir \\
    --dataset-year 2023 \\
    --subset news \\
    --query_limit 20
"""

import os
import json
import argparse
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent

from tqdm import tqdm

from indexing_corpus_dataset.index_builder import Index_Builder, MODEL2PATH, MODEL2POOLING
from indexing_corpus_dataset.layout import DATA_ROOT, corpus_name, trqa_corpus_name
from indexing_corpus_dataset.dataset_loaders import load_qrels, load_queries, resolve_split_id, resolve_data_path


def evaluate_entity_retrieval(retrieved_ids: list, gold_ids: list, k_values: list) -> dict:
    """
    Evaluate entity recall metrics for a single query.

    Args:
        retrieved_ids: List of retrieved passage IDs (in rank order)
        gold_ids: List of gold (relevant) passage IDs
        k_values: List of k values to compute recall@k for

    Returns:
        Dictionary with entity_recall (overall) and entity_recall@k for each k
    """
    gold_ids_set = set(str(gid) for gid in gold_ids)
    retrieved_ids_str = [str(rid) for rid in retrieved_ids]

    # Compute recall@k for each k
    metrics = {}
    for k in k_values:
        retrieved_at_k = set(retrieved_ids_str[:k])
        if gold_ids_set:
            recall_at_k = len(retrieved_at_k & gold_ids_set) / len(gold_ids_set)
        else:
            recall_at_k = 0.0
        metrics[f'entity_recall@{k}'] = recall_at_k

    # Overall entity recall (all retrieved documents)
    retrieved_all = set(retrieved_ids_str)
    if gold_ids_set:
        entity_recall = len(retrieved_all & gold_ids_set) / len(gold_ids_set)
    else:
        entity_recall = 0.0
    metrics['entity_recall'] = entity_recall

    return metrics


def get_gold_passage_ids_for_queries(qrels: dict, query_ids: set) -> set:
    """Get all unique gold passage IDs for the given query IDs."""
    gold_ids = set()
    for qid in tqdm(query_ids, desc="Collecting gold passage IDs", unit="query"):
        gold_ids.update(qrels.get(qid, []))
    return gold_ids


def create_small_corpus(full_corpus_path: str, gold_passage_ids: set, output_path: str, max_size: int = None):
    """
    Create sub corpus: gold passages plus distractors up to max_size.
    - Gold passages: all relevant passages for the query subset (from qrels).
    - If max_size is set: add non-gold passages (distractors) until total length = max_size.
    - If max_size is None: sub corpus = only gold passages (no limit).
    """
    gold_ids = frozenset(gold_passage_ids)
    found_gold = set()
    written = 0

    with open(full_corpus_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        # Phase 1: write all gold passages
        for line in tqdm(infile, desc="Phase 1: extracting gold passages", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = doc.get('id', doc.get('doc_id', doc.get('passage_id')))
            if doc_id is not None and str(doc_id) in gold_ids:
                outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
                written += 1
                found_gold.add(str(doc_id))

        # Phase 2: if max_size set, add distractors (non-gold) until we reach max_size
        if max_size is not None and written < max_size:
            infile.seek(0)
            for line in tqdm(infile, desc="Phase 2: adding distractors", unit="line"):
                if written >= max_size:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = doc.get('id', doc.get('doc_id', doc.get('passage_id')))
                if doc_id is not None and str(doc_id) not in gold_ids:
                    outfile.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    written += 1

    missing = gold_ids - found_gold
    if missing:
        print(f"Warning: {len(missing)} gold passage IDs not found in corpus (e.g. {list(missing)[:5]})")
    n_gold = len(found_gold)
    n_distractors = written - n_gold
    if max_size is not None:
        print(f"Small corpus: {written} passages total ({n_gold} gold, {n_distractors} distractors) -> {output_path}")
    else:
        print(f"Small corpus: {written} passages (gold only) -> {output_path}")
    return written


def create_qrels_subset(qrels: dict, query_ids: set, output_path: str, qrel_file_format: str = "trec"):
    """Write qrels filtered to the query subset (TREC format)."""
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in sorted(query_ids):
            for passage_id in qrels.get(qid, []):
                # TREC format: query_id 0 passage_id relevance
                f.write(f"{qid} 0 {passage_id} 1\n")
                count += 1
    print(f"Qrels subset: {count} rows for {len(query_ids)} queries -> {output_path}")
    return count


def _get_passage_id(doc):
    """Extract passage ID from retrieved doc."""
    for key in ('id', 'doc_id', 'passage_id', 'docid'):
        val = doc.get(key)
        if val is not None:
            return str(val)
    if 'title' in doc:
        return str(doc['title']).strip()
    raise KeyError("Retrieved doc has no id field. Keys: %s" % list(doc.keys()))


def run_retrieval_and_evaluate(retriever, queries: list, qrels: dict, k_values: list):
    """Run retrieval for each query and compute entity recall metrics."""
    results = []
    for qid, query_text in tqdm(queries, desc="Retrieval", unit="query"):
        gold_ids = qrels.get(qid, [])
        try:
            docs, scores = retriever.search(query_text, return_score=True)
        except Exception as e:
            print(f"Retrieval error for qid={qid}: {e}")
            docs, scores = [], []
        retrieved_ids = [_get_passage_id(d) for d in docs]
        metrics = evaluate_entity_retrieval(retrieved_ids, gold_ids, k_values)
        metrics['qid'] = qid
        results.append(metrics)
    return results


def _str2bool(v):
    """Accept 'true'/'false' strings (SageMaker passes all hyperparams as strings)."""
    if isinstance(v, bool):
        return v
    return v.strip().lower() in ('true', '1', 'yes')


def _parse_args():
    """Build and parse the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Index builder test: small corpus -> index -> retrieval -> recall")

    # Index builder args (same as index_builder.py).
    # NOTE: use hyphens in option strings (argparse standard); dest uses underscores.
    parser.add_argument('--retrieval-method', type=str, default='e5', dest='retrieval_method', choices=['bm25', 'spladepp', 'spladev3', 'contriever', 'dpr', 'e5', 'bge', 'qwen3_emb_0.6b', 'qwen3_emb_4b', 'qwen3_emb_8b', 'rerank_l6', 'rerank_l12'])
    parser.add_argument('--corpus-path', type=str, default=None, dest='corpus_path', help='Path to FULL corpus JSONL (auto-selected from DATA_ROOT if not set)')
    parser.add_argument('--save-dir', type=str, default=None, dest='save_dir', help='Directory for test index and outputs (default: temp dir)')
    parser.add_argument('--max-length', type=int, default=512, dest='max_length')
    parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
    parser.add_argument('--faiss-type', type=str, default='Flat', dest='faiss_type')
    parser.add_argument('--embedding-path', type=str, default=None, dest='embedding_path')
    # Boolean flags accept "true"/"false" strings so SageMaker hyperparameters work.
    parser.add_argument('--save-embedding', type=_str2bool, default=True, dest='save_embedding')
    parser.add_argument('--use-fp16', type=_str2bool, default=True, dest='use_fp16')
    parser.add_argument('--faiss-gpu', type=_str2bool, default=False, dest='faiss_gpu')
    parser.add_argument('--device-id', type=int, default=None, dest='device_id', help='GPU device ID for CUDA systems (e.g., 0, 1). Ignored for MPS.')

    # Dataset args (following run_sequential_pipeline.py conventions)
    parser.add_argument('--dataset', type=str, default='trqa', choices=['trqa', 'neuclir', 'browsecomp_plus'], help='Dataset name')
    # Accept both --dataset-year and --data-set (submit_index_builder_test.py uses the latter).
    parser.add_argument('--dataset-year', '--data-set', type=str, default=None, dest='dataset_year', help='Dataset year (e.g. 2023, 2024); for trqa, the eval split (test/validation). Auto-selected if omitted.')
    parser.add_argument('--subset', type=str, default=None, help='Dataset subset (news/technical for neuclir; wiki1/wiki2/ecommerce for trqa). Auto-selected if omitted.')
    parser.add_argument('--query-key', type=str, default=None, dest='query_key', help='Key in query JSONL records to use as query text. Auto-selected if omitted.')
    parser.add_argument('--min-relevance-score', type=int, default=None, dest='min_relevance_score', help='Minimum relevance score to treat as relevant (default: 3 for neuclir)')
    parser.add_argument('--query-limit', type=int, default=10, dest='query_limit', help='Max number of queries to use')
    parser.add_argument('--sub-corpus-max-size', type=int, default=3000, dest='sub_corpus_max_size', help='Total sub corpus size: gold passages + distractors up to this many (default: gold only, no limit)')
    parser.add_argument('--data-path', type=str, default=None, dest='data_path', help='Root directory for queries/qrels (local or s3://). When set, overrides the default local data/<dataset>/ lookup. load_queries and load_qrels support S3 paths directly.')

    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Auto-select dataset args ──
    if args.dataset_year is None and args.dataset == "neuclir":
        args.dataset_year = "2024"
        print(f"Auto-selected dataset year: {args.dataset_year}")

    # TRQA carries the evaluation split (test/validation) in dataset_year.
    if args.dataset_year is None and args.dataset == "trqa":
        args.dataset_year = "test"
        print(f"Auto-selected eval split: {args.dataset_year}")

    if args.subset is None:
        if args.dataset == "neuclir":
            args.subset = "news"
        elif args.dataset == "trqa":
            args.subset = "wiki1"
        if args.subset is not None:
            print(f"Auto-selected subset: {args.subset}")

    if args.query_key is None:
        # Downloaders flatten each dataset's best query text into the "text"
        # field (NeuCLIR's topic_* fields are collapsed at download time), so
        # "text" is the correct key for every dataset.
        args.query_key = "text"
        print(f"Auto-selected query key: {args.query_key}")

    # Auto-adjust max_length for browsecomp_plus + qwen3_emb
    if args.dataset == "browsecomp_plus" and args.retrieval_method.startswith("qwen3_emb"):
        args.max_length = 4096
        print(f"Auto-adjusted max_length to {args.max_length} for browsecomp_plus + {args.retrieval_method}")

    if args.min_relevance_score is None:
        if args.dataset == "neuclir":
            args.min_relevance_score = 3
        elif args.dataset == "browsecomp_plus":
            args.min_relevance_score = 1  # gold=2, evidence=1, skip hard negatives=0
        elif args.dataset == "trqa":
            args.min_relevance_score = 1  # binary qrels

    # ── Resolve file_data_set ──
    file_data_set = resolve_split_id(args.dataset, args.dataset_year, args.subset)

    # ── Auto-select corpus_path (following setup_public_dataset_paths) ──
    if args.corpus_path is None:
        _ir_root = DATA_ROOT
        if args.dataset == "neuclir":
            _corpus_file = f"{corpus_name(args.subset)}.jsonl"
            args.corpus_path = str(_ir_root / "neuclir" / "corpus" / _corpus_file)
        elif args.dataset == "trqa":
            _corpus_file = f"{trqa_corpus_name(args.subset)}.jsonl"
            args.corpus_path = str(_ir_root / "trqa" / "corpus" / _corpus_file)
        elif args.dataset == "browsecomp_plus":
            args.corpus_path = str(_ir_root / "browsecomp_plus" / "corpus" / "corpus.jsonl")
        else:
            print(f"Warning: corpus_path not auto-selected for dataset '{args.dataset}'. Please set --corpus_path.")
            return 1
        print(f"Auto-selected corpus path: {args.corpus_path}")

    if args.save_dir is None:
        args.save_dir = tempfile.mkdtemp(prefix='index_builder_test_')
        print(f"Using temp save_dir: {args.save_dir}")

    # Resolve model path
    args.model_path = MODEL2PATH.get(args.retrieval_method, '')
    pooling_method = MODEL2POOLING.get(args.retrieval_method)

    print("=" * 70)
    print("INDEX BUILDER TEST")
    print("=" * 70)
    print(f"Dataset:       {args.dataset} / {file_data_set}")
    print(f"Corpus path:   {args.corpus_path}")
    print(f"Save dir:      {args.save_dir}")
    print(f"Retriever:     {args.retrieval_method}")

    # ── 1. Load queries and qrels ──
    print("\n=== 1. Loading queries and qrels ===")
    if args.data_path:
        data_path = args.data_path  # S3 path or explicit local path
    else:
        # /dev/shm → NVME → repo-local data/{dataset}/ (shared resolver).
        data_path = resolve_data_path(args.dataset, project_root=_project_root)

    print(f"Loading dataset from {data_path}, split: {file_data_set}")
    queries_dict = load_queries(str(data_path), file_data_set, query_key=args.query_key)
    qrels_nested = load_qrels(str(data_path), file_data_set, min_relevance_score=args.min_relevance_score)

    if args.min_relevance_score is not None:
        print(f"Loaded {len(queries_dict)} queries and {len(qrels_nested)} qrels (min relevance >= {args.min_relevance_score})")
    else:
        print(f"Loaded {len(queries_dict)} queries and {len(qrels_nested)} qrels")

    # Convert qrels to simple format: {qid: [docid1, docid2, ...]}
    qrels = {qid: list(doc_rels.keys()) for qid, doc_rels in qrels_nested.items()}

    # Filter queries to those with qrels, then apply limit
    queries_with_qrels = [(qid, qtext) for qid, qtext in queries_dict.items() if qid in qrels]
    if not queries_with_qrels:
        print("Error: No queries have qrels. Check dataset/year/subset and data_path.")
        return 1
    queries = queries_with_qrels[:args.query_limit]
    print(f"Queries: {len(queries)} (limited to {args.query_limit}, with qrels)")

    # 2. Get gold passage IDs and create small corpus (or use existing)
    os.makedirs(args.save_dir, exist_ok=True)
    small_corpus_path = os.path.join(args.save_dir, "small_corpus.jsonl")
    if os.path.exists(small_corpus_path):
        n_lines = sum(1 for _ in open(small_corpus_path, 'r', encoding='utf-8'))
        print(f"\n=== 2. Small corpus (using existing) ===")
        print(f"Small corpus exists: {small_corpus_path} ({n_lines} passages)")
    else:
        print("\n=== 2. Creating small corpus (gold passages + distractors) ===")
        gold_ids = get_gold_passage_ids_for_queries(qrels, {qid for qid, _ in queries})
        if not gold_ids:
            print("Error: No gold passage IDs found")
            return 1
        # Expand corpus_path glob if needed (when a *.jsonl pattern is passed)
        import glob
        corpus_files = sorted(glob.glob(args.corpus_path)) if '*' in args.corpus_path else [args.corpus_path]
        if not corpus_files:
            print(f"Error: No corpus files found matching: {args.corpus_path}")
            return 1
        if len(corpus_files) == 1:
            create_small_corpus(corpus_files[0], gold_ids, small_corpus_path, max_size=args.sub_corpus_max_size)
        else:
            # Multi-file corpus: stream through all files
            import tempfile as _tf
            _tmp = _tf.NamedTemporaryFile(delete=False, suffix='.jsonl')
            _tmp_path = _tmp.name
            _tmp.close()
            with open(_tmp_path, 'w', encoding='utf-8') as _out:
                for _f in corpus_files:
                    with open(_f, 'r', encoding='utf-8') as _in:
                        for _line in _in:
                            _out.write(_line)
            create_small_corpus(_tmp_path, gold_ids, small_corpus_path, max_size=args.sub_corpus_max_size)
            os.unlink(_tmp_path)

    # 3. Build index on small corpus (following index_builder.py main())
    print("\n=== 3. Building index ===")
    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=small_corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        device_id=args.device_id,
    )
    index_builder.build_index()

    # 4. Initialize retriever and run retrieval
    print("\n=== 4. Running retrieval ===")
    from searcher_component.retriever import BM25Retriever, DenseRetriever, SPLADERetriever

    # Build minimal config object for retriever
    class RetrieverConfig:
        pass
    config = RetrieverConfig()
    config.retriever_name = args.retrieval_method
    config.corpus_path = small_corpus_path
    config.index_dir = args.save_dir
    config.retrieval_topk = 1000  # retrieve enough for recall@k
    config.retrieval_batch_size = 32
    if args.dataset == "browsecomp_plus" and args.retrieval_method.startswith("qwen3_emb"):
        config.retrieval_query_max_length = 8196
    else:
        config.retrieval_query_max_length = 512
    config.retrieval_use_fp16 = args.use_fp16
    config.bm25_k1 = 0.9
    config.bm25_b = 0.4
    config.faiss_gpu = args.faiss_gpu
    config.device = __import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    config.splade_max_length = args.max_length  # Match index build for SPLADE

    if args.retrieval_method == 'bm25':
        retriever = BM25Retriever(config)
    elif args.retrieval_method in ('spladepp', 'spladev3'):
        retriever = SPLADERetriever(config)
    elif args.retrieval_method in ['contriever', 'dpr', 'e5', 'bge'] or args.retrieval_method.startswith('qwen3_emb'):
        retriever = DenseRetriever(config)
    else:
        print(f"Error: Unsupported retriever for test: {args.retrieval_method}")
        return 1

    k_values = [1, 3, 10, 100]
    results = run_retrieval_and_evaluate(retriever, queries, qrels, k_values)

    # 5. Aggregate and report recall
    print("\n=== 5. Recall evaluation ===")
    n = len(results)
    if n == 0:
        print("No results to aggregate")
        return 1

    recall_all = sum(r['entity_recall'] for r in results) / n
    recall_by_k = {k: sum(r[f'entity_recall@{k}'] for r in results) / n for k in k_values}

    print(f"\nEntity Recall (index builder test) over {n} queries:")
    for k in k_values:
        print(f"  Entity Recall@{k:3d}: {recall_by_k[k]:.4f} ({recall_by_k[k]*100:.2f}%)")
    print(f"  Entity Recall (all): {recall_all:.4f} ({recall_all*100:.2f}%)")

    # Save metrics
    metrics_path = os.path.join(args.save_dir, "index_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'n': n,
            'dataset': args.dataset,
            'file_data_set': file_data_set,
            'retrieval_method': args.retrieval_method,
            'entity_recall': recall_all,
            'entity_recall_by_k': recall_by_k,
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Interpretation
    print("\n" + "=" * 70)
    if recall_all >= 0.9:
        print("PASS: High recall indicates the indexing pipeline is working correctly.")
    elif recall_all >= 0.5:
        print("PARTIAL: Moderate recall. Check index build parameters and model compatibility.")
    else:
        print("LOW RECALL: Index may have issues. Verify corpus format, model, and index/query encoder alignment.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# python src/agentic_retrieval_research/indexing/index_builder_test.py
# python src/agentic_retrieval_research/indexing/index_builder_test.py --dataset neuclir --dataset-year 2023 --subset news --retrieval_method bge --query_limit 20
# python src/agentic_retrieval_research/indexing/index_builder_test.py --dataset browsecomp_plus --retrieval_method bge --query_limit 20
