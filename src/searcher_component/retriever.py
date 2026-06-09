# ===========================
# === Src: https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
# ===========================
import os

# Disable safetensors auto-conversion (avoids background thread errors when
# HuggingFace Hub is unreachable or returns empty responses).
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import json
import faiss
import torch
import warnings
import datasets
import numpy as np
from tqdm import tqdm
from typing import List
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# Import S3 utilities
from agentic_retrieval_research.utils.s3_utils import is_s3_path, get_s3_fs


MODEL2PATH = {
    "contriever": "facebook/contriever-msmarco",
    "dpr": "facebook/dpr-question_encoder-single-nq-base",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "reasonir": 'reasonir/ReasonIR-8B',
    "spladepp": "naver/splade-cocondenser-ensembledistil",
    "spladev3": "naver/splade-v3",
    "qwen3_emb": "Qwen/Qwen3-Embedding-4B",
    "agentir_4b": "Tevatron/AgentIR-4B",
}

QWEN3_EMB_SIZES = {
    "0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "4B": "Qwen/Qwen3-Embedding-4B",
    "8B": "Qwen/Qwen3-Embedding-8B",
}

def get_qwen3_emb_path(size: str = "4B") -> str:
    """Return the HuggingFace model path for a Qwen3-Embedding size variant."""
    if size not in QWEN3_EMB_SIZES:
        raise ValueError(f"Invalid Qwen3 embedding size '{size}'. Choose from: {list(QWEN3_EMB_SIZES.keys())}")
    return QWEN3_EMB_SIZES[size]

MODEL2POOLING = {
    "contriever": "mean",
    "dpr": "pooler",
    "e5": "mean",
    "bge": "cls",
    "reasonir": 'mean',
    "spladepp": None,
    "spladev3": None,
    "qwen3_emb": "last_token",
    "agentir_4b": "last_token",
}

def get_device(device_id=None):
    """Get the best available device (CUDA, MPS, or CPU).

    Args:
        device_id: Optional GPU device ID for CUDA systems (e.g., 0, 1, 2).
                   Ignored for MPS (Apple Silicon uses all cores automatically).
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        # Always use an explicit ordinal so HuggingFace device_map resolves
        # to the correct GPU (important when CUDA_VISIBLE_DEVICES didn't
        # take effect in spawn'd worker processes).
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    elif torch.backends.mps.is_available():
        # MPS automatically uses all GPU cores, device_id is ignored
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(retrieval_method, model_path: str, use_fp16: bool = False, device=None):
    if device is None:
        device = get_device()

    # Use torch_dtype and device_map to load directly onto the target device.
    # This avoids the "Cannot copy out of meta tensor" error that occurs when
    # accelerate is installed and from_pretrained() initialises weights on the
    # meta device before model.to(device) is called.
    torch_dtype = torch.float16 if (use_fp16 and str(device) != "cpu" and
                                     not str(device).startswith("mps")) else torch.float32
    device_map = {"": str(device)}

    if retrieval_method == 'dpr':
        tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path)
        model = DPRQuestionEncoder.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch_dtype, device_map=device_map
        )

    # Qwen3 embedding models and AgentIR require left padding for last-token pooling
    if "qwen3" in retrieval_method or "agentir" in retrieval_method:
        tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
        # Qwen3-style: use hidden state at the last real (EOS) token position.
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_corpus(corpus_path: str):
    """Load corpus from local filesystem or S3.

    For local paths, checks for a pre-built Arrow cache in the same directory
    as the corpus file (``<corpus_dir>/arrow_cache``). If found, loads from
    there via memory-mapping (near-instant, regardless of corpus size).
    To build the cache once, run::

        python -c "
        import datasets
        corpus = datasets.load_dataset(
            'json',
            data_files='<corpus_path>',
            split='train'
        )
        corpus.save_to_disk('<corpus_dir>/arrow_cache')
        "

        TMPDIR=/mnt/sagemaker-nvme/tmp \
        HF_DATASETS_CACHE=/mnt/sagemaker-nvme/hf_cache \
        python -c "
        import datasets
        print('Loading JSONL (this takes a while, one-time cost)...')
        corpus = datasets.load_dataset(
            'json',
            data_files='/mnt/sagemaker-nvme/ir_datasets/trec_rag/corpus/msmarco_v2.1_doc_segmented_pyserini_format.jsonl',
            split='train'
        )
        print(f'Loaded {len(corpus)} docs. Saving Arrow cache...')
        corpus.save_to_disk('/mnt/sagemaker-nvme/ir_datasets/trec_rag/corpus/arrow_cache')
        print('Done. Future loads will use the cache.')
        "

    Args:
        corpus_path: Path to corpus file (can be local or S3 path like 's3://bucket/key')

    Returns:
        HuggingFace Dataset object
    """
    if is_s3_path(corpus_path):
        # For S3 paths, use s3fs with datasets
        # The datasets library can work with s3fs for remote files
        import s3fs
        fs = get_s3_fs()

        # datasets.load_dataset can work with s3fs by passing a file-like object
        # or by using the path directly if s3fs is configured
        corpus = datasets.load_dataset(
            'json',
            data_files=corpus_path,
            split="train",
            storage_options={'anon': False}  # Use AWS credentials
        )
    else:
        import os
        # Check for pre-built Arrow cache next to the corpus file.
        # Supports two naming conventions:
        #   1. arrow_cache_<stem>  (per-file, e.g. arrow_cache_corpus_en_news)
        #   2. arrow_cache         (legacy single-corpus layout)
        corpus_dir = os.path.dirname(corpus_path)
        stem = os.path.splitext(os.path.basename(corpus_path))[0]
        arrow_cache_per_file = os.path.join(corpus_dir, f"arrow_cache_{stem}")
        arrow_cache_legacy = os.path.join(corpus_dir, "arrow_cache")
        if os.path.isdir(arrow_cache_per_file):
            arrow_cache = arrow_cache_per_file
        elif os.path.isdir(arrow_cache_legacy):
            arrow_cache = arrow_cache_legacy
        else:
            arrow_cache = None

        if arrow_cache:
            print(f"Loading corpus from Arrow cache (memory-mapped): {arrow_cache}")
            corpus = datasets.load_from_disk(arrow_cache)
        else:
            suggested = arrow_cache_per_file
            print(f"Loading corpus from JSONL (slow for large corpora). "
                  f"Run save_to_disk('{suggested}') once to build a fast Arrow cache.")
            corpus = datasets.load_dataset(
                'json',
                data_files=corpus_path,
                split="train"
            )
    return corpus

def load_or_build_id2idx(corpus, corpus_path, desc="Building id2idx mapping"):
    """Load or build mapping from document ID to corpus position.

    Caches the mapping in the same directory as the corpus for faster loading.

    Args:
        corpus: HuggingFace dataset with 'id' field
        corpus_path: Path to corpus file (used to determine cache location)
        desc: Description for progress bar

    Returns:
        Dict mapping document ID (str) to corpus position (int)
    """
    import os
    import pickle

    # Determine cache file path
    if is_s3_path(corpus_path):
        # For S3 paths, use a local cache directory
        cache_dir = os.path.expanduser("~/.cache/agentic_retrieval")
        os.makedirs(cache_dir, exist_ok=True)
        # Create a filename based on the S3 path
        cache_filename = corpus_path.replace("s3://", "").replace("/", "_") + "_id2idx.pkl"
        cache_path = os.path.join(cache_dir, cache_filename)
    else:
        # For local paths, put cache next to corpus file
        corpus_dir = os.path.dirname(corpus_path)
        cache_path = os.path.join(corpus_dir, "id2idx_cache.pkl")

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading id2idx mapping from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), rebuilding...")

    # Build the mapping
    print(f"{desc}: building mapping from document IDs to positions...")
    id2idx = {}

    # Check if corpus has an 'id' field
    if hasattr(corpus, 'column_names') and 'id' in corpus.column_names:
        id_field = 'id'
    elif hasattr(corpus, 'column_names') and 'doc_id' in corpus.column_names:
        id_field = 'doc_id'
    elif hasattr(corpus, 'column_names') and 'passage_id' in corpus.column_names:
        id_field = 'passage_id'
    else:
        # Fallback: use position as ID
        print("Warning: No 'id' field found in corpus. Using positions as IDs.")
        return {str(i): i for i in range(len(corpus))}

    for idx in tqdm(range(len(corpus)), desc=desc):
        doc = corpus[idx]
        doc_id = doc.get(id_field)
        if doc_id is not None:
            id2idx[str(doc_id)] = idx

    # Save to cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(id2idx, f)
        print(f"Saved id2idx mapping to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache ({e})")

    return id2idx

# def load_docs(corpus, doc_idxs):
#     results = [corpus[str(idx)] for idx in doc_idxs]
#     # results = [corpus[idx] for idx in doc_idxs]
#     return results

def load_docs(corpus, doc_idxs, id2idx=None):
    """Load documents from corpus by indices or IDs.

    Args:
        corpus: The corpus dataset
        doc_idxs: List of document indices or IDs
        id2idx: Optional mapping from document IDs (strings) to corpus positions (ints)

    Returns:
        List of documents
    """
    # If id2idx is provided, convert string IDs to integer positions
    if id2idx is not None:
        doc_idxs = [id2idx.get(str(idx), id2idx.get(idx, idx)) for idx in doc_idxs]

    # doc_idxs can be a NumPy array; make sure it's a plain list of ints
    doc_idxs = [int(i) if isinstance(i, (int, np.integer)) else i for i in doc_idxs]

    # pandas DataFrame: select rows by position
    if hasattr(corpus, "iloc"):
        # returns a DataFrame of the selected rows
        return corpus.iloc[doc_idxs]

    # HuggingFace Dataset: use .select() for efficient batch access
    if hasattr(corpus, "select"):
        try:
            selected = corpus.select(doc_idxs)
            # Convert to list of dicts for consistent format
            return [dict(item) for item in selected]
        except Exception as e:
            print(f"Warning: Error using dataset.select(): {e}. Falling back to individual access.")
            return [dict(corpus[int(i)]) for i in doc_idxs]

    # list-like
    if isinstance(corpus, list):
        return [corpus[i] for i in doc_idxs]

    # dict keyed by integer ids
    if isinstance(corpus, dict):
        # only use this if your keys are ints that align with positions
        return [corpus[i] for i in doc_idxs]

    # Fallback: try positional access
    return [corpus[int(i)] for i in doc_idxs]

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device=None):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device if device is not None else get_device()

        self.model, self.tokenizer = load_model(
            retrieval_method=self.model_name,
            model_path = self.model_path,
            use_fp16 = self.use_fp16,
            device = self.device
        )
        self.model.eval()
        print(f"Encoder loaded on device: {self.device}")

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        name = self.model_name.lower()

        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in name:
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]
        if "bge" in name:
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]
        if "agentir" in name:
            if is_query:
                task = "Given a user's reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user's reasoning"
                query_list = [f"Instruct: {task}\nQuery:{query}" for query in query_list]
            # Documents: no prefix
        elif "qwen3" in name:
            if is_query:
                task = "Given a web search query, retrieve relevant passages that answer the query"
                query_list = [f"Instruct: {task}\nQuery:{query}" for query in query_list]
            # Documents: no prefix
        # DPR/Contriever: no prefixes

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ----- forward + pooling -----
        model_cls = type(self.model).__name__

        if "t5" in model_cls.lower():
            # T5-based retrieval model: take first token of decoder output
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        
        elif "reasonir" in model_cls.lower() or "reasonir" in name:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                None,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        
        elif "dpr" in name or "dpr" in model_cls.lower(): 
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                None,
                None,
                self.pooling_method
            )
            # do NOT normalize for DPR   
        
        elif "contriever" in name or "contriever" in model_cls.lower():
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                None,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        elif "qwen3" in name or "agentir" in name:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                None,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retriever_name = config.retriever_name
        self.corpus_path = config.corpus_path
        self.topk = config.retrieval_topk
        self._docid_to_doc = None
        corpus_path = getattr(config, 'corpus_path', None)
        if corpus_path and not corpus_path.endswith("*.jsonl"):
            from pathlib import Path as _P
            corpus_stem = _P(corpus_path).stem  # e.g. "corpus_en_news"
        else:
            corpus_stem = None

        if config.retriever_name in ['bm25', 'rerank_l6', 'rerank_l12']:
            suffix = f"{corpus_stem}_bm25_index" if corpus_stem else "bm25_index"
            self.index_path = f"{config.index_dir}/{suffix}"
            self.pooling_method = None
            self.retrieval_model_path = "cross-encoder/ms-marco-MiniLM-L-6-v2" # "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L12-v2"
        elif config.retriever_name in ('spladepp', 'spladev3'):
            suffix = f"{corpus_stem}_{config.retriever_name}_index" if corpus_stem else f"{config.retriever_name}_index"
            self.index_path = f"{config.index_dir}/{suffix}"
            self.pooling_method = None
            self.retrieval_model_path = MODEL2PATH[config.retriever_name]
        else:
            # Use size-qualified name for qwen3_emb (e.g. "qwen3_4B_emb")
            qwen3_size = getattr(config, 'qwen3_size', None)
            file_method = config.retriever_name
            if config.retriever_name == 'qwen3_emb' and qwen3_size:
                file_method = f"qwen3_emb_{qwen3_size}"
            if corpus_stem:
                self.index_path = f"{config.index_dir}/{corpus_stem}_{file_method}_Flat.index"
            else:
                self.index_path = f"{config.index_dir}/{file_method}_Flat.index"
            self.retrieval_model_path = MODEL2PATH[config.retriever_name]
            self.pooling_method = MODEL2POOLING[config.retriever_name]

        # Override Qwen3 embedding model path if a size variant is specified
        qwen3_size = getattr(config, 'qwen3_size', None)
        if qwen3_size and config.retriever_name == 'qwen3_emb':
            self.retrieval_model_path = get_qwen3_emb_path(qwen3_size)

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

    def retrieve(self, query: str, num: int = None, return_score: bool = False):
        """Alias for search method to unify interface with endpoint retrievers."""
        return self.search(query, num, return_score)

    def batch_retrieve(self, query_list: List[str], num: int = None, return_score: bool = False):
        """Alias for batch_search method to unify interface with endpoint retrievers."""
        return self.batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            results = []
            for hit in hits:
                raw = json.loads(self.searcher.doc(hit.docid).raw())
                content = raw.get('contents', '')
                results.append({
                    'id': raw.get('id', raw.get('_id', hit.docid)),
                    'title': content.split("\n")[0].strip("\"") if content else '',
                    'text': "\n".join(content.split("\n")[1:]) if content else '',
                    'contents': content
                })
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        # Attach retriever score and rank to each doc
        for rank, (doc, score) in enumerate(zip(results, scores), 1):
            if isinstance(doc, dict):
                doc["score"] = score
                doc["rank"] = rank

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class _SPLADEQueryEncoderForImpact:
    """Query encoder for LuceneImpactSearcher: encode(query) -> Dict[str, int].

    Weights are raw ``log(1 + ReLU(logits))`` values truncated to int, matching
    the float weights written by ``index_builder._process_splade_batch`` (Pyserini
    truncates floats to ints when building the impact index).
    """

    def __init__(self, model_path: str, device, max_length: int = 256, use_fp16: bool = False):
        self.device = device
        self.max_length = max_length
        print(f"Loading SPLADE model for impact queries: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        self.model = self.model.to(device)
        if use_fp16:
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, query: str):
        """Return Dict[str, int] for LuceneImpactSearcher (token -> int weight)."""
        inputs = self.tokenizer(
            query,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        mlm_logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(mlm_logits))
        mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand_as(relu_log)
        relu_log = relu_log * mask_expanded
        logits = torch.max(relu_log, dim=1)[0]  # [1, vocab_size]
        non_zero_indices = torch.nonzero(logits[0] > 0, as_tuple=True)[0]
        if len(non_zero_indices) == 0:
            return {}
        tokens = self.tokenizer.convert_ids_to_tokens(non_zero_indices.cpu().tolist())
        weights = logits[0][non_zero_indices].cpu().tolist()
        # Truncate to int — same scale as index builder (Pyserini truncates floats
        # to ints when building the impact index, so we do the same here).
        out = {}
        for token, w in zip(tokens, weights):
            q = int(w)
            if q >= 1:
                out[token] = q
        return out

class SPLADERetriever(BaseRetriever):
    """SPLADE retriever using Pyserini impact index with LuceneImpactSearcher."""
    def __init__(self, config):
        super().__init__(config)
        device = getattr(config, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        splade_max_length = getattr(config, 'splade_max_length', 256)
        use_fp16 = getattr(config, 'retrieval_use_fp16', False)
        query_encoder = _SPLADEQueryEncoderForImpact(
            self.retrieval_model_path, device, max_length=splade_max_length, use_fp16=use_fp16
        )
        # Impact index must be searched with LuceneImpactSearcher, not LuceneSearcher (BM25)
        self.searcher = LuceneImpactSearcher(
            self.index_path, query_encoder=query_encoder, min_idf=0, encoder_type='pytorch'
        )
        # SPLADE impact index stores docs with empty 'contents'; LuceneImpactSearcher returns
        # external docids (strings). Pyserini's doc() expects internal Lucene docid, so we
        # always resolve hits via corpus + id2idx instead of reading from index.
        self.contain_doc = False
        self._corpus_id2idx = None
        self.corpus = load_corpus(self.corpus_path)
        self._build_corpus_id2idx()
        self.max_process_num = 8

    def _build_corpus_id2idx(self):
        """Build or load mapping from doc id to corpus position (cached in corpus dir)."""
        if self._corpus_id2idx is not None:
            return
        self._corpus_id2idx = load_or_build_id2idx(
            self.corpus, self.corpus_path, desc="Corpus id mapping (SPLADE retriever)"
        )

    def _check_contain_doc(self):
        try:
            return self.searcher.doc(0).raw() is not None
        except Exception:
            return False

    def _search(self, query: str, num: int = None, return_score: bool = False, qid: str = None):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, k=num)
        if len(hits) < 1:
            return ([], []) if return_score else []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]
        if self.contain_doc:
            raw_docs = [
                json.loads(self.searcher.doc(hit.docid).raw())
                for hit in hits
            ]
            results = []
            for raw_doc in raw_docs:
                content = raw_doc.get('contents', '')
                result = {
                    'title': content.split("\n")[0].strip("\"") if content else '',
                    'text': "\n".join(content.split("\n")[1:]) if content else '',
                    'contents': content
                }
                if 'id' in raw_doc:
                    result['id'] = raw_doc['id']
                elif 'doc_id' in raw_doc:
                    result['doc_id'] = raw_doc['doc_id']
                elif 'passage_id' in raw_doc:
                    result['passage_id'] = raw_doc['passage_id']
                results.append(result)
        else:
            # LuceneImpactSearcher returns external docids (strings); need id2idx for lookup
            results = load_docs(self.corpus, [hit.docid for hit in hits], id2idx=self._corpus_id2idx)
        # Attach retriever score and rank to each doc
        for rank, (doc, score) in enumerate(zip(results, scores), 1):
            if isinstance(doc, dict):
                doc["score"] = score
                doc["rank"] = rank
        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        return results

class RerankRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # Frist-stage
        self.searcher = LuceneSearcher(self.index_path)
        self.searcher.set_bm25(config.bm25_k1, config.bm25_b)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

        # Second-stage (with GPU support)
        self.device = get_device()
        # Update model path based on retriever name
        if config.retriever_name == "rerank_l12":
            self.retrieval_model_path = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        elif config.retriever_name == "rerank_l6":
            self.retrieval_model_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        print(f"Loading CrossEncoder: {self.retrieval_model_path} on device: {self.device}")
        self.cross_encoder = CrossEncoder(
            self.retrieval_model_path,
            max_length=config.retrieval_query_max_length,
            device=str(self.device)  # CrossEncoder accepts device as string
        )
    
    def set_topk(self, new_k):
        self.topk = new_k
      
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None
    
    def _rerank_documents(self, query, contents):
        # Build query-document pairs for reranking
        query_doc_pairs = []
        for i, doc in enumerate(contents):
            # Handle different document formats
            if isinstance(doc, dict):
                doc_text = doc.get('contents', doc.get('text', ''))
            else:
                # Handle HuggingFace Dataset objects
                try:
                    doc_text = doc['contents']
                except (KeyError, TypeError):
                    print(f"Warning: Document {i} missing 'contents' field. Using empty string.")
                    doc_text = ''

            query_doc_pairs.append((query, doc_text))

        # Predict relevance scores using CrossEncoder
        scores = self.cross_encoder.predict(query_doc_pairs)

        # Sort by score and return top-k
        reranked_docs = sorted(zip(scores, contents), key=lambda x: x[0], reverse=True)[:self.topk]
        scores, sorted_contents = zip(*reranked_docs) if reranked_docs else ([], [])
        return list(sorted_contents), list(scores)
    
    def _search(self, query: str, num: int = None, return_score: bool = False):
        first_stage_num = 1000
        if num is None:
            num = self.topk

        # First-stage
        hits = self.searcher.search(query, first_stage_num)
        if len(hits) < 1:
            return ([], []) if return_score else []

        if len(hits) < first_stage_num:
            warnings.warn('Not enough documents retrieved for first-stage!')
        else:
            hits = hits[:first_stage_num]

        if self.contain_doc:
            # all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw()) for hit in hits]
        else:
            docids = [hit.docid for hit in hits]
            all_contents = load_docs(self.corpus, docids)

        # Second-stage reranking
        if len(all_contents) > 0:
            results, scores = self._rerank_documents(query, all_contents)
        else:
            # Handle case where no documents were loaded
            results, scores = [], []

        # Attach reranker score and rank to each doc
        for rank, (doc, score) in enumerate(zip(results, scores), 1):
            if isinstance(doc, dict):
                doc["score"] = float(score)
                doc["rank"] = rank

        return (results, scores) if return_score else results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        pass

def _attach_embeddings(index, idxs: list, results: list) -> None:
    """Reconstruct doc embeddings from a FAISS Flat index and attach to result dicts.

    Each result dict gets an ``'_emb'`` key with shape ``(1, dim)`` so it can be
    passed directly to sklearn's ``cosine_similarity``.  Silently skipped for GPU
    indices or non-Flat index types that don't support ``reconstruct_batch``.
    """
    if not results:
        return
    try:
        doc_embs = index.reconstruct_batch(np.array(idxs, dtype=np.int64))
        for doc, emb in zip(results, doc_embs):
            if isinstance(doc, dict):
                doc["_emb"] = emb.reshape(1, -1)
    except Exception:
        pass  # GPU index or compressed index: reconstruct not available


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print('loading index ...')
        # Use memory mapping for faster loading of large indices (29GB+)
        self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)
        # FAISS GPU is only supported for CUDA (not MPS)
        if config.faiss_gpu and torch.cuda.is_available():
            # Check if GPU features are available
            if hasattr(faiss, 'GpuMultipleClonerOptions'):
                print("Using FAISS with CUDA GPU (Multi-GPU mode)...")
                # --- Multi-GPUs: Only run with A100 (2 GPUs), H100 leads to an error
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
                num_gpus = torch.cuda.device_count()
                print(f"  → FAISS index sharded across {num_gpus} GPUs")
            elif hasattr(faiss, 'StandardGpuResources'):
                # Fall back to single GPU if multi-GPU not available
                print("Using FAISS with CUDA GPU (Single-GPU mode)...")
                device_id = torch.cuda.current_device()
                print(f'  → Using GPU {device_id}')
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(res, device_id, self.index, co)
            else:
                # faiss-cpu package installed, GPU features not available
                print("WARNING: faiss_gpu=True but faiss-cpu package detected.")
                print("         Install faiss-gpu for GPU acceleration.")
                print("         Falling back to CPU index with GPU-accelerated encoder.")
                print("         To install: pip uninstall faiss-cpu && conda install -c conda-forge faiss-gpu")
        elif config.faiss_gpu and torch.backends.mps.is_available():
            print("Note: FAISS GPU not supported on MPS. Using CPU FAISS with MPS-accelerated encoder.")

        print('loading corpus ...')
        self.corpus = load_corpus(self.corpus_path)

        # Get device for encoder
        self.device = get_device()
        self.encoder = Encoder(
            model_name = config.retriever_name,
            model_path = self.retrieval_model_path,
            pooling_method = self.pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = self.device
        )
        self.topk = config. retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        results = load_docs(self.corpus, idxs)
        _attach_embeddings(self.index, idxs, results)
        # Attach retriever score and rank to each doc
        for rank, (doc, score) in enumerate(zip(results, scores), 1):
            if isinstance(doc, dict):
                doc["score"] = score
                doc["rank"] = rank
        return (results, scores) if return_score else results

        # results = load_docs(self.corpus, idxs)
        # if return_score:
        #     return results, scores.tolist()
        # else:
        #     return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            flat_results = load_docs(self.corpus, flat_idxs)
            _attach_embeddings(self.index, flat_idxs, flat_results)
            # Attach retriever score and rank to each doc
            flat_scores = sum(batch_scores, [])
            for rank_in_flat, (doc, sc) in enumerate(zip(flat_results, flat_scores)):
                if isinstance(doc, dict):
                    doc["score"] = sc
                    doc["rank"] = (rank_in_flat % num) + 1
            # chunk them back
            batch_results = [flat_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            
        if return_score:
            return results, scores
        else:
            return results
