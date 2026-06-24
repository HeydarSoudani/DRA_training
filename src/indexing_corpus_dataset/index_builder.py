import os

# Disable safetensors auto-conversion (avoids background thread errors when
# HuggingFace Hub is unreachable or returns empty responses).
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gc
import json
import torch
import shutil
import argparse
import warnings
import subprocess
import numpy as np
import importlib.util
from tqdm import tqdm
from typing import cast
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


# =============================
MODEL2POOLING = {
    "bm25": "",
    "contriever": "mean",
    "dpr": "pooler",
    "e5": "mean",
    "bge": "cls",
    "reasonir": 'mean',
    "spladepp": None,
    "spladev3": None,
    "qwen3_emb_0.6b": "last_token",
    "qwen3_emb_4b": "last_token",
    "qwen3_emb_8b": "last_token",
}

MODEL2PATH = {
    "bm25": "",
    "contriever": "facebook/contriever-msmarco",
    "dpr": "facebook/dpr-ctx_encoder-single-nq-base",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
    "reasonir": 'reasonir/ReasonIR-8B',
    "spladepp": "naver/splade-cocondenser-ensembledistil",
    "spladev3": "naver/splade-v3",
    "qwen3_emb_0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3_emb_4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3_emb_8b": "Qwen/Qwen3-Embedding-8B",
}

# Canonical dataset root (single source of truth in layout.py).
try:
    from indexing_corpus_dataset.layout import DATA_ROOT
except ImportError:  # running as a plain script from inside the package dir
    from layout import DATA_ROOT


def get_device(device_id=None):
    """Get the best available device (CUDA, MPS, or CPU).

    Args:
        device_id: Optional GPU device ID for CUDA systems (e.g., 0, 1, 2).
                   Ignored for MPS (Apple Silicon uses all cores automatically).
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(retrieval_method, model_path: str, use_fp16: bool = False, device=None):
    if device is None:
        device = get_device()

    if retrieval_method == 'dpr':
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path)
        model = DPRContextEncoder.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # Qwen3 embedding models require left padding for last-token pooling
    if "qwen3" in retrieval_method:
        tokenizer.padding_side = "left"

    model.eval()
    model.to(device)

    # Note: MPS doesn't support float16 as well as CUDA, so only use fp16 on CUDA
    if use_fp16 and device.type == "cuda":
        model = model.half()

    print(f"Model loaded on device: {device}")
    return model, tokenizer

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
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

def resolve_corpus_files(corpus_path: str) -> list[str]:
    """Resolve corpus_path to a sorted list of .jsonl file paths.

    Accepts a single file or a directory (all *.jsonl inside).
    """
    if os.path.isdir(corpus_path):
        files = sorted(str(p) for p in Path(corpus_path).glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in corpus directory: {corpus_path}")
        print(f"Found {len(files)} corpus file(s) in {corpus_path}:")
        for f in files:
            print(f"  {f}")
        return files
    return [corpus_path]

def count_corpus(corpus_files: list[str]) -> int:
    """Count total non-empty lines across corpus files."""
    total = 0
    for f in corpus_files:
        with open(f, encoding='utf-8') as fh:
            total += sum(1 for line in fh if line.strip())
    return total

def peek_corpus_has_title(corpus_files: list[str]) -> bool:
    """Check whether the first document in the corpus has a 'title' field."""
    for f in corpus_files:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return 'title' in json.loads(line)
    return False

def iter_corpus_batches(corpus_files: list[str], batch_size: int):
    """Stream corpus in batches of plain dicts, never writing to disk."""
    batch = []
    for f in corpus_files:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


@torch.no_grad()
def _encode_batch_core(encoder, tokenizer, batch_texts, retrieval_method, pooling_method, max_length, device):
    """Encode a single batch of texts, returning a numpy float32 array of embeddings.

    Shared by both single-GPU and multi-GPU (per-worker) encoding paths.
    """
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if "t5" in retrieval_method:
        decoder_input_ids = torch.zeros(
            (inputs['input_ids'].shape[0], 1), dtype=torch.long
        ).to(device)
        output = encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
        embeddings = output.last_hidden_state[:, 0, :]
    elif "reasonir" in retrieval_method:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(None, output.last_hidden_state, inputs['attention_mask'], pooling_method)
    elif "dpr" in retrieval_method:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(output.pooler_output, None, None, pooling_method)
    elif "contriever" in retrieval_method:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(None, output.last_hidden_state, inputs['attention_mask'], pooling_method)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    elif "qwen3" in retrieval_method:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(None, output.last_hidden_state, inputs['attention_mask'], pooling_method)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    else:
        output = encoder(**inputs, return_dict=True)
        embeddings = pooling(output.pooler_output, output.last_hidden_state, inputs['attention_mask'], pooling_method)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    return cast(torch.Tensor, embeddings).detach().cpu().numpy()


def _encode_shard_worker(rank, world_size, corpus_files, corpus_size, corpus_has_title,
                         retrieval_method, model_path, use_fp16, pooling_method,
                         max_length, batch_size, output_path):
    """Worker process: load model on GPU ``rank`` and encode a contiguous shard of the corpus.

    Results are saved to ``output_path`` (numpy memmap) with shape metadata in
    ``output_path + '.shape.npy'``.
    """
    device = torch.device(f"cuda:{rank}")

    # Compute contiguous shard boundaries
    docs_per_gpu = (corpus_size + world_size - 1) // world_size
    start_idx = rank * docs_per_gpu
    end_idx = min(start_idx + docs_per_gpu, corpus_size)
    n_docs = end_idx - start_idx

    if n_docs <= 0:
        print(f"[GPU {rank}] No documents in shard, skipping")
        np.save(output_path + '.shape.npy', np.array([0, 0]))
        return

    print(f"[GPU {rank}] Loading model on cuda:{rank}")
    encoder, tokenizer = load_model(retrieval_method, model_path, use_fp16, device)

    print(f"[GPU {rank}] Encoding docs [{start_idx:,}, {end_idx:,}) — {n_docs:,} documents")
    all_embeddings = []
    batch_texts = []
    doc_idx = 0
    pbar = tqdm(total=n_docs, desc=f'GPU-{rank}', miniters=max(1, n_docs // 100),
                 position=rank, leave=True)

    for f in corpus_files:
        if doc_idx >= end_idx:
            break
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                if doc_idx >= end_idx:
                    break
                line_s = line.strip()
                if not line_s:
                    continue
                if doc_idx >= start_idx:
                    doc = json.loads(line_s)
                    if corpus_has_title:
                        text = '"' + doc.get('title', '') + '"\n' + doc.get('contents', '')
                    else:
                        text = doc.get('contents', '')
                    if retrieval_method == "e5":
                        text = f"passage: {text}"
                    batch_texts.append(text)

                    if len(batch_texts) >= batch_size:
                        emb = _encode_batch_core(encoder, tokenizer, batch_texts,
                                                 retrieval_method, pooling_method, max_length, device)
                        all_embeddings.append(emb)
                        pbar.update(len(batch_texts))
                        batch_texts = []
                doc_idx += 1

    if batch_texts:
        emb = _encode_batch_core(encoder, tokenizer, batch_texts,
                                 retrieval_method, pooling_method, max_length, device)
        all_embeddings.append(emb)
        pbar.update(len(batch_texts))
    pbar.close()

    all_embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    # Save to memmap + shape metadata so the main process can reassemble
    memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=all_embeddings_np.shape)
    memmap[:] = all_embeddings_np
    memmap.flush()
    del memmap
    np.save(output_path + '.shape.npy', np.array(all_embeddings_np.shape))

    print(f"[GPU {rank}] Done — {all_embeddings_np.shape[0]:,} embeddings saved")

    del encoder, tokenizer, all_embeddings, all_embeddings_np
    torch.cuda.empty_cache()


class Index_Builder:
    r"""A tool class used to build an index used in retrieval."""
    def __init__(self, retrieval_method, model_path, corpus_path, save_dir, max_length, batch_size, use_fp16, pooling_method, faiss_type=None, index_path=None, embedding_path=None, save_embedding=False, faiss_gpu=False, device_id=None):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.index_save_path = None
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.device_id = device_id
        # Derive corpus stem for naming index/embedding files
        self.corpus_stem = Path(corpus_path).stem if os.path.isfile(corpus_path) else os.path.basename(corpus_path.rstrip(os.sep))

        # Set device
        self.device = get_device(device_id)
        self.gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Print device info
        print(f"\n{'='*50}")
        print(f"Device Configuration:")
        print(f"{'='*50}")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"Number of CUDA GPUs available: {self.gpu_num}")
            if device_id is not None:
                print(f"Selected GPU ID: {device_id}")
        elif self.device.type == "mps":
            print(f"MPS (Metal Performance Shaders) enabled")
            print(f"Note: MPS automatically uses all GPU cores")
        print(f"Batch size: {self.batch_size}")
        print(f"FP16 enabled: {self.use_fp16 and self.device.type == 'cuda'}")
        print(f"{'='*50}\n")
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = index_path if index_path is not None else self._resolve_index_path()
        self.embedding_save_path = os.path.join(self.save_dir, f"{self.corpus_stem}_{self.retrieval_method}.memmap")
        self.corpus_files = None
        self.corpus_size = 0
        self.corpus_has_title = False
        if self.retrieval_method in {"bm25", "spladepp", "spladev3"}:
            print(f"Skipping corpus load for {self.retrieval_method}.")
        elif self.embedding_path is not None:
            print(f"Skipping corpus load - using pre-computed embeddings from {self.embedding_path}")
        else:
            self.corpus_files = resolve_corpus_files(self.corpus_path)
            print("Counting corpus documents (streaming, no disk cache)...")
            self.corpus_size = count_corpus(self.corpus_files)
            self.corpus_has_title = peek_corpus_has_title(self.corpus_files)
            print(f"Corpus: {self.corpus_size:,} documents, has_title={self.corpus_has_title}")

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    @staticmethod
    def _ensure_module_available(module_name, extra_message=""):
        if importlib.util.find_spec(module_name) is None:
            hint = f" Please install it via `{extra_message}`." if extra_message else ""
            raise ModuleNotFoundError(f"Required module '{module_name}' is not installed.{hint}")

    def _resolve_index_path(self):
        if self.retrieval_method == "bm25":
            return os.path.join(self.save_dir, f"{self.corpus_stem}_bm25_index")
        if self.retrieval_method == "spladepp":
            return os.path.join(self.save_dir, f"{self.corpus_stem}_spladepp_index")
        if self.retrieval_method == "spladev3":
            return os.path.join(self.save_dir, f"{self.corpus_stem}_spladev3_index")
        return os.path.join(self.save_dir, f"{self.corpus_stem}_{self.retrieval_method}_{self.faiss_type}.index")

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        elif self.retrieval_method in ("spladepp", "spladev3"):
            self.build_splade_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Build BM25 index using Pyserini.
        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md
        """
        index_dir = self.index_save_path
        os.makedirs(index_dir, exist_ok=True)

        if os.path.isdir(self.corpus_path):
            corpus_dir = os.path.abspath(self.corpus_path)
        else:
            corpus_dir = os.path.dirname(os.path.abspath(self.corpus_path))

        print("Start building bm25 index...")
        pyserini_args = [
            "--collection", "JsonCollection",
            "--input", corpus_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args, check=True)
        print(f"Finish! BM25 index stored at {index_dir}")

    def build_splade_index(self):
        """Build a SPLADE impact index using direct encoding or existing vectors.jsonl."""
        self._ensure_module_available("pyserini", "pip install pyserini[impact]")

        index_dir = self.index_save_path
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        os.makedirs(index_dir, exist_ok=True)

        # Use existing vectors if embedding_path points to vectors.jsonl or a dir containing it
        use_existing_vectors = False
        if self.embedding_path is not None:
            if os.path.isfile(self.embedding_path):
                if os.path.basename(self.embedding_path) == "vectors.jsonl":
                    vector_dir = os.path.dirname(os.path.abspath(self.embedding_path))
                    use_existing_vectors = True
            elif os.path.isdir(self.embedding_path):
                vectors_file = os.path.join(self.embedding_path, "vectors.jsonl")
                if os.path.isfile(vectors_file):
                    vector_dir = os.path.abspath(self.embedding_path.rstrip(os.sep))
                    use_existing_vectors = True
            if use_existing_vectors:
                print(f"Using existing {self.retrieval_method} vectors from {vector_dir} (indexing only).")

        if not use_existing_vectors:
            if not self.model_path:
                raise ValueError(f"{self.retrieval_method} requires a valid model path or --embedding_path to vectors.jsonl.")
            vector_dir = os.path.join(self.save_dir, f"{self.retrieval_method}_vectors")
            if os.path.exists(vector_dir):
                shutil.rmtree(vector_dir)
            os.makedirs(vector_dir, exist_ok=True)

            print(f"Loading SPLADE model: {self.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            model.eval()
            model.to(self.device)
            if self.use_fp16 and self.device.type == "cuda":
                model = model.half()
            if self.gpu_num > 1 and self.device.type == "cuda":
                print(f"Using DataParallel for {self.retrieval_method} encoding across {self.gpu_num} GPUs")
                model = torch.nn.DataParallel(model)
                self.batch_size = self.batch_size * self.gpu_num

            print(f"Encoding corpus with {self.retrieval_method} (batch size: {self.batch_size})...")
            self._encode_splade_corpus(model, tokenizer, vector_dir)

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        cpu_threads = max(1, min(os.cpu_count() or 1, 32))
        index_cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonVectorCollection",
            "--input", vector_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(cpu_threads),
            "--impact",
            "--pretokenized"
        ]

        print(f"Building Lucene impact index for {self.retrieval_method} outputs...")
        subprocess.run(index_cmd, check=True)

        if not use_existing_vectors:
            if not self.save_embedding and os.path.isdir(vector_dir):
                shutil.rmtree(vector_dir)
                print("Temporary SPLADE vector directory removed.")
            else:
                print(f"SPLADE vector dumps preserved at {vector_dir}")

        print(f"Finish! {self.retrieval_method} index stored at {index_dir}")

    @torch.no_grad()
    def _encode_splade_corpus(self, model, tokenizer, output_dir):
        """Encode corpus with SPLADE model and save as Pyserini JsonVectorCollection format."""
        corpus_files = resolve_corpus_files(self.corpus_path)

        print("Counting documents...")
        total_docs = sum(
            1 for corpus_file in corpus_files
            for line in open(corpus_file, encoding='utf-8')
            if line.strip()
        )
        print(f"Total documents: {total_docs:,}")

        output_file = os.path.join(output_dir, "vectors.jsonl")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            batch_texts, batch_ids = [], []
            doc_count = 0
            pbar = tqdm(total=total_docs, desc='Encoding')

            for corpus_file in corpus_files:
                with open(corpus_file, encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        batch_texts.append(doc.get('contents', ''))
                        batch_ids.append(doc.get('id', str(doc_count)))
                        doc_count += 1

                        if len(batch_texts) >= self.batch_size:
                            self._process_splade_batch(batch_texts, batch_ids, model, tokenizer, out_f)
                            pbar.update(len(batch_texts))
                            batch_texts, batch_ids = [], []

            if batch_texts:
                self._process_splade_batch(batch_texts, batch_ids, model, tokenizer, out_f)
                pbar.update(len(batch_texts))
            pbar.close()

        print(f"Encoded vectors saved to {output_file}")

    def _process_splade_batch(self, batch_texts, batch_ids, model, tokenizer, output_file):
        """Process a batch of documents with SPLADE encoding."""
        inputs = tokenizer(
            batch_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # SPLADE: log(1 + ReLU(MLM_logits)) with max pooling over sequence
        relu_log = torch.log(1 + torch.relu(outputs.logits))
        relu_log = relu_log * inputs['attention_mask'].unsqueeze(-1).expand_as(relu_log)
        logits = torch.max(relu_log, dim=1)[0]

        for doc_id, doc_logits in zip(batch_ids, logits):
            non_zero = torch.nonzero(doc_logits > 0, as_tuple=True)[0]
            if len(non_zero) > 0:
                tokens = tokenizer.convert_ids_to_tokens(non_zero.cpu().tolist())
                weights = doc_logits[non_zero].cpu().tolist()
                vector = {t: float(w) for t, w in zip(tokens, weights)}
            else:
                vector = {}
            output_file.write(json.dumps({'id': doc_id, 'contents': '', 'vector': vector}) + '\n')

        del inputs, outputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        return np.memmap(
            embedding_path,
            mode="r",
            dtype=np.float32
        ).reshape(corpus_size, hidden_size)

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        """Encode all corpus documents. Dispatches to multi-GPU sharding when available."""
        use_multi_gpu = (
            self.gpu_num > 1
            and self.device.type == "cuda"
            and self.device_id is None
        )
        if use_multi_gpu:
            return self._encode_all_multi_gpu()

        # --- Single device path (GPU / CPU / MPS) ---
        all_embeddings = []
        n_batches = (self.corpus_size + self.batch_size - 1) // self.batch_size

        for batch_docs in tqdm(iter_corpus_batches(self.corpus_files, self.batch_size),
                               total=n_batches, desc='Inference Embeddings:'):
            if self.corpus_has_title:
                batch_data = ['"' + doc.get('title', '') + '"\n' + doc.get('contents', '')
                              for doc in batch_docs]
            else:
                batch_data = [doc.get('contents', '') for doc in batch_docs]

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            embeddings = _encode_batch_core(
                self.encoder, self.tokenizer, batch_data,
                self.retrieval_method, self.pooling_method,
                self.max_length, self.device,
            )
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings.astype(np.float32)

    def _encode_all_multi_gpu(self):
        """Encode corpus using process-based sharding — one model replica per GPU.

        Each GPU independently encodes a contiguous shard of the corpus.
        This avoids the GPU-0 memory bottleneck and communication overhead of
        DataParallel, giving near-linear scaling.
        """
        import tempfile
        import torch.multiprocessing as mp

        print(f"\n{'='*50}")
        print(f"Multi-GPU Encoding (process-based sharding)")
        print(f"GPUs: {self.gpu_num}, Batch size per GPU: {self.batch_size}")
        print(f"Corpus: {self.corpus_size:,} documents")
        print(f"{'='*50}\n")

        # Use a sub-directory under save_dir (fast storage) for shard files
        shard_dir = os.path.join(self.save_dir, '_encoding_shards')
        os.makedirs(shard_dir, exist_ok=True)
        shard_paths = [os.path.join(shard_dir, f'shard_{i}.memmap') for i in range(self.gpu_num)]

        ctx = mp.get_context('spawn')
        processes = []
        for rank in range(self.gpu_num):
            p = ctx.Process(
                target=_encode_shard_worker,
                args=(
                    rank, self.gpu_num,
                    self.corpus_files, self.corpus_size, self.corpus_has_title,
                    self.retrieval_method, self.model_path, self.use_fp16, self.pooling_method,
                    self.max_length, self.batch_size, shard_paths[rank],
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Move cursor below all progress bars
        print('\n' * self.gpu_num)

        # Check for worker failures
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                shutil.rmtree(shard_dir, ignore_errors=True)
                raise RuntimeError(f"GPU {i} worker exited with code {p.exitcode}")

        # Reassemble shard embeddings in order
        print("Concatenating embeddings from all GPUs...")
        shards = []
        for path in shard_paths:
            shape = tuple(np.load(path + '.shape.npy'))
            if shape[0] == 0:
                continue
            shard = np.memmap(path, dtype=np.float32, mode='r', shape=shape)
            shards.append(np.array(shard))  # copy into regular array
            del shard

        all_embeddings = np.concatenate(shards, axis=0).astype(np.float32)
        del shards

        # Clean up temporary shard files
        shutil.rmtree(shard_dir, ignore_errors=True)

        print(f"Total embeddings shape: {all_embeddings.shape}")
        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Encode documents with a dense model and build a FAISS index."""
        import faiss
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        use_multi_gpu = (
            self.gpu_num > 1
            and self.device.type == "cuda"
            and self.device_id is None
        )

        if self.embedding_path is not None:
            # Need model briefly to read hidden_size from config
            self.encoder, self.tokenizer = load_model(
                retrieval_method=self.retrieval_method,
                model_path=self.model_path,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            hidden_size = self.encoder.config.hidden_size
            del self.encoder, self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            embedding_file_size = os.path.getsize(self.embedding_path)
            corpus_size = embedding_file_size // (hidden_size * 4)
            print(f"Inferred corpus size from embedding file: {corpus_size:,} documents")
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            if not use_multi_gpu:
                # Single device — load model here; encode_all uses self.encoder
                self.encoder, self.tokenizer = load_model(
                    retrieval_method=self.retrieval_method,
                    model_path=self.model_path,
                    use_fp16=self.use_fp16,
                    device=self.device,
                )
            # Multi-GPU workers load their own model replicas
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            self.corpus_files = None  # release file handles / metadata

        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        del all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


def main():
    parser = argparse.ArgumentParser(description="Creating index...")
    parser.add_argument('--corpus_path', type=str, default=f'{DATA_ROOT}/browsecomp_plus/corpus/corpus.jsonl', help='Path to corpus .jsonl file or directory of .jsonl files.')
    parser.add_argument('--retrieval_method', type=str, default='e5', choices=['bm25', 'spladepp', 'spladev3', 'contriever', 'dpr', 'e5', 'bge', 'reasonir', 'qwen3_emb_0.6b', 'qwen3_emb_4b', 'qwen3_emb_8b'])
    parser.add_argument('--index_path', type=str, default=None, help='Override path for the output index file/dir. Auto-derived from corpus_path if not set.')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--faiss_type', type=str, default='Flat')
    parser.add_argument('--embedding_path', type=str, default=None, help='Path to pre-computed embeddings (.memmap). Skips encoding if set.')
    parser.add_argument('--save_embedding', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--use_fp16', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--faiss_gpu', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--device_id', type=int, default=None, help='GPU device ID for CUDA systems (e.g., 0, 1). Ignored for MPS.')

    args = parser.parse_args()

    # Auto-adjust max_length for browsecomp_plus + qwen3_emb
    if 'browsecomp_plus' in args.corpus_path and args.retrieval_method.startswith('qwen3_emb'):
        args.max_length = 4096
        print(f"Auto-adjusted max_length to {args.max_length} for browsecomp_plus + {args.retrieval_method}")

    # Derive save_dir as {dataset_root}/indices/ (sibling of the corpus/ directory)
    corpus_dir = os.path.dirname(os.path.abspath(args.corpus_path))
    save_dir = os.path.join(os.path.dirname(corpus_dir), "indices")

    args.model_path = MODEL2PATH[args.retrieval_method]
    pooling_method = MODEL2POOLING.get(args.retrieval_method)

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        index_path=args.index_path,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        device_id=args.device_id,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()


# --- Example usage ---
#
# Index and embedding files are saved in {dataset_root}/indices/ (sibling of corpus/).
# Naming: {corpus_stem}_{retrieval_method}_{faiss_type}.index / .memmap
#
# python src/agentic_retrieval_research/indexing/index_builder.py
