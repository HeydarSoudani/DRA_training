"""Pipeline constants and lightweight config objects.

Shared by general_utils, worker, and CLI setup helpers.
No heavy imports — this module must be safe to import from anywhere
without triggering circular dependencies.
"""

from pathlib import Path
from typing import Dict


# Maps agentic-model name → HuggingFace model ID for vLLM-served finetuned models
FINETUNED_MODEL_MAP: Dict[str, str] = {
    "drtulu":    "rl-research/DR-Tulu-8B",
    "webweaver": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tongyi":    "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "cpm_report": "openbmb/AgentCPM-Report",
}


def is_local_finetuned(agentic_model: str, llm_model: str) -> bool:
    """True when llm_model matches the agent's known HF model name."""
    return llm_model == FINETUNED_MODEL_MAP.get(agentic_model)


_IR_ROOT = Path("/mnt/sagemaker-nvme/ir_datasets")

# Agents that manage their own LLM connection (direct vLLM/OpenAI clients)
SELF_MANAGED_LLM_AGENTS = frozenset({"oss", "tongyi", "glm", "cpm_explore", "cpm_report"})


class _RetrieverConfig:
    """Lightweight config object for local retrievers (BM25, Dense, Rerank, SPLADE)."""

    def __init__(self, retriever_name, index_dir, corpus_path, topk):
        self.retriever_name = retriever_name
        self.index_dir = index_dir
        self.corpus_path = corpus_path
        self.retrieval_topk = topk
        self.bm25_k1 = 0.9
        self.bm25_b = 0.4
        self.faiss_gpu = False
        self.retrieval_query_max_length = 512
        self.retrieval_use_fp16 = True
        self.retrieval_batch_size = 32
        # SPLADE-specific
        self.splade_max_length = 256
        self.device = None  # auto-detected by SPLADERetriever


def get_reranker_configs(rerank_top_k: int = 100) -> dict:
    """Return the canonical reranker configuration dict."""
    return {
        "batched_reranker_config": {
            "reranker_model": "claude-sonnet-4-5",
            "max_chars_per_document": 4096,
        },
        "rankllama_reranker_config": {
            "model_name": "castorini/rankllama-v1-7b-lora-passage",
            "top_k": rerank_top_k,
            "batch_size": 1,
            "rerank_max_len": 256,
        },
        "rank1_reranker_config": {
            "model_name": "jhu-clsp/rank1-7b",
            "backend": "api",
            "api_url": "http://localhost:8000/v1",
            "api_model_name": "jhu-clsp/rank1-7b",
            "top_k": rerank_top_k,
            "batch_size": 100,
            "context_size": 128,
            "max_output_tokens": 200,
        },
        "qwen3_reranker_config": {
            "model_name": None,
            "size": "4B",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "batch_size": 32,
            "top_k": rerank_top_k,
            "enable_thinking": False,
        },
    }
