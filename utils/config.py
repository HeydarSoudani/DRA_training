"""Pipeline constants, lightweight config objects, and inference configuration.

No heavy imports — this module must be safe to import from anywhere
without triggering circular dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Model maps and agent categories
# ---------------------------------------------------------------------------

# Maps agentic-model name → HuggingFace model ID for vLLM-served finetuned models
FINETUNED_MODEL_MAP: Dict[str, str] = {
    "drtulu":     "rl-research/DR-Tulu-8B",
    "webweaver":  "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "tongyi":     "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "cpm_report": "openbmb/AgentCPM-Report",
    "searchr1":   "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3",
    "research":   "agentrl/ReSearch-Qwen-7B-Instruct",
    "stepsearch": "Zill1/StepSearch-7B-Instruct",
}


def is_local_finetuned(agentic_model: str, llm_model: str) -> bool:
    """True when llm_model matches the agent's known HF model name."""
    return llm_model == FINETUNED_MODEL_MAP.get(agentic_model)


# Single source of truth: CLI --agentic-model value → the LLM it runs on.
# --llm-model is no longer a CLI input; it is derived from this map.
# The `oss` agent is exposed as two CLI names (oss_20b / oss_120b) so both
# GPT-OSS variants are selectable; AGENTIC_MODEL_ALIAS maps them back to the
# internal "oss" agent used by AGENT_MAP / SELF_MANAGED_LLM_AGENTS.
AGENTIC_MODEL_TO_LLM: Dict[str, str] = {
    # self-managed (vLLM)
    "cpm_report":  "openbmb/AgentCPM-Report",
    "cpm_explore": "openbmb/AgentCPM-Explore",
    "glm":         "zai-org/GLM-4.7-Flash",
    "tongyi":      "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "oss_20b":     "gpt-oss-20b",
    "oss_120b":    "gpt-oss-120b",
    # local-finetuned (vLLM)
    "webweaver":   "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
    "drtulu":      "rl-research/DR-Tulu-8B",
    "searchr1":    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3",
    "research":    "agentrl/ReSearch-Qwen-7B-Instruct",
    "stepsearch":  "Zill1/StepSearch-7B-Instruct",
    # API-backed instruction-tuned reasoning agents
    "react":       "claude-sonnet-4-6",
    "selfask":     "claude-sonnet-4-6",
    "searcho1":    "claude-sonnet-4-6",
}

# CLI --agentic-model value → internal agent name (when they differ).
AGENTIC_MODEL_ALIAS: Dict[str, str] = {
    "oss_20b":  "oss",
    "oss_120b": "oss",
}


_IR_ROOT = Path("/mnt/sagemaker-nvme/ir_datasets")

# Root for criteria-augmented datasets that live outside _IR_ROOT.
# Hardcoded (like _IR_ROOT) rather than derived from __file__ depth, so it is
# stable regardless of where this package is imported from.
_DATA_ROOT = Path("/gpfs/home6/data")

# Agents that manage their own LLM connection (direct vLLM/OpenAI clients)
SELF_MANAGED_LLM_AGENTS = frozenset({"oss", "tongyi", "glm", "cpm_explore", "cpm_report"})


# ---------------------------------------------------------------------------
# Retriever config
# ---------------------------------------------------------------------------

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
            # Optional score-threshold selector (dynamic-length output). Disabled
            # by default; set enable_selector=True to keep only high-scoring docs.
            "enable_selector": False,
            "selector_score_threshold": 3.0,
            "selector_min_keep": 0,
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
        "listwise_reranker_config": {
            "model_name": "castorini/rank_zephyr_7b_v1_full",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "template": "rankzephyr",
            "top_k": rerank_top_k,
            "window_size": 20,
            "stride": 10,
            "use_sliding_window": True,
            "max_passage_words": 300,
            "max_tokens": 200,
            "temperature": 0.0,
            "enable_thinking": False,
        },
        "rank_r1_reranker_config": {
            "api_url": "http://localhost:8001/v1",
            "api_model_name": "rank-r1",
            "api_key": "EMPTY",
            "num_child": 19,
            "k": 10,
            "max_tokens": 2048,
            "top_k": rerank_top_k,
            "context_size": 450,
        },
        "monot5_reranker_config": {
            "model_name": "castorini/monot5-base-msmarco",
            "use_mt5": False,
            "batch_size": 32,
            "top_k": rerank_top_k,
        },
    }


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Immutable bundle of parameters shared by the main loop, force answer,
    and answer candidate components of an agent."""

    # API dispatch
    api_type: str = "chat_completion"  # "responses_api" | "chat_completion"

    # Model identity
    model_name: str = ""
    api_base: Optional[str] = None
    api_key: str = "EMPTY"

    # Generation limits
    max_output_tokens: int = 20000

    # Reasoning (Responses API only; None disables the reasoning block)
    reasoning_effort: Optional[str] = "high"

    # Prompt scaffolding
    system_prompt: Optional[str] = None
    format_instructions: str = ""
