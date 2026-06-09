"""Efficiency tracking for retrieval pipelines."""

from typing import Dict, Any


class EfficiencyTracker:
    """Track efficiency metrics for retrieval pipelines.

    Tracks LLM calls, tokens, retriever calls, and latency breakdown.
    """

    def __init__(self, enabled: bool = False):
        """Initialize efficiency tracker.

        Args:
            enabled: Whether to track efficiency metrics
        """
        self.enabled = enabled
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.llm_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.retriever_calls = 0
        # Timing fields (can be customized per pipeline)
        self.stage_times = {}
        self.total_time = 0.0

    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record an LLM call.

        Args:
            input_tokens: Number of input tokens (estimated if not provided)
            output_tokens: Number of output tokens (estimated if not provided)
        """
        if not self.enabled:
            return
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_retriever_call(self):
        """Record a retriever call."""
        if not self.enabled:
            return
        self.retriever_calls += 1

    def record_stage_time(self, stage_name: str, time_seconds: float):
        """Record time for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage (e.g., 'query_processing', 'retrieval', 'reranking')
            time_seconds: Time taken in seconds
        """
        if not self.enabled:
            return
        if stage_name not in self.stage_times:
            self.stage_times[stage_name] = 0.0
        self.stage_times[stage_name] += time_seconds

    def get_summary(self) -> Dict[str, Any]:
        """Get efficiency summary.

        Returns:
            Dictionary with efficiency metrics
        """
        if not self.enabled:
            return {}

        summary = {
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "retriever_calls": self.retriever_calls,
        }

        # Add latency breakdown if stage times were recorded
        if self.stage_times:
            summary["latency_seconds"] = {
                stage: round(time_val, 3)
                for stage, time_val in self.stage_times.items()
            }
            summary["latency_seconds"]["total"] = round(self.total_time, 3)

        return summary

    def get_per_query_metrics(self, num_queries: int) -> Dict[str, Any]:
        """Get per-query efficiency metrics.

        Args:
            num_queries: Number of queries processed

        Returns:
            Dictionary with per-query metrics
        """
        if not self.enabled or num_queries == 0:
            return {}

        return {
            "avg_llm_calls_per_query": round(self.llm_calls / num_queries, 2),
            "avg_tokens_per_query": round((self.total_input_tokens + self.total_output_tokens) / num_queries, 2),
            "avg_retriever_calls_per_query": round(self.retriever_calls / num_queries, 2),
            "avg_latency_per_query_seconds": round(self.total_time / num_queries, 3),
        }
