"""Snippet search tool using the Semantic Scholar snippet API (S2_API_KEY)."""

import os
from typing import Any, Dict, List, Optional

import requests


class SnippetSearchTool:
    """Search for paper snippets using the Semantic Scholar snippet search API.

    Maps to the ``snippet_search`` tool in the DR-Tulu system prompt.

    Args:
        api_key: Semantic Scholar API key.  Defaults to the ``S2_API_KEY``
                 environment variable (works without a key but at lower rate limits).
        timeout: HTTP request timeout in seconds.
    """

    _S2_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10) -> None:
        self.api_key = api_key or os.getenv("S2_API_KEY")
        self.timeout = timeout

    def execute( self, query: str, limit: int = 10, year: Optional[str] = None, ) -> List[Dict[str, Any]]:
        """Search for paper snippets matching *query*.

        Args:
            query:  Plain-text search query.
            limit:  Maximum number of snippets to return (default: 10).
            year:   Optional publication-year filter, e.g. ``"2021-2025"`` or
                    ``"2023-"``.  Passed directly to the S2 API.

        Returns:
            Normalised doc list, each entry containing:
                title  – paper title
                url    – Semantic Scholar paper URL
                text   – snippet text
                doc_id – Semantic Scholar corpus ID
        """
        params: Dict[str, Any] = {"query": query, "limit": limit}
        if year:
            params["year"] = year

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            resp = requests.get(
                self._S2_SNIPPET_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception:
            return []

        docs: List[Dict[str, Any]] = []
        for item in data:
            snippet = item.get("snippet", {})
            paper = item.get("paper", {})
            corpus_id = paper.get("corpusId", "")
            docs.append({
                "title":  paper.get("title", ""),
                "url":    f"https://www.semanticscholar.org/paper/{corpus_id}",
                "text":   snippet.get("text", ""),
                "doc_id": str(corpus_id),
            })
        return docs
