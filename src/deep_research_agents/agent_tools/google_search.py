"""Google web search tool supporting Serper.dev and Tavily backends."""

import json
import os
from typing import Any, Dict, List, Literal, Optional

import requests


class GoogleSearchTool:
    """Web search with a selectable backend: Serper.dev or Tavily.

    Maps to the ``google_search`` tool in the DR-Tulu system prompt.

    Args:
        backend:  Which search service to use.  Either ``"serper"`` (default)
                  or ``"tavily"``.
        api_key:  API key for the chosen backend.  Falls back to the
                  ``SERPER_API_KEY`` or ``TAVILY_API_KEY`` environment
                  variable respectively.
        timeout:  HTTP request timeout in seconds.
    """

    _SERPER_URL = "https://google.serper.dev/search"
    _TAVILY_URL = "https://api.tavily.com/search"

    def __init__( self, backend: Literal["serper", "tavily"] = "serper", api_key: Optional[str] = None, timeout: int = 10, ) -> None:
        if backend not in ("serper", "tavily"):
            raise ValueError(f"Unknown backend {backend!r}. Choose 'serper' or 'tavily'.")
        self.backend = backend
        self.timeout = timeout

        if api_key:
            self.api_key = api_key
        elif backend == "serper":
            self.api_key = os.getenv("SERPER_API_KEY")
        else:
            self.api_key = os.getenv("TAVILY_API_KEY")

    def execute( self, query: str, num_results: int = 10, ) -> List[Dict[str, Any]]:
        """Search for *query* using the configured backend.

        Args:
            query:       Search query string.
            num_results: Number of results to request (default: 10).

        Returns:
            Normalised doc list, each entry containing:
                title  – page title
                url    – page URL
                text   – result snippet / description
                doc_id – page URL (same as url)
        """
        if not self.api_key:
            return []

        if self.backend == "serper":
            return self._search_serper(query, num_results)
        return self._search_tavily(query, num_results)

    # ── Serper backend ────────────────────────────────────────────────────────

    def _search_serper(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = json.dumps({"q": query, "num": num_results})

        try:
            resp = requests.post(
                self._SERPER_URL,
                headers=headers,
                data=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            organic = resp.json().get("organic", [])
        except Exception:
            return []

        docs: List[Dict[str, Any]] = []
        for result in organic:
            link = result.get("link", "")
            docs.append({
                "title":  result.get("title", ""),
                "url":    link,
                "text":   result.get("snippet", ""),
                "doc_id": link,
            })
        return docs

    # ── Tavily backend ────────────────────────────────────────────────────────

    def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "max_results": num_results,
            "api_key": self.api_key,
        }

        try:
            resp = requests.post(
                self._TAVILY_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception:
            return []

        docs: List[Dict[str, Any]] = []
        for result in results:
            url = result.get("url", "")
            docs.append({
                "title":  result.get("title", ""),
                "url":    url,
                "text":   result.get("content", ""),
                "doc_id": url,
            })
        return docs
