"""Webpage browsing tool using the Jina Reader API (JINA_API_KEY)."""

import os
from typing import Any, Dict, List, Optional

import requests


class BrowseWebpageTool:
    """Fetch and parse a webpage using the Jina Reader API.

    Maps to the ``browse_webpage`` tool in the DR-Tulu system prompt.
    The tool input is expected to be a URL; the API returns clean text/markdown.

    Args:
        api_key: Jina API key.  Defaults to the ``JINA_API_KEY`` environment
                 variable.
        timeout: HTTP request timeout in seconds.
    """

    _JINA_BASE_URL = "https://r.jina.ai/"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30) -> None:
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.timeout = timeout

    def execute(self, url: str) -> List[Dict[str, Any]]:
        """Fetch the webpage at *url* and return it as a single-item doc list.

        Args:
            url: The URL to fetch.

        Returns:
            Normalised doc list with one entry containing:
                title  – page title (from Jina response)
                url    – the original URL
                text   – page content as clean text / markdown
                doc_id – the original URL
        """
        if not self.api_key:
            return []

        jina_url = f"{self._JINA_BASE_URL}{url}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

        try:
            resp = requests.get(jina_url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("data", {})
        except Exception:
            return []

        return [{
            "title":  data.get("title", url),
            "url":    data.get("url", url),
            "text":   data.get("content", ""),
            "doc_id": url,
        }]
