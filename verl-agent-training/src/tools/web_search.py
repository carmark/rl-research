"""Web search tool for the Agentic RL environment."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Simulated web search tool.

    In production, this would call a real search API (e.g. SerpAPI, Bing).
    For RL training, we provide a pluggable backend: either a real API or a
    mock that returns pre-defined results from a corpus.
    """

    name = "web_search"
    description = "Search the web for information related to a query."
    parameters = {
        "query": {"type": "string", "description": "The search query"},
    }

    def __init__(
        self,
        backend: str = "mock",
        api_key: str | None = None,
        corpus: dict[str, list[dict]] | None = None,
        max_results: int = 5,
    ):
        self.backend = backend
        self.api_key = api_key
        self.corpus = corpus or {}
        self.max_results = max_results

    def execute(self, query: str, **kwargs: Any) -> list[dict]:
        if self.backend == "mock":
            return self._mock_search(query)
        elif self.backend == "serpapi":
            return self._serpapi_search(query)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _mock_search(self, query: str) -> list[dict]:
        """Return mock search results from the pre-loaded corpus."""
        results = []
        query_lower = query.lower()
        for key, entries in self.corpus.items():
            if any(word in key.lower() for word in query_lower.split()):
                results.extend(entries)
        if not results:
            results = [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"This is a mock search result for the query '{query}'.",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                }
            ]
        return results[: self.max_results]

    def _serpapi_search(self, query: str) -> list[dict]:
        """Call the SerpAPI search endpoint."""
        try:
            import requests
        except ImportError:
            raise RuntimeError("requests package required for serpapi backend")

        if not self.api_key:
            raise ValueError("api_key required for serpapi backend")

        resp = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": self.api_key, "num": self.max_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic_results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                }
            )
        return results[: self.max_results]
