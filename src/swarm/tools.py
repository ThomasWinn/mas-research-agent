from __future__ import annotations

from typing import Any

try:
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TavilySearchAPIWrapper = None  # type: ignore


class SearchClient:
    """Light wrapper around a web search provider."""

    def __init__(
        self,
        provider: str = "tavily",
        api_key: str | None = None,
        default_k: int = 5,
    ) -> None:
        self.provider = provider
        self.default_k = default_k
        self._wrapper: Any | None = None

        if provider == "tavily":
            if TavilySearchAPIWrapper is None:
                raise ImportError(
                    "langchain-community is required for Tavily search integration."
                )
            if not api_key:
                self.provider = "noop"
                self._wrapper = None
            else:
                self._wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        elif provider == "noop":
            self._wrapper = None
        else:
            raise ValueError(f"Unsupported search provider: {provider}")

    def search(self, query: str, k: int | None = None) -> list[dict[str, str]]:
        if self.provider == "noop" or not self._wrapper:
            return [
                {
                    "title": "Search provider not configured",
                    "url": "",
                    "snippet": (
                        "No web search provider is configured. Configure TAVILY_API_KEY "
                        "or supply a custom SearchClient."
                    ),
                }
            ]
        size = k or self.default_k
        results = self._wrapper.results(query, max_results=size)
        normalized: list[dict[str, str]] = []
        for item in results:
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )
        return normalized
