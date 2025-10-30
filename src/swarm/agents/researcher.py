from __future__ import annotations

from typing import Any, Iterable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore
from ..tools import SearchClient


class ResearcherAgent:
    """Fetches evidence for each subtopic and drafts focused notes."""

    def __init__(
        self,
        llm: Runnable,
        search_client: SearchClient,
        memory: MemoryStore,
        batch_size: int = 3,
    ) -> None:
        self.llm = llm
        self.search_client = search_client
        self.memory = memory
        self.batch_size = batch_size
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research specialist. Given source snippets, craft a concise factual summary "
                    "highlighting key findings, data points, and differing perspectives.",
                ),
                (
                    "user",
                    "Topic: {topic}\n"
                    "Sources:\n{sources}\n"
                    "Write a structured summary (3-5 sentences) citing sources inline as [1], [2], etc.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        subtopics: Iterable[str] = state.get("subtopics", [])
        drafts: dict[str, dict[str, Any]] = {}
        for idx, topic in enumerate(subtopics, start=1):
            sources = self.search_client.search(topic, k=self.batch_size)
            formatted_sources = self._format_sources(sources)
            summary = (self.prompt | self.llm | self.parser).invoke(
                {"topic": topic, "sources": formatted_sources}
            )
            drafts[topic] = {"summary": summary, "sources": sources}
            self.memory.write(query, f"research:{idx}", drafts[topic])
        self.memory.write(query, "drafts", drafts)
        return {"drafts": drafts}

    def _format_sources(self, sources: list[dict[str, str]]) -> str:
        formatted = []
        for idx, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            formatted.append(f"[{idx}] {title}\nURL: {url}\n{snippet}")
        return "\n\n".join(formatted) if formatted else "No sources found."
