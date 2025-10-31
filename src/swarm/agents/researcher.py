from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore
from ..tools import SearchClient


class ResearcherAgent:
    """Single specialist responsible for drafting notes on assigned subtopics."""

    def __init__(
        self,
        name: str,
        llm: Runnable,
        search_client: SearchClient,
        *,
        batch_size: int = 3,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.search_client = search_client
        self.batch_size = batch_size
        system_msg = system_prompt or (
            "You are a research specialist. Given source snippets, craft a concise factual summary "
            "highlighting key findings, data points, and differing perspectives."
        )
        user_msg = user_prompt or (
            "Topic: {topic}\n"
            "Sources:\n{sources}\n"
            "Write a structured summary (3-5 sentences) citing sources inline as [1], [2], etc."
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("user", user_msg),
            ]
        )
        self.parser = StrOutputParser()

    def research_topic(self, topic: str, *, query: str) -> dict[str, Any]:
        sources = self.search_client.search(topic, k=self.batch_size)
        formatted_sources = self._format_sources(sources)
        summary = (self.prompt | self.llm | self.parser).invoke(
            {"topic": topic, "sources": formatted_sources}
        )
        return {"summary": summary, "sources": sources, "agent": self.name}

    def _format_sources(self, sources: list[dict[str, str]]) -> str:
        formatted = []
        for idx, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            formatted.append(f"[{idx}] {title}\nURL: {url}\n{snippet}")
        return "\n\n".join(formatted) if formatted else "No sources found."


class ResearchTeamAgent:
    """Coordinates multiple research specialists and merges their work."""

    def __init__(self, members: Iterable[ResearcherAgent], memory: MemoryStore) -> None:
        self.members = list(members)
        if not self.members:
            raise ValueError("ResearchTeamAgent requires at least one researcher.")
        self.memory = memory

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        subtopics = list(state.get("subtopics", []))
        if not subtopics:
            self.memory.write(query, "drafts", {})
            return {"drafts": {}}

        drafts: dict[str, dict[str, Any]] = {}
        ordered_results: list[tuple[str, dict[str, Any]] | None] = [None] * len(subtopics)
        member_cycle = itertools.cycle(self.members)

        def run_research(idx: int, topic: str, researcher: ResearcherAgent) -> tuple[int, str, dict[str, Any]]:
            payload = researcher.research_topic(topic, query=query)
            payload = {**payload, "topic": topic}
            return idx, topic, payload

        worker_count = min(len(self.members), len(subtopics))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_research, idx, topic, next(member_cycle))
                for idx, topic in enumerate(subtopics, start=1)
            ]
            for future in as_completed(futures):
                idx, topic, payload = future.result()
                ordered_results[idx - 1] = (topic, payload)

        for idx, result in enumerate(ordered_results, start=1):
            if result is None:
                continue
            topic, payload = result
            drafts[topic] = payload
            self.memory.write(query, f"research:{idx}", payload)

        self.memory.write(query, "drafts", drafts)
        return {"drafts": drafts}
