from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore


class SynthesizerAgent:
    """Combines researcher notes into a cohesive narrative."""

    def __init__(
        self,
        llm: Runnable,
        memory: MemoryStore,
        enable_evaluator: bool = False,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.enable_evaluator = enable_evaluator
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a lead analyst. Merge the researcher summaries into a unified deliverable. "
                    "Highlight consensus, disagreements, and notable data with citations.",
                ),
                (
                    "user",
                    "Original question: {query}\n"
                    "Research notes:\n{notes}\n"
                    "Citation map (use these ids for inline references):\n{citation_map}\n"
                    "Craft a comprehensive yet digestible synthesis. Include an executive summary, "
                    "key insights, and opportunities for further investigation. Append bracketed citation ids "
                    "immediately after every sentence that draws on sourced information, using only ids from the map.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        drafts: dict[str, dict[str, Any]] = state.get("drafts") or self.memory.read(
            query, "drafts", {}
        )
        notes = self._format_notes(drafts)
        citation_entries = self._build_citation_entries(drafts)
        citation_map = json.dumps(citation_entries, ensure_ascii=False, indent=2) if citation_entries else "[]"
        synthesis = (self.prompt | self.llm | self.parser).invoke(
            {"query": query, "notes": notes, "citation_map": citation_map}
        )
        self.memory.write(query, "synthesis", synthesis)
        if citation_entries:
            self.memory.write(query, "citation_entries", citation_entries)
        state_update: dict[str, Any] = {"synthesis": synthesis}
        if citation_entries:
            state_update["citation_entries"] = citation_entries
        return state_update

    def needs_evaluation(self, _: dict[str, Any]) -> Literal["evaluate", "publish"]:
        return "evaluate" if self.enable_evaluator else "publish"

    def _format_notes(self, drafts: dict[str, dict[str, Any]]) -> str:
        if not drafts:
            return "No research drafts available."
        lines: list[str] = []
        for idx, (topic, payload) in enumerate(drafts.items(), start=1):
            summary = payload.get("summary", "")
            lines.append(f"{idx}. {topic}\n{summary}")
        return "\n\n".join(lines)

    def _build_citation_entries(self, drafts: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        counter = 1
        for payload in drafts.values():
            for source in payload.get("sources", []):
                url = source.get("url", "").strip()
                title = source.get("title", "").strip() or "Untitled"
                if not url or url in seen_urls:
                    continue
                entries.append({"id": counter, "title": title, "url": url})
                seen_urls.add(url)
                counter += 1
        return entries
