from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore


class PublisherAgent:
    """Generates a concise report title and exports the synthesis to Markdown."""

    def __init__(
        self,
        llm: Runnable,
        memory: MemoryStore,
        output_dir: Path,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You craft concise report titles. Keep titles under 8 words, "
                    "informative, and free of punctuation except hyphens. Use underscores for spaces.",
                ),
                (
                    "user",
                    "Research question: {query}\n"
                    "Executive summary (truncated): {summary}\n"
                    "Return only the title text.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        synthesis = state.get("synthesis") or self.memory.read(query, "synthesis", "")
        citations = state.get("citation_entries") or self.memory.read(
            query, "citation_entries", []
        )
        synthesis_with_links = self._inject_citation_links(synthesis, citations)
        summary = synthesis_with_links[:500]
        title = (self.prompt | self.llm | self.parser).invoke(
            {"query": query, "summary": summary}
        ).strip()
        if not title:
            title = "Research Summary"

        filename = self._build_filename(title)
        report_path = self.output_dir / filename
        content = self._build_markdown(title, query, synthesis_with_links)
        report_path.write_text(content, encoding="utf-8")

        self.memory.write(query, "report_path", str(report_path))
        return {"report_path": str(report_path), "report_title": title}

    def _build_filename(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if not slug:
            slug = "research-summary"
        slug = slug[:60].rstrip("-")
        candidate = self.output_dir / f"{slug}.md"
        counter = 2
        while candidate.exists():
            candidate = self.output_dir / f"{slug}-{counter}.md"
            counter += 1
        return candidate.name

    def _build_markdown(self, title: str, query: str, synthesis: str) -> str:
        body = synthesis.strip() or "_No synthesis available._"
        return (
            f"# {title}\n\n"
            f"**Query:** {query}\n\n"
            f"{body}\n"
        )

    def _inject_citation_links(
        self, synthesis: str, citations: list[dict[str, Any]] | None
    ) -> str:
        if not synthesis or not citations:
            return synthesis
        url_map = {
            int(entry.get("id", 0)): entry.get("url", "").strip()
            for entry in citations
            if entry.get("id") and entry.get("url")
        }
        if not url_map:
            return synthesis

        def replacer(match: re.Match[str]) -> str:
            citation_id = int(match.group(1))
            url = url_map.get(citation_id)
            if not url:
                return match.group(0)
            return f"[{citation_id}](<{url}>)"

        return re.sub(r"\[([0-9]+)\](?!\()", replacer, synthesis)
