from __future__ import annotations

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
                    "Craft a comprehensive yet digestible synthesis. Include an executive summary, "
                    "key insights, and opportunities for further investigation.",
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
        synthesis = (self.prompt | self.llm | self.parser).invoke(
            {"query": query, "notes": notes}
        )
        self.memory.write(query, "synthesis", synthesis)
        return {"synthesis": synthesis}

    def needs_evaluation(self, _: dict[str, Any]) -> Literal["evaluate", "end"]:
        return "evaluate" if self.enable_evaluator else "end"

    def _format_notes(self, drafts: dict[str, dict[str, Any]]) -> str:
        if not drafts:
            return "No research drafts available."
        lines: list[str] = []
        for idx, (topic, payload) in enumerate(drafts.items(), start=1):
            summary = payload.get("summary", "")
            lines.append(f"{idx}. {topic}\n{summary}")
        return "\n\n".join(lines)
