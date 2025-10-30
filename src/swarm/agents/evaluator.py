from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore


class EvaluatorAgent:
    """Reviews the synthesis for factual gaps, bias, or unanswered questions."""

    def __init__(self, llm: Runnable, memory: MemoryStore) -> None:
        self.llm = llm
        self.memory = memory
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a critical reviewer. Inspect the synthesis for unsupported claims, "
                    "missing evidence, and potential bias.",
                ),
                (
                    "user",
                    "Original question: {query}\n"
                    "Synthesis:\n{synthesis}\n"
                    "Research notes:\n{notes}\n"
                    "Provide:\n"
                    "1. Validation of well-supported insights.\n"
                    "2. Flagged claims needing verification.\n"
                    "3. Missing perspectives or follow-up questions.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        synthesis: str = state.get("synthesis") or self.memory.read(query, "synthesis", "")
        drafts = state.get("drafts") or self.memory.read(query, "drafts", {})
        notes = self._format_notes(drafts)
        critique = (self.prompt | self.llm | self.parser).invoke(
            {"query": query, "synthesis": synthesis, "notes": notes}
        )
        self.memory.write(query, "critique", critique)
        return {"critique": critique}

    def _format_notes(self, drafts: dict[str, dict[str, Any]]) -> str:
        if not drafts:
            return "No researcher notes available."
        lines: list[str] = []
        for idx, (topic, payload) in enumerate(drafts.items(), start=1):
            summary = payload.get("summary", "")
            lines.append(f"{idx}. {topic}\n{summary}")
        return "\n\n".join(lines)
