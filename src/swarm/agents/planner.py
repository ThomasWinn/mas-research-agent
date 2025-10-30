from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..memory import MemoryStore


class PlannerAgent:
    """Breaks a user query into focused research subtopics."""

    def __init__(
        self,
        llm: Runnable,
        memory: MemoryStore,
        max_subtopics: int = 5,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.max_subtopics = max_subtopics
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research planner. Break the user request into concise subtopics "
                    "that will guide researchers. Focus on coverage and avoid redundancy.",
                ),
                (
                    "user",
                    "User request: {query}\n"
                    "Provide between 3 and {max_subtopics} bullet points outlining the research plan.",
                ),
            ]
        )
        self.parser = StrOutputParser()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        raw_plan: str = (self.prompt | self.llm | self.parser).invoke(
            {"query": query, "max_subtopics": self.max_subtopics}
        )
        subtopics = self._parse_plan(raw_plan)
        self.memory.write(query, "subtopics", subtopics)
        return {"subtopics": subtopics}

    def _parse_plan(self, raw_plan: str) -> list[str]:
        bullet_markers = ("- ", "* ", "• ", "1.", "2.", "3.", "4.", "5.")
        subtopics: list[str] = []
        for line in raw_plan.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for marker in bullet_markers:
                if stripped.startswith(marker):
                    stripped = stripped[len(marker) :].strip()
                    break
            subtopics.append(stripped)
            if len(subtopics) >= self.max_subtopics:
                break
        if not subtopics:
            subtopics = [raw_plan.strip()]
        return subtopics
