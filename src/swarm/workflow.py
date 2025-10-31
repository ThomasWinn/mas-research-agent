from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agents import (
    EvaluatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ResearchTeamAgent,
    SynthesizerAgent,
)
from .config import Settings, settings
from .memory import MemoryStore
from .tools import SearchClient

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "langchain-openai is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


class ResearchState(TypedDict, total=False):
    query: str
    subtopics: list[str]
    drafts: dict[str, dict[str, Any]]
    synthesis: str
    critique: str


@dataclass(slots=True)
class ResearchResult:
    state: ResearchState
    memory: MemoryStore
    graph: Any


def build_research_graph(
    planner: PlannerAgent,
    researcher: ResearcherAgent | ResearchTeamAgent,
    synthesizer: SynthesizerAgent,
    evaluator: EvaluatorAgent | None = None,
) -> Any:
    builder = StateGraph(ResearchState)
    builder.add_node("plan", planner)
    builder.add_node("research", researcher)
    builder.add_node("synthesize", synthesizer)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")

    if evaluator:
        builder.add_node("evaluate", evaluator)
        builder.add_conditional_edges(
            "synthesize",
            synthesizer.needs_evaluation,
            {"evaluate": "evaluate", "end": END},
        )
        builder.add_edge("evaluate", END)
    else:
        builder.add_edge("synthesize", END)

    return builder.compile()


def run_research_workflow(
    query: str,
    *,
    config: Settings | None = None,
    search_provider: Literal["tavily", "noop"] = "tavily",
) -> ResearchResult:
    cfg = config or settings
    memory = MemoryStore()
    search_client = SearchClient(
        provider=search_provider,
        api_key=cfg.tavily_api_key,
        default_k=cfg.researcher_batch_size,
    )

    planner_model = _create_llm(cfg, temperature=0.2)
    synthesizer_model = _create_llm(cfg, temperature=0.3)
    evaluator_model = _create_llm(cfg, temperature=0.0)

    planner = PlannerAgent(
        planner_model,
        memory=memory,
        max_subtopics=cfg.max_subtopics,
    )
    researcher_profiles = [
        {
            "name": "scout-alpha",
            "model": cfg.researcher_4bit_model,
            "temperature": 0.15,
            "system_prompt": (
                "You are a fast-turnaround evidence scout using a compressed Qwen 7B model. "
                "Stay grounded in the provided snippets and favor concise factual statements."
            ),
            "user_prompt": (
                "Topic: {topic}\n"
                "Sources:\n{sources}\n"
                "Produce 3 short factual sentences with inline citations [1], [2], etc."
            ),
        },
        {
            "name": "scout-beta",
            "model": cfg.researcher_4bit_model,
            "temperature": 0.18,
            "system_prompt": (
                "You are a coverage-focused researcher operating on a 4-bit Qwen model. "
                "Capture the biggest takeaways and surface divergent viewpoints without embellishment."
            ),
            "user_prompt": (
                "Topic: {topic}\n"
                "Sources:\n{sources}\n"
                "Summarize the top findings in 3-5 sentences. Cite sources inline as [1], [2], etc."
            ),
        },
        {
            "name": "scout-gamma",
            "model": cfg.researcher_4bit_model,
            "temperature": 0.12,
            "system_prompt": (
                "You are a precision note-taker running on a lightweight Qwen configuration. "
                "Prioritize crisp facts, statistics, and dates drawn directly from the snippets."
            ),
            "user_prompt": (
                "Topic: {topic}\n"
                "Sources:\n{sources}\n"
                "List 3-4 key facts with inline citations [1], [2], etc."
            ),
        },
        {
            "name": "analyst-delta",
            "model": cfg.researcher_8bit_model,
            "temperature": 0.1,
            "system_prompt": (
                "You are a senior analyst with more capacity (8-bit Qwen). "
                "Synthesize nuanced insights, note contradictions, and connect themes."
            ),
            "user_prompt": (
                "Topic: {topic}\n"
                "Sources:\n{sources}\n"
                "Deliver a cohesive mini-brief (4-6 sentences) with inline citations, highlighting agreements or conflicts."
            ),
        },
        {
            "name": "analyst-epsilon",
            "model": cfg.researcher_8bit_model,
            "temperature": 0.08,
            "system_prompt": (
                "You are a verification-minded analyst operating on an 8-bit Qwen model. "
                "Cross-check the evidence, call out uncertainties, and emphasize reliability."
            ),
            "user_prompt": (
                "Topic: {topic}\n"
                "Sources:\n{sources}\n"
                "Write a cautious summary (4-5 sentences) that distinguishes well-supported facts from tentative items. "
                "Use inline citations [1], [2], etc."
            ),
        },
    ]
    researcher_members = [
        ResearcherAgent(
            profile["name"],
            llm=_create_llm(
                cfg,
                temperature=profile["temperature"],
                model=profile["model"],
            ),
            search_client=search_client,
            batch_size=cfg.researcher_batch_size,
            system_prompt=profile["system_prompt"],
            user_prompt=profile["user_prompt"],
        )
        for profile in researcher_profiles
    ]
    researcher = ResearchTeamAgent(researcher_members, memory=memory)
    synthesizer = SynthesizerAgent(
        synthesizer_model,
        memory=memory,
        enable_evaluator=cfg.enable_evaluator,
    )
    evaluator = (
        EvaluatorAgent(evaluator_model, memory=memory) if cfg.enable_evaluator else None
    )

    graph = build_research_graph(planner, researcher, synthesizer, evaluator)
    state: ResearchState = {
        "query": query,
        "subtopics": [],
        "drafts": {},
        "synthesis": "",
        "critique": "",
    }
    final_state = graph.invoke(state)
    memory.write(query, "final_state", final_state)
    return ResearchResult(state=final_state, memory=memory, graph=graph)


def _create_llm(cfg: Settings, *, temperature: float, model: str | None = None) -> ChatOpenAI:
    if not cfg.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in your environment or a .env file."
        )
    return ChatOpenAI(
        model=model or cfg.openai_model,
        temperature=temperature,
        api_key=cfg.openai_api_key,
    )
