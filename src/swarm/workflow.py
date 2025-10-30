from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agents import EvaluatorAgent, PlannerAgent, ResearcherAgent, SynthesizerAgent
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
    researcher: ResearcherAgent,
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
    researcher_model = _create_llm(cfg, temperature=0.1)
    synthesizer_model = _create_llm(cfg, temperature=0.3)
    evaluator_model = _create_llm(cfg, temperature=0.0)

    planner = PlannerAgent(
        planner_model,
        memory=memory,
        max_subtopics=cfg.max_subtopics,
    )
    researcher = ResearcherAgent(
        researcher_model,
        search_client=search_client,
        memory=memory,
        batch_size=cfg.researcher_batch_size,
    )
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


def _create_llm(cfg: Settings, *, temperature: float) -> ChatOpenAI:
    if not cfg.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in your environment or a .env file."
        )
    return ChatOpenAI(
        model=cfg.openai_model,
        temperature=temperature,
        api_key=cfg.openai_api_key,
    )
