from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agents import (
    EvaluatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ResearchTeamAgent,
    SynthesizerAgent,
    PublisherAgent,
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

BASE_SYSTEM = dedent(
    """You are a research agent. Follow these guardrails:
• Ground every claim in the provided snippets only; never infer beyond them.
• Quote short spans when possible and include inline citations like [1], [2].
• Use the exact JSON schema:
{{
"subtopic":"...",
"claims":[{{"text":"...","quote":"...","citation_id":1}}],
"citation_map":[{{"id":1,"url":"...","title":"..."}}],
"confidence":0.0_to_1.0,
"tokens_read":1234
}}
• Keep outputs concise; no intro/outro prose; no markdown fences.
• Respect per-doc token caps; do not exceed the context budget."""
)


class ResearchState(TypedDict, total=False):
    query: str
    subtopics: list[str]
    drafts: dict[str, dict[str, Any]]
    synthesis: str
    citation_entries: list[dict[str, Any]]
    critique: str
    report_path: str
    report_title: str


@dataclass(slots=True)
class ResearchResult:
    state: ResearchState
    memory: MemoryStore
    graph: Any


def build_research_graph(
    planner: PlannerAgent,
    researcher: ResearcherAgent | ResearchTeamAgent,
    synthesizer: SynthesizerAgent,
    publisher: PublisherAgent,
    evaluator: EvaluatorAgent | None = None,
) -> Any:
    builder = StateGraph(ResearchState)
    builder.add_node("plan", planner)
    builder.add_node("research", researcher)
    builder.add_node("synthesize", synthesizer)
    builder.add_node("publish", publisher)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "synthesize")

    if evaluator:
        builder.add_node("evaluate", evaluator)
        builder.add_conditional_edges(
            "synthesize",
            synthesizer.needs_evaluation,
            {"evaluate": "evaluate", "publish": "publish"},
        )
        builder.add_edge("evaluate", "publish")
    else:
        builder.add_edge("synthesize", "publish")

    builder.add_edge("publish", END)

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

    planner_model = _create_llm(cfg, temperature=0.2, model=cfg.researcher_8bit_model)
    synthesizer_model = _create_llm(cfg, temperature=0.3, model=cfg.synth_eval_model)
    evaluator_model = _create_llm(cfg, temperature=0.0, model=cfg.synth_eval_model)
    output_dir = Path(__file__).resolve().parent.parent

    planner = PlannerAgent(
        planner_model,
        memory=memory,
        max_subtopics=cfg.max_subtopics,
    )
    researcher_profiles = [
    # === Scouts (4-bit, breadth, low-noise, same temp) ===
    {
        "name": "scout-alpha",
        "model": cfg.researcher_4bit_model,
        "temperature": 0.14,
        "top_p": 0.95,
        "system_prompt": BASE_SYSTEM + "\nROLE: Fast-turnaround evidence scout; prefer crisp, atomic facts.",
        "user_prompt": dedent(
            """\
            Topic: {topic}

            Sources:
            {sources}
            Task: Produce EXACTLY three short factual sentences, each with an inline citation [id].
            Return the JSON schema only."""
        ),
    },
    {
        "name": "scout-beta",
        "model": cfg.researcher_4bit_model,
        "temperature": 0.14,
        "top_p": 0.95,
        "system_prompt": BASE_SYSTEM + "\nROLE: Coverage-focused scout; surface the biggest takeaways and divergent viewpoints.",
        "user_prompt": dedent(
            """\
            Topic: {topic}

            Sources:
            {sources}
            Task: Summarize the top findings in 3-5 sentences with inline citations [id].
            Return the JSON schema only."""
        ),
    },
    {
        "name": "scout-gamma",
        "model": cfg.researcher_4bit_model,
        "temperature": 0.14,
        "top_p": 0.95,
        "system_prompt": BASE_SYSTEM + "\nROLE: Precision note-taker; prioritize numbers, dates, and named entities.",
        "user_prompt": dedent(
            """\
            Topic: {topic}

            Sources:
            {sources}
            Task: List 3-4 key facts emphasizing statistics/dates, with inline citations [id].
            Return the JSON schema only."""
        ),
    },

    # === Heavies (8-bit, depth/verification, even lower temp) ===
    {
        "name": "analyst-delta",
        "model": cfg.researcher_8bit_model,
        "temperature": 0.09,
        "top_p": 0.90,
        "system_prompt": BASE_SYSTEM + "\nROLE: Senior analyst; connect themes, call out contradictions, keep to evidence.",
        "user_prompt": dedent(
            """\
            Topic: {topic}

            Sources:
            {sources}
            Task: Write a cohesive mini-brief (4-6 sentences) that highlights agreements/conflicts.
            Every sentence must have at least one inline citation [id]. Return the JSON schema only."""
        ),
    },
    {
        "name": "analyst-epsilon",
        "model": cfg.researcher_8bit_model,
        "temperature": 0.09,
        "top_p": 0.90,
        "system_prompt": BASE_SYSTEM + "\nROLE: Verification-minded analyst; separate well-supported facts from tentative items.",
        "user_prompt": dedent(
            """\
            Topic: {topic}

            Sources:
            {sources}
            Task: Produce a cautious summary (4-5 sentences) that labels uncertainties and grades reliability.
            Cite each sentence with [id]. Return the JSON schema only."""
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
    publisher = PublisherAgent(
        synthesizer_model,
        memory=memory,
        output_dir=output_dir,
    )
    evaluator = (
        EvaluatorAgent(evaluator_model, memory=memory) if cfg.enable_evaluator else None
    )

    graph = build_research_graph(planner, researcher, synthesizer, publisher, evaluator)
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
    base_url = cfg.lm_studio_server
    api_key = cfg.openai_api_key or ("lm-studio" if base_url else None)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in your environment or a .env file."
        )
    return ChatOpenAI(
        model=model or cfg.openai_model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
