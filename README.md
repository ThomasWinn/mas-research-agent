# Collaborative Research Swarm

This project demonstrates a multi-agent architecture for collaborative research using [LangGraph](https://github.com/langchain-ai/langgraph). A planner agent decomposes a user's request, specialized researchers gather evidence, a synthesizer merges the insights, and an optional evaluator critiques the final report for factual gaps or bias.

## Features

- **Planner Agent** – generates a coverage-oriented research plan of subtopics.
- **Researcher Agents** – query the web (Tavily by default) and draft concise summaries with citations.
- **Synthesizer Agent** – produces an executive summary and key insights grounded in researcher notes.
- **Evaluator Agent (optional)** – reviews the synthesis for unsupported claims and missing perspectives.
- **Shared Memory** – persisted in Redis when available or an in-memory fallback for local runs.
- **LangGraph Orchestration** – declarative coordination of agent workflow.

## Getting Started

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (use a `.env` file or export them directly):
   ```bash
   export OPENAI_API_KEY="sk-..."           # Required
   export TAVILY_API_KEY="tvly-..."         # Optional but recommended for live web search
   # Optional overrides
   export OPENAI_MODEL="gpt-4o-mini"
   export MAX_SUBTOPICS=5
   export RESEARCHER_BATCH_SIZE=3
   export ENABLE_EVALUATOR=true
   ```

4. Run the swarm against a topic:
   ```bash
   python -m src.run_research "How will multi-agent systems transform scientific research?"
   ```

   Use `--provider noop` to skip web search, or `--no-evaluate` to disable the evaluator.

## Project Layout

- `src/swarm/agents/` – individual agent implementations (planner, researchers, synthesizer, evaluator).
- `src/swarm/memory.py` – shared memory abstraction with Redis support.
- `src/swarm/tools.py` – search client wrapper (Tavily by default).
- `src/swarm/workflow.py` – LangGraph assembly and workflow runner.
- `src/run_research.py` – CLI entry point for end-to-end execution.

## Extending the Swarm

- Swap in custom LLMs by editing `_create_llm` in `src/swarm/workflow.py`.
- Replace the search client with a tool better suited to your domain (e.g., scholarly APIs).
- Persist intermediary artifacts by pointing `MemoryStore` at a Redis instance.
- Add specialized researcher agents (e.g., data extraction, sentiment analysis) and route subtopics accordingly.

## Notes

The repository ships with sensible defaults for experimentation. API usage costs will apply when invoking hosted models or search services. Ensure you comply with provider terms of service.
