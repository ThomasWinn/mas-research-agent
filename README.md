# Collaborative Research Swarm

This project demonstrates a LangGraph-powered, multi-agent workflow for evidence-based research. A planner breaks down the question, specialized researchers gather evidence in parallel, a synthesizer produces a structured report, an optional evaluator critiques it, and a publisher exports the result as Markdown.

## Features
- **Planner Agent** – decomposes the request into coverage-focused subtopics.
- **Parallel Research Team** – five specialists pull fresh sources (Tavily by default) and draft JSON-structured notes simultaneously.
- **Synthesizer Agent** – merges researcher drafts into an executive summary with inline citations.
- **Publisher Agent** – generates a concise title and saves the synthesis as a Markdown file with clickable citations.
- **Evaluator Agent (optional)** – flags unsupported claims or missing perspectives before publishing.
- **Shared Memory** – transparently backed by Redis when available; falls back to in-process storage.
- **LM Studio Compatibility** – ships with defaults for running Qwen models locally via the LM Studio server.

## Getting Started
1. **Install LM Studio and models**
   - Download [LM Studio](https://lmstudio.ai/) and load:
     - `qwen2.5-7b-instruct-mlx@8bit`
     - `qwen2.5-32b-instruct-mlx`
   - Start the local server (`Settings → Developer → Enable local server`) and ensure it listens at `http://127.0.0.1:1234/v1`.

2. **Configure environment variables**
   - Copy `.env.template` to `.env` and set:
     ```bash
     OPENAI_API_KEY=placeholder-key          # LM Studio ignores the value but it must be non-empty
     LM_STUDIO_SERVER=http://127.0.0.1:1234/v1
     RESEARCHER_4BIT_MODEL=qwen2.5-7b-instruct-mlx@4bit
     RESEARCHER_8BIT_MODEL=qwen2.5-7b-instruct-mlx@8bit
     SYNTH_EVAL_4BIT_MODEL=qwen2.5-32b-instruct-mlx
     TAVILY_API_KEY=tvly-...                 # Optional; needed for live web search
     ```

3. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Run the workflow**
   ```bash
   cd src
   python run_research.py "How will multi-agent systems transform scientific research?"
   ```
   Use `--provider noop` to skip web search or `--no-evaluate` to bypass the critique pass. The run saves a Markdown report alongside `src/run_research.py`.

## Project Layout
- `src/run_research.py` – CLI entry point for end-to-end execution.
- `src/swarm/workflow.py` – LangGraph assembly and LLM/search configuration.
- `src/swarm/agents/` – planner, researcher, synthesizer, evaluator, and publisher implementations.
- `src/swarm/tools.py` – Tavily search wrapper; swaps to a noop client when no key is supplied.
- `src/swarm/memory.py` – Redis-backed shared memory with in-memory fallback.

## Researcher Parallelization
`ResearchTeamAgent` (`src/swarm/agents/researcher.py`) spins up a `ThreadPoolExecutor` sized to `min(team_size, subtopic_count)` and cycles through the researcher profiles. Each subtopic is assigned round-robin to an agent, executed concurrently, and the results are written back into a deterministic order before being persisted to shared memory. This keeps sourcing fast even with larger plans while preserving the planner’s original ordering.

## Extending the Swarm
- Swap the LM Studio models for hosted APIs by updating `_create_llm` in `src/swarm/workflow.py`.
- Add additional specialists by appending profiles to `researcher_profiles`.
- Persist outputs elsewhere by overriding `PublisherAgent` or pointing `MemoryStore` at Redis.

API usage costs apply when invoking external providers such as Tavily. Ensure you comply with each provider’s terms of service.
