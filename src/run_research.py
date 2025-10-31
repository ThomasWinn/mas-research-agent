from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from swarm.config import Settings, settings
from swarm.workflow import run_research_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the collaborative research swarm on a topic."
    )
    parser.add_argument("query", nargs="?", help="Research question or topic.")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to a text file containing the research question.",
    )
    parser.add_argument(
        "--provider",
        choices=("tavily", "noop"),
        default="tavily",
        help="Search provider to use. 'noop' runs without web search.",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable the evaluator agent for critique.",
    )
    return parser.parse_args()


def load_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query.strip()
    if args.file:
        content = args.file.read_text(encoding="utf-8")
        return content.strip()
    raise ValueError("Provide a query argument or --file pointing to a prompt.")


def main() -> None:
    args = parse_args()
    try:
        topic = load_query(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cfg: Settings = settings
    if args.evaluate is not None:
        cfg = replace(cfg, enable_evaluator=args.evaluate)

    try:
        result = run_research_workflow(
            topic,
            config=cfg,
            search_provider=args.provider,
        )
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    state = result.state
    print("=== Research Plan ===")
    for idx, item in enumerate(state.get("subtopics", []), start=1):
        print(f"{idx}. {item}")

    print("\n=== Synthesis ===")
    print(state.get("synthesis", "(empty)"))

    if critique := state.get("critique"):
        print("\n=== Critique ===")
        print(critique)

    if report_path := state.get("report_path"):
        print(f"\nMarkdown report saved to: {report_path}")


if __name__ == "__main__":
    main()
