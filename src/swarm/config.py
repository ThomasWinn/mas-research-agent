from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the research swarm."""

    openai_model: str = field(default="gpt-4o-mini")
    openai_api_key: str | None = field(default=None)
    tavily_api_key: str | None = field(default=None)
    max_subtopics: int = field(default=5)
    researcher_batch_size: int = field(default=3)
    researcher_4bit_model: str = field(default="qwen2.5-7b-instruct-q4")
    researcher_8bit_model: str = field(default="qwen2.5-7b-instruct-q8")
    enable_evaluator: bool = field(default=False)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            max_subtopics=_env_int("MAX_SUBTOPICS", 5),
            researcher_batch_size=_env_int("RESEARCHER_BATCH_SIZE", 3),
            researcher_4bit_model=os.getenv("RESEARCHER_4BIT_MODEL", "qwen2.5-7b-instruct-q4"),
            researcher_8bit_model=os.getenv("RESEARCHER_8BIT_MODEL", "qwen2.5-7b-instruct-q8"),
            enable_evaluator=_env_flag("ENABLE_EVALUATOR", False),
        )


settings = Settings.from_env()
