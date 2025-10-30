from __future__ import annotations

import json
from typing import Any

try:
    import redis
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    redis = None


class MemoryStore:
    """Simple shared memory store backed by Redis when available."""

    def __init__(self, namespace: str = "research", redis_url: str | None = None) -> None:
        self.namespace = namespace
        self._redis = None
        if redis_url and redis:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._cache: dict[str, Any] = {}

    def _compose_key(self, query: str, name: str) -> str:
        return f"{self.namespace}:{hash(query)}:{name}"

    def write(self, query: str, name: str, value: Any) -> None:
        payload = json.dumps(value)
        key = self._compose_key(query, name)
        if self._redis:
            self._redis.set(key, payload)
        else:
            self._cache[key] = payload

    def read(self, query: str, name: str, default: Any = None) -> Any:
        key = self._compose_key(query, name)
        payload: str | None
        if self._redis:
            payload = self._redis.get(key)
        else:
            payload = self._cache.get(key)
        if payload is None:
            return default
        return json.loads(payload)

    def clear(self, query: str) -> None:
        prefix = f"{self.namespace}:{hash(query)}:"
        if self._redis:
            keys = list(self._redis.scan_iter(f"{prefix}*"))
            if keys:
                self._redis.delete(*keys)
        else:
            to_remove = [key for key in self._cache if key.startswith(prefix)]
            for key in to_remove:
                self._cache.pop(key, None)
