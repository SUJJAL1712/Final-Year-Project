"""LLM response caching to minimize API costs."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.config import DATA_DIR


class LLMCache:
    """File-based cache for LLM responses. Saves money on repeated queries."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate a deterministic cache key from the request parameters."""
        content = json.dumps(
            {"prompt": prompt, "model": model, **kwargs},
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str, **kwargs) -> dict | None:
        """Retrieve cached response if it exists."""
        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, prompt: str, model: str, response: dict, **kwargs) -> None:
        """Cache an LLM response."""
        key = self._make_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, default=str)

    def clear(self) -> int:
        """Clear all cached responses. Returns count of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
