"""
Claude API client wrapper with caching and rate limiting.

Provides a unified interface for all LLM calls in the project,
with built-in caching to minimize API costs during development.
"""

import os
import time
import json
from typing import Any

import anthropic
from loguru import logger

from src.utils.cache import LLMCache
from src.utils.config import get_config


class ClaudeClient:
    """Wrapper around Anthropic's Claude API with caching and rate limiting."""

    def __init__(self):
        config = get_config().llm_config
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        self.requests_per_minute = config["requests_per_minute"]

        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.cache = LLMCache() if config["cache_enabled"] else None
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def query(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a query to Claude and return the text response.

        Args:
            prompt: The user message
            system: System prompt for context
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Claude's text response
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        # Check cache
        if self.cache:
            cached = self.cache.get(prompt, self.model, system=system, temperature=temp)
            if cached:
                logger.debug("Cache hit for LLM query")
                return cached["response"]

        self._rate_limit()

        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": tokens,
                "temperature": temp,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)
            text = response.content[0].text

            # Cache response
            if self.cache:
                self.cache.set(
                    prompt, self.model,
                    {"response": text, "usage": {"input": response.usage.input_tokens,
                                                  "output": response.usage.output_tokens}},
                    system=system, temperature=temp,
                )

            logger.debug(f"LLM query: {response.usage.input_tokens}in/{response.usage.output_tokens}out tokens")
            return text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def query_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> dict | list:
        """
        Query Claude and parse the response as JSON.
        The prompt should instruct Claude to respond in JSON format.
        """
        response = self.query(prompt, system, temperature)

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            return {"error": "json_parse_failed", "raw_response": response}

    def batch_query(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float | None = None,
    ) -> list[str]:
        """Process multiple prompts sequentially with rate limiting."""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch query {i+1}/{len(prompts)}")
            result = self.query(prompt, system, temperature)
            results.append(result)
        return results
