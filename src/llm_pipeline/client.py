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
        parsed = self._parse_json_response(response)
        if parsed is not None:
            return parsed

        # One strict retry with an explicit JSON-only suffix.
        # This is a real recovery attempt (not suppression): we ask the model
        # to regenerate strictly valid JSON when the first response is malformed.
        logger.warning("Initial JSON parse failed, retrying with strict JSON instruction")
        strict_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Return ONLY valid JSON. "
            "No markdown, no code fences, no explanation text."
        )
        retry_temp = 0.0 if temperature is None else temperature
        retry_response = self.query(strict_prompt, system, retry_temp)
        parsed_retry = self._parse_json_response(retry_response)
        if parsed_retry is not None:
            logger.info("Recovered valid JSON on retry")
            return parsed_retry

        logger.warning("Failed to parse JSON response after retry")
        logger.debug(f"Raw response (first try): {response[:500]}")
        logger.debug(f"Raw response (retry): {retry_response[:500]}")
        return {
            "error": "json_parse_failed",
            "raw_response": response,
            "retry_raw_response": retry_response,
        }

    def _parse_json_response(self, response: str) -> dict | list | None:
        """Parse JSON from raw model output, tolerating wrapper noise."""
        text = self._strip_markdown_fences(response.strip())

        # Fast path: whole payload is JSON.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Recovery path: find the first valid JSON object/array in the text.
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove optional markdown code fences around a JSON payload."""
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

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
