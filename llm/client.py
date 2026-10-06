"""
LLM Client — OpenAI-compatible wrapper with function-calling support.

Works with any OpenAI-compatible API (OpenAI, Azure, local Ollama, etc.)
by accepting a configurable base_url. The client handles tool/function
calling and returns structured results.
"""

import json
import os
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible LLM client with function calling support."""

    def __init__(self, api_key=None, model=None, base_url=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None  # lazy-initialised on first use

    def _get_client(self):
        """Lazy-initialise the OpenAI client so the server can start
        without an API key and accept it later."""
        if self._client is None:
            # Re-read env in case it was set after init
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "No API key configured. Set OPENAI_API_KEY in your .env "
                    "file or pass it when creating the LLMClient."
                )
            kwargs = {"api_key": api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def set_api_key(self, api_key: str):
        """Allow setting the API key at runtime (e.g. from the UI)."""
        self.api_key = api_key
        self._client = None  # force re-init

    def chat(self, system_prompt, messages, tools=None, temperature=0.7):
        """Send a chat completion request with optional function calling.

        Args:
            system_prompt: The composed multi-layer system prompt.
            messages: Conversation history (list of role/content dicts).
            tools: Phase-gated list of tool schemas (or None).
            temperature: Sampling temperature.

        Returns:
            dict with:
              - content  (str): The natural-language response text.
              - tool_calls (list[dict]): Each has id, name, arguments.
        """
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("LLM request: model=%s, tools=%d", self.model, len(tools or []))

        response = self._get_client().chat.completions.create(**kwargs)
        message = response.choices[0].message

        result = {"content": message.content or "", "tool_calls": []}

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result["tool_calls"].append(
                    {"id": tc.id, "name": tc.function.name, "arguments": args}
                )

        logger.debug(
            "LLM response: content_len=%d, tool_calls=%d",
            len(result["content"]),
            len(result["tool_calls"]),
        )

        return result
