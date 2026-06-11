"""Local OAuth-bridge AI client for trading analysis.

The public class keeps the historical ``OpenAIClient`` name so existing modules
do not need a noisy import rewrite, but it no longer uses the OpenAI Platform
SDK or an API key. It sends OpenAI-compatible chat requests to Sueun's local
OAuth bridge.
"""

import json
from typing import Any, Dict, Optional

import requests

from src.infrastructure.config.settings import (
    AI_BRIDGE_API_KEY,
    AI_BRIDGE_BASE_URL,
    AI_BRIDGE_TIMEOUT_SECONDS,
    AI_MAX_TOKENS,
    AI_MODEL,
    AI_REASONING_EFFORT,
)

DEFAULT_TEMPERATURE = 0.3
CHAT_COMPLETIONS_PATH = "/chat/completions"


class OpenAIClient:
    """Client for local OpenAI-compatible OAuth bridge interactions.

    This class provides the one AI boundary used by trading analysis. The bridge
    can be reauthenticated or replaced without touching the analysis modules.

    Attributes:
        base_url: Local bridge base URL, usually ``http://127.0.0.1:8787/v1``.
        model: Bridge model ID to request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """Initialize local OAuth bridge client.

        Args:
            api_key: Optional placeholder token for OpenAI-compatible clients.
                It is not required by the local bridge. Kept for backward
                compatibility with older call sites and tests.
            base_url: OpenAI-compatible bridge base URL.
            model: Bridge model ID.
            reasoning_effort: Bridge reasoning effort hint.
            timeout_seconds: HTTP timeout for a single AI request.
        """
        self.base_url = (base_url or AI_BRIDGE_BASE_URL).rstrip("/")
        self.model = model or AI_MODEL
        self.reasoning_effort = reasoning_effort or AI_REASONING_EFFORT
        self.timeout_seconds = timeout_seconds or AI_BRIDGE_TIMEOUT_SECONDS
        self.api_key = api_key or AI_BRIDGE_API_KEY

    def analyze_with_prompt(
        self, prompt: str, system_message: str, temperature: float = DEFAULT_TEMPERATURE
    ) -> Dict[str, Any]:
        """Analyze data using the local OAuth bridge with a custom prompt.

        Args:
            prompt: User prompt for analysis.
            system_message: System message defining AI role.
            temperature: Randomness in AI response (0.0 - 1.0).

        Returns:
            Parsed JSON response from OpenAI.

        Raises:
            Exception: If analysis fails or response is invalid.
        """
        response_payload = self._post_chat_completion(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": AI_MAX_TOKENS,
                "reasoning_effort": self.reasoning_effort,
            }
        )

        return self._parse_response(response_payload)

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST a chat completion request to the local bridge."""
        url = f"{self.base_url}{CHAT_COMPLETIONS_PATH}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"AI bridge request failed: {exc}. Is the OAuth bridge running at {self.base_url}?"
            ) from exc

        if response.status_code >= 400:
            message = self._extract_error_message(response)
            raise RuntimeError(
                f"AI bridge request failed with HTTP {response.status_code}: {message}"
            )

        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError("AI bridge returned non-JSON response") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("AI bridge returned an unexpected response shape")
        return parsed

    def _extract_error_message(self, response: requests.Response) -> str:
        """Extract a concise error message from a failed bridge response."""
        try:
            payload = response.json()
        except ValueError:
            return response.text[:500]

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
        return str(payload)[:500]

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse OpenAI-compatible API response.

        Args:
            response: Raw bridge response dictionary or old SDK-like test object.

        Returns:
            Parsed response as dictionary.
        """
        content = self._extract_content(response)

        # Try to extract JSON from markdown code blocks
        json_content = self._extract_json_from_markdown(content)

        try:
            parsed = json.loads(json_content)
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {"response": content}
        # Only dict-shaped JSON is a valid decision/result payload; wrap anything else.
        return parsed if isinstance(parsed, dict) else {"response": content}

    def _extract_content(self, response: Any) -> str:
        """Extract assistant text from bridge dicts or old SDK-like objects."""
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        return str(message.get("content") or "")
                    if "text" in first_choice:
                        return str(first_choice.get("text") or "")

            if "output_text" in response:
                return str(response.get("output_text") or "")
            if "response" in response:
                return str(response.get("response") or "")

            raise RuntimeError(
                "AI bridge response is missing choices[0].message.content"
            )

        # Backward-compatible path for the existing parser unit tests.
        return str(response.choices[0].message.content)

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks.

        Args:
            content: Raw content that may contain JSON in markdown.

        Returns:
            Extracted JSON string or original content.
        """
        # Look for ```json ... ``` or ``` ... ``` patterns
        lines = content.strip().split("\n")

        # Find start and end of code block
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if start_idx is None:
                    start_idx = i + 1  # Skip the ``` line
                else:
                    end_idx = i  # Don't include the closing ``` line
                    break

        if start_idx is not None and end_idx is not None:
            # Extract content between code blocks
            json_lines = lines[start_idx:end_idx]
            return "\n".join(json_lines)

        # If no code blocks found, return original content
        return content
