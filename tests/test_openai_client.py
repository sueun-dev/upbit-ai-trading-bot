"""Tests for OpenAIClient response parsing (no network calls)."""

import json
import unittest
from unittest.mock import Mock

from src.shared.openai_client import OpenAIClient


def _client() -> OpenAIClient:
    # A dummy key is enough; no request is made during these tests.
    return OpenAIClient(api_key="test-key")


def _response(content: str) -> Mock:
    resp = Mock()
    resp.choices = [Mock(message=Mock(content=content))]
    return resp


class TestExtractJsonFromMarkdown(unittest.TestCase):
    def test_extracts_fenced_json_block(self) -> None:
        client = _client()
        raw = 'prefix\n```json\n{"action": "buy"}\n```\nsuffix'
        extracted = client._extract_json_from_markdown(raw)
        self.assertEqual(json.loads(extracted), {"action": "buy"})

    def test_returns_original_when_no_fence(self) -> None:
        client = _client()
        raw = '{"action": "hold"}'
        self.assertEqual(client._extract_json_from_markdown(raw), raw)


class TestParseResponse(unittest.TestCase):
    def test_parses_dict_json(self) -> None:
        client = _client()
        out = client._parse_response(_response('{"BTC": {"action": "buy"}}'))
        self.assertEqual(out, {"BTC": {"action": "buy"}})

    def test_non_dict_json_is_wrapped(self) -> None:
        # A JSON list is valid JSON but not a decision payload; it must be wrapped
        # so callers always receive a dict (mypy contract + downstream .get()).
        client = _client()
        out = client._parse_response(_response("[1, 2, 3]"))
        self.assertEqual(out, {"response": "[1, 2, 3]"})

    def test_plain_text_is_wrapped(self) -> None:
        client = _client()
        out = client._parse_response(_response("not json at all"))
        self.assertEqual(out, {"response": "not json at all"})


if __name__ == "__main__":
    unittest.main()
