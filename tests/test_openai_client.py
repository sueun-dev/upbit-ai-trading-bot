"""Tests for OpenAIClient response parsing (no network calls)."""

import json
import unittest
from unittest.mock import Mock, patch

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

    def test_parses_openai_compatible_bridge_dict(self) -> None:
        client = _client()
        out = client._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"BTC": {"action": "hold"}}',
                        }
                    }
                ]
            }
        )
        self.assertEqual(out, {"BTC": {"action": "hold"}})


class TestBridgeHttpClient(unittest.TestCase):
    @patch("src.shared.openai_client.requests.post")
    def test_posts_to_local_bridge_chat_completions(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}}]
        }
        mock_post.return_value = response

        client = OpenAIClient(
            base_url="http://127.0.0.1:8787/v1",
            model="gpt-5.5",
            reasoning_effort="xhigh",
            timeout_seconds=3,
        )
        result = client.analyze_with_prompt("hi", "Return JSON only.", temperature=0)

        self.assertEqual(result, {"ok": True})
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(
            mock_post.call_args.args[0],
            "http://127.0.0.1:8787/v1/chat/completions",
        )
        self.assertEqual(kwargs["json"]["model"], "gpt-5.5")
        self.assertEqual(kwargs["json"]["reasoning_effort"], "xhigh")

    @patch("src.shared.openai_client.requests.post")
    def test_bridge_http_errors_are_clear(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 401
        response.json.return_value = {
            "error": {"message": "Provided authentication token is expired."}
        }
        mock_post.return_value = response

        client = OpenAIClient(base_url="http://127.0.0.1:8787/v1")

        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            client.analyze_with_prompt("hi", "system")


if __name__ == "__main__":
    unittest.main()
