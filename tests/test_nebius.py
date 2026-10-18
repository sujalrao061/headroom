import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from headroom.nebius import _strategy_json, propose, propose_attention


class NebiusAdapterTests(unittest.TestCase):
    def test_unconfigured_call_is_labeled(self):
        with patch.dict(os.environ, {"NEBIUS_API_KEY": ""}):
            result = propose("nearest", "code", "indices", 256, {})
        self.assertEqual(result["status"], "not_configured")

    def test_live_response_is_bounded_to_strategy(self):
        response = {"choices": [{"message": {"content": json.dumps({"strategy": "no_repair", "explanation": "Full matrix required"})}}], "usage": {"total_tokens": 25}}
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as called:
            result = propose("matrix", "code", "full matrix", 256, {"peak_allocated_mb": 70})
        self.assertEqual(result["status"], "live")
        self.assertEqual(result["strategy"], "no_repair")
        self.assertIn("Bearer test-token", called.call_args.args[0].headers["Authorization"])

    def test_http_error_reports_status_without_server_body(self):
        error = urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, io.BytesIO(b"sensitive response"))
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", side_effect=error):
            result = propose("nearest", "code", "indices", 256, {})
        self.assertEqual(result["http_status"], 403)
        self.assertNotIn("sensitive response", json.dumps(result))

    def test_strategy_object_inside_markdown_is_parsed(self):
        response = {"choices": [{"message": {"content": "Here is the strategy:\n```json\n{\"strategy\":\"tile_nearest\",\"explanation\":\"Bound intermediates\"}\n```"}}]}
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            result = propose("nearest", "code", "indices", 256, {})
        self.assertEqual(result["status"], "live")
        self.assertEqual(result["strategy"], "tile_nearest")

    def test_truncated_json_recovers_only_explicit_allowlisted_strategy(self):
        self.assertEqual(_strategy_json('{"strategy":"no_repair","explanation":"need a sentinel') ["strategy"], "no_repair")
        with self.assertRaises(ValueError):
            _strategy_json("I think no_repair is probably right")

    def test_request_allows_reasoning_budget_and_names_extra_semantics(self):
        response = {"choices": [{"message": {"content": '{"strategy":"no_repair","explanation":"Conditional sentinel output"}'}}]}
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as called:
            propose("nearest", "code", "conditional index or -1", 256, {}, mode="source_only")
        body = json.loads(called.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertIn("sentinels", body["messages"][0]["content"])

    def test_radius_strategy_is_accepted(self):
        response = {"choices": [{"message": {"content": '{"strategy":"tile_radius_count","explanation":"Accumulate counts"}'}}]}
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            result = propose("radius", "code", "counts", 256, {})
        self.assertEqual(result["status"], "live")
        self.assertEqual(result["strategy"], "tile_radius_count")

    def test_attention_choice_is_bounded_and_does_not_send_verdict(self):
        report = {"shape": {"sequence_length": 64}, "memory_budget_mib": 8, "checks": [],
                  "baseline": {"latency_ms": 2}, "candidate": {"latency_ms": 50}, "library": {"latency_ms": 1},
                  "baseline_within_budget": False, "candidate_within_budget": True, "library_within_budget": True, "recommended": "pytorch_sdpa"}
        response = {"choices": [{"message": {"content": '{"strategy":"pytorch_sdpa","explanation":"Faster validated option"}'}}]}
        with patch.dict(os.environ, {"NEBIUS_API_KEY": "test-token"}), patch("headroom.nebius.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as called:
            result = propose_attention(report)
        self.assertEqual(result["strategy"], "pytorch_sdpa")
        sent = json.loads(called.call_args.args[0].data)
        self.assertNotIn('"recommended"', sent["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
