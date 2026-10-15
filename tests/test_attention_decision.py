import json
import unittest
from pathlib import Path

from scripts.evaluate_attention_decision import evaluate, independent_choice


class AttentionDecisionTests(unittest.TestCase):
    def setUp(self):
        self.raw = Path("evidence/attention-gpu.json").read_bytes()
        self.report = json.loads(self.raw)

    def test_offline_choice_is_independent_of_saved_recommendation(self):
        self.report["recommended"] = "tiled_python"
        self.assertEqual(independent_choice(self.report), "pytorch_sdpa")

    def test_no_model_result_is_claimed_offline(self):
        result = evaluate(self.raw)
        self.assertEqual(result["independent_choice"], "pytorch_sdpa")
        self.assertIsNone(result["model"])
        self.assertIsNone(result["model_agrees"])

    def test_evidence_hash_is_stable_across_windows_line_endings(self):
        lf_result = evaluate(self.raw.replace(b"\r\n", b"\n"))
        crlf_result = evaluate(self.raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertEqual(lf_result["source_sha256"], crlf_result["source_sha256"])

    def test_no_verified_option(self):
        self.report["memory_budget_mib"] = 0
        self.assertEqual(independent_choice(self.report), "no_repair")


if __name__ == "__main__":
    unittest.main()
