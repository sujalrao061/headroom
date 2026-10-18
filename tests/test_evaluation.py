import unittest
import json
from unittest.mock import patch
from pathlib import Path

from scripts.evaluate_strategies import evaluate, rule_strategy
from scripts.evaluate_heldout import evaluate as evaluate_heldout


class EvaluationTests(unittest.TestCase):
    def test_rule_covers_the_three_known_contracts(self):
        self.assertEqual(rule_strategy("One reference index per query point"), "tile_nearest")
        self.assertEqual(rule_strategy("Every pairwise distance, shape [N, M]"), "no_repair")
        self.assertEqual(rule_strategy("Number of reference points within radius 1.0 for each query"), "tile_radius_count")

    def test_offline_report_has_no_fake_model_results(self):
        saved = {"results": [{"case_id": "nearest", "contract": "One reference index per query point"}]}
        row = evaluate(saved)["rows"][0]
        self.assertTrue(row["rule"]["correct"])
        self.assertNotIn("evidence", row)
        self.assertNotIn("source_only", row)

    def test_heldout_labels_and_missing_model_results(self):
        fixtures = json.loads(Path("evidence/heldout-contracts.json").read_text(encoding="utf-8"))
        report = evaluate_heldout(fixtures)
        self.assertEqual(report["total_cases"], 12)
        self.assertIsNone(report["model_correct"])
        self.assertTrue(all("nemotron" not in row for row in report["rows"]))

    def test_heldout_rejects_duplicate_ids(self):
        fixtures = {"cases": [{"id": "H01", "contract": "x", "expected_strategy": "no_repair"}] * 2}
        with self.assertRaises(ValueError):
            evaluate_heldout(fixtures)

    def test_heldout_retries_transient_failure_and_records_attempts(self):
        fixtures = {"protocol": "test", "shared_source": "source", "size": 8,
                    "cases": [{"id": "H01", "contract": "nearest index per query", "expected_strategy": "tile_nearest"}]}
        responses = [
            {"status": "error", "strategy": "tile_nearest", "http_status": 429, "explanation": "rate limited"},
            {"status": "live", "strategy": "tile_nearest", "model": "test-model", "usage": {}}
        ]
        with patch("scripts.evaluate_heldout.propose", side_effect=responses) as mocked, patch("scripts.evaluate_heldout.time.sleep"):
            report = evaluate_heldout(fixtures, live=True)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(report["model_correct"], 1)
        self.assertEqual(report["model_completed"], 1)
        self.assertEqual(report["rows"][0]["nemotron"]["attempt_count"], 2)
        self.assertEqual(report["rows"][0]["nemotron"]["attempts"][0]["http_status"], 429)

    def test_heldout_does_not_retry_wrong_model_answer(self):
        fixtures = {"protocol": "test", "shared_source": "source", "size": 8,
                    "cases": [{"id": "H01", "contract": "nearest index per query", "expected_strategy": "tile_nearest"}]}
        wrong = {"status": "live", "strategy": "no_repair", "model": "test-model", "usage": {}}
        with patch("scripts.evaluate_heldout.propose", return_value=wrong) as mocked:
            report = evaluate_heldout(fixtures, live=True)
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(report["rows"][0]["nemotron"]["correct"])


if __name__ == "__main__":
    unittest.main()
