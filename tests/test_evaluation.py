import unittest
import json
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


if __name__ == "__main__":
    unittest.main()
