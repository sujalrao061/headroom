"""Score a frozen, hand-authored contract set before claiming model benefit."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from headroom.nebius import propose
from scripts.evaluate_strategies import rule_strategy

STRATEGIES = {"tile_nearest", "tile_radius_count", "no_repair"}


def evaluate(fixtures: dict, live: bool = False) -> dict:
    cases = fixtures["cases"]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)) or not cases:
        raise ValueError("Cases need distinct IDs and at least one entry")
    rows = []
    for case in cases:
        expected = case["expected_strategy"]
        if expected not in STRATEGIES:
            raise ValueError("Invalid expected strategy")
        selected = rule_strategy(case["contract"])
        row = {"id": case["id"], "expected_strategy": expected,
               "rule": {"strategy": selected, "correct": selected == expected}}
        if live:
            start = time.perf_counter()
            response = propose(case["id"], fixtures["shared_source"], case["contract"], fixtures["size"], {}, mode="source_only")
            row["nemotron"] = {"status": response["status"], "strategy": response["strategy"] if response["status"] == "live" else None,
                               "correct": response["strategy"] == expected if response["status"] == "live" else None,
                               "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                               "model": response.get("model"), "usage": response.get("usage"),
                               "error": response.get("explanation") if response["status"] == "error" else None,
                               "http_status": response.get("http_status") if response["status"] == "error" else None}
        rows.append(row)
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": fixtures["protocol"],
            "live_model_requested": live, "total_cases": len(rows),
            "rule_correct": sum(row["rule"]["correct"] for row in rows),
            "model_correct": sum(row["nemotron"]["correct"] is True for row in rows) if live else None,
            "model_completed": sum(row["nemotron"]["status"] == "live" for row in rows) if live else 0,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use an already configured private Nebius key for 12 calls")
    parser.add_argument("--input", default="evidence/heldout-contracts.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.live and not os.getenv("NEBIUS_API_KEY"):
        parser.error("--live needs NEBIUS_API_KEY in this process environment; do not put it in the command line")
    report = evaluate(json.loads(Path(args.input).read_text(encoding="utf-8")), live=args.live)
    output = Path(args.output or ("evidence/heldout-live.json" if args.live else "evidence/heldout-results.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
