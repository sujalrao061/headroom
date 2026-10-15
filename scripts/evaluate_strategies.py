"""Compare strategy selection on saved local evidence without exposing an API key."""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from headroom.nebius import propose

EXPECTED = {"nearest": "tile_nearest", "matrix": "no_repair", "radius": "tile_radius_count"}


def rule_strategy(contract: str) -> str:
    """A deliberately simple contract-text baseline, fixed before evaluation."""
    if "Every pairwise distance" in contract:
        return "no_repair"
    if "within radius" in contract:
        return "tile_radius_count"
    if "index per query" in contract:
        return "tile_nearest"
    return "no_repair"


def evaluate(saved: dict, live: bool = False) -> dict:
    rows = []
    for result in saved["results"]:
        case_id = result["case_id"]
        expected = EXPECTED[case_id]
        rule = rule_strategy(result["contract"])
        row = {"case_id": case_id, "expected_strategy": expected,
               "rule": {"strategy": rule, "correct": rule == expected}}
        if live:
            evidence = {key: result[key] for key in ("baseline", "candidate", "library", "checks", "memory_limit_mb", "latency_limit_ms", "failure_kind")}
            for mode in ("source_only", "evidence"):
                start = time.perf_counter()
                response = propose(case_id, result["source"], result["contract"], result["size"], evidence, mode=mode)
                row[mode] = {"status": response["status"], "strategy": response["strategy"],
                             "correct": response["strategy"] == expected if response["status"] == "live" else None,
                             "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                             "model": response.get("model"), "usage": response.get("usage")}
        rows.append(row)
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Three known built-in workloads; exploratory strategy-selection comparison, not unfamiliar-workload generalization",
            "live_model_requested": live, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call Nebius with the private key already set in this process environment")
    parser.add_argument("--input", default="evidence/local-gpu.json")
    parser.add_argument("--output", default="evidence/strategy-comparison.json")
    args = parser.parse_args()
    if args.live and not os.getenv("NEBIUS_API_KEY"):
        parser.error("--live needs NEBIUS_API_KEY in this process environment; do not put it in the command line")
    report = evaluate(json.loads(Path(args.input).read_text(encoding="utf-8")), live=args.live)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
