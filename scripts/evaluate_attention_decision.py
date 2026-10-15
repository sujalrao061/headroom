"""Compare a deterministic measured choice with one private Nemotron call."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from headroom.nebius import propose_attention


def independent_choice(report: dict) -> str:
    budget = report["memory_budget_mib"]
    if report["device"].startswith("CPU"):
        return "no_repair"
    checks = report["checks"]
    options = []
    for name, field, valid in (
        ("keep_baseline", "baseline", True),
        ("tiled_python", "candidate", all(c["passed"] for c in checks)),
        ("pytorch_sdpa", "library", all(c["library_passed"] for c in checks)),
    ):
        metrics = report[field]
        memory = metrics["peak_allocated_mb"]
        if valid and memory is not None and memory <= budget:
            options.append((metrics["latency_ms"], name))
    return min(options)[1] if options else "no_repair"


def evaluate(raw: bytes, live: bool = False) -> dict:
    report = json.loads(raw)
    selected = independent_choice(report)
    result = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "scope": "One saved local attention measurement; strategy selection only, not a new GPU run",
              "independent_choice": selected, "live_model_requested": live,
              "model": None, "model_agrees": None}
    if live:
        # Older saved reports did not include this field. Derive it from raw metrics.
        report["baseline_within_budget"] = bool(report["baseline"]["peak_allocated_mb"] is not None
                                                 and report["baseline"]["peak_allocated_mb"] <= report["memory_budget_mib"]
                                                 and not report["device"].startswith("CPU"))
        response = propose_attention(report)
        result["model"] = {"status": response["status"],
                           "strategy": response["strategy"] if response["status"] == "live" else None,
                           "model_id": response.get("model"), "usage": response.get("usage"),
                           "error": response.get("explanation") if response["status"] == "error" else None}
        result["model_agrees"] = response["strategy"] == selected if response["status"] == "live" else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Make one Nebius call with a private key already in the process environment")
    parser.add_argument("--input", default="evidence/attention-gpu.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.live and not os.getenv("NEBIUS_API_KEY"):
        parser.error("--live needs NEBIUS_API_KEY in this process environment; do not put it in the command line")
    result = evaluate(Path(args.input).read_bytes(), live=args.live)
    output = Path(args.output or ("evidence/attention-decision-live.json" if args.live else "evidence/attention-decision-offline.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
