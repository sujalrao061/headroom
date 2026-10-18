"""Save CPU correctness checks separately from hash-bound GPU/model evidence."""
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark_attention import validate


def capture() -> dict:
    checks = validate()
    passed = sum(row["passed"] and row["library_passed"] for row in checks)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Nine deterministic CPU float32 output comparisons for full attention, Python tiling, and PyTorch SDPA. These validate sampled forward outputs only.",
        "checks_passed": passed,
        "checks_total": len(checks),
        "passed": passed == len(checks),
        "limitations": "Nine finite CPU correctness shapes and separate local GPU measurements; no masks, dropout, gradients, mixed precision, or production kernel comparison.",
        "checks": checks,
    }


if __name__ == "__main__":
    report = capture()
    output = Path("evidence/attention-contracts.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())
    if not report["passed"]:
        raise SystemExit(1)
