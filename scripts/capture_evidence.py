"""Save a fresh local run without secrets or personally identifying data."""
import json
from datetime import datetime, timezone
from pathlib import Path

from headroom.engine import run


def main():
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local fixed-workload GPU evidence; no live Nebius call in this capture",
        "results": [run("nearest"), run("matrix"), run("radius")],
    }
    path = Path("evidence/local-gpu.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
