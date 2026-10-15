"""Capture independent local GPU evidence for a new workload family."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from headroom.attention import full_attention, tiled_attention, library_attention
from headroom.engine import _measure, _synchronize


def inputs(n: int, m: int, d: int, device: torch.device, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(torch.randn(shape, generator=generator).to(device) for shape in ((n, d), (m, d), (m, d)))


def validate() -> list[dict]:
    rows = []
    for n, m, d, seed in ((7, 11, 8, 1), (65, 129, 16, 2), (131, 257, 32, 3)):
        q, k, v = inputs(n, m, d, torch.device("cpu"), seed)
        expected = full_attention(q, k, v)
        actual = tiled_attention(q, k, v)
        library = library_attention(q, k, v)
        difference = (expected - actual).abs().max().item()
        rows.append({"shape": [n, m, d], "max_absolute_error": round(difference, 8),
                     "passed": bool(torch.allclose(expected, actual, rtol=1e-4, atol=1e-5)),
                     "library_max_absolute_error": round((expected - library).abs().max().item(), 8),
                     "library_passed": bool(torch.allclose(expected, library, rtol=1e-4, atol=1e-5))})
    return rows


def capture(size: int = 1536, width: int = 64, memory_budget_mib: int = 8) -> dict:
    checks = validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q, k, v = inputs(size, size, width, device, 7)
    baseline = _measure(lambda a, b: full_attention(a, b, v), q, k, device)
    baseline.pop("output")
    candidate = _measure(lambda a, b: tiled_attention(a, b, v), q, k, device)
    candidate.pop("output")
    library = _measure(lambda a, b: library_attention(a, b, v), q, k, device)
    library.pop("output")
    _synchronize(device)
    candidate_ok = bool(device.type == "cuda" and all(c["passed"] for c in checks)
                        and candidate["peak_allocated_mb"] is not None
                        and candidate["peak_allocated_mb"] <= memory_budget_mib)
    library_ok = bool(device.type == "cuda" and all(c["library_passed"] for c in checks)
                      and library["peak_allocated_mb"] is not None
                      and library["peak_allocated_mb"] <= memory_budget_mib)
    baseline_ok = bool(device.type == "cuda" and baseline["peak_allocated_mb"] is not None
                       and baseline["peak_allocated_mb"] <= memory_budget_mib)
    viable = []
    if baseline_ok:
        viable.append(("keep_baseline", baseline["latency_ms"]))
    if candidate_ok:
        viable.append(("tiled_python", candidate["latency_ms"]))
    if library_ok:
        viable.append(("pytorch_sdpa", library["latency_ms"]))
    recommended = min(viable, key=lambda item: item[1])[0] if viable else "none_verified"
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Independent single-head float32 attention workload; no Nebius call or arbitrary code execution",
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU (GPU unverified)",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "shape": {"sequence_length": size, "head_dimension": width},
            "memory_budget_mib": memory_budget_mib, "checks": checks,
            "baseline": baseline, "candidate": candidate, "library": library,
            "baseline_within_budget": baseline_ok, "candidate_within_budget": candidate_ok, "library_within_budget": library_ok,
            "recommended": recommended,
            "selection_policy": f"Among validated implementations within the {memory_budget_mib} MiB additional-allocation budget, select the lowest measured median latency. No model chose this result.",
            "limitations": "Three finite CPU contract checks and local GPU measurements; no masks, dropout, gradients, mixed precision, or production kernel comparison."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1536)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--output", default="evidence/attention-gpu.json")
    args = parser.parse_args()
    if not 64 <= args.size <= 1536 or not 8 <= args.width <= 128:
        parser.error("Dimensions outside bounded benchmark range")
    result = capture(args.size, args.width)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
