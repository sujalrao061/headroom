"""Reproducible randomized and boundary-focused checks for fixed tiled workloads."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from headroom.workloads import (
    nearest_baseline,
    radius_baseline,
    tiled_nearest,
    tiled_radius_count,
)


SIZES = [(1, 3), (7, 11), (31, 33), (63, 65), (127, 129), (128, 130), (129, 131), (255, 257)]
TILES = [7, 31, 64, 127, 128]
SEEDS = [3, 17, 41]


def _equal(left, right):
    return left.shape == right.shape and torch.equal(left, right)


def _boundary_inputs(device, radius):
    query = torch.zeros((130, 3), device=device)
    reference = torch.full((131, 3), 2.0, device=device)
    # Exact threshold and equal-distance cases straddle the default tile edge.
    reference[127] = torch.tensor([1.0, 0.0, 0.0], device=device)
    reference[128] = torch.tensor([-1.0, 0.0, 0.0], device=device)
    query[128] = torch.tensor([10.0, 0.0, 0.0], device=device)
    reference[130] = torch.tensor([10.0, 0.0, 0.0], device=device)
    if radius:
        # Make the exact-radius points unambiguously included for the target rows.
        query[0] = 0
    return query, reference


def capture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials = []
    for case_id, reference_fn, candidate_fn in (
        ("nearest", nearest_baseline, tiled_nearest),
        ("radius", radius_baseline, tiled_radius_count),
    ):
        for n, m in SIZES:
            for seed in SEEDS:
                generator = torch.Generator(device="cpu").manual_seed(seed * 100_000 + n * 100 + m)
                query = torch.randn((n, 3), generator=generator).to(device)
                reference = torch.randn((m, 3), generator=generator).to(device)
                expected = reference_fn(query, reference)
                for tile in TILES:
                    actual = candidate_fn(query, reference, tile=tile)
                    trials.append({
                        "case": case_id,
                        "shape": [n, m],
                        "seed": seed,
                        "tile": tile,
                        "passed": _equal(actual, expected),
                    })

        q, r = _boundary_inputs(device, radius=case_id == "radius")
        expected = reference_fn(q, r)
        for tile in TILES:
            actual = candidate_fn(q, r, tile=tile)
            trials.append({
                "case": case_id,
                "shape": [130, 131],
                "seed": None,
                "tile": tile,
                "kind": "adversarial-boundary",
                "passed": _equal(actual, expected),
            })

    # Confirm the suite itself detects representative regressions.
    def first_tile_only(query, reference, tile=128):
        return nearest_baseline(query, reference[:tile])

    q, r = _boundary_inputs(device, radius=False)
    nearest_bug_detected = not _equal(first_tile_only(q, r), nearest_baseline(q, r))

    def exclusive_radius(query, reference, tile=128):
        rows = []
        for q0 in range(0, query.shape[0], tile):
            qtile = query[q0:q0 + tile]
            counts = torch.zeros((qtile.shape[0],), device=query.device, dtype=torch.long)
            for r0 in range(0, reference.shape[0], tile):
                rt = reference[r0:r0 + tile]
                distance = torch.sqrt(torch.sum((qtile[:, None, :] - rt[None, :, :]) ** 2, dim=-1))
                counts += torch.sum(distance < 1.0, dim=1)
            rows.append(counts)
        return torch.cat(rows)

    q, r = _boundary_inputs(device, radius=True)
    radius_bug_detected = not _equal(exclusive_radius(q, r), radius_baseline(q, r))

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Deterministic fixed-workload contract stress checks; correctness evidence only, not a performance benchmark or proof for all inputs.",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU (correctness only)",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "parameters": {"random_shapes": len(SIZES), "seeds_per_shape": len(SEEDS), "tile_sizes": TILES, "cases": ["nearest", "radius"]},
        "passed": all(row["passed"] for row in trials) and nearest_bug_detected and radius_bug_detected,
        "checks_passed": sum(row["passed"] for row in trials),
        "checks_total": len(trials),
        "regression_sensitivity": {
            "nearest_that_ignores_later_reference_tiles_detected": nearest_bug_detected,
            "radius_that_excludes_exact_boundary_detected": radius_bug_detected,
        },
        "trials": trials,
    }


if __name__ == "__main__":
    report = capture()
    output = Path("evidence/contract-gauntlet.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "checks_passed", "checks_total", "device", "regression_sensitivity")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
