"""Fixed-input runner. Never executes uploaded code or model-authored Python."""
import gc
import platform
import time

import torch

from .workloads import CASES, tiled_nearest, tiled_radius_count


PATCH = """--- a/workload.py
+++ b/workload.py
@@
-    delta = query[:, None, :] - reference[None, :, :]
-    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))
-    return torch.argmin(distance, dim=1)
+    rows = []
+    for q0 in range(0, len(query), 128):
+        q = query[q0:q0 + 128]
+        best_distance = torch.full((len(q),), float('inf'), device=q.device, dtype=q.dtype)
+        best_index = torch.zeros((len(q),), device=q.device, dtype=torch.long)
+        for r0 in range(0, len(reference), 128):
+            r = reference[r0:r0 + 128]
+            distance = torch.sqrt(torch.sum((q[:, None, :] - r[None, :, :]) ** 2, dim=-1))
+            local_distance, local_index = torch.min(distance, dim=1)
+            improve = local_distance < best_distance
+            best_distance = torch.where(improve, local_distance, best_distance)
+            best_index = torch.where(improve, local_index + r0, best_index)
+        rows.append(best_index)
+    return torch.cat(rows)
"""

PATCH_RADIUS = """--- a/workload.py
+++ b/workload.py
@@
-    delta = query[:, None, :] - reference[None, :, :]
-    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))
-    return torch.sum(distance <= 1.0, dim=1)
+    rows = []
+    for q0 in range(0, len(query), 128):
+        q = query[q0:q0 + 128]
+        counts = torch.zeros((len(q),), device=q.device, dtype=torch.long)
+        for r0 in range(0, len(reference), 128):
+            r = reference[r0:r0 + 128]
+            distance = torch.sqrt(torch.sum((q[:, None, :] - r[None, :, :]) ** 2, dim=-1))
+            counts += torch.sum(distance <= 1.0, dim=1)
+        rows.append(counts)
+    return torch.cat(rows)
"""

REGRESSION_NEAREST = """import torch
from headroom.workloads import nearest_baseline, tiled_nearest

for n, m, seed in [(7, 11, 1), (33, 19, 2), (129, 131, 3)]:
    torch.manual_seed(seed)
    q, r = torch.randn(n, 3), torch.randn(m, 3)
    assert torch.equal(nearest_baseline(q, r), tiled_nearest(q, r))

q = torch.tensor([[0., 0., 0.]])
r = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
assert torch.equal(nearest_baseline(q, r), tiled_nearest(q, r, tile=1))

# The equal-distance winners straddle the default 128-reference tile.
q = torch.zeros(130, 3)
r = torch.full((131, 3), 2.0)
r[127], r[128] = torch.tensor([1., 0., 0.]), torch.tensor([-1., 0., 0.])
q[128], r[130] = torch.tensor([10., 0., 0.]), torch.tensor([10., 0., 0.])
assert torch.equal(nearest_baseline(q, r), tiled_nearest(q, r))
"""

REGRESSION_MATRIX = """import torch
from headroom.workloads import matrix_baseline, tiled_nearest

q, r = torch.randn(7, 3), torch.randn(11, 3)
assert matrix_baseline(q, r).shape == (7, 11)
assert tiled_nearest(q, r).shape == (7,)
"""

REGRESSION_RADIUS = """import torch
from headroom.workloads import radius_baseline, tiled_radius_count

for n, m, seed in [(7, 11, 1), (33, 19, 2), (129, 131, 3)]:
    torch.manual_seed(seed)
    q, r = torch.randn(n, 3), torch.randn(m, 3)
    assert torch.equal(radius_baseline(q, r), tiled_radius_count(q, r))

q = torch.tensor([[0., 0., 0.]])
r = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [2., 0., 0.]])
assert torch.equal(radius_baseline(q, r), tiled_radius_count(q, r, tile=1))

# Exact-radius points straddle the default 128-reference tile.
q = torch.zeros(130, 3)
r = torch.full((131, 3), 2.0)
r[127], r[128] = torch.tensor([1., 0., 0.]), torch.tensor([-1., 0., 0.])
q[128], r[130] = torch.tensor([10., 0., 0.]), torch.tensor([10., 0., 0.])
assert torch.equal(radius_baseline(q, r), tiled_radius_count(q, r))
"""


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, query, reference, device, repetitions=3):
    fn(query, reference)
    _synchronize(device)
    base_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = fn(query, reference)
        _synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "latency_ms": round(sorted(times)[len(times) // 2], 3),
        "peak_allocated_mb": round((peak - base_allocated) / 2**20, 3) if device.type == "cuda" else None,
        "peak_total_allocated_mb": round(peak / 2**20, 3) if device.type == "cuda" else None,
        "base_allocated_mb": round(base_allocated / 2**20, 3) if device.type == "cuda" else None,
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 3) if device.type == "cuda" else None,
        "output": result,
    }


def _inputs(n, m, device, seed=7):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn((n, 3), generator=generator).to(device),
            torch.randn((m, 3), generator=generator).to(device))


def _check_contract(case_id, candidate):
    checks = []
    for n, m, seed in [(7, 11, 1), (33, 19, 2), (129, 131, 3)]:
        q, r = _inputs(n, m, torch.device("cpu"), seed)
        expected = CASES[case_id]["baseline"](q, r)
        actual = candidate(q, r)
        passed = actual.shape == expected.shape and (torch.equal(actual, expected) if case_id in ("nearest", "radius") else torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
        checks.append({"shape": f"{n}×{m}", "passed": bool(passed), "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)})
    adversarial = [
        ("exact-radius boundary" if case_id == "radius" else "equal-distance tie",
         torch.tensor([[0., 0., 0.]]),
         torch.tensor([[1., 0., 0.], [-1., 0., 0.], [2., 0., 0.]])),
    ]
    # Put equal-distance and exact-radius values on opposite sides of both
    # the default reference-tile and query-tile boundaries.
    q = torch.zeros(130, 3)
    r = torch.full((131, 3), 2.0)
    r[127] = torch.tensor([1., 0., 0.])
    r[128] = torch.tensor([-1., 0., 0.])
    q[128] = torch.tensor([10., 0., 0.])
    r[130] = torch.tensor([10., 0., 0.])
    adversarial.append(("tile edges: 130×131", q, r))
    for label, q, r in adversarial:
        actual = candidate(q, r)
        expected = CASES[case_id]["baseline"](q, r)
        checks.append({"shape": label, "passed": bool(actual.shape == expected.shape and torch.equal(actual, expected)), "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)})
    return checks


def _frontier(size, memory_limit_mb, latency_limit_ms, device, selected_measurement, candidate):
    """Probe finite increasing sizes; never extrapolate beyond them."""
    sizes = sorted({s for s in (256, 512, 1024, size) if s <= size})
    rows = []
    for n in sizes:
        if n == size:
            metric = selected_measurement
        else:
            q, r = _inputs(n, n, device)
            metric = _measure(candidate, q, r, device)
            metric.pop("output")
        within = metric["peak_allocated_mb"] is not None and metric["peak_allocated_mb"] <= memory_limit_mb and metric["latency_ms"] <= latency_limit_ms
        rows.append({"size": n, "latency_ms": metric["latency_ms"], "peak_allocated_mb": metric["peak_allocated_mb"], "within_limits": bool(within)})
    valid = [row["size"] for row in rows if row["within_limits"]]
    return {"tested": rows, "largest_tested_within_limits": max(valid) if valid else None, "untested_sizes": "unknown"}


def run(case_id="nearest", size=1536, memory_limit_mb=8, latency_limit_ms=1000):
    if case_id not in CASES:
        raise ValueError("Unknown built-in workload")
    if not 64 <= size <= 1536 or not 1 <= memory_limit_mb <= 4096 or not 1 <= latency_limit_ms <= 30000:
        raise ValueError("Inputs outside allowed demo range")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_fn = CASES[case_id]["candidate"]
    q, r = _inputs(size, size, device)
    baseline = _measure(CASES[case_id]["baseline"], q, r, device)
    baseline.pop("output")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checks = _check_contract(case_id, candidate_fn)
    candidate = _measure(candidate_fn, q, r, device)
    candidate.pop("output")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    library_checks = _check_contract(case_id, CASES[case_id]["library"])
    library = _measure(CASES[case_id]["library"], q, r, device)
    library.pop("output")
    library["contract_passed"] = all(c["passed"] for c in library_checks)
    library["within_limits"] = bool(library["contract_passed"] and library["peak_allocated_mb"] is not None and library["peak_allocated_mb"] <= memory_limit_mb and library["latency_ms"] <= latency_limit_ms)
    contract_ok = all(c["passed"] for c in checks)
    memory_ok = candidate["peak_allocated_mb"] is not None and candidate["peak_allocated_mb"] <= memory_limit_mb
    latency_ok = candidate["latency_ms"] <= latency_limit_ms
    accepted = bool(contract_ok and memory_ok and latency_ok and device.type == "cuda")
    reasons = []
    if not contract_ok: reasons.append("Output contract failed: the tiled candidate returns indices, but this workload requires the full matrix.")
    if device.type != "cuda": reasons.append("GPU acceptance is unverified: CUDA PyTorch is unavailable.")
    elif not memory_ok: reasons.append("Measured peak allocated memory exceeds the application limit.")
    if not latency_ok: reasons.append("Measured latency exceeds the application limit.")
    if accepted: reasons.append("All sampled output, GPU memory, and latency checks passed. Untested inputs remain unknown.")
    frontier = _frontier(size, memory_limit_mb, latency_limit_ms, device, candidate, candidate_fn) if accepted else None
    return {
        "case_id": case_id, "name": CASES[case_id]["name"], "contract": CASES[case_id]["contract"],
        "source": CASES[case_id]["code"], "hot_line": 2,
        "size": size, "memory_limit_mb": memory_limit_mb, "latency_limit_ms": latency_limit_ms,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU (GPU unverified)",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
        "baseline": baseline, "candidate": candidate, "library": library, "checks": checks,
        "frontier": frontier,
        "accepted": accepted, "reasons": reasons, "repetitions": 3, "warmups": 1,
        "failure_kind": "application memory budget" if baseline["peak_allocated_mb"] is not None and baseline["peak_allocated_mb"] > memory_limit_mb else "none observed",
        "patch": "Replace full pairwise allocation with a tiled radius count; preserve the inclusive radius boundary." if case_id == "radius" else "Replace full pairwise allocation with two-axis tiled nearest-neighbor reduction; preserve first-index tie behavior.",
        "patch_diff": PATCH_RADIUS if case_id == "radius" else PATCH if case_id == "nearest" else "The same nearest-index patch was evaluated but rejected: it removes required matrix values.",
        "regression_test": "assert torch.equal(radius_baseline(q, r), tiled_radius_count(q, r))" if case_id == "radius" else "assert torch.equal(nearest_baseline(q, r), tiled_nearest(q, r))",
        "regression_test_source": REGRESSION_RADIUS if case_id == "radius" else REGRESSION_NEAREST if case_id == "nearest" else REGRESSION_MATRIX,
        "method": "fixed built-in workload; no arbitrary code execution",
    }
