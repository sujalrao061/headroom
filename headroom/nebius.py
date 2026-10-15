"""Server-side Nemotron strategy proposal; credentials never reach the browser."""
import json
import os
import urllib.error
import urllib.request


def _strategy_json(content: str, allowed=("tile_nearest", "tile_radius_count", "no_repair")) -> dict:
    decoder = json.JSONDecoder()
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("strategy") in allowed:
            return value
    raise ValueError("Nemotron did not return a supported strategy object")


def propose(case_id: str, source: str, contract: str, size: int, evidence: dict, mode: str = "evidence") -> dict:
    if mode not in ("evidence", "source_only"):
        raise ValueError("Unsupported proposal mode")
    fallback = "tile_radius_count" if case_id == "radius" else "tile_nearest"
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        return {"status": "not_configured", "model": None, "strategy": fallback, "explanation": "Local fixed candidate only. Live Nemotron diagnosis has not run."}
    model = os.getenv("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "You diagnose GPU memory pressure. Return JSON only with keys strategy and explanation. Strategy must be tile_nearest, tile_radius_count, or no_repair. Choose tile_nearest only when the contract requires one nearest index per query. Choose tile_radius_count only when it requires a count of references within a radius per query. Choose no_repair when the contract requires every pairwise distance or when neither tiled reduction preserves the output. Never write executable code. A separate validator decides acceptance."},
            {"role": "user", "content": f"Workload: {case_id}; size: {size}x{size}; output contract: {contract}. Source:\n{source}\n" + (f"Measured evidence and independent candidate feedback: {json.dumps(evidence)}\nThe physical GPU did not necessarily OOM. " if mode == "evidence" else "No measured execution evidence is supplied. ") + "Which bounded strategy fits the contract?"},
        ],
    }
    request = urllib.request.Request(
        "https://api.tokenfactory.us-central1.nebius.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response)
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        proposal = _strategy_json(content)
        strategy = proposal.get("strategy")
        if strategy not in ("tile_nearest", "tile_radius_count", "no_repair"):
            raise ValueError("Unsupported strategy")
        return {"status": "live", "model": model, "strategy": strategy, "explanation": str(proposal.get("explanation", ""))[:500], "usage": data.get("usage")}
    except urllib.error.HTTPError as exc:
        return {"status": "error", "model": model, "strategy": fallback, "http_status": exc.code, "explanation": f"Nemotron request returned HTTP {exc.code}; local fixed candidate shown separately."}
    except Exception as exc:
        return {"status": "error", "model": model, "strategy": fallback, "explanation": f"Nemotron request failed ({type(exc).__name__}); local fixed candidate shown separately."}


def propose_attention(report: dict) -> dict:
    """Ask for a bounded choice; independent measurements remain authoritative."""
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        return {"status": "not_configured", "model": None, "strategy": None,
                "explanation": "Local measured recommendation only. Live Nemotron comparison has not run."}
    model = os.getenv("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    evidence = {name: report[name] for name in ("shape", "memory_budget_mib", "checks", "baseline", "candidate", "library", "baseline_within_budget", "candidate_within_budget", "library_within_budget")}
    body = {"model": model, "temperature": 0, "max_tokens": 384, "messages": [
        {"role": "system", "content": "Select exactly one strategy for a single-head float32 attention workload. Reply as JSON with strategy and explanation. Allowed strategies: keep_baseline, pytorch_sdpa, tiled_python, no_repair. Recommend the lowest measured median latency among options that pass output checks and the additional-allocation memory budget. If none passes, choose no_repair. Do not treat an unmeasured algorithm as proven or write executable code."},
        {"role": "user", "content": "The baseline materializes a full attention score matrix. Tiled Python uses online softmax. PyTorch SDPA is the built-in library alternative. Measured evidence: " + json.dumps(evidence) + " Which strategy should be recommended?"},
    ]}
    request = urllib.request.Request(
        "https://api.tokenfactory.us-central1.nebius.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response)
        proposal = _strategy_json((data["choices"][0]["message"].get("content") or "").strip(),
                                  ("keep_baseline", "pytorch_sdpa", "tiled_python", "no_repair"))
        return {"status": "live", "model": model, "strategy": proposal["strategy"],
                "explanation": str(proposal.get("explanation", ""))[:500], "usage": data.get("usage")}
    except urllib.error.HTTPError as exc:
        return {"status": "error", "model": model, "strategy": None, "http_status": exc.code,
                "explanation": f"Nemotron request returned HTTP {exc.code}; measured recommendation is shown separately."}
    except Exception as exc:
        return {"status": "error", "model": model, "strategy": None,
                "explanation": f"Nemotron request failed ({type(exc).__name__}); measured recommendation is shown separately."}
