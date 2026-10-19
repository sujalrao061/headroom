# Headroom

Headroom is a fixed-workload proof of concept for evidence-backed PyTorch GPU memory repair. Three point-cloud functions materialize the same pairwise tensor but promise different outputs: nearest indices, every distance, or counts within a radius. A bounded candidate must preserve the selected contract.

## Current status

The first vertical slice includes a Next.js workbench, a Python/PyTorch runner, independent finite contract checks, synchronized timing and CUDA peak-memory measurement when CUDA PyTorch is available, a bounded Nemotron strategy request through Nebius Token Factory when `NEBIUS_API_KEY` is present, and a downloadable JSON report. The model chooses from a bounded strategy set and can prevent a repair from being accepted; no model-authored code is executed. Local GPU measurements are recorded in `evidence/local-gpu.json`. A live Nebius call was verified on October 13, 2026: Nemotron selected `tile_nearest` for the default 1,536-point nearest-neighbor case, and independent checks accepted the measured candidate. Some responses failed JSON parsing; the app labeled those runs as fixed local candidates, and the parser has since been hardened for a follow-up live check. The runner currently uses built-in workloads only and binds to loopback.
When configured, the server sends measured evidence and independent feedback to Nemotron. A live `tile_nearest` strategy is eligible for acceptance only if the validator's checks passed; a `no_repair` strategy cannot be silently relabeled as a model repair. Without a key or after an API failure, the measured candidate is identified as a fixed local candidate.

The [public demo](https://headroom-beta-tan.vercel.app), [public repository](https://github.com/sujalrao061/headroom), and [1:42 YouTube demo](https://youtu.be/RYXefc-9gwc) are available. The demo includes English captions and YouTube's AI-use disclosure for its generated narration. The deployed server-side Nebius key has been verified in a live model call; no key is in this repository. A benchmark on unfamiliar executable workloads has not been run, so the current strategy-selection evidence does not establish generalization or a model advantage over the simple rule. The local runner reports only a finite tested-size frontier for accepted nearest-neighbor repairs.

The public workbench also displays a reproducible [correctness gauntlet](evidence/contract-gauntlet.json): the latest RTX 3050 Ti capture covers 490 nearest-neighbor and radius-count comparisons across 14 uneven/asymmetric shapes, three deterministic seeds, five tile sizes, duplicate-heavy and tightly clustered inputs, and adversarial tile-boundary values. The harness confirms it catches two injected regressions (ignoring later nearest tiles and excluding exact-radius matches). These are finite correctness checks for the fixed operators, not a proof for every possible input, unfamiliar workload, or a performance benchmark. Regenerate with `python -m scripts.run_contract_gauntlet`.

The site also surfaces the frozen [12-case contract wording evaluation](evidence/heldout-live.json): a simple keyword rule matched 5/12 labels; Nemotron produced 11 correct usable decisions, with one request error. The cases vary wording around the same known allocation and output contracts. They do not test unfamiliar executable workloads, GPU correctness, or superiority to a stronger baseline; interpret this as a small diagnostic, not a general model benchmark.

## Judge walkthrough

The hosted workbench opens with the saved nearest-neighbor GPU result. Start there: compare the baseline's 65.012 MiB peak additional allocation with the tiled candidate's 0.463 MiB, then inspect the five finite checks and the measured latency cost. Switch to **Full distance matrix** and load its saved run to see why the same index-only repair is rejected when every pairwise distance is part of the requested output. Use **Ask Nemotron now** for a fresh, bounded model choice about the saved evidence; the GPU measurements are recorded, and the independent validator is still the acceptance authority. Finish in **Attention** to see why PyTorch SDPA wins the measured memory-and-latency tradeoff over the hand-written Python tile implementation.

The website can now be deployed without the GPU runner as a clearly labeled **recorded-evidence demo**. Judges can choose any of the three workloads and open a saved local GPU run. The fresh GPU run button is disabled when the runner is offline. With a server-side `NEBIUS_API_KEY`, “Ask Nemotron now” makes a new Token Factory runtime call about that fixed saved evidence and labels the model response separately. The separately hash-verified attention model decision is shown as recorded evidence. A public GPU-backed run needs a separately hosted, constrained runner and a reachable `HEADROOM_RUNNER_URL`. See [DEPLOY.md](DEPLOY.md).

## Run locally

1. Install Node.js 20+ and pnpm. Run `pnpm install`, then `pnpm dev`.
2. Install a CUDA-compatible PyTorch build for your GPU/driver, following PyTorch's official selector. Run `python -m headroom.server` in a separate terminal.
3. Open `http://127.0.0.1:3000`.
4. On Windows, start the Python runner with `powershell -File .\scripts\start_runner.ps1`. Its hidden prompt accepts a **new, private** Nebius key without putting it in chat, command history, or a project file. The key is held temporarily in the local PowerShell and Python process environments while the runner runs; the script removes its PowerShell copy when the runner exits. Set `NEBIUS_MODEL` if your Token Factory account exposes a different NVIDIA Nemotron model ID. Never put this key in a `NEXT_PUBLIC_` variable or browser code.

Run `python -m unittest discover -s tests` to check the three output contracts. No phone number, account, uploaded code, or customer data is required.

Run `python -m scripts.capture_evidence` to refresh the checked-in local GPU evidence after installing PyTorch. On this development machine, `start_runner.ps1` selects the bundled Python 3.12 and the project-local `.torch-deps` installation of CUDA-enabled PyTorch. A standard virtual environment can install PyTorch normally.

Run `python -m scripts.evaluate_strategies` to score a fixed contract-text rule against the saved three-case evidence without any API call. The output is `evidence/strategy-comparison.json`. It leaves model columns absent when no live call was made. `--live` adds source-only and evidence-grounded Nemotron calls when a private `NEBIUS_API_KEY` is already present in that process environment. These three known cases are a smoke test, not evidence of generalization; a held-out workload set is still required.

Run `python -m scripts.evaluate_heldout` for 12 hand-authored variations of output contracts. The fixed simple rule gets 5 of 12. A live Nebius run returned 11 usable decisions, all correct against the frozen labels, with one request error; see `evidence/heldout-live.json`. This tests wording robustness and unsupported outputs using one shared allocation pattern, not unfamiliar executable workloads. The cases and expected labels were frozen in `evidence/heldout-contracts.json` before the live model run. `--live` issues 12 source-only Nebius calls when a private key is already in the process environment; it records strategy, status, latency, and token usage without saving the key.

Run `python -m scripts.benchmark_attention` with CUDA-enabled PyTorch for a separate, executable attention workload. It compares full-matrix attention, a bounded Python online-softmax implementation, and PyTorch's built-in scaled-dot-product attention (SDPA). The local [attention report](evidence/attention-gpu.json) shows that both bounded methods meet the 8 MiB additional-allocation budget, but SDPA is dramatically faster; the independent benchmark recommends SDPA. The companion [correctness report](evidence/attention-contracts.json) covers nine CPU output shapes, including single-token, asymmetric, and tile-edge cases. Regenerate it with `python -m scripts.capture_attention_validation`. A separate [live Nebius decision](evidence/attention-decision-live.json) also selected SDPA. This agreement does not show that Nemotron improves on the deterministic measured-selection rule. GPU performance remains a single float32 forward case without masks, dropout, gradients, or mixed precision.

The website shows saved point-cloud measurements and the saved attention measurement with its hash-verified Nemotron decision as **recorded local runs**. It offers new GPU runs only when `/health` confirms the runner is online. On a new attention run, the server measures all three options, selects the fastest independently validated option within the memory budget, then asks Nemotron for a separate bounded choice through Nebius. The UI distinguishes recorded evidence from a fresh run.

For a one-call model check without restarting the server, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_live_attention_decision.ps1` in a separate PowerShell tab. It privately prompts for the key, sends the saved raw metrics without the independently selected verdict, and writes `evidence/attention-decision-live.json`. The offline counterpart is `evidence/attention-decision-offline.json` and records only the deterministic choice. The saved reports include a hash tying the decision to the source measurements, not the key.

The original 12-case live report scored 11/12 for Nemotron and 5/12 for the simple rule. A second, separately frozen set scored 10/12 for Nemotron (11/12 requests completed) and 6/12 for the rule; one response failed parsing and one extra-semantics contract was misclassified. After tightening the supported-contract instructions and response handling, a third fresh set scored 12/12 for Nemotron and 4/12 for the simple rule; all 12 requests completed. See [`evidence/heldout-v3-live.json`](evidence/heldout-v3-live.json) and the deployed evidence panel. This is a small wording-robustness test on one fixed workload pattern, not evidence of generalization to unfamiliar executable workloads, GPU correctness, or superiority to a stronger baseline. The hidden key prompt does not save the key. Never paste the key into chat or a command argument.

## Measurement protocol

The runner generates deterministic float32 `[N,3]` points with a CPU seed and moves them to the selected device. Each function gets one warm-up call, then three synchronized timed calls. It reports median latency, PyTorch peak *additional* allocated bytes above the allocation present after warm-up (the per-operation budget metric), absolute peak allocated, absolute peak reserved, device, Python/PyTorch/CUDA versions, input size, and output checks. Peak memory measures the whole three-call sequence after resetting counters, not one isolated call. The additional metric avoids charging a later operation for a persistent CUDA library workspace allocated by an earlier one. CPU runs do not count as GPU acceptance. `failure_kind` distinguishes an application budget violation from an actual CUDA out-of-memory exception; this prototype does not deliberately exhaust the device. Finite tests do not prove universal equivalence.

For accepted nearest-neighbor repairs, the runner separately measures candidate sizes 256, 512, 1024, and the selected size when each is no larger than the selection. It reports the largest **tested** size within the chosen limits; it does not extrapolate beyond that range.

## Security boundary

The runner accepts only a case ID and constrained numeric parameters. It never executes uploaded source or generated patches. The Nebius key stays in the runner's environment, away from browser code and model-visible execution. A production public deployment needs an isolated disposable GPU worker, per-job resource limits, authentication/ownership for saved jobs, rate limits, and a security review before arbitrary code ingestion is enabled.

## Project layout

- `app/` — Next.js workbench and server-side runner proxy.
- `headroom/workloads.py` — three output contracts and their bounded candidates.
- `headroom/engine.py` — independent checks and measurement.
- `headroom/nebius.py` — optional live Token Factory call.
- `tests/` — contract tests.

Licensed under MIT. Tiling is a standard method, not an algorithm invented by this project.
