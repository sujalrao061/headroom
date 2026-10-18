# Headroom — submission draft

**Recommended track:** Best Apps and Agents

**One-line pitch:** Make a PyTorch workload fit its GPU, prove the required outputs survived, and leave a reviewable regression test.

## Project description

GPU out-of-memory advice often starts with generic suggestions: reduce batch size, use mixed precision, or try another kernel. Those changes can silently violate the output a developer actually needs. Headroom starts from an explicit output contract and a memory and latency budget. It links the expensive pairwise-distance allocation to its source line, tests a bounded repair on real hardware, and rejects a candidate if the output contract fails.

The demo now includes three point-cloud workloads. All materialize a large pairwise-distance tensor. Nearest neighbor requires one index per query; radius count requires one count per query; a full matrix requires every distance. The two reductions need different bounded repairs, while applying either to the full-matrix contract must be rejected. The workbench shows measurements, each contract check, the patch, a tested operating range, and a downloadable report and regression test. It also measures PyTorch's standard `torch.cdist` as an honest alternative; tiling is not presented as a new algorithm.

NVIDIA Nemotron reviews measured execution evidence through Nebius Token Factory and chooses from a bounded strategy set. Its answer cannot edit the independent tests or execute arbitrary code. Live Token Factory calls were verified on October 13, 2026: for the 1,536-point nearest case, Nemotron selected `tile_nearest` and the independent validator accepted it; for the full-matrix case, Nemotron selected `no_repair` and the index-only candidate was rejected. The adapter can still receive malformed model output and correctly labels those runs as fixed local candidates. The saved [`evidence/live-nebius.json`](evidence/live-nebius.json) records one such nearest-case parse failure alongside a live matrix response; a clean, repeatable two-case capture is still pending.

## What is measured now

The reproducible report is [`evidence/local-gpu.json`](evidence/local-gpu.json). It records a local NVIDIA GeForce RTX 3050 Ti Laptop GPU run, PyTorch 2.7.1+cu118, float32 inputs, deterministic seeds, one warm-up, three synchronized timed repetitions, and peak additional/total/reserved PyTorch memory. At 1,536 × 1,536 points and an 8 MiB **per-operation additional-allocation** budget, the nearest-index and radius-count candidates each pass five finite contract checks and the memory/latency limits. The fifth check puts equal-distance and exact-radius values across the 128-element query and reference tile boundaries; a deliberately broken first-tile-only nearest repair fails it. The nearest-index candidate fails the full-matrix contract checks. Timings and allocation values vary by run; the JSON file is the source of exact numbers. This was an application budget violation, **not a physical CUDA out-of-memory event**.

The radius case has local GPU evidence, but its live Nebius response still needs capture after the runner is restarted. These finite checks do not prove equivalence for every input. The displayed maximum is the largest size tested in the selected range, not the true maximum feasible size. A three-case local rule comparison is recorded in [`evidence/strategy-comparison.json`](evidence/strategy-comparison.json): the simple rule selects all three correctly, so these familiar examples do not establish that Nemotron improves strategy selection. No unfamiliar-workload benchmark or live three-way model comparison has been run yet.

A separate frozen, hand-authored [contract set](evidence/heldout-contracts.json) varies the wording and includes unsupported outputs. The simple rule gets 5/12. In the [live Nebius report](evidence/heldout-live.json), Nemotron returned 11 usable decisions and all 11 matched the labels; one of 12 requests failed. This is a wording test against a deliberately simple rule. The model prompt describes the allowed strategies, and the cases do not contain unfamiliar executable workloads, so this does not establish generalization or end-to-end repair quality.

A separate [attention GPU report](evidence/attention-gpu.json) exercises a second executable workload family on the local RTX 3050 Ti. At sequence length 1,536 and width 64, full attention used 18.75 MiB additional allocation. A tiled Python online-softmax version used about 1.2 MiB but took about 91 ms; PyTorch SDPA used about 0.75 MiB and took about 0.55 ms in the saved run. A companion [correctness report](evidence/attention-contracts.json) contains nine finite float32 forward-output checks across single-token, asymmetric, tile-edge, and larger sequence shapes for both alternatives. The independent measured recommendation is SDPA. This comparison prevents presenting a slower hand-written implementation as superior to the standard library. It does not cover masks, dropout, backward passes, or mixed precision.

The [live attention decision](evidence/attention-decision-live.json) shows that NVIDIA Nemotron, called through Nebius Token Factory, selected SDPA from the saved raw measurements, agreeing with the independent measured choice. The decision report's SHA-256 matches the current GPU evidence file, and no key is stored in it. This one-case agreement does not establish a model advantage over the deterministic selector. The workbench displays this as a recorded local run; a fresh model-backed run requires a runner started with a private key.

The public [correctness gauntlet](evidence/contract-gauntlet.json) adds 490 passing comparisons for nearest-neighbor and radius-count candidates on an RTX 3050 Ti. It spans 14 uneven and asymmetric shapes, three deterministic seeds, five tile sizes, duplicate-heavy and tightly clustered inputs, and adversarial cross-tile boundary cases. It also verifies that injected regressions for skipped nearest tiles and an exclusive radius boundary are detected. This increases confidence in those fixed implementations; it is not universal correctness proof, a performance measurement, or an unfamiliar-workload generalization result. Reproduce it with `python -m scripts.run_contract_gauntlet`.

The demo also exposes a frozen [12-case wording evaluation](evidence/heldout-live.json): a simple keyword rule matched 5 labels, while Nemotron returned 11 correct usable decisions and one request failed. Each case varies the wording of the same known allocation/output-contract task. This is limited evidence of wording robustness, not arbitrary workload reasoning, GPU correctness, or a fair comparison with a stronger baseline.

A second frozen wording set scored Nemotron 10/12 (11/12 requests completed) against 6/12 for the simple rule; one model response failed parsing and one extra-semantics contract was misclassified. The report is [`evidence/heldout-v2-live.json`](evidence/heldout-v2-live.json). After tightening the supported-contract instructions and response handling, a third fresh set was frozen in [`evidence/heldout-contracts-v3.json`](evidence/heldout-contracts-v3.json), with a 4/12 rule baseline. Its live run has not yet been captured. All these tests use wording variants for one fixed allocation pattern; they do not evaluate unfamiliar executable workloads or general GPU repair.

## Architecture and safety

- **Frontend:** Next.js, React, TypeScript. Guest use needs no login, phone number, email address, or file upload.
- **GPU runner:** Python/PyTorch, fixed built-in workloads, bounded numeric inputs, one active job, loopback service.
- **Model:** NVIDIA Nemotron via Nebius Token Factory, server-side key only. The model chooses a named strategy after seeing measured evidence; the independent validator remains the acceptance authority.
- **Data:** No database or personal data in the current demo. If saved jobs are added, use private storage with ownership checks, retention/deletion controls, and a separately reviewed isolation design before accepting arbitrary code.

## Demo video storyboard, 1:42

| Time | Show | Narration point |
| --- | --- | --- |
| 0:00–0:14 | GPU memory result first | Recorded RTX 3050 Ti allocation falls from about 65 MB to under 0.5 MB for nearest-neighbor output. |
| 0:14–0:24 | Problem and product | Headroom checks whether a memory repair preserves the answer. |
| 0:24–0:36 | Nearest-neighbor output contract | The workload needs one nearest index per query, not every pairwise distance. |
| 0:36–0:47 | Five checks and measured tradeoff | The candidate preserves tested outputs, while taking longer than the baseline. |
| 0:47–0:59 | Full-matrix counterexample | The nearest-index repair is rejected when every distance is required. |
| 0:59–1:14 | Live NVIDIA Nemotron via Nebius Token Factory | The model reviews saved GPU evidence and chooses from bounded strategies. |
| 1:14–1:24 | Model decision and independent validator | The captured model choice agrees with the validator; GPU measurements remain recorded evidence. |
| 1:24–1:33 | Attention comparison | PyTorch SDPA wins the measured memory and latency comparison. |
| 1:33–1:42 | Public demo and repository | Invite judges to explore the live site and source. |

## Minimum before submission

- [x] Configure a private runner key and verify live Nemotron responses without saving the key.
- [x] Prepare a recorded-evidence website that works when the local runner is offline and labels saved measurements clearly.
- [x] Implement a server-side Token Factory runtime decision about fixed saved GPU evidence, without exposing the key to the browser.
- [x] Publish the MIT-licensed repository and add its URL here.
- [x] Deploy the web frontend and verify the public saved-evidence flow. See [DEPLOY.md](DEPLOY.md).
- [x] Configure the server-side Nebius key and verify the hosted “Ask Nemotron now” action produces a live Token Factory response. On October 14, 2026, the public site returned `tile_nearest` for the saved nearest-neighbor run, agreeing with the independent validator. The GPU measurements remain labeled as recorded evidence.
- [x] Publish the prepared 1:42 demo video publicly on YouTube with English captions and the AI-use disclosure.
- [ ] Enter the project description, recommended track, working demo URL, repository URL, video URL, and honest Nebius/NVIDIA feedback on Devpost.
- [x] Review the official [rules](https://nebiusglobalaihackathon.devpost.com/rules) and [FAQ](https://nebiusglobalaihackathon.devpost.com/details/faqs).

## Improvements if time remains

- Capture a clean, repeatable two-case live trace after hardening response parsing.
- Run unfamiliar fixed workloads and the three-way comparison: deterministic rules, code/error-only Nemotron, evidence-grounded Nemotron, with comparable budgets and inference cost. A 12-case wording check has been captured separately.
- Deploy a constrained public GPU worker so judges can run fresh GPU measurements from the hosted site.

**Public demo URL:** https://headroom-beta-tan.vercel.app

**Public repository URL:** https://github.com/sujalrao061/headroom

**Public video URL:** https://youtu.be/RYXefc-9gwc
