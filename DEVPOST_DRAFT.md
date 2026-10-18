# Devpost submission draft — Headroom

**Recommended track:** Best Apps and Agents

**Tagline:** Make a PyTorch workload fit its GPU, then prove the required output still survives.

**Demo:** https://headroom-beta-tan.vercel.app

**Source:** https://github.com/sujalrao061/headroom

**Video:** https://youtu.be/RYXefc-9gwc (public, 1:42, English captions; AI narration disclosed). GPU results are recorded from the local RTX 3050 Ti; only the NVIDIA Nemotron decision is live on the hosted site.

## Inspiration

“Reduce memory use” is incomplete advice when a smaller result is the wrong result. A nearest-neighbor query needs one index per query, while a full distance-matrix request needs every pairwise value. Both can create the same large intermediate tensor, but only one can safely be replaced by an index-only reduction. Headroom makes that output contract explicit before proposing a repair.

## What it does

Headroom shows the source line responsible for a large GPU allocation, a memory and latency budget, and a bounded repair candidate. It runs independent contract checks and reports the measured tradeoff. The point-cloud demo includes nearest-neighbor lookup, radius counting, and a full-matrix counterexample. A separate attention comparison measures full attention, Python tiling, and PyTorch SDPA, recommending the standard library when it wins.

On the public website, GPU measurements are **recorded evidence** from a local NVIDIA GPU. The “Ask Nemotron now” button makes a **new live NVIDIA Nemotron decision through Nebius Token Factory** about that saved evidence. The model chooses only from named strategies; it cannot execute code or change the independent validator. The site labels recorded measurements and live model output separately.

## How we built it

The interface uses Next.js, React, and TypeScript. A separate Python/PyTorch runner executes fixed, bounded workloads on an NVIDIA GeForce RTX 3050 Ti Laptop GPU. The runner uses deterministic seeds, synchronized timing, memory measurements, and finite output checks. The hosted server route calls NVIDIA Nemotron through Nebius Token Factory with a private server-side key. Vercel hosts the public app and limits requests to the live model endpoint.

The current public site does not accept user code, files, accounts, email addresses, or phone numbers. It does not offer fresh hosted GPU benchmarking; that would require a separate constrained GPU worker.

## Challenges and lessons

The central challenge was separating a memory improvement from an output-contract violation. An index-only candidate can appear successful on memory while failing a full-matrix request. The attention comparison exposed another trap: hand-written tiling used little memory but was far slower than PyTorch SDPA in the recorded run. We also found that model output may be malformed, so the application parses only an allowlisted strategy and leaves acceptance to independent checks.

## What worked

For the saved 1,536 × 1,536 nearest-neighbor case, the tiled candidate passed five finite output checks, including tie and tile-edge cases, while reducing peak additional allocation under the configured 8 MiB per-operation budget. The public “Ask Nemotron now” action was verified on October 14, 2026: the live model selected `tile_nearest`, agreeing with the independent validator. The displayed timing penalty and tested-size range remain visible.

We expanded the reproducible GPU correctness gauntlet to 490 passing nearest-neighbor and radius-count comparisons across 14 uneven/asymmetric shapes, three seeds, five tile sizes, duplicate-heavy and tightly clustered inputs, and explicit tile-boundary cases. It also confirms that two deliberately injected regressions are caught. For the separate attention workload, nine finite float32 correctness shapes pass for both Python tiling and PyTorch SDPA; SDPA remains the measured recommendation. These finite tests strengthen confidence in the fixed examples but do not establish correctness for arbitrary inputs or generalization to unfamiliar workloads.

## Limitations and next steps

The GPU results are from one local laptop GPU and a finite set of inputs; they do not prove correctness for every workload or the largest feasible size. The public site replays those measurements. Future work is a constrained hosted GPU worker, unfamiliar executable workloads, and a matched-cost comparison between deterministic selection and Nemotron with and without measurement evidence.

## Feedback on Nebius and NVIDIA tools

Nebius Token Factory made it practical to call an NVIDIA Nemotron model from a small server-side route while keeping the credential off the browser. The model's useful role here is a bounded decision over measured evidence, with an independent validator as the final authority. The local NVIDIA GPU and PyTorch measurements made the product honest about memory, latency, and standard-library alternatives. In building the demo, we would have benefited from easier-to-find guidance on structured responses and per-key spending controls.

## Technologies

Next.js, React, TypeScript, Python, PyTorch, CUDA, NVIDIA Nemotron, Nebius Token Factory, Vercel.
