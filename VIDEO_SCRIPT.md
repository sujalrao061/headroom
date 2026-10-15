# Headroom demo video (target: 2 minutes 30 seconds)

Record the public site at https://headroom-beta-tan.vercel.app. Keep the browser zoom high enough to read measurements. Do not show the Nebius or Vercel dashboards, API keys, or local terminal prompts.

| Time | Screen | Narration |
| --- | --- | --- |
| 0:00–0:20 | Hero, then three workload choices | “A GPU memory fix is only valid if it preserves the output a developer needs. Headroom starts with that contract.” |
| 0:20–0:48 | Nearest neighbor selected; click **Explore saved GPU run** | “This workload materializes every pairwise distance. On an RTX 3050 Ti, the recorded full allocation used about 65 MB. A bounded tiled candidate used less than half a megabyte of additional allocation.” |
| 0:48–1:15 | Scroll through contract checks, timing, and patch | “The candidate passed five finite output checks, including tie and tile-edge cases. The tradeoff is visible: it is slower than the baseline, and we only claim the measured operating range.” |
| 1:15–1:42 | Select full distance matrix; open saved GPU run | “The same memory-saving idea fails when every distance is required. Headroom rejects that repair instead of presenting a smaller but incorrect output as success.” |
| 1:42–2:05 | Select nearest again; click **Ask Nemotron now** | “NVIDIA Nemotron, through Nebius Token Factory, makes a new bounded strategy choice from the saved measurements. The independent validator remains the authority; the model cannot execute code.” |
| 2:05–2:25 | Attention comparison | “For attention, we compare hand-written tiling with PyTorch SDPA. SDPA meets this memory budget and is much faster in the recorded run, so Headroom recommends the standard library.” |
| 2:25–2:30 | End card with public URL and GitHub URL | “Explore the evidence and code at Headroom.” |

The GPU measurements shown on the public website are recorded from a local run. Only the Nemotron decision made by **Ask Nemotron now** is live on the hosted site. If a live call fails while recording, retry after the firewall cooldown; do not edit a failure into a claimed success. Publish a video under three minutes to YouTube and add its URL to `SUBMISSION.md` and Devpost.
