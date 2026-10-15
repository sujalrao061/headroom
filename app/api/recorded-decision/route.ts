import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const expected = {
  nearest: "tile_nearest",
  matrix: "no_repair",
  radius: "tile_radius_count",
} as const;
type CaseId = keyof typeof expected;
let nextAllowedAt = 0;

function parseStrategy(content: string): string | null {
  const allowed = new Set(["tile_nearest", "tile_radius_count", "no_repair"]);
  for (const match of content.matchAll(/\{[^{}]*\}/g)) {
    try {
      const value = JSON.parse(match[0]);
      if (value && allowed.has(value.strategy)) return value.strategy;
    } catch { /* Continue to the next object. */ }
  }
  return null;
}

export async function POST(request: NextRequest) {
  const origin = request.headers.get("origin");
  let originHost: string | null = null;
  try { originHost = origin ? new URL(origin).host : null; } catch { /* Reject below. */ }
  if (!originHost || originHost !== request.headers.get("host")) {
    return NextResponse.json({ error: "Use the Headroom website to request a decision." }, { status: 403 });
  }
  if (Number(request.headers.get("content-length") ?? 0) > 256) {
    return NextResponse.json({ error: "Request too large." }, { status: 413 });
  }
  let caseId: CaseId;
  try {
    const body = await request.json();
    if (!body || typeof body.case_id !== "string" || !Object.hasOwn(expected, body.case_id)) throw new Error();
    caseId = body.case_id as CaseId;
  } catch {
    return NextResponse.json({ error: "Choose a supported saved workload." }, { status: 400 });
  }
  const key = process.env.NEBIUS_API_KEY;
  if (!key) {
    return NextResponse.json({ error: "Live Nemotron is not configured on this website." }, { status: 503 });
  }
  const now = Date.now();
  if (now < nextAllowedAt) {
    return NextResponse.json({ error: "Please wait a few seconds before requesting another model decision." }, { status: 429 });
  }
  nextAllowedAt = now + 10_000; // Best-effort local throttle; apply durable rate limits before broad public use.
  try {
    const report = JSON.parse(await readFile(join(process.cwd(), "evidence", "local-gpu.json"), "utf8"));
    const result = report.results.find((item: { case_id: string }) => item.case_id === caseId);
    if (!result) throw new Error("Saved evidence missing");
    const model = process.env.NEBIUS_MODEL ?? "nvidia/nemotron-3-super-120b-a12b";
    const response = await fetch("https://api.tokenfactory.us-central1.nebius.com/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        // Nemotron may use part of this budget for reasoning before emitting JSON.
        model, temperature: 0, max_tokens: 1024,
        messages: [
          { role: "system", content: "Diagnose a fixed PyTorch GPU workload. Return one JSON object with only strategy and explanation. Strategy must be tile_nearest, tile_radius_count, or no_repair. Choose tile_nearest only for a nearest-index output contract, tile_radius_count only for a radius-count output contract, and no_repair if neither preserves the required output. Do not write code. Independent validation decides acceptance." },
          { role: "user", content: JSON.stringify({
            contract: result.contract,
            source: result.source,
            measured_baseline: result.baseline,
            measured_candidate: result.candidate,
            candidate_contract_checks: result.checks,
            memory_limit_mb: result.memory_limit_mb,
            latency_limit_ms: result.latency_limit_ms,
          }) },
        ],
      }),
      cache: "no-store",
      signal: AbortSignal.timeout(25_000),
    });
    if (!response.ok) {
      return NextResponse.json({ error: `Nebius returned HTTP ${response.status}.` }, { status: 502 });
    }
    const data = await response.json();
    const strategy = parseStrategy(String(data?.choices?.[0]?.message?.content ?? ""));
    if (!strategy) {
      return NextResponse.json({ error: "Nemotron returned no supported strategy." }, { status: 502 });
    }
    return NextResponse.json({
      status: "live", model, strategy,
      agrees_with_validator: strategy === expected[caseId],
      evidence_captured_at: report.captured_at_utc,
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "The live model decision could not complete." }, { status: 502 });
  }
}
