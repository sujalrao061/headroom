import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const evidenceDir = join(process.cwd(), "evidence");
    const gpuBytes = await readFile(join(evidenceDir, "attention-gpu.json"));
    const gpu = JSON.parse(gpuBytes.toString("utf8"));
    const digest = createHash("sha256").update(gpuBytes).digest("hex");
    const decision = JSON.parse(await readFile(join(evidenceDir, "attention-decision-live.json"), "utf8"));
    const verified = decision.source_sha256 === digest && decision.model?.status === "live";
    const result = {
      ...gpu,
      recorded_at: gpu.captured_at_utc,
      evidence_verified: verified,
      model_agrees: verified ? decision.model_agrees : null,
      nemotron: verified
        ? { status: "live", strategy: decision.model.strategy, model: decision.model.model_id, explanation: "Recorded Nebius response tied to this GPU report." }
        : { status: "not_verified", strategy: null, model: null, explanation: "No verified model decision is attached to this GPU report." },
    };
    return NextResponse.json(result, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "Recorded attention evidence is unavailable." }, { status: 404 });
  }
}
