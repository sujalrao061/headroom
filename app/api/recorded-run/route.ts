import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const cases = new Set(["nearest", "matrix", "radius"]);

export async function GET(request: NextRequest) {
  const caseId = request.nextUrl.searchParams.get("case_id") ?? "nearest";
  if (!cases.has(caseId)) {
    return NextResponse.json({ error: "Unknown recorded workload." }, { status: 400 });
  }
  try {
    const report = JSON.parse(await readFile(join(process.cwd(), "evidence", "local-gpu.json"), "utf8"));
    const result = report.results.find((item: { case_id: string }) => item.case_id === caseId);
    if (!result) throw new Error("Missing evidence");
    return NextResponse.json({
      ...result,
      recorded_at: report.captured_at_utc,
      candidate_origin: "fixed local candidate in a recorded GPU run",
      nemotron: {
        status: "recorded_no_call",
        model: null,
        strategy: null,
        explanation: "This saved GPU run did not call Nemotron. Separate recorded model evaluations are documented in the evidence folder.",
      },
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "Recorded GPU evidence is unavailable." }, { status: 404 });
  }
}
