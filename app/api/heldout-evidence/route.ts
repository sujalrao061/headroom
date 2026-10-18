import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const report = JSON.parse(await readFile(join(process.cwd(), "evidence", "heldout-live.json"), "utf8"));
    return NextResponse.json({
      captured_at_utc: report.captured_at_utc,
      total_cases: report.total_cases,
      rule_correct: report.rule_correct,
      model_correct: report.model_correct,
      model_completed: report.model_completed,
      model_id: report.rows?.find((row: { nemotron?: { model?: string } }) => row.nemotron?.model)?.nemotron.model ?? null,
      scope: report.protocol,
      cases: report.rows?.map((row: { id: string; expected_strategy: string; rule: { correct: boolean }; nemotron: { status: string; correct: boolean | null } }) => ({
        id: row.id,
        expected_strategy: row.expected_strategy,
        rule_correct: row.rule.correct,
        model_status: row.nemotron.status,
        model_correct: row.nemotron.correct,
      })),
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "Saved model evaluation is unavailable." }, { status: 404 });
  }
}
