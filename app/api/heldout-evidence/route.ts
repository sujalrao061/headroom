import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function readReport(file: string, label: string) {
  const report = JSON.parse(await readFile(join(process.cwd(), "evidence", file), "utf8"));
  return {
    label,
    file,
    captured_at_utc: report.captured_at_utc,
    total_cases: report.total_cases,
    rule_correct: report.rule_correct,
    model_correct: report.model_correct,
    model_completed: report.model_completed,
    model_id: report.rows?.find((row: { nemotron?: { model?: string } }) => row.nemotron?.model)?.nemotron.model ?? null,
    cases: report.rows?.map((row: { id: string; expected_strategy: string; rule: { correct: boolean }; nemotron: { status: string; correct: boolean | null; attempt_count?: number } }) => ({
      id: row.id,
      expected_strategy: row.expected_strategy,
      rule_correct: row.rule.correct,
      model_status: row.nemotron.status,
      model_correct: row.nemotron.correct,
      attempt_count: row.nemotron.attempt_count ?? 1,
    })),
  };
}

export async function GET() {
  try {
    const original = await readReport("heldout-live.json", "Original frozen set");
    const followUp = await readReport("heldout-v2-live.json", "Fresh follow-up set");
    return NextResponse.json({ reports: [original, followUp] }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ error: "Saved model evaluations are unavailable." }, { status: 404 });
  }
}
