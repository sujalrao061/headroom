import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const report = JSON.parse(await readFile(join(process.cwd(), "evidence", "contract-gauntlet.json"), "utf8"));
    return NextResponse.json(report, { headers: { "Cache-Control": "public, max-age=60, stale-while-revalidate=300" } });
  } catch {
    return NextResponse.json({ error: "Correctness gauntlet evidence is unavailable." }, { status: 404 });
  }
}
