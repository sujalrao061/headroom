import { NextResponse } from "next/server";
import report from "../../../evidence/contract-gauntlet.json";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(report, { headers: { "Cache-Control": "public, max-age=60, stale-while-revalidate=300" } });
}
