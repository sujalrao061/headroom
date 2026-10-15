import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  try {
    const response = await fetch(`${process.env.HEADROOM_RUNNER_URL ?? "http://127.0.0.1:8000"}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error("Runner unavailable");
    const data = await response.json();
    return NextResponse.json({ runner_online: true, supported_cases: Array.isArray(data.supported_cases) ? data.supported_cases : ["nearest", "matrix", "radius"], attention_supported: data.attention_supported === true });
  } catch {
    return NextResponse.json({ runner_online: false, supported_cases: ["nearest", "matrix", "radius"], attention_supported: false });
  }
}
