import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 45_000);
    try {
      const response = await fetch(`${process.env.HEADROOM_RUNNER_URL ?? "http://127.0.0.1:8000"}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
        cache: "no-store",
      });
      const data = await response.json();
      return NextResponse.json(data, { status: response.status });
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return NextResponse.json({ error: "GPU runner is unavailable. Start the Python runner and try again." }, { status: 503 });
  }
}
