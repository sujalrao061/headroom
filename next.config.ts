import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  // Keep production builds from replacing chunks used by a running dev server.
  distDir: process.env.NODE_ENV === "production" ? ".next-build" : ".next",
  outputFileTracingIncludes: {
    "/api/recorded-run": ["./evidence/local-gpu.json"],
    "/api/attention-evidence": ["./evidence/attention-gpu.json", "./evidence/attention-decision-live.json"],
  },
};
export default nextConfig;
