import type { NextConfig } from "next";

const config: NextConfig = {
  // The default 30 s proxy timeout is shorter than a slow LLM call.
  experimental: { proxyTimeout: 120_000 },
  async rewrites() {
    const api = process.env.API_URL ?? "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
  },
};

export default config;
