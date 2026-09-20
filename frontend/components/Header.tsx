"use client";

import { useEffect, useState } from "react";
import { health } from "../lib/api";

export function Header() {
  const [status, setStatus] = useState<"connecting" | "offline" | number>("connecting");

  useEffect(() => {
    health().then((h) => setStatus(h ? h.points : "offline"));
  }, []);

  const label =
    status === "connecting" ? "Connecting…" : status === "offline" ? "Offline" : `Knowledge base ready · ${status} chunks`;

  return (
    <header>
      <div className="brand">
        <div className="brand-badge">✚</div>
        <span>MediRAG</span>
      </div>
      <div className={"status" + (status === "offline" ? " offline" : "")}>
        <span className="dot" />
        <span>{label}</span>
      </div>
    </header>
  );
}
