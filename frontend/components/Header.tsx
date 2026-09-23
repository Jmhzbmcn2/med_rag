"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { health } from "../lib/api";

export function Header() {
  const pathname = usePathname();
  const [status, setStatus] = useState<"connecting" | "offline" | number>("connecting");

  useEffect(() => {
    health().then((h) => setStatus(h ? h.points : "offline"));
  }, []);

  const label =
    status === "connecting" ? "Connecting…" : status === "offline" ? "Offline" : `Knowledge base ready · ${status} chunks`;

  return (
    <header>
      <div className="header-left">
        <div className="brand">
          <div className="brand-badge">✚</div>
          <span>MediRAG</span>
        </div>
        <nav className="app-nav" aria-label="Primary">
          <Link className={pathname === "/" ? "active" : ""} href="/">Chat</Link>
          <Link className={pathname === "/retrieval" ? "active" : ""} href="/retrieval">Retrieval</Link>
        </nav>
      </div>
      <div className={"status" + (status === "offline" ? " offline" : "")}>
        <span className="dot" />
        <span>{label}</span>
      </div>
    </header>
  );
}
