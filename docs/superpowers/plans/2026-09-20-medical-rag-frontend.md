# Medical RAG Next.js Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vanilla `backend/medical_rag/static/index.html` with a Next.js app that keeps its behaviour and adds a local multi-conversation sidebar.

**Architecture:** `frontend/` is a separate Next 15 (App Router, TypeScript) app on port 3000. The browser only calls same-origin `/api/*`; `next.config.ts` rewrites proxy it to FastAPI on port 8000, so there is no CORS and no backend API change. All state is client side; pure logic lives in `lib/` and is covered by Vitest, React components stay thin. Conversations persist in `localStorage`.

**Tech Stack:** Next 15, React 19, TypeScript, plain global CSS (copied from the template), Vitest, npm.

**Spec:** `docs/superpowers/specs/2026-09-20-medical-rag-frontend-design.md`. Visual template: `medical_rag_ui.html` (repo root). Existing behaviour to preserve: `backend/medical_rag/static/index.html` (read its `<script>` before deleting it in Task 7).

## Global Constraints

- Backend API is unchanged: `POST /api/chat` `{question}` (1 to 1000 chars) returns `{answer, sources: [{n, title, url, section_path, type, text, score}]}`; `GET /api/health` returns `{status, points}`. `score` is `null` without a reranker.
- The browser calls only relative `/api/*` URLs. `API_URL` (server side env, default `http://localhost:8000`) is read only in `next.config.ts`, never exposed to the client.
- `rewrites` proxy timeout: `experimental.proxyTimeout: 120000` (default 30 s is shorter than a slow LLM call). Client request timeout: 60 s (as in the old UI).
- Never use `dangerouslySetInnerHTML`. Model and user text render as React text so it is escaped.
- `localStorage` key `medirag.conversations.v1`. Every `localStorage` access is inside `try/catch`. Pending messages are never saved.
- UI copy stays English as in the template; `<html lang="vi">` as in the spec, with `suppressHydrationWarning` on `<html>` and `<body>` (browser extensions inject attributes). System font stack, no `next/font`.
- Disclaimer text: `⚠️ AI-generated information — not a medical diagnosis.` Rendered once at the bottom of the MessageList result feed.
- Do not modify or delete `medical_rag_ui.html` (untracked reference file).
- Run frontend commands from `frontend/`, backend commands from `backend/`.
- Commit messages end with the trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (second `-m`).

## File Structure

```
frontend/
  package.json, tsconfig.json, next.config.ts, vitest.config.ts, .gitignore
  app/layout.tsx, app/page.tsx, app/globals.css
  components/Header.tsx, Sidebar.tsx, MessageList.tsx, Message.tsx, Composer.tsx, SourcesPanel.tsx, Disclaimer.tsx
  hooks/useChat.ts
  lib/api.ts, lib/citations.ts, lib/sources.ts, lib/conversations.ts
  lib/api.test.ts, lib/citations.test.ts, lib/sources.test.ts, lib/conversations.test.ts
```

One responsibility per file: `api.ts` is the only file that calls `fetch`; `conversations.ts` is the only file that touches storage and owns the state transitions; components are presentational; `useChat.ts` glues them.

---

### Task 1: Scaffold the Next.js app

**Files:**
- Create: `frontend/package.json`, `frontend/tsconfig.json`, `frontend/next.config.ts`, `frontend/vitest.config.ts`, `frontend/.gitignore`, `frontend/app/layout.tsx`, `frontend/app/page.tsx`

**Interfaces:**
- Produces: `npm run dev|build|start|test` scripts; `/api/*` proxy to `API_URL`; empty `app/page.tsx` that later tasks replace.

- [ ] **Step 1: Create the manifest and install dependencies**

Create `frontend/package.json`:

```json
{
  "name": "medirag-frontend",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "vitest run"
  }
}
```

Run:

```bash
cd frontend
npm install next@15 react@19 react-dom@19
npm install -D typescript @types/node @types/react @types/react-dom vitest
```

Expected: `package.json` gains `dependencies` and `devDependencies`, `package-lock.json` is created.

- [ ] **Step 2: Add config files**

`frontend/.gitignore`:

```
node_modules/
.next/
next-env.d.ts
```

`frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`frontend/next.config.ts`:

```ts
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
```

`frontend/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "node", include: ["lib/**/*.test.ts"] },
});
```

- [ ] **Step 3: Add a placeholder layout and page**

`frontend/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "MediRAG - Medical AI Assistant" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
```

`frontend/app/globals.css` (replaced in Task 5):

```css
body{margin:0}
```

`frontend/app/page.tsx` (replaced in Task 5):

```tsx
export default function Page() {
  return <main>MediRAG</main>;
}
```

- [ ] **Step 4: Verify the build**

Run: `cd frontend && npm run build`
Expected: build succeeds, output lists route `/`. If Next rewrites `tsconfig.json` on first build, keep its changes.

- [ ] **Step 5: Commit**

```bash
git add frontend
git commit -m "feat: scaffold Next.js frontend with /api proxy" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: API client

**Files:**
- Create: `frontend/lib/api.ts`
- Test: `frontend/lib/api.test.ts`

**Interfaces:**
- Produces:
  - `type Source = { n: number; title: string; url: string; section_path: string; type: string; text: string; score: number | null }`
  - `type ChatResponse = { answer: string; sources: Source[] }`
  - `chat(question: string): Promise<ChatResponse>` throws `Error` with a user-facing message.
  - `health(): Promise<{ points: number } | null>` never throws.

- [ ] **Step 1: Write the failing tests**

`frontend/lib/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { chat, health } from "./api";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("chat", () => {
  it("posts the question and returns the payload", async () => {
    const payload = { answer: "ok [1]", sources: [] };
    const fetchMock = vi.fn().mockResolvedValue(json(payload));
    vi.stubGlobal("fetch", fetchMock);

    expect(await chat("đau đầu?")).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/chat");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "đau đầu?" });
  });

  it("uses the backend detail for a 502", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "LLM error: boom" }, 502)));
    await expect(chat("q")).rejects.toThrow("LLM error: boom");
  });

  it("falls back to a generic message when detail is not a string (422)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: [{ msg: "too short" }] }, 422)));
    await expect(chat("q")).rejects.toThrow("Request failed (422)");
  });

  it("falls back to a generic message when the body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 500 })));
    await expect(chat("q")).rejects.toThrow("Request failed (500)");
  });

  it("maps a timeout to a readable message", async () => {
    const timeout = Object.assign(new Error("timed out"), { name: "TimeoutError" });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(timeout));
    await expect(chat("q")).rejects.toThrow("Request timed out");
  });

  it("passes network errors through", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(chat("q")).rejects.toThrow("Failed to fetch");
  });
});

describe("health", () => {
  it("returns the chunk count", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ status: "ok", points: 1234 })));
    expect(await health()).toEqual({ points: 1234 });
  });

  it("returns null on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({}, 500)));
    expect(await health()).toBeNull();
  });

  it("returns null on a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    expect(await health()).toBeNull();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run lib/api.test.ts`
Expected: FAIL, cannot resolve `./api`.

- [ ] **Step 3: Implement**

`frontend/lib/api.ts`:

```ts
export type Source = {
  n: number;
  title: string;
  url: string;
  section_path: string;
  type: string;
  text: string;
  score: number | null;
};

export type ChatResponse = { answer: string; sources: Source[] };

const CHAT_TIMEOUT_MS = 60_000;

export async function chat(question: string): Promise<ChatResponse> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(CHAT_TIMEOUT_MS),
    });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") throw new Error("Request timed out");
    throw err;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
  }
  return data as ChatResponse;
}

export async function health(): Promise<{ points: number } | null> {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return null;
    const data = await res.json();
    return typeof data.points === "number" ? { points: data.points } : null;
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run lib/api.test.ts`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/api.ts frontend/lib/api.test.ts
git commit -m "feat: frontend API client with typed errors" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Citation and source helpers

**Files:**
- Create: `frontend/lib/citations.ts`, `frontend/lib/sources.ts`
- Test: `frontend/lib/citations.test.ts`, `frontend/lib/sources.test.ts`

**Interfaces:**
- Produces:
  - `type Part = { kind: "text"; value: string } | { kind: "cite"; n: number }`
  - `splitCitations(text: string, sourceCount: number): Part[]`
  - `passageOf(text: string): string` drops the leading `[Chủ đề: … | Nguồn: …]` header, cuts to 300 characters plus `…`.
  - `isHttpUrl(url: string): boolean`

- [ ] **Step 1: Write the failing tests**

`frontend/lib/citations.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { splitCitations } from "./citations";

const cites = (text: string, count: number) =>
  splitCitations(text, count).flatMap((p) => (p.kind === "cite" ? [p.n] : []));
const joinedText = (text: string, count: number) =>
  splitCitations(text, count)
    .map((p) => (p.kind === "text" ? p.value : `<${p.n}>`))
    .join("");

describe("splitCitations", () => {
  it("turns valid [n] into cites", () => {
    expect(splitCitations("Sốt cao [1] và ho [2].", 2)).toEqual([
      { kind: "text", value: "Sốt cao " },
      { kind: "cite", n: 1 },
      { kind: "text", value: " và ho " },
      { kind: "cite", n: 2 },
      { kind: "text", value: "." },
    ]);
  });

  it("keeps out-of-range and zero markers as text", () => {
    expect(cites("a [0] b [3] c", 2)).toEqual([]);
    expect(joinedText("a [0] b [3] c", 2)).toBe("a [0] b [3] c");
  });

  it("handles adjacent citations", () => {
    expect(cites("x [1][2]", 2)).toEqual([1, 2]);
  });

  it("returns a single text part when there are no markers", () => {
    expect(splitCitations("plain", 3)).toEqual([{ kind: "text", value: "plain" }]);
  });

  it("returns no parts for empty text", () => {
    expect(splitCitations("", 3)).toEqual([]);
  });

  it("passes HTML through untouched as text (React escapes it)", () => {
    expect(splitCitations("<img src=x onerror=alert(1)> [1]", 1)).toEqual([
      { kind: "text", value: "<img src=x onerror=alert(1)> " },
      { kind: "cite", n: 1 },
    ]);
  });

  it("treats every marker as text when there are no sources", () => {
    expect(cites("a [1]", 0)).toEqual([]);
  });
});
```

`frontend/lib/sources.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { isHttpUrl, passageOf } from "./sources";

describe("passageOf", () => {
  it("drops the leading bracket header", () => {
    expect(passageOf("[Chủ đề: COPD | Nguồn: YouMed] Ho kéo dài.")).toBe("Ho kéo dài.");
  });

  it("keeps brackets that are not at the start", () => {
    expect(passageOf("Ho [kéo dài]")).toBe("Ho [kéo dài]");
  });

  it("cuts long text to 300 characters plus an ellipsis", () => {
    const out = passageOf("a".repeat(400));
    expect(out).toBe("a".repeat(300) + "…");
  });

  it("leaves text of exactly 300 characters alone", () => {
    expect(passageOf("a".repeat(300))).toBe("a".repeat(300));
  });
});

describe("isHttpUrl", () => {
  it("accepts http and https", () => {
    expect(isHttpUrl("https://youmed.vn/x")).toBe(true);
    expect(isHttpUrl("http://youmed.vn/x")).toBe(true);
  });

  it("rejects other schemes and empty values", () => {
    expect(isHttpUrl("javascript:alert(1)")).toBe(false);
    expect(isHttpUrl("")).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run lib/citations.test.ts lib/sources.test.ts`
Expected: FAIL, cannot resolve `./citations` and `./sources`.

- [ ] **Step 3: Implement**

`frontend/lib/citations.ts`:

```ts
export type Part = { kind: "text"; value: string } | { kind: "cite"; n: number };

// A [n] becomes a citation chip only when it points at a real source.
export function splitCitations(text: string, sourceCount: number): Part[] {
  return text.split(/(\[\d+\])/).flatMap((part): Part[] => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (m && +m[1] >= 1 && +m[1] <= sourceCount) return [{ kind: "cite", n: +m[1] }];
    return part ? [{ kind: "text", value: part }] : [];
  });
}
```

`frontend/lib/sources.ts`:

```ts
const MAX_PASSAGE = 300;

// Chunks start with a "[Chủ đề: … | Nguồn: …]" header that the source card already shows elsewhere.
export function passageOf(text: string): string {
  const body = text.replace(/^\[[^\]]*\]\s*/, "");
  return body.length > MAX_PASSAGE ? body.slice(0, MAX_PASSAGE) + "…" : body;
}

export const isHttpUrl = (url: string): boolean => /^https?:\/\//.test(url);
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run lib/citations.test.ts lib/sources.test.ts`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/citations.ts frontend/lib/sources.ts frontend/lib/citations.test.ts frontend/lib/sources.test.ts
git commit -m "feat: citation splitting and source passage helpers" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Conversation state and storage

**Files:**
- Create: `frontend/lib/conversations.ts`
- Test: `frontend/lib/conversations.test.ts`

**Interfaces:**
- Consumes: `Source` from `./api`.
- Produces:
  - `type Message = { id: string; role: "user" | "assistant"; text: string; sources?: Source[]; error?: boolean; pending?: boolean }`
  - `type Conversation = { id: string; title: string; createdAt: number; messages: Message[] }`
  - `type StorageLike = Pick<Storage, "getItem" | "setItem">`
  - `uid(): string`
  - `titleOf(question: string): string`
  - `groupByDay(cs: Conversation[], now: number): { today: Conversation[]; previous: Conversation[] }` (each newest first)
  - `startTurn(cs, convId, question, now, userId, pendingId): Conversation[]`
  - `finishTurn(cs, convId, messageId, patch: Partial<Message>): Conversation[]`
  - `sourcesFor(conv: Conversation | undefined, selectedId: string | null): Source[]`
  - `load(storage?: StorageLike): Conversation[]`, `save(cs: Conversation[], storage?: StorageLike): void`

- [ ] **Step 1: Write the failing tests**

`frontend/lib/conversations.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { Source } from "./api";
import {
  type Conversation,
  type StorageLike,
  finishTurn,
  groupByDay,
  load,
  save,
  sourcesFor,
  startTurn,
  titleOf,
  uid,
} from "./conversations";

const src = (n: number): Source => ({
  n, title: `t${n}`, url: "https://x", section_path: "s", type: "disease", text: "x", score: null,
});
const conv = (id: string, createdAt: number, messages: Conversation["messages"] = []): Conversation => ({
  id, title: id, createdAt, messages,
});
const memory = (): StorageLike & { data: Map<string, string> } => {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v) };
};

describe("uid", () => {
  it("returns distinct ids", () => {
    expect(uid()).not.toBe(uid());
  });
});

describe("titleOf", () => {
  it("trims and keeps short questions", () => {
    expect(titleOf("  đau đầu?  ")).toBe("đau đầu?");
  });

  it("cuts to 40 characters plus an ellipsis", () => {
    expect(titleOf("a".repeat(50))).toBe("a".repeat(40) + "…");
  });
});

describe("groupByDay", () => {
  it("splits at local midnight and sorts newest first", () => {
    const now = new Date(2026, 8, 20, 0, 30).getTime();
    const yesterday = conv("y", new Date(2026, 8, 19, 23, 59).getTime());
    const early = conv("t1", new Date(2026, 8, 20, 0, 1).getTime());
    const later = conv("t2", new Date(2026, 8, 20, 0, 20).getTime());
    const g = groupByDay([yesterday, early, later], now);
    expect(g.today.map((c) => c.id)).toEqual(["t2", "t1"]);
    expect(g.previous.map((c) => c.id)).toEqual(["y"]);
  });
});

describe("startTurn", () => {
  it("creates a titled conversation when the id is new", () => {
    const cs = startTurn([], "c1", "đau đầu?", 1000, "u1", "p1");
    expect(cs).toHaveLength(1);
    expect(cs[0]).toMatchObject({ id: "c1", title: "đau đầu?", createdAt: 1000 });
    expect(cs[0].messages).toEqual([
      { id: "u1", role: "user", text: "đau đầu?" },
      { id: "p1", role: "assistant", text: "", pending: true },
    ]);
  });

  it("appends to an existing conversation and leaves others alone", () => {
    const other = conv("c2", 1);
    const first = startTurn([other], "c1", "q1", 1000, "u1", "p1");
    const both = startTurn(first, "c1", "q2", 2000, "u2", "p2");
    expect(both.find((c) => c.id === "c2")).toBe(other);
    expect(both.find((c) => c.id === "c1")!.messages.map((m) => m.id)).toEqual(["u1", "p1", "u2", "p2"]);
    expect(both.find((c) => c.id === "c1")!.title).toBe("q1");
  });
});

describe("finishTurn", () => {
  it("patches only the target message in the target conversation", () => {
    const a = startTurn([], "a", "qa", 1, "ua", "pa");
    const cs = startTurn(a, "b", "qb", 2, "ub", "pb");
    const done = finishTurn(cs, "a", "pa", { text: "answer", sources: [src(1)], pending: false });
    const ca = done.find((c) => c.id === "a")!;
    expect(ca.messages[1]).toMatchObject({ text: "answer", pending: false, sources: [src(1)] });
    expect(ca.messages[0].text).toBe("qa");
    expect(done.find((c) => c.id === "b")).toBe(cs.find((c) => c.id === "b"));
  });

  it("is a no-op for an unknown conversation", () => {
    const cs = startTurn([], "a", "qa", 1, "ua", "pa");
    expect(finishTurn(cs, "zzz", "pa", { text: "x" })).toEqual(cs);
  });
});

describe("sourcesFor", () => {
  const c = conv("c", 1, [
    { id: "u1", role: "user", text: "q1" },
    { id: "a1", role: "assistant", text: "r1", sources: [src(1)] },
    { id: "u2", role: "user", text: "q2" },
    { id: "a2", role: "assistant", text: "r2", sources: [src(2), src(3)] },
    { id: "u3", role: "user", text: "q3" },
    { id: "a3", role: "assistant", text: "boom", error: true },
  ]);

  it("returns the selected message's sources", () => {
    expect(sourcesFor(c, "a1")).toEqual([src(1)]);
  });

  it("defaults to the latest message that has sources, skipping errors", () => {
    expect(sourcesFor(c, null)).toEqual([src(2), src(3)]);
  });

  it("falls back to the latest when the selection has no sources", () => {
    expect(sourcesFor(c, "a3")).toEqual([src(2), src(3)]);
  });

  it("returns an empty list without a conversation", () => {
    expect(sourcesFor(undefined, null)).toEqual([]);
  });
});

describe("load / save", () => {
  it("round-trips and drops pending messages", () => {
    const store = memory();
    const cs = startTurn([], "c1", "q", 1, "u1", "p1");
    save(cs, store);
    const loaded = load(store);
    expect(loaded[0].messages.map((m) => m.id)).toEqual(["u1"]);
  });

  it("returns [] for corrupt JSON", () => {
    const store = memory();
    store.data.set("medirag.conversations.v1", "{not json");
    expect(load(store)).toEqual([]);
  });

  it("returns [] when the stored value is not an array", () => {
    const store = memory();
    store.data.set("medirag.conversations.v1", '{"a":1}');
    expect(load(store)).toEqual([]);
  });

  it("returns [] and does not throw when storage is blocked", () => {
    const blocked: StorageLike = {
      getItem: () => { throw new Error("blocked"); },
      setItem: () => { throw new Error("blocked"); },
    };
    expect(load(blocked)).toEqual([]);
    expect(() => save([], blocked)).not.toThrow();
  });

  it("returns [] when no storage exists at all (node has no localStorage)", () => {
    expect(load()).toEqual([]);
    expect(() => save([])).not.toThrow();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run lib/conversations.test.ts`
Expected: FAIL, cannot resolve `./conversations`.

- [ ] **Step 3: Implement**

`frontend/lib/conversations.ts`:

```ts
import type { Source } from "./api";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  error?: boolean;
  pending?: boolean;
};

export type Conversation = { id: string; title: string; createdAt: number; messages: Message[] };

export type StorageLike = Pick<Storage, "getItem" | "setItem">;

const KEY = "medirag.conversations.v1";
const TITLE_MAX = 40;

// crypto.randomUUID needs a secure context; the app may be opened over plain http on a LAN address.
export const uid = (): string => Math.random().toString(36).slice(2) + Date.now().toString(36);

export function titleOf(question: string): string {
  const q = question.trim();
  return q.length > TITLE_MAX ? q.slice(0, TITLE_MAX) + "…" : q;
}

export function groupByDay(cs: Conversation[], now: number) {
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  const sorted = [...cs].sort((a, b) => b.createdAt - a.createdAt);
  return {
    today: sorted.filter((c) => c.createdAt >= midnight.getTime()),
    previous: sorted.filter((c) => c.createdAt < midnight.getTime()),
  };
}

// Adds the user question and a pending assistant message; creates the conversation if convId is new.
export function startTurn(
  cs: Conversation[],
  convId: string,
  question: string,
  now: number,
  userId: string,
  pendingId: string,
): Conversation[] {
  const turn: Message[] = [
    { id: userId, role: "user", text: question },
    { id: pendingId, role: "assistant", text: "", pending: true },
  ];
  if (cs.some((c) => c.id === convId)) {
    return cs.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, ...turn] } : c));
  }
  return [...cs, { id: convId, title: titleOf(question), createdAt: now, messages: turn }];
}

// Targets the conversation by id, so an answer lands where it was asked even if the user moved away.
export function finishTurn(
  cs: Conversation[],
  convId: string,
  messageId: string,
  patch: Partial<Message>,
): Conversation[] {
  return cs.map((c) =>
    c.id !== convId
      ? c
      : { ...c, messages: c.messages.map((m) => (m.id === messageId ? { ...m, ...patch } : m)) },
  );
}

export function sourcesFor(conv: Conversation | undefined, selectedId: string | null): Source[] {
  if (!conv) return [];
  const picked = conv.messages.find((m) => m.id === selectedId);
  if (picked?.sources) return picked.sources;
  const latest = [...conv.messages].reverse().find((m) => m.sources);
  return latest?.sources ?? [];
}

export function load(storage?: StorageLike): Conversation[] {
  try {
    const raw = (storage ?? localStorage).getItem(KEY);
    const data = raw ? JSON.parse(raw) : [];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

export function save(cs: Conversation[], storage?: StorageLike): void {
  try {
    const clean = cs.map((c) => ({ ...c, messages: c.messages.filter((m) => !m.pending) }));
    (storage ?? localStorage).setItem(KEY, JSON.stringify(clean));
  } catch {
    // storage blocked or full: the app keeps working, history just is not kept
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run lib/conversations.test.ts`
Expected: 17 passed.

- [ ] **Step 5: Run the whole suite and commit**

Run: `cd frontend && npm test`
Expected: all test files pass (39 tests total).

```bash
git add frontend/lib/conversations.ts frontend/lib/conversations.test.ts
git commit -m "feat: conversation state helpers and localStorage persistence" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: UI (styles, components, hook, page)

**Files:**
- Modify: `frontend/app/globals.css`, `frontend/app/page.tsx`
- Create: `frontend/hooks/useChat.ts`, `frontend/components/Header.tsx`, `Sidebar.tsx`, `MessageList.tsx`, `Message.tsx`, `Composer.tsx`, `SourcesPanel.tsx`, `Disclaimer.tsx`

**Interfaces:**
- Consumes: everything from Tasks 2 to 4.
- Produces: the working page. `useChat()` returns `{ conversations, activeId, active, selectedId, sources, pending, send(q), newChat(), select(id), selectMessage(id) }`.

No unit tests here (components and hook are thin glue; the logic they call is tested in Tasks 2 to 4). Verification is `tsc`, `build` and the manual check in Task 6.

- [ ] **Step 1: Replace `frontend/app/globals.css`**

Template CSS unchanged, plus the additions the old UI needed (`pending`, `error`, `badge`, `offline`, `empty`) and button resets for the sidebar items:

```css
:root{
  --bg:#f5f8fb;
  --panel:#ffffff;
  --border:#e4eaf0;
  --text:#1f2937;
  --muted:#6b7280;
  --primary:#1477e6;
  --primary-soft:#eaf4ff;
  --success:#16a34a;
  --warning:#b45309;
  --shadow:0 8px 24px rgba(15,23,42,.06);
}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--text);
  background:var(--bg);
  height:100vh;
  overflow:hidden;
}
.app{
  height:100vh;
  display:grid;
  grid-template-rows:64px 1fr;
}
header{
  background:rgba(255,255,255,.95);
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 22px;
  backdrop-filter:blur(10px);
}
.brand{display:flex;align-items:center;gap:11px;font-weight:800;font-size:19px}
.brand-badge{
  width:36px;height:36px;border-radius:12px;background:var(--primary);
  display:grid;place-items:center;color:#fff;font-size:20px;
  box-shadow:0 6px 16px rgba(20,119,230,.25)
}
.status{
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;border-radius:999px;
  color:#166534;background:#effcf3;border:1px solid #ccefd7;
  font-size:13px;font-weight:700;
}
.dot{width:8px;height:8px;border-radius:50%;background:var(--success)}
.status.offline{color:#991b1b;background:#fef2f2;border-color:#fecaca}
.status.offline .dot{background:#dc2626}
.layout{
  min-height:0;
  display:grid;
  grid-template-columns:240px minmax(420px,1fr) 300px;
}
aside,.sources{
  background:var(--panel);
  min-height:0;
  overflow:auto;
}
aside{border-right:1px solid var(--border);padding:16px}
.sources{border-left:1px solid var(--border);padding:16px}
.new-chat{
  width:100%;border:0;border-radius:12px;padding:11px 14px;
  background:var(--primary);color:#fff;font-weight:800;cursor:pointer;
  box-shadow:0 6px 16px rgba(20,119,230,.18)
}
.section-title{
  margin:20px 4px 8px;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:#9ca3af;font-weight:800;
}
.chat-item{
  display:block;width:100%;text-align:left;border:0;background:none;font:inherit;
  padding:11px 12px;border-radius:10px;margin-bottom:5px;
  font-size:14px;cursor:pointer;color:#374151;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.chat-item:hover{background:#f3f6fa}
.chat-item.active{background:var(--primary-soft);color:#0f5db6;font-weight:700}
.main{
  min-width:0;min-height:0;
  display:grid;grid-template-rows:auto 1fr auto;
  background:linear-gradient(180deg,#f9fbfd 0%,#f4f7fa 100%);
}
.main-head{
  padding:18px 24px;border-bottom:1px solid var(--border);
  background:rgba(255,255,255,.78);
}
.main-head h1{font-size:18px;margin:0 0 4px}
.main-head p{font-size:13px;color:var(--muted);margin:0}
.messages{
  overflow:auto;padding:24px;display:flex;flex-direction:column;gap:18px;
}
.row{display:flex;gap:10px;align-items:flex-start}
.row.user{justify-content:flex-end}
.avatar{
  width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
  background:#e9eef5;flex:0 0 auto;
}
.assistant .avatar{background:#eaf4ff;color:#0f5db6}
.bubble{
  max-width:72%;padding:13px 15px;border-radius:16px;
  line-height:1.55;font-size:14px;box-shadow:var(--shadow);
  white-space:pre-wrap;overflow-wrap:anywhere;
}
.user .bubble{
  background:var(--primary);color:white;border-bottom-right-radius:5px;
}
.assistant .bubble{
  background:white;border:1px solid var(--border);border-bottom-left-radius:5px;
}
.bubble.clickable{cursor:pointer}
.bubble.selected{border-color:var(--primary)}
.bubble.pending{color:var(--muted)}
.bubble.error{background:#fef2f2;border-color:#fecaca;color:#991b1b}
.citation{
  display:inline-flex;align-items:center;justify-content:center;
  height:20px;min-width:20px;padding:0 6px;margin-left:3px;
  border-radius:7px;background:var(--primary-soft);color:#0f5db6;
  font-size:11px;font-weight:800;vertical-align:1px;
}
.composer{
  padding:14px 20px 18px;border-top:1px solid var(--border);
  background:rgba(255,255,255,.88);
}
.input-wrap{
  display:flex;align-items:center;gap:10px;background:white;
  border:1px solid #d9e2ec;border-radius:14px;padding:8px 8px 8px 14px;
  box-shadow:var(--shadow);
}
input{
  flex:1;border:0;outline:none;font-size:14px;background:transparent;
}
.send{
  width:38px;height:38px;border:0;border-radius:10px;
  background:var(--primary);color:white;font-size:18px;cursor:pointer;
}
.send:disabled{opacity:.5;cursor:default}
.sources h2{font-size:16px;margin:2px 0 14px}
.source-card{
  border:1px solid var(--border);border-radius:14px;padding:14px;
  margin-bottom:12px;background:#fff;box-shadow:var(--shadow);
}
.source-top{display:flex;gap:10px;align-items:flex-start}
.doc-icon{
  width:34px;height:34px;border-radius:9px;background:#f1f5f9;
  display:grid;place-items:center;flex:0 0 auto;
}
.source-card h3{font-size:13px;margin:0 0 4px}
.source-card h3 a{color:inherit;text-decoration:none}
.source-card h3 a:hover{text-decoration:underline}
.meta{font-size:12px;color:var(--muted)}
.badge{
  display:inline-block;margin-top:4px;padding:1px 8px;border-radius:999px;
  background:#f1f5f9;color:#475569;font-size:11px;font-weight:700;
}
.score{
  margin-top:12px;padding-top:10px;border-top:1px solid #eef2f7;
  display:flex;justify-content:space-between;font-size:12px
}
.score strong{color:#0f5db6}
.passage{
  margin-top:9px;font-size:12px;color:#4b5563;line-height:1.5;
  background:#f8fafc;padding:9px;border-radius:9px;
}
.empty{font-size:13px;color:var(--muted)}
.message-disclaimer{
  text-align:center;
  font-size:12px;
  color:var(--muted);
  padding:12px 0 4px;
  margin-top:4px;
  user-select:none;
}
@media (max-width:980px){
  .layout{grid-template-columns:200px 1fr}
  .sources{display:none}
}
@media (max-width:700px){
  .layout{grid-template-columns:1fr}
  aside{display:none}
  .bubble{max-width:88%}
}
```

- [ ] **Step 2: Add the hook**

`frontend/hooks/useChat.ts`:

```ts
"use client";

import { useCallback, useEffect, useState } from "react";
import { chat } from "../lib/api";
import { finishTurn, load, save, sourcesFor, startTurn, uid } from "../lib/conversations";
import type { Conversation } from "../lib/conversations";

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null); // null = new, empty chat
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // Read storage after mount so server and client render the same first frame.
  useEffect(() => {
    setConversations(load());
    setHydrated(true);
  }, []);

  // Skip until hydrated, otherwise the initial [] would overwrite the stored history.
  useEffect(() => {
    if (hydrated) save(conversations);
  }, [conversations, hydrated]);

  const send = useCallback(
    async (raw: string) => {
      const question = raw.trim();
      if (!question || pending) return;
      const convId = activeId ?? uid();
      const pendingId = uid();
      setActiveId(convId);
      setSelectedId(null);
      setPending(true);
      setConversations((cs) => startTurn(cs, convId, question, Date.now(), uid(), pendingId));
      try {
        const data = await chat(question);
        setConversations((cs) =>
          finishTurn(cs, convId, pendingId, { text: data.answer, sources: data.sources, pending: false }),
        );
      } catch (err) {
        const text = err instanceof Error ? err.message : "Request failed";
        setConversations((cs) => finishTurn(cs, convId, pendingId, { text, error: true, pending: false }));
      } finally {
        setPending(false);
      }
    },
    [activeId, pending],
  );

  const newChat = useCallback(() => {
    setActiveId(null);
    setSelectedId(null);
  }, []);

  const select = useCallback((id: string) => {
    setActiveId(id);
    setSelectedId(null);
  }, []);

  const active = conversations.find((c) => c.id === activeId);

  return {
    conversations,
    activeId,
    active,
    selectedId,
    sources: sourcesFor(active, selectedId),
    pending,
    send,
    newChat,
    select,
    selectMessage: setSelectedId,
  };
}
```

- [ ] **Step 3: Add the presentational components**

`frontend/components/Header.tsx`:

```tsx
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
```

`frontend/components/Sidebar.tsx`:

```tsx
import { groupByDay } from "../lib/conversations";
import type { Conversation } from "../lib/conversations";

type Props = {
  conversations: Conversation[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
};

export function Sidebar({ conversations, activeId, onNew, onSelect }: Props) {
  const { today, previous } = groupByDay(conversations, Date.now());
  const group = (title: string, items: Conversation[]) =>
    items.length > 0 && (
      <>
        <div className="section-title">{title}</div>
        {items.map((c) => (
          <button
            key={c.id}
            className={"chat-item" + (c.id === activeId ? " active" : "")}
            onClick={() => onSelect(c.id)}
          >
            {c.title}
          </button>
        ))}
      </>
    );

  return (
    <aside>
      <button className="new-chat" onClick={onNew}>
        ＋ New Chat
      </button>
      {group("Today", today)}
      {group("Previous", previous)}
    </aside>
  );
}
```

`frontend/components/Message.tsx`:

```tsx
import { splitCitations } from "../lib/citations";
import type { Message as Msg } from "../lib/conversations";

type Props = { message: Msg; selected: boolean; onSelect: () => void };

export function Message({ message: m, selected, onSelect }: Props) {
  if (m.role === "user") {
    return (
      <div className="row user">
        <div className="bubble">{m.text}</div>
        <div className="avatar">👤</div>
      </div>
    );
  }

  const clickable = !!m.sources?.length;
  const classes = ["bubble", m.pending && "pending", m.error && "error", clickable && "clickable", selected && "selected"];
  const interactive = clickable
    ? {
        role: "button",
        tabIndex: 0,
        onClick: onSelect,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect();
          }
        },
      }
    : {};

  return (
    <div className="row assistant">
      <div className="avatar">🩺</div>
      <div className={classes.filter(Boolean).join(" ")} {...interactive}>
        {m.pending
          ? "Thinking…"
          : m.error
            ? m.text
            : splitCitations(m.text, m.sources?.length ?? 0).map((p, i) =>
                p.kind === "cite" ? (
                  <span key={i} className="citation">
                    {p.n}
                  </span>
                ) : (
                  p.value
                ),
              )}
      </div>
    </div>
  );
}
```

`frontend/components/MessageList.tsx`:

```tsx
"use client";

import { useEffect, useRef } from "react";
import type { Message as Msg } from "../lib/conversations";
import { Disclaimer } from "./Disclaimer";
import { Message } from "./Message";

type Props = { messages: Msg[]; selectedId: string | null; onSelect: (id: string) => void };

export function MessageList({ messages, selectedId, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight });
  }, [messages]);

  // One disclaimer at the end of the feed once any answer (or error) has been shown.
  const hasAssistant = messages.some((m) => m.role === "assistant" && !m.pending);

  return (
    <div className="messages" ref={box}>
      {messages.map((m) => (
        <Message key={m.id} message={m} selected={m.id === selectedId} onSelect={() => onSelect(m.id)} />
      ))}
      {hasAssistant && <Disclaimer />}
    </div>
  );
}
```

`frontend/components/Composer.tsx`:

```tsx
"use client";

import { useState } from "react";

type Props = { pending: boolean; onSend: (question: string) => void };

export function Composer({ pending, onSend }: Props) {
  const [value, setValue] = useState("");
  const empty = !value.trim();

  const submit = () => {
    if (empty || pending) return;
    onSend(value);
    setValue("");
  };

  return (
    <div className="composer">
      <div className="input-wrap">
        <input
          value={value}
          maxLength={1000}
          placeholder="Ask a medical question..."
          aria-label="Question"
          onChange={(e) => setValue(e.target.value)}
          // isComposing: Vietnamese IMEs confirm text with Enter
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) submit();
          }}
        />
        <button className="send" aria-label="Send" disabled={pending || empty} onClick={submit}>
          ➤
        </button>
      </div>
    </div>
  );
}
```

`frontend/components/SourcesPanel.tsx`:

```tsx
import type { Source } from "../lib/api";
import { isHttpUrl, passageOf } from "../lib/sources";

export function SourcesPanel({ sources }: { sources: Source[] }) {
  return (
    <section className="sources">
      <h2>Sources</h2>
      {sources.length === 0 && <div className="empty">Sources appear here after you ask a question.</div>}
      {sources.map((s) => {
        const label = `[${s.n}] ${s.title}`;
        return (
          <div className="source-card" key={s.n}>
            <div className="source-top">
              <div className="doc-icon">📄</div>
              <div>
                <h3>
                  {isHttpUrl(s.url) ? (
                    <a href={s.url} target="_blank" rel="noopener noreferrer">
                      {label}
                    </a>
                  ) : (
                    label
                  )}
                </h3>
                <div className="meta">{s.section_path}</div>
                <span className="badge">{s.type}</span>
              </div>
            </div>
            {s.score !== null && (
              <div className="score">
                <span>Reranker score</span>
                <strong>{s.score.toFixed(2)}</strong>
              </div>
            )}
            <div className="passage">{passageOf(s.text)}</div>
          </div>
        );
      })}
    </section>
  );
}
```

`frontend/components/Disclaimer.tsx`:

```tsx
export function Disclaimer() {
  return <div className="message-disclaimer">⚠️ AI-generated information — not a medical diagnosis.</div>;
}
```

- [ ] **Step 4: Replace `frontend/app/page.tsx`**

```tsx
"use client";

import { Composer } from "../components/Composer";
import { Header } from "../components/Header";
import { MessageList } from "../components/MessageList";
import { Sidebar } from "../components/Sidebar";
import { SourcesPanel } from "../components/SourcesPanel";
import { useChat } from "../hooks/useChat";

export default function Page() {
  const chat = useChat();

  return (
    <div className="app">
      <Header />
      <div className="layout">
        <Sidebar
          conversations={chat.conversations}
          activeId={chat.activeId}
          onNew={chat.newChat}
          onSelect={chat.select}
        />
        <main className="main">
          <div className="main-head">
            <h1>Medical AI Assistant</h1>
            <p>Ask questions grounded in your medical knowledge base.</p>
          </div>
          <MessageList
            messages={chat.active?.messages ?? []}
            selectedId={chat.selectedId}
            onSelect={chat.selectMessage}
          />
          <Composer pending={chat.pending} onSend={chat.send} />
        </main>
        <SourcesPanel sources={chat.sources} />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Type-check and build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: no type errors, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat: chat UI with sources panel and local conversation history" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Manual end-to-end check

**Files:** none (verification only; fix anything found in the files above, with a test first if the bug is in `lib/`).

Needs the backend prerequisites from `README.md`: `.env` filled in, the embedding server running on `EMBED_URL`, an ingested `qdrant_data/`.

- [ ] **Step 1: Start both servers**

```bash
# terminal 1, backend/
cd backend && uvicorn medical_rag.api:app --env-file ../.env --port 8000
# terminal 2, frontend/
cd frontend && npm run dev
```

Note: the API resolves `QDRANT_PATH` relative to its working directory. If it refuses to start, run from the repo root as in the README (`uvicorn medical_rag.api:app --env-file .env --port 8000 --app-dir backend`) or set `QDRANT_PATH` to an absolute path.

- [ ] **Step 2: Check the proxy**

Run: `curl http://localhost:3000/api/health`
Expected: `{"status":"ok","points":<N>}`.

- [ ] **Step 3: Walk the UI at http://localhost:3000**

Check each item, note failures:
1. Header pill shows `Knowledge base ready · N chunks`.
2. Ask a Vietnamese question: user bubble, `Thinking…` bubble, then the answer with `[n]` chips and source cards on the right (title link opens a new tab, `type` badge, section path, passage without the `[Chủ đề: …]` header; "Reranker score" only if `RERANK_URL` is set).
3. Ask a second question in the same chat; the panel switches to the new sources. Click the first answer; the panel shows its sources and the bubble gets a blue border.
4. The sidebar shows the chat under "Today" with the first question as title.
5. Reload the page: the conversation is still there and clickable.
6. Click "New Chat": messages clear, sources show the empty text, no empty item appears in the sidebar. Ask something: a second sidebar item appears.
7. Switch conversation while a request is pending: the answer lands in the original conversation.
8. Type text with a Vietnamese IME (Telex) and confirm with Enter: it must not send half-composed text.
9. Stop uvicorn, reload: pill shows `Offline`; sending gives a red error bubble and the send button works again after.
10. Narrow the window below 980 px and 700 px: sources column, then sidebar, hide as in the template.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A frontend
git commit -m "fix: frontend issues found in manual check" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

Skip this step if nothing needed fixing.

---

### Task 7: Remove the old UI and update docs

**Files:**
- Delete: `backend/medical_rag/static/index.html`
- Modify: `backend/medical_rag/api.py`, `backend/tests/test_api.py`, `backend/pyproject.toml`, `README.md`

**Interfaces:**
- Removes: `GET /` route, `INDEX` constant. Nothing else in the codebase reads them (grep confirmed only `api.py`, `test_api.py`, `pyproject.toml`).

- [ ] **Step 1: Delete the old page and its route**

```bash
git rm backend/medical_rag/static/index.html
```

In `backend/medical_rag/api.py`:
- delete the line `from fastapi.responses import FileResponse`
- delete the line `INDEX = Path(__file__).parent / "static" / "index.html"` and the blank line after it (keep two blank lines before `@asynccontextmanager`)
- delete the last route:

```python
@app.get("/")
def index():
    return FileResponse(INDEX)
```

Keep `from pathlib import Path` (still used for `QDRANT_PATH`).

In `backend/tests/test_api.py` delete:

```python
def test_index_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MediRAG" in resp.text
```

In `backend/pyproject.toml` delete these two lines and the blank line after them:

```toml
[tool.setuptools.package-data]
medical_rag = ["static/*.html"]
```

- [ ] **Step 2: Run the backend tests**

Run: `cd backend && python -m pytest -q`
Expected: all pass, one test fewer than before.

- [ ] **Step 3: Update `README.md`**

Repo tree block, replace the `backend/medical_rag/` line and add a `frontend/` line:

```
backend/medical_rag/   ingestion/ (chunking + ingest), encoders, store, retrieval, generation, llm, api
frontend/              Next.js chat UI (proxies /api to the backend)
```

Run section, replace step 3 (`API and UI, then open http://localhost:8000/`) with:

````
3. API on port 8000:
   ```bash
   uvicorn medical_rag.api:app --env-file .env --port 8000
   ```
4. Frontend, then open http://localhost:3000/:
   ```bash
   cd frontend && npm install && npm run dev        # production: npm run build && npm start
   ```
````

API table: delete the row `| GET / | the single-page UI |`.

Configuration table: add a row

```
| `API_URL` | no (frontend, server side) | `http://localhost:8000` |
```

- [ ] **Step 4: Verify nothing references the old UI**

Run: `grep -rn "static/index\|FileResponse\|localhost:8000/" README.md backend --include=*.py --include=*.toml --include=*.md`
Expected: no matches (matches in `docs/superpowers/` are historical and stay).

- [ ] **Step 5: Commit**

```bash
git add -A backend README.md
git commit -m "refactor: remove the vanilla UI now that the Next.js frontend replaces it" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage**
- Separate Next server, `/api` rewrites, `API_URL`, `proxyTimeout` → Task 1.
- `lib/api.ts` chat/health/types → Task 2.
- `splitCitations`, no `dangerouslySetInnerHTML`, `pre-wrap` → Task 3, Task 5 (`.bubble`, `Message.tsx`).
- Conversations: `titleOf`, `groupByDay`, `load`/`save` with try/catch, pending never saved → Task 4.
- `useChat`: hydrate in `useEffect`, write result by conversation id, send disabled while pending, `newChat`, `select`, `selectMessage` → Task 5 (logic in Task 4 `startTurn`/`finishTurn`).
- Header pill, Sidebar groups, Message click-to-select, Composer, SourcesPanel (link, badge, path, 300 chars, score only when not null), Disclaimer (once at the end of the message feed), `suppressHydrationWarning`, responsive CSS → Task 5.
- Vitest for citations, conversations, api → Tasks 2 to 4. Manual E2E → Task 6.
- Cleanup list (static page, `GET /`, `INDEX`, test, `package-data`, README) and keep `medical_rag_ui.html` → Task 7 and Global Constraints.
- Run instructions and `API_URL` in README → Task 7.

**Deliberate deviations from the spec (update the spec to match)**
- Plain global `globals.css` instead of per-component CSS Modules (the template CSS ports 1:1, fewer files).
- "New Chat" sets `activeId = null` and the conversation is created on first send, so empty conversations never appear in the sidebar (spec said create an empty one).
- Extra file `lib/sources.ts` (`passageOf`, `isHttpUrl`) and pure helpers `startTurn`/`finishTurn`/`sourcesFor`/`uid` in `lib/conversations.ts`: they carry behaviour of the old UI the spec did not list (header strip, http(s)-only links, 60 s timeout, IME Enter guard) and make the by-id write testable.

**Type consistency:** `Source`, `ChatResponse` (api.ts) are used by `conversations.ts`, `SourcesPanel.tsx`, `useChat.ts`. `Message`, `Conversation`, `StorageLike` (conversations.ts) are used by components and tests. `startTurn(cs, convId, question, now, userId, pendingId)` and `finishTurn(cs, convId, messageId, patch)` match between Task 4 tests, implementation and `useChat.ts`. `useChat` return keys match `page.tsx` usage.
