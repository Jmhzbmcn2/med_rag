# Medical RAG — Next.js Frontend Design

Scope: replace the vanilla `backend/medical_rag/static/index.html` with a Next.js app that keeps the same behaviour and adds a local multi-conversation sidebar. The visual template is `medical_rag_ui.html` (repo root). The backend API is unchanged.

## Decisions

- **Separate Next server.** `frontend/` runs on its own Node process (`next dev` / `next start`, port 3000). The browser only calls same-origin `/api/*`; `next.config.ts` `rewrites` proxy it to FastAPI. No CORS, no backend change.
- **Multi-conversation history in `localStorage`.** The backend stays single-turn: every question is independent, history is only for viewing again.
- **TypeScript, plain CSS, npm.** Template CSS is copied almost as is (`:root` tokens kept), split per component with CSS Modules; layout and tokens in `globals.css`. No Tailwind, no UI library.
- **Old UI is deleted** once the new one is verified (see Cleanup).

## Architecture

```
frontend/
  app/layout.tsx, page.tsx, globals.css
  components/  Header, Sidebar, MessageList, Message, Composer, SourcesPanel, Disclaimer
  hooks/useChat.ts          # state, API calls, persistence
  lib/api.ts                # chat(question), health()
  lib/citations.ts          # splitCitations(text, sourceCount)
  lib/conversations.ts      # types, load/save, groupByDay, titleOf
  lib/*.test.ts             # Vitest, environment node
  next.config.ts
```

Next 15, App Router. `page.tsx` is a client component tree (all state is client side). `lang="vi"`, system font stack, no `next/font`.

### next.config.ts

`rewrites`: `/api/:path*` → `${process.env.API_URL ?? "http://localhost:8000"}/api/:path*`. `API_URL` is read on the server, never exposed to the client. `experimental.proxyTimeout: 120000`: the default 30 s is shorter than a slow LLM call and would surface as a proxy error while FastAPI is still working.

### lib/api.ts

- `chat(question) -> {answer, sources}`: `POST /api/chat`. On a non-2xx response throw `Error` with the backend `detail` (502/422) or a generic message; network errors throw too.
- `health() -> {points} | null`: `GET /api/health`, `null` on any failure.
- Types `Source {n, title, url, section_path, type, text, score: number | null}` and `ChatResponse` mirror the backend.

### lib/citations.ts

`splitCitations(text, sourceCount) -> Array<{kind:"text", value} | {kind:"cite", n}>`. Splits on `[n]`; an `[n]` becomes a `cite` only when `1 <= n <= sourceCount`, otherwise it stays as text. Rendering uses React elements only (no `dangerouslySetInnerHTML`), so user and model text are escaped. Bubbles use `white-space: pre-wrap`.

### lib/conversations.ts

- `Message = {id, role: "user" | "assistant", text, sources?: Source[], error?: boolean}`; `Conversation = {id, title, createdAt, messages}`.
- `titleOf(question)`: the first question, cut to 40 characters.
- `groupByDay(conversations, now)`: `{today, previous}`, newest first.
- `load()` / `save()` on key `medirag.conversations.v1`, each wrapped in `try/catch`. Corrupt JSON or blocked storage gives an empty list and the app still works. Pending messages are never saved.

### hooks/useChat.ts

State: `conversations`, `activeId`, `selectedMessageId`, `pending`. Hydrated from `load()` in `useEffect` (not during render, to avoid an SSR mismatch).

- `send(question)`: append the user message and a pending assistant message to the active conversation (create one on first send), call `chat`, then replace the pending message with the answer or an error message. The result is written to the conversation that sent it, by id, even if the user switched conversations meanwhile. Send is disabled while `pending`.
- `newChat()`: create an empty conversation and make it active.
- `select(conversationId)`, `selectMessage(messageId)`.

### Components

- `Header`: brand, health pill. On mount `health()`; "Knowledge Base Ready · N chunks" on success, "Offline" (muted colours) on failure.
- `Sidebar`: New Chat button, "Today" and "Previous" groups, active item highlighted.
- `MessageList` / `Message`: user and assistant rows as in the template; assistant text through `splitCitations`; pending shows a typing indicator; error shows an error bubble. Clicking an assistant message selects it. Auto-scroll to the bottom on new messages.
- `Composer`: text input, Enter to send, disabled send button while pending or when empty.
- `SourcesPanel`: sources of the selected assistant message (default: the latest). Card: title as a link to `url` (new tab, `rel="noopener noreferrer"`), `type` badge, `section_path`, first 300 characters of `text`. The "Reranker score" row appears only when `score != null`, 2 decimals. Empty state text when there are no sources.
- `Disclaimer`: footer "AI-generated information — not a medical diagnosis." (same as the template).

Responsive as in the template: ≤980 px hides the sources column, ≤700 px hides the sidebar.

## Testing

Vitest, no component or browser tests.

- `citations.test.ts`: valid `[n]` becomes a cite, out-of-range and `[0]` stay text, HTML in text is passed through unchanged as a text part, adjacent citations, no citations.
- `conversations.test.ts`: `titleOf` truncation, `groupByDay` around midnight, `load` on corrupt JSON and on a throwing storage.
- `api.test.ts`: fake `fetch`; 200 shape, 502 with `detail`, 422, network failure, `health` failure returns `null`.
- Final check: run FastAPI and Next together, ask one question, confirm answer, citation chips, source cards, a reload restores the conversation, and stopping FastAPI shows "Offline" and an error bubble.

## Run

```
# terminal 1, repo root
cd backend && uvicorn medical_rag.api:app --env-file ../.env --port 8000
# terminal 2
cd frontend && npm install && npm run dev        # http://localhost:3000
# production: npm run build && npm start
```

Env: `API_URL` (optional, default `http://localhost:8000`). README gets a Frontend section with these commands and the variable.

## Cleanup (last implementation step, after the manual check passes)

The user asked to remove the old UI. Delete:

- `backend/medical_rag/static/index.html`
- `GET /` route, the `INDEX` constant and the unused `FileResponse` import in `backend/medical_rag/api.py`
- `test_index_page_is_served` in `backend/tests/test_api.py`
- `package-data` line `medical_rag = ["static/*.html"]` in `backend/pyproject.toml`
- the "UI" mentions in `README.md` (`static/index.html`, `GET /` row, "open http://localhost:8000/") replaced by the frontend instructions

`medical_rag_ui.html` at the repo root is the design template, untracked, and is **kept** as the reference for the CSS port (only the backend's old UI was asked to go).

## Out of scope

Real multi-turn (client sends history, server rewrites follow-ups), streaming, delete or rename conversation, dark mode, a mobile sources view, component tests.
