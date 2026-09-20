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
