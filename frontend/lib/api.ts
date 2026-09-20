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
