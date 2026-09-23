import { afterEach, describe, expect, it, vi } from "vitest";
import { chat, health, retrieve } from "./api";

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

describe("retrieve", () => {
  it("posts the selected mode and fixed result count", async () => {
    const payload = {
      mode: "sparse",
      hits: [
        {
          id: "1",
          text: "Nội dung",
          type: "disease",
          article_title: "Bài 1",
          article_url: "https://example.test/1",
          section_path: "Mục",
          retrieval_score: 1.25,
          rerank_score: null,
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue(json(payload));
    vi.stubGlobal("fetch", fetchMock);

    expect(await retrieve("đau đầu?", "sparse")).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/retrieve");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "đau đầu?", mode: "sparse", k: 10 });
  });

  it("uses a backend detail error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "embedding service error: down" }, 502)));
    await expect(retrieve("q", "dense")).rejects.toThrow("embedding service error: down");
  });

  it("uses a generic error for a non-string detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: [] }, 422)));
    await expect(retrieve("q", "hybrid")).rejects.toThrow("Request failed (422)");
  });

  it("maps a timeout to a readable message", async () => {
    const timeout = Object.assign(new Error("timed out"), { name: "TimeoutError" });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(timeout));
    await expect(retrieve("q", "hybrid")).rejects.toThrow("Request timed out");
  });
});
