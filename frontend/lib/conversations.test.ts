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
