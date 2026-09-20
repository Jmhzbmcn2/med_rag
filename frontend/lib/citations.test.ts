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
