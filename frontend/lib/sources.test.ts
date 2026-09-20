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
