const MAX_PASSAGE = 300;

// Chunks start with a "[Chủ đề: … | Nguồn: …]" header that the source card already shows elsewhere.
export function passageOf(text: string): string {
  const body = text.replace(/^\[[^\]]*\]\s*/, "");
  return body.length > MAX_PASSAGE ? body.slice(0, MAX_PASSAGE) + "…" : body;
}

export const isHttpUrl = (url: string): boolean => /^https?:\/\//.test(url);
