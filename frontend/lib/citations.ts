export type Part = { kind: "text"; value: string } | { kind: "cite"; n: number };

// A [n] becomes a citation chip only when it points at a real source.
export function splitCitations(text: string, sourceCount: number): Part[] {
  return text.split(/(\[\d+\])/).flatMap((part): Part[] => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (m && +m[1] >= 1 && +m[1] <= sourceCount) return [{ kind: "cite", n: +m[1] }];
    return part ? [{ kind: "text", value: part }] : [];
  });
}
