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
