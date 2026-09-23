"use client";

import { type FormEvent, useState } from "react";
import { isHttpUrl } from "../lib/sources";
import { retrieve } from "../lib/api";
import type { RetrievalMode, RetrievalResponse } from "../lib/api";
import { Header } from "./Header";

export function RetrievalApp() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [result, setResult] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const value = question.trim();
    if (!value || pending) return;
    setPending(true);
    setError("");
    setResult(null);
    try {
      setResult(await retrieve(value, mode));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="app">
      <Header />
      <main className="retrieval-main">
        <section className="retrieval-shell">
          <div className="retrieval-heading">
            <h1>Retrieval Inspector</h1>
            <p>Inspect ranked chunks without generating an answer.</p>
          </div>

          <form className="retrieval-form" onSubmit={submit}>
            <div className="retrieval-field">
              <label htmlFor="retrieval-question">Question</label>
              <input
                id="retrieval-question"
                value={question}
                maxLength={1000}
                required
                placeholder="Enter a medical query..."
                onChange={(event) => setQuestion(event.target.value)}
              />
            </div>
            <div className="retrieval-field">
              <label htmlFor="retrieval-mode">Mode</label>
              <select
                id="retrieval-mode"
                value={mode}
                onChange={(event) => setMode(event.target.value as RetrievalMode)}
              >
                <option value="hybrid">Hybrid</option>
                <option value="dense">Dense</option>
                <option value="sparse">Sparse</option>
              </select>
            </div>
            <button type="submit" disabled={pending || !question.trim()}>
              {pending ? "Searching…" : "Search"}
            </button>
          </form>

          <p className="score-note">
            Retrieval scores use the selected mode's scale and are not comparable across modes.
          </p>
          {error && <div className="retrieval-error" role="alert">{error}</div>}
          {!result && !error && !pending && (
            <div className="retrieval-empty">Results appear here after you run a query.</div>
          )}
          {result && (
            <section className="retrieval-results" aria-live="polite">
              <h2>{result.hits.length} chunks</h2>
              {result.hits.length === 0 && <div className="retrieval-empty">No matching chunks found.</div>}
              {result.hits.map((hit, index) => (
                <article className="retrieval-card" key={hit.id}>
                  <div className="retrieval-card-head">
                    <span className="retrieval-rank">#{index + 1}</span>
                    <div>
                      <h3>
                        {isHttpUrl(hit.article_url) ? (
                          <a href={hit.article_url} target="_blank" rel="noopener noreferrer">
                            {hit.article_title}
                          </a>
                        ) : hit.article_title}
                      </h3>
                      <div className="meta">{hit.section_path} · {hit.type}</div>
                    </div>
                  </div>
                  <div className="retrieval-scores">
                    <span>{result.mode} score <strong>{hit.retrieval_score.toFixed(6)}</strong></span>
                    {hit.rerank_score !== null && (
                      <span>reranker score <strong>{hit.rerank_score.toFixed(4)}</strong></span>
                    )}
                  </div>
                  <div className="retrieval-text">{hit.text}</div>
                </article>
              ))}
            </section>
          )}
        </section>
      </main>
    </div>
  );
}
