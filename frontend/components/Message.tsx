import { splitCitations } from "../lib/citations";
import type { Message as Msg } from "../lib/conversations";

type Props = { message: Msg; selected: boolean; onSelect: () => void };

export function Message({ message: m, selected, onSelect }: Props) {
  if (m.role === "user") {
    return (
      <div className="row user">
        <div className="bubble">{m.text}</div>
        <div className="avatar">👤</div>
      </div>
    );
  }

  const clickable = !!m.sources?.length;
  const classes = ["bubble", m.pending && "pending", m.error && "error", clickable && "clickable", selected && "selected"];
  const interactive = clickable
    ? {
        role: "button",
        tabIndex: 0,
        onClick: onSelect,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect();
          }
        },
      }
    : {};

  return (
    <div className="row assistant">
      <div className="avatar">🩺</div>
      <div className={classes.filter(Boolean).join(" ")} {...interactive}>
        {m.pending
          ? "Thinking…"
          : m.error
            ? m.text
            : splitCitations(m.text, m.sources?.length ?? 0).map((p, i) =>
                p.kind === "cite" ? (
                  <span key={i} className="citation">
                    {p.n}
                  </span>
                ) : (
                  p.value
                ),
              )}
      </div>
    </div>
  );
}
