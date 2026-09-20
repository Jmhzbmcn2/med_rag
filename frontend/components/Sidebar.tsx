import { groupByDay } from "../lib/conversations";
import type { Conversation } from "../lib/conversations";

type Props = {
  conversations: Conversation[];
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
};

export function Sidebar({ conversations, activeId, onNew, onSelect }: Props) {
  const { today, previous } = groupByDay(conversations, Date.now());
  const group = (title: string, items: Conversation[]) =>
    items.length > 0 && (
      <>
        <div className="section-title">{title}</div>
        {items.map((c) => (
          <button
            key={c.id}
            className={"chat-item" + (c.id === activeId ? " active" : "")}
            onClick={() => onSelect(c.id)}
          >
            {c.title}
          </button>
        ))}
      </>
    );

  return (
    <aside>
      <button className="new-chat" onClick={onNew}>
        ＋ New Chat
      </button>
      {group("Today", today)}
      {group("Previous", previous)}
    </aside>
  );
}
