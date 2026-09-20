"use client";

import { useEffect, useRef } from "react";
import type { Message as Msg } from "../lib/conversations";
import { Message } from "./Message";

type Props = { messages: Msg[]; selectedId: string | null; onSelect: (id: string) => void };

export function MessageList({ messages, selectedId, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight });
  }, [messages]);

  return (
    <div className="messages" ref={box}>
      {messages.map((m) => (
        <Message key={m.id} message={m} selected={m.id === selectedId} onSelect={() => onSelect(m.id)} />
      ))}
    </div>
  );
}
