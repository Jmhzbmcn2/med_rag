"use client";

import { useCallback, useEffect, useState } from "react";
import { chat } from "../lib/api";
import { finishTurn, load, save, sourcesFor, startTurn, uid } from "../lib/conversations";
import type { Conversation } from "../lib/conversations";

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null); // null = new, empty chat
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // Read storage after mount so server and client render the same first frame.
  useEffect(() => {
    setConversations(load());
    setHydrated(true);
  }, []);

  // Skip until hydrated, otherwise the initial [] would overwrite the stored history.
  useEffect(() => {
    if (hydrated) save(conversations);
  }, [conversations, hydrated]);

  const send = useCallback(
    async (raw: string) => {
      const question = raw.trim();
      if (!question || pending) return;
      const convId = activeId ?? uid();
      const pendingId = uid();
      setActiveId(convId);
      setSelectedId(null);
      setPending(true);
      setConversations((cs) => startTurn(cs, convId, question, Date.now(), uid(), pendingId));
      try {
        const data = await chat(question);
        setConversations((cs) =>
          finishTurn(cs, convId, pendingId, { text: data.answer, sources: data.sources, pending: false }),
        );
      } catch (err) {
        const text = err instanceof Error ? err.message : "Request failed";
        setConversations((cs) => finishTurn(cs, convId, pendingId, { text, error: true, pending: false }));
      } finally {
        setPending(false);
      }
    },
    [activeId, pending],
  );

  const newChat = useCallback(() => {
    setActiveId(null);
    setSelectedId(null);
  }, []);

  const select = useCallback((id: string) => {
    setActiveId(id);
    setSelectedId(null);
  }, []);

  const active = conversations.find((c) => c.id === activeId);

  return {
    conversations,
    activeId,
    active,
    selectedId,
    sources: sourcesFor(active, selectedId),
    pending,
    send,
    newChat,
    select,
    selectMessage: setSelectedId,
  };
}
