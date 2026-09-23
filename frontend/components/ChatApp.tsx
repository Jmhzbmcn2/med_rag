"use client";

import { Composer } from "./Composer";
import { Header } from "./Header";
import { MessageList } from "./MessageList";
import { Sidebar } from "./Sidebar";
import { SourcesPanel } from "./SourcesPanel";
import { useChat } from "../hooks/useChat";

export function ChatApp() {
  const chat = useChat();

  return (
    <div className="app">
      <Header />
      <div className="layout">
        <Sidebar
          conversations={chat.conversations}
          activeId={chat.activeId}
          onNew={chat.newChat}
          onSelect={chat.select}
        />
        <main className="main">
          <div className="main-head">
            <h1>Medical AI Assistant</h1>
            <p>Ask questions grounded in your medical knowledge base.</p>
          </div>
          <MessageList
            messages={chat.active?.messages ?? []}
            selectedId={chat.selectedId}
            onSelect={chat.selectMessage}
          />
          <Composer pending={chat.pending} onSend={chat.send} />
        </main>
        <SourcesPanel sources={chat.sources} />
      </div>
    </div>
  );
}
