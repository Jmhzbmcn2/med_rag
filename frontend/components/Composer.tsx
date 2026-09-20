"use client";

import { useState } from "react";
import { Disclaimer } from "./Disclaimer";

type Props = { pending: boolean; onSend: (question: string) => void };

export function Composer({ pending, onSend }: Props) {
  const [value, setValue] = useState("");
  const empty = !value.trim();

  const submit = () => {
    if (empty || pending) return;
    onSend(value);
    setValue("");
  };

  return (
    <div className="composer">
      <Disclaimer />
      <div className="input-wrap">
        <input
          value={value}
          maxLength={1000}
          placeholder="Ask a medical question..."
          aria-label="Question"
          onChange={(e) => setValue(e.target.value)}
          // isComposing: Vietnamese IMEs confirm text with Enter
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) submit();
          }}
        />
        <button className="send" aria-label="Send" disabled={pending || empty} onClick={submit}>
          ➤
        </button>
      </div>
    </div>
  );
}
