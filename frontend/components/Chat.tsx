"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";

import type { ChatMessage, FastApiQueryResponse, SourceInfo } from "@/types/chat";

interface SourceGroup {
  key: string;
  source: string;
  page: number;
  chunks: SourceInfo[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getErrorMessage(payload: unknown, status: number): string {
  if (isRecord(payload) && typeof payload.detail === "string") {
    return payload.detail;
  }
  return `Request failed with HTTP ${status}.`;
}

function groupSources(sources: SourceInfo[]): SourceGroup[] {
  const groups = new Map<string, SourceGroup>();

  for (const source of sources) {
    const key = JSON.stringify([source.source, source.page]);
    const existing = groups.get(key);

    if (existing) {
      existing.chunks.push(source);
    } else {
      groups.set(key, {
        key,
        source: source.source,
        page: source.page,
        chunks: [source],
      });
    }
  }

  return Array.from(groups.values());
}

export function Chat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const messagesElement = messagesRef.current;
    if (messagesElement) {
      messagesElement.scrollTo({
        top: messagesElement.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages, isLoading, error]);

  function startNewChat() {
    if (isLoading) {
      return;
    }

    setMessages([]);
    setQuestion("");
    setError(null);
  }

  function handleQuestionKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isLoading) {
      return;
    }

    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmedQuestion,
      },
    ]);
    setQuestion("");
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      const payload: unknown = await response.json();
      if (!response.ok) {
        throw new Error(getErrorMessage(payload, response.status));
      }

      const result = payload as FastApiQueryResponse;
      if (typeof result.answer !== "string" || !Array.isArray(result.sources)) {
        throw new Error("The server returned an unexpected response.");
      }

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "An unexpected error occurred.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="chat-body">
      <div className="chat-toolbar">
        <span>{messages.length > 0 ? `${messages.length} messages` : "New conversation"}</span>
        <button type="button" onClick={startNewChat} disabled={isLoading || messages.length === 0}>
          + New Chat
        </button>
      </div>

      <div ref={messagesRef} className="messages" aria-live="polite">
        {messages.length === 0 ? (
          <div className="welcome-state">
            <span aria-hidden="true">✦</span>
            <h2>Ask your knowledge base</h2>
            <p>Enter a question in Chinese or English. The answer will include the retrieved sources.</p>
          </div>
        ) : (
          messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <p className="message-role">{message.role === "user" ? "You" : "MAX"}</p>
              <div className="message-content">{message.content}</div>
              {message.role === "assistant" && message.sources && message.sources.length > 0
                ? (() => {
                    const sourceGroups = groupSources(message.sources);
                    return (
                      <details className="sources">
                        <summary>
                          {sourceGroups.length} source location{sourceGroups.length === 1 ? "" : "s"}
                          {` · ${message.sources.length} passage${message.sources.length === 1 ? "" : "s"}`}
                        </summary>
                        <div className="source-list">
                          {sourceGroups.map((group) => (
                            <section className="source-card" key={group.key}>
                              <div className="source-heading">
                                <strong>{group.source}</strong>
                                <span>Page {group.page}</span>
                              </div>
                              <div className="source-chunks">
                                {group.chunks.map((chunk, index) => (
                                  <div className="source-chunk" key={`${group.key}-${index}`}>
                                    <div className="source-chunk-meta">
                                      <span>Passage {index + 1}</span>
                                      <span>Distance {chunk.distance.toFixed(3)}</span>
                                    </div>
                                    <p>{chunk.text}</p>
                                  </div>
                                ))}
                              </div>
                            </section>
                          ))}
                        </div>
                      </details>
                    );
                  })()
                : null}
            </article>
          ))
        )}

        {isLoading ? (
          <div className="loading-row" role="status">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span>Searching documents and preparing an answer…</span>
          </div>
        ) : null}
      </div>

      {error ? <p className="error-banner" role="alert">{error}</p> : null}

      <form className="composer" onSubmit={handleSubmit}>
        <label htmlFor="question">Question</label>
        <div className="composer-row">
          <textarea
            id="question"
            name="question"
            placeholder="Ask something about your documents…"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleQuestionKeyDown}
            rows={2}
            maxLength={4000}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !question.trim()}>
            {isLoading ? "Waiting…" : "Ask"}
          </button>
        </div>
        <p className="composer-hint">Enter to send · Shift+Enter for a new line</p>
      </form>
    </div>
  );
}
