import { Chat } from "@/components/Chat";

export default function Home() {
  return (
    <main className="page-shell">
      <section className="chat-shell" aria-labelledby="page-title">
        <header className="chat-header">
          <div className="brand-mark" aria-hidden="true">
            M
          </div>
          <div>
            <p className="eyebrow">AI KNOWLEDGE SERVER</p>
            <h1 id="page-title">MAX Knowledge Assistant</h1>
            <p>Answers grounded in your uploaded documents</p>
          </div>
        </header>
        <Chat />
      </section>
    </main>
  );
}
