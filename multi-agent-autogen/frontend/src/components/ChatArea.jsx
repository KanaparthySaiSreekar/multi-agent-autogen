import { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import AgentProgress from "./AgentProgress";
import TypingIndicator from "./TypingIndicator";
import ChatInput from "./ChatInput";
import WelcomeScreen from "./WelcomeScreen";
import "./ChatArea.css";

export default function ChatArea({ chat, onMessageSent }) {
  const bottomRef = useRef(null);

  const {
    messages,
    isStreaming,
    activeAgents,
    currentAgent,
    sendMessage,
  } = chat;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming, activeAgents]);

  const handleSend = async (text) => {
    await sendMessage(text);
    onMessageSent?.();
  };

  const hasMessages = messages.length > 0;

  return (
    <main className="chat-area">
      <header className="chat-header">
        <h1>Document Intelligence</h1>
        <p>Multi-Agent Analysis powered by AG2</p>
      </header>

      <div className="chat-messages">
        {!hasMessages && !isStreaming && (
          <WelcomeScreen onPromptClick={handleSend} />
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isStreaming && (
          <>
            {(currentAgent || activeAgents.length > 0) && (
              <AgentProgress
                activeAgents={activeAgents}
                currentAgent={currentAgent}
              />
            )}
            <TypingIndicator />
          </>
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={handleSend} disabled={isStreaming} />
    </main>
  );
}
