import { useState, useCallback, useRef } from "react";
import { streamChat, fetchHistory } from "../api";

export default function useChat() {
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeAgents, setActiveAgents] = useState([]);
  const [currentAgent, setCurrentAgent] = useState(null);
  const abortRef = useRef(null);

  const loadHistory = useCallback(async (convId) => {
    setConversationId(convId);
    try {
      const history = await fetchHistory(convId);
      setMessages(
        history.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          component: m.component,
          payload: m.payload,
        }))
      );
    } catch {
      /* ignore */
    }
  }, []);

  const resetChat = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setIsStreaming(false);
    setActiveAgents([]);
    setCurrentAgent(null);
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      if (!text.trim() || isStreaming) return null;

      // Add user message
      const userMsg = { id: Date.now(), role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setActiveAgents([]);
      setCurrentAgent(null);

      let newConvId = conversationId;

      try {
        const response = await streamChat(text, conversationId);
        if (!response.ok) throw new Error("Stream request failed");
        if (!response.body) throw new Error("No response body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        abortRef.current = reader;

        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data: ")) continue;
            const jsonStr = trimmed.slice(6);
            let event;
            try {
              event = JSON.parse(jsonStr);
            } catch {
              continue;
            }

            switch (event.type) {
              case "status":
                setCurrentAgent(event.agent);
                break;

              case "agent_done":
                setActiveAgents((prev) =>
                  prev.includes(event.agent)
                    ? prev
                    : [...prev, event.agent]
                );
                break;

              case "end":
                if (event.conversation_id) {
                  newConvId = event.conversation_id;
                  setConversationId(event.conversation_id);
                }
                setMessages((prev) => [
                  ...prev,
                  {
                    id: Date.now() + 1,
                    role: "assistant",
                    content: event.full_reply || null,
                    component: event.component || null,
                    payload: event.payload || null,
                  },
                ]);
                break;

              case "error":
                setMessages((prev) => [
                  ...prev,
                  {
                    id: Date.now() + 2,
                    role: "assistant",
                    content: `Error: ${event.content}`,
                  },
                ]);
                break;
            }
          }
        }
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 3,
            role: "assistant",
            content: `Error: ${err.message}`,
          },
        ]);
      } finally {
        setIsStreaming(false);
        setCurrentAgent(null);
        abortRef.current = null;
      }

      return newConvId;
    },
    [conversationId, isStreaming]
  );

  return {
    messages,
    conversationId,
    isStreaming,
    activeAgents,
    currentAgent,
    sendMessage,
    loadHistory,
    resetChat,
  };
}
