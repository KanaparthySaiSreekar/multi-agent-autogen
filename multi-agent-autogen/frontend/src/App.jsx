import { useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import useChat from "./hooks/useChat";
import useConversations from "./hooks/useConversations";

export default function App() {
  const chat = useChat();
  const { conversations, refresh } = useConversations();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSelectConversation = (id) => {
    chat.loadHistory(id);
  };

  const handleNewChat = () => {
    chat.resetChat();
  };

  const handleMessageSent = async () => {
    // Refresh conversation list after a message is sent
    setTimeout(refresh, 500);
  };

  return (
    <div className="app-layout">
      <Sidebar
        conversations={conversations}
        activeId={chat.conversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((v) => !v)}
      />
      <ChatArea chat={chat} onMessageSent={handleMessageSent} />
    </div>
  );
}
