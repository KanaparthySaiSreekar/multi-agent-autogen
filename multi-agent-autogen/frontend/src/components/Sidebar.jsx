import "./Sidebar.css";

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNewChat,
  open,
  onToggle,
}) {
  return (
    <>
      <button className="sidebar-toggle" onClick={onToggle}>
        {open ? "\u2039" : "\u203A"}
      </button>
      <aside className={`sidebar ${open ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <h2>Conversations</h2>
          <button className="new-chat-btn" onClick={onNewChat}>
            + New Chat
          </button>
        </div>
        <div className="sidebar-list">
          {conversations.length === 0 && (
            <p className="sidebar-empty">No conversations yet</p>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`sidebar-item ${c.id === activeId ? "active" : ""}`}
              onClick={() => onSelect(c.id)}
            >
              <span className="sidebar-item-name">{c.name}</span>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}
