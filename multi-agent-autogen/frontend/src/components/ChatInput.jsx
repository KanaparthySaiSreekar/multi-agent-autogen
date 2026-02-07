import { useState, useRef } from "react";
import { uploadFile } from "../api";
import "./ChatInput.css";

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState("");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);
  const textareaRef = useRef(null);

  const handleSubmit = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await uploadFile(file);
      setText(
        (prev) =>
          prev +
          (prev ? "\n\n" : "") +
          `[Attached: ${result.filename}]\n\n${result.content}`
      );
      textareaRef.current?.focus();
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setUploading(false);
      // Reset file input
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleInput = (e) => {
    setText(e.target.value);
    // Auto-resize
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  return (
    <div className="chat-input-bar">
      <button
        className="attach-btn"
        onClick={() => fileRef.current?.click()}
        disabled={disabled || uploading}
        title="Attach .txt or .pdf file"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
        </svg>
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".txt,.pdf"
        onChange={handleFileChange}
        hidden
      />
      <textarea
        ref={textareaRef}
        className="chat-textarea"
        value={text}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder={uploading ? "Uploading file..." : "Type a message or paste a document..."}
        disabled={disabled || uploading}
        rows={1}
      />
      <button
        className="send-btn"
        onClick={handleSubmit}
        disabled={!text.trim() || disabled}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="22" y1="2" x2="11" y2="13" />
          <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
      </button>
    </div>
  );
}
