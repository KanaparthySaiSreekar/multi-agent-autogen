const BASE = "/api/chat";

export async function fetchConversations() {
  const res = await fetch(BASE);
  if (!res.ok) throw new Error("Failed to load conversations");
  return res.json();
}

export async function fetchHistory(conversationId) {
  const res = await fetch(`${BASE}/${conversationId}/history`);
  if (!res.ok) throw new Error("Failed to load history");
  return res.json();
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error("File upload failed");
  return res.json();
}

/**
 * POST /api/chat as SSE stream. Returns a ReadableStream reader.
 * Caller iterates lines and parses JSON.
 */
export function streamChat(message, conversationId) {
  return fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId || null }),
  });
}
