import { useState, useEffect, useCallback } from "react";
import { fetchConversations } from "../api";

export default function useConversations() {
  const [conversations, setConversations] = useState([]);

  const load = useCallback(async () => {
    try {
      const data = await fetchConversations();
      setConversations(data);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { conversations, refresh: load };
}
