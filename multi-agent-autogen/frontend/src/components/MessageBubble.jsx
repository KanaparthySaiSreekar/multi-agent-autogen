import AnalysisCard from "./AnalysisCard";
import "./MessageBubble.css";

export default function MessageBubble({ message }) {
  const { role, content, component, payload } = message;
  const isUser = role === "user";

  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        {content && <p className="bubble-text">{content}</p>}
        {component === "analysis_card" && payload && (
          <AnalysisCard data={payload} />
        )}
      </div>
    </div>
  );
}
