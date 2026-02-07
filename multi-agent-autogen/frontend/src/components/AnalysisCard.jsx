import "./AnalysisCard.css";

export default function AnalysisCard({ data }) {
  if (!data) return null;

  const { summary, key_decisions, action_items, risks } = data;

  return (
    <div className="analysis-card">
      {/* Summary */}
      <div className="ac-section ac-summary">
        <h3>Summary</h3>
        <p>{summary}</p>
        {key_decisions?.length > 0 && (
          <>
            <h4>Key Decisions</h4>
            <ul>
              {key_decisions.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          </>
        )}
      </div>

      {/* Action Items */}
      {action_items?.length > 0 && (
        <div className="ac-section ac-actions">
          <h3>Action Items</h3>
          {action_items.map((item, i) => (
            <div key={i} className="ac-action-item">
              <div className="ac-task">{item.task}</div>
              <div className="ac-meta">
                <span className={`ac-badge ac-badge-${(item.priority || "medium").toLowerCase()}`}>
                  {item.priority}
                </span>
                {item.owner && <span>Owner: {item.owner}</span>}
                {item.deadline && <span>Deadline: {item.deadline}</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Risks */}
      {risks?.length > 0 && (
        <div className="ac-section ac-risks">
          <h3>Risks &amp; Open Issues</h3>
          {risks.map((r, i) => (
            <div key={i} className="ac-risk-item">
              <div className="ac-desc">{r.description}</div>
              <div className="ac-meta">
                <span className={`ac-badge ac-badge-${(r.severity || "medium").toLowerCase()}`}>
                  {r.severity}
                </span>
                <span className="ac-category">{r.category}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
