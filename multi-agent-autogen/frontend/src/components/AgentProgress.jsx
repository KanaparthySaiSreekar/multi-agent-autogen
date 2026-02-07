import "./AgentProgress.css";

const AGENTS = ["SummaryAgent", "ActionAgent", "RiskAgent"];
const LABELS = {
  SummaryAgent: "Summary",
  ActionAgent: "Actions",
  RiskAgent: "Risks",
};

export default function AgentProgress({ activeAgents, currentAgent }) {
  return (
    <div className="agent-progress">
      {AGENTS.map((agent, i) => {
        const done = activeAgents.includes(agent);
        const active = agent === currentAgent && !done;

        return (
          <div key={agent} className="agent-step-wrapper">
            <div
              className={`agent-step ${done ? "done" : ""} ${active ? "active" : ""}`}
            >
              <div className="agent-dot">
                {done ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  <span>{i + 1}</span>
                )}
              </div>
              <span className="agent-label">{LABELS[agent]}</span>
            </div>
            {i < AGENTS.length - 1 && (
              <div className={`agent-connector ${done ? "done" : ""}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}
