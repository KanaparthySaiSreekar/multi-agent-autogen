import "./WelcomeScreen.css";

const EXAMPLES = [
  {
    title: "Analyze a meeting transcript",
    prompt:
      "Analyze the following meeting transcript:\n\nProject Alpha Kickoff Meeting - Jan 15, 2025\n\nAttendees: Sarah (PM), Mike (Dev Lead), Lisa (Design), Tom (QA)\n\nSarah: Let's define our MVP scope. We need user auth, dashboard, and reporting by March 1st.\nMike: Auth and dashboard are feasible. Reporting might slip - we need to finalize the data model first.\nLisa: I'll have wireframes for dashboard by next Friday. Need content specs from Sarah.\nTom: We should set up CI/CD before sprint 2. I'll need access to the staging environment.\nSarah: Agreed. Mike, can you handle staging setup? Budget approved for AWS.\nMike: Yes, I'll set it up this week. We should also decide on PostgreSQL vs MongoDB.\nSarah: Let's go with PostgreSQL for now. Tom, plan regression tests for sprint 3.\nTom: Will do. What about load testing?\nSarah: Good point - let's discuss that in the next meeting. Meeting adjourned.",
  },
  {
    title: "Review a project brief",
    prompt:
      "Analyze the following project brief:\n\nProject: Customer Portal Redesign\nSponsor: VP of Customer Success\nBudget: $150,000\nTimeline: Q2 2025\n\nObjective: Redesign the customer self-service portal to reduce support ticket volume by 30% and improve customer satisfaction scores.\n\nKey Requirements:\n1. Modern responsive design with accessibility compliance (WCAG 2.1 AA)\n2. Self-service knowledge base with AI-powered search\n3. Ticket tracking dashboard with real-time status updates\n4. Integration with existing Salesforce CRM and Zendesk\n5. Single sign-on (SSO) via Okta\n\nConstraints:\n- Must maintain backward compatibility with existing API consumers\n- Cannot exceed 2-second page load time\n- Data migration from legacy portal must be seamless\n\nStakeholders: Customer Success team, Engineering, Product, Legal (for privacy review)\n\nRisks noted: Legacy system documentation is incomplete. Key developer leaving in April.",
  },
  {
    title: "Assess a legal draft",
    prompt:
      'Analyze the following contract excerpt:\n\nSERVICE LEVEL AGREEMENT (SLA)\n\nBetween: TechCorp Inc. ("Provider") and GlobalRetail Ltd. ("Client")\nEffective Date: February 1, 2025\n\n1. SERVICE AVAILABILITY\nProvider guarantees 99.9% uptime for all production services, measured monthly. Scheduled maintenance windows (Sundays 2-6 AM EST) are excluded from uptime calculations.\n\n2. RESPONSE TIMES\n- Critical (P1): 15-minute response, 4-hour resolution\n- High (P2): 1-hour response, 8-hour resolution\n- Medium (P3): 4-hour response, 48-hour resolution\n- Low (P4): Next business day response\n\n3. PENALTIES\nFor each 0.1% below the 99.9% target, Provider shall credit Client 5% of monthly fees, up to a maximum of 30% of monthly fees.\n\n4. DATA HANDLING\nProvider shall process Client data in compliance with GDPR and CCPA. Data residency: US-East region only. Backup frequency: Every 6 hours with 30-day retention.\n\n5. TERMINATION\nEither party may terminate with 90 days written notice. Upon termination, Provider shall return all Client data within 30 days in an industry-standard format.\n\n6. LIMITATION OF LIABILITY\nProvider\'s total liability shall not exceed 12 months of fees paid under this agreement.',
  },
];

export default function WelcomeScreen({ onPromptClick }) {
  return (
    <div className="welcome-screen">
      <h2>Welcome to Document Intelligence</h2>
      <p>
        Paste a document or choose an example below. Our AI agents will analyze
        it for summaries, action items, and risks.
      </p>
      <div className="welcome-examples">
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            className="welcome-card"
            onClick={() => onPromptClick(ex.prompt)}
          >
            <span className="welcome-card-title">{ex.title}</span>
            <span className="welcome-card-hint">Click to try</span>
          </button>
        ))}
      </div>
    </div>
  );
}
