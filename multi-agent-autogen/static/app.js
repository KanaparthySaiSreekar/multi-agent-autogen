const API = "/api";

async function submitDocument() {
  const title = document.getElementById("doc-title").value.trim();
  const content = document.getElementById("doc-content").value.trim();

  if (!title || !content) {
    alert("Please enter both a title and document content.");
    return;
  }

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  document.getElementById("results").classList.add("hidden");
  document.getElementById("loading").classList.remove("hidden");

  try {
    const res = await fetch(`${API}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content }),
    });
    const data = await res.json();
    pollForResult(data.id);
  } catch (err) {
    alert("Error submitting document: " + err.message);
    btn.disabled = false;
    document.getElementById("loading").classList.add("hidden");
  }
}

async function pollForResult(docId) {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`${API}/documents/${docId}`);
      const data = await res.json();

      if (data.analysis_status === "completed" && data.analysis) {
        clearInterval(interval);
        renderResults(data.analysis);
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("analyze-btn").disabled = false;
        loadHistory();
      } else if (data.analysis_status === "failed") {
        clearInterval(interval);
        alert("Analysis failed. Please try again.");
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("analyze-btn").disabled = false;
      }
    } catch (err) {
      clearInterval(interval);
      alert("Error polling results: " + err.message);
      document.getElementById("loading").classList.add("hidden");
      document.getElementById("analyze-btn").disabled = false;
    }
  }, 3000);
}

function renderResults(analysis) {
  // Summary
  document.getElementById("summary-text").textContent =
    analysis.summary || "No summary available.";

  const decisionsList = document.getElementById("decisions-list");
  decisionsList.innerHTML = "";
  (analysis.key_decisions || []).forEach((d) => {
    const li = document.createElement("li");
    li.textContent = d;
    decisionsList.appendChild(li);
  });

  // Action items
  const actionsDiv = document.getElementById("actions-list");
  actionsDiv.innerHTML = "";
  (analysis.action_items || []).forEach((item) => {
    const el = document.createElement("div");
    el.className = "action-item";
    const priority = (item.priority || "medium").toLowerCase();
    el.innerHTML = `
      <div class="task">${escapeHtml(item.task)}</div>
      <div class="meta">
        <span class="badge badge-${priority}">${priority}</span>
        ${item.owner ? `<span>Owner: ${escapeHtml(item.owner)}</span>` : ""}
        ${item.deadline ? `<span>Deadline: ${escapeHtml(item.deadline)}</span>` : ""}
        ${item.dependencies && item.dependencies.length ? `<span>Depends on: ${item.dependencies.map(escapeHtml).join(", ")}</span>` : ""}
      </div>
    `;
    actionsDiv.appendChild(el);
  });

  // Risks
  const risksDiv = document.getElementById("risks-list");
  risksDiv.innerHTML = "";
  (analysis.risks || []).forEach((item) => {
    const el = document.createElement("div");
    el.className = "risk-item";
    const severity = (item.severity || "medium").toLowerCase();
    el.innerHTML = `
      <div class="desc">${escapeHtml(item.description)}</div>
      <div class="meta">
        <span class="badge badge-${severity}">${severity}</span>
        <span class="category-badge">${escapeHtml(item.category || "risk")}</span>
      </div>
    `;
    risksDiv.appendChild(el);
  });

  document.getElementById("results").classList.remove("hidden");
}

async function loadHistory() {
  try {
    const res = await fetch(`${API}/documents`);
    const docs = await res.json();
    const container = document.getElementById("history-list");
    container.innerHTML = "";

    docs.forEach((doc) => {
      const el = document.createElement("div");
      el.className = "history-item";
      el.onclick = () => loadDocument(doc.id);
      el.innerHTML = `
        <div>
          <span class="title">${escapeHtml(doc.title)}</span>
          <span class="info"> — ${doc.word_count} words</span>
        </div>
        <span class="status-${doc.analysis_status || "pending"}">${doc.analysis_status || "pending"}</span>
      `;
      container.appendChild(el);
    });
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

async function loadDocument(docId) {
  try {
    const res = await fetch(`${API}/documents/${docId}`);
    const data = await res.json();

    document.getElementById("doc-title").value = data.title;
    document.getElementById("doc-content").value = data.content;

    if (data.analysis) {
      renderResults(data.analysis);
    }
  } catch (err) {
    alert("Error loading document: " + err.message);
  }
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Load history on page load
document.addEventListener("DOMContentLoaded", loadHistory);
