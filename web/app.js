"use strict";

/* ------------------------------------------------------------------ */
/* Tabs                                                                */
/* ------------------------------------------------------------------ */

const tabs = document.querySelectorAll(".tab");
const views = { studio: document.getElementById("view-studio"), optimize: document.getElementById("view-optimize") };

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    Object.values(views).forEach((v) => v.classList.remove("active"));
    views[tab.dataset.view].classList.add("active");
    if (tab.dataset.view === "optimize") loadResults();
  });
});

/* ------------------------------------------------------------------ */
/* Studio: domains                                                     */
/* ------------------------------------------------------------------ */

const domainRow = document.getElementById("domain-row");
const goalInput = document.getElementById("goal-input");
const goalPanel = document.getElementById("goal-panel");
const toolPreview = document.getElementById("tool-preview");
const customChip = document.getElementById("custom-chip");
const customPanel = document.getElementById("custom-tools-panel");
const customInput = document.getElementById("custom-tools-input");
const errorBanner = document.getElementById("studio-error");

let domains = [];
let selectedDomainId = null;
let customMode = false;

goalInput.addEventListener("focus", () => goalPanel.classList.add("focused"));
goalInput.addEventListener("blur", () => goalPanel.classList.remove("focused"));

async function loadDomains() {
  const res = await fetch("/api/domains");
  const data = await res.json();
  domains = data.domains || [];
  domains.forEach((d) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = d.label;
    chip.dataset.domainId = d.id;
    chip.addEventListener("click", () => selectDomain(d.id));
    domainRow.insertBefore(chip, domainRow.querySelector(".spacer"));
  });
}

function selectDomain(id) {
  selectedDomainId = id;
  customMode = false;
  customChip.classList.remove("selected");
  customPanel.classList.remove("visible");
  domainRow.querySelectorAll(".chip[data-domain-id]").forEach((c) => {
    c.classList.toggle("selected", c.dataset.domainId === id);
  });
  const domain = domains.find((d) => d.id === id);
  if (domain) {
    if (!goalInput.value.trim()) goalInput.value = domain.goal_hint;
    renderToolPreview(domain.tools);
  }
}

customChip.addEventListener("click", () => {
  customMode = true;
  selectedDomainId = null;
  domainRow.querySelectorAll(".chip[data-domain-id]").forEach((c) => c.classList.remove("selected"));
  customChip.classList.add("selected");
  customPanel.classList.add("visible");
  toolPreview.classList.remove("visible");
  toolPreview.innerHTML = "";
});

function renderToolPreview(tools) {
  toolPreview.innerHTML = "";
  (tools || []).forEach((t) => {
    const pill = document.createElement("span");
    pill.className = "tool-pill";
    pill.textContent = t.name;
    pill.title = t.description || "";
    toolPreview.appendChild(pill);
  });
  toolPreview.classList.toggle("visible", (tools || []).length > 0);
}

/* ------------------------------------------------------------------ */
/* Studio: stepper                                                     */
/* ------------------------------------------------------------------ */

const STAGES = ["perceive", "retrieve", "select", "synthesize", "verify"];
const stepper = document.getElementById("stepper");
const archivePanel = document.getElementById("archive-panel");
const archiveHeading = document.getElementById("archive-heading");
const archiveEntries = document.getElementById("archive-entries");
const candidatesEl = document.getElementById("candidates");
const generateBtn = document.getElementById("generate-btn");

function resetStepper() {
  stepper.classList.add("visible");
  STAGES.forEach((s) => {
    const el = stepper.querySelector(`.step[data-stage="${s}"]`);
    el.classList.remove("active", "done", "error");
  });
}

function markStage(stage, status) {
  const idx = STAGES.indexOf(stage);
  STAGES.forEach((s, i) => {
    const el = stepper.querySelector(`.step[data-stage="${s}"]`);
    if (i < idx) el.classList.add("done");
  });
  const el = stepper.querySelector(`.step[data-stage="${stage}"]`);
  if (!el) return;
  el.classList.remove("active", "done", "error");
  el.classList.add(status);
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.add("visible");
  const active = stepper.querySelector(".step.active");
  if (active) { active.classList.remove("active"); active.classList.add("error"); }
}

/* ------------------------------------------------------------------ */
/* Studio: archive retrieval panel                                     */
/* ------------------------------------------------------------------ */

function renderArchive(retrieved) {
  archivePanel.classList.add("visible");
  archiveEntries.innerHTML = "";
  if (!retrieved.length) {
    archiveHeading.innerHTML = "No similar past attempts in the archive yet -- this design starts cold.";
    return;
  }
  archiveHeading.innerHTML = `<strong>${retrieved.length}</strong> past attempt${retrieved.length === 1 ? "" : "s"} informed this design`;
  retrieved.forEach((e) => {
    const card = document.createElement("div");
    card.className = "archive-entry";
    const acc = e.accuracy;
    const accClass = acc == null ? "" : acc >= 0.6 ? "acc-good" : "acc-bad";
    const accText = acc == null ? "n/a" : Math.round(acc * 100) + "%";
    card.innerHTML = `
      <p class="goal-snip">${escapeHtml(e.goal)}</p>
      <div class="meta">
        <span>${escapeHtml(e.design_philosophy || "?")}</span>
        <span class="${accClass}">acc ${accText}</span>
        <span>sim ${Math.round(e.similarity * 100)}%</span>
      </div>`;
    archiveEntries.appendChild(card);
  });
}

/* ------------------------------------------------------------------ */
/* Studio: candidate cards                                             */
/* ------------------------------------------------------------------ */

function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderRoleGraph(spec) {
  const names = spec.roles.map((r) => r.name);
  const entry = spec.orchestration.entry_role;
  const order = [entry, ...names.filter((n) => n !== entry)];
  return order
    .map((n, i) => {
      const node = `<span class="role-node ${n === entry ? "entry" : ""}">${escapeHtml(n)}</span>`;
      return i === 0 ? node : `<span class="arrow">&rarr;</span>${node}`;
    })
    .join("");
}

function renderCandidate(spec, slot) {
  const card = document.createElement("div");
  card.className = "candidate-card";
  card.dataset.slot = slot;

  const rolesHtml = spec.roles
    .map(
      (r) => `
      <div class="role-block">
        <div class="role-name">${escapeHtml(r.name)}</div>
        <p class="prompt">${escapeHtml(r.system_prompt)}</p>
        <div class="role-meta">
          ${(r.tools || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("") || '<span class="tag">no tools</span>'}
          <span class="tag">mem: ${escapeHtml(r.memory.kind)}</span>
          <span class="tag">retry: ${r.retry.max_retries}x ${escapeHtml(r.retry.backoff)}</span>
          <span class="tag">${r.max_tokens} tok</span>
          <span class="tag">temp ${r.temperature}</span>
          ${r.critic && r.critic.enabled ? '<span class="tag">critic on</span>' : ""}
        </div>
      </div>`
    )
    .join("");

  card.innerHTML = `
    <div class="candidate-head">
      <div>
        <h3>${escapeHtml(spec.design_philosophy)}</h3>
        <p>${escapeHtml(PHILOSOPHY_BLURB[spec.design_philosophy] || "")}</p>
      </div>
      <span class="topology-badge">${escapeHtml(spec.orchestration.topology)}</span>
    </div>
    <div class="budget-row">
      <div class="budget-stat"><div class="num tabular">${spec.orchestration.max_steps}</div><div class="lbl">max steps</div></div>
      <div class="budget-stat"><div class="num tabular">${spec.orchestration.budget_tokens.toLocaleString()}</div><div class="lbl">token budget</div></div>
      <div class="budget-stat"><div class="num tabular">${spec.roles.length}</div><div class="lbl">role${spec.roles.length === 1 ? "" : "s"}</div></div>
    </div>
    <div class="graph">${renderRoleGraph(spec)}</div>
    <div class="roles">${rolesHtml}</div>
    <div class="json-view mono"></div>
    <div class="card-actions">
      <button class="btn-ghost btn-raw" type="button">Raw JSON</button>
      <button class="btn-ghost btn-download" type="button">Download</button>
    </div>`;

  const jsonView = card.querySelector(".json-view");
  jsonView.textContent = JSON.stringify(spec, null, 2);
  card.querySelector(".btn-raw").addEventListener("click", () => jsonView.classList.toggle("visible"));
  card.querySelector(".btn-download").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${spec.design_philosophy}-spec.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  return card;
}

const PHILOSOPHY_BLURB = {
  react: "Interleaves reasoning and tool calls in a single loop until the goal is satisfied.",
  plan_execute: "A planner decomposes the goal into tasks; an executor carries each one out with tools.",
  plan_critic_reflect: "A planner drafts a solution and a critic reviews it before it's finalized.",
};

/* ------------------------------------------------------------------ */
/* Studio: generate                                                    */
/* ------------------------------------------------------------------ */

generateBtn.addEventListener("click", runGenerate);

function runGenerate() {
  errorBanner.classList.remove("visible");
  archivePanel.classList.remove("visible");
  candidatesEl.innerHTML = "";
  resetStepper();

  const goal = goalInput.value.trim();
  if (!goal) {
    showError("Type a goal before generating.");
    return;
  }

  const params = new URLSearchParams({ goal, n_seeds: document.getElementById("n-seeds-select").value });
  if (customMode) {
    const raw = customInput.value.trim();
    if (!raw) {
      showError("Add at least one custom tool, or pick a domain instead.");
      return;
    }
    params.set("tools", raw);
  } else if (selectedDomainId) {
    params.set("domain", selectedDomainId);
  } else {
    showError("Choose a domain or add custom tools.");
    return;
  }

  generateBtn.disabled = true;
  const source = new EventSource(`/api/generate/stream?${params.toString()}`);

  source.onmessage = (evt) => {
    const payload = JSON.parse(evt.data);
    handleStreamEvent(payload, source);
  };

  source.onerror = () => {
    source.close();
    generateBtn.disabled = false;
  };
}

function handleStreamEvent(payload, source) {
  switch (payload.stage) {
    case "perceive":
      markStage("perceive", "active");
      break;
    case "retrieve":
      markStage("retrieve", "active");
      renderArchive(payload.retrieved);
      break;
    case "select":
      markStage("select", "active");
      break;
    case "synthesize":
      markStage("synthesize", "active");
      break;
    case "verify":
      markStage("verify", "active");
      break;
    case "done":
      markStage("verify", "done");
      payload.specs.forEach((spec, i) => candidatesEl.appendChild(renderCandidate(spec, i + 1)));
      generateBtn.disabled = false;
      source.close();
      break;
    case "error":
      showError(payload.message);
      generateBtn.disabled = false;
      source.close();
      break;
  }
}

/* ------------------------------------------------------------------ */
/* Optimization view                                                    */
/* ------------------------------------------------------------------ */

let resultsCache = null;

async function loadResults() {
  if (resultsCache) { renderOptimize(resultsCache); return; }
  const res = await fetch("/api/results");
  const data = await res.json();
  resultsCache = data;
  renderOptimize(data);
}

function renderOptimize(data) {
  const emptyEl = document.getElementById("optimize-empty");
  const contentEl = document.getElementById("optimize-content");
  if (data.empty) {
    emptyEl.style.display = "block";
    contentEl.style.display = "none";
    return;
  }
  emptyEl.style.display = "none";
  contentEl.style.display = "block";

  const iters = data.iterations;

  document.getElementById("chart-accuracy").innerHTML = svgLineChart({
    iterations: iters,
    series: [
      { key: "dev_mean_score", label: "dev", color: "var(--slot-2)" },
      { key: "test_mean_score", label: "test", color: "var(--slot-1)" },
    ],
    valueFormatter: fmtScore,
  });

  document.getElementById("chart-reliability").innerHTML = svgLineChart({
    iterations: iters,
    series: [{ key: "score_stddev", label: "stddev", color: "var(--slot-1)" }],
    valueFormatter: (v) => v.toFixed(3),
  });

  document.getElementById("chart-cost").innerHTML = svgLineChart({
    iterations: iters,
    series: [
      { key: "input_tokens", label: "input", color: "var(--slot-2)" },
      { key: "output_tokens", label: "output", color: "var(--slot-1)" },
    ],
    valueFormatter: fmtCompact,
  });

  document.getElementById("chart-speed").innerHTML = svgLineChart({
    iterations: iters,
    series: [{ key: "latency_ms", label: "latency", color: "var(--slot-1)" }],
    valueFormatter: (v) => Math.round(v) + "ms",
  });

  renderIterPicker(iters);
}

function fmtScore(v) {
  if (v == null) return "--";
  const pct = v <= 1.5 ? v * 100 : v;
  return Math.round(pct) + "%";
}

function fmtCompact(v) {
  if (v == null) return "--";
  if (v >= 1000) return (v / 1000).toFixed(1) + "K";
  return String(Math.round(v));
}

/* ---- generic SVG line chart, no dependencies ------------------------ */

function svgLineChart({ iterations, series, valueFormatter }) {
  const width = 520, height = 220;
  const pad = { top: 18, right: 92, bottom: 30, left: 46 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const xs = iterations.map((r) => r.iteration);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const xScale = (x) => (xs.length <= 1 ? pad.left + innerW / 2 : pad.left + ((x - xMin) / (xMax - xMin || 1)) * innerW);

  let allVals = [];
  series.forEach((s) => iterations.forEach((r) => { const v = r[s.key]; if (v != null) allVals.push(v); }));
  if (!allVals.length) allVals = [0, 1];
  let yMin = Math.min(0, ...allVals);
  let yMax = Math.max(...allVals);
  if (yMax === yMin) yMax = yMin + 1;
  yMax += (yMax - yMin) * 0.15;
  const yScale = (y) => pad.top + innerH - ((y - yMin) / (yMax - yMin)) * innerH;

  const ticks = 4;
  let gridLines = "", yTickLabels = "";
  for (let i = 0; i <= ticks; i++) {
    const v = yMin + (yMax - yMin) * (i / ticks);
    const y = yScale(v);
    gridLines += `<line x1="${pad.left}" y1="${y}" x2="${pad.left + innerW}" y2="${y}" stroke="var(--gridline)" stroke-width="1"/>`;
    yTickLabels += `<text x="${pad.left - 8}" y="${y + 4}" text-anchor="end" font-family="var(--font-mono)" font-size="11" fill="var(--ink-muted)">${valueFormatter ? valueFormatter(v) : Math.round(v)}</text>`;
  }

  const showEvery = xs.length > 10 ? Math.ceil(xs.length / 8) : 1;
  let xTickLabels = "";
  xs.forEach((x, i) => {
    if (i % showEvery !== 0 && i !== xs.length - 1) return;
    xTickLabels += `<text x="${xScale(x)}" y="${pad.top + innerH + 20}" text-anchor="middle" font-family="var(--font-mono)" font-size="11" fill="var(--ink-muted)">${x}</text>`;
  });

  let linesSvg = "";
  const labelAnchors = [];
  series.forEach((s) => {
    const pts = iterations.map((r) => ({ x: r.iteration, y: r[s.key] })).filter((p) => p.y != null);
    if (!pts.length) return;
    const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`).join(" ");
    linesSvg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>`;
    const last = pts[pts.length - 1];
    const lx = xScale(last.x), ly = yScale(last.y);
    linesSvg += `<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="4.5" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
    labelAnchors.push({ label: s.label, color: s.color, x: lx, realY: ly, y: ly, value: valueFormatter ? valueFormatter(last.y) : last.y });
  });

  labelAnchors.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labelAnchors.length; i++) {
    if (labelAnchors[i].y - labelAnchors[i - 1].y < 26) {
      labelAnchors[i].y = labelAnchors[i - 1].y + 26;
    }
  }

  let labelsSvg = "";
  labelAnchors.forEach((a) => {
    if (Math.abs(a.y - a.realY) > 4) {
      labelsSvg += `<line x1="${a.x + 6}" y1="${a.realY.toFixed(1)}" x2="${a.x + 10}" y2="${a.y.toFixed(1)}" stroke="var(--ink-muted)" stroke-width="1"/>`;
    }
    labelsSvg += `<text x="${a.x + 10}" y="${(a.y - 2).toFixed(1)}" font-family="var(--font-display)" font-size="12.5" font-weight="600" fill="var(--ink-secondary)">${a.label}</text>`;
    labelsSvg += `<text x="${a.x + 10}" y="${(a.y + 11).toFixed(1)}" font-family="var(--font-mono)" font-size="11" fill="var(--ink-muted)">${a.value}</text>`;
  });

  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
    ${gridLines}
    <line x1="${pad.left}" y1="${pad.top + innerH}" x2="${pad.left + innerW}" y2="${pad.top + innerH}" stroke="var(--baseline)" stroke-width="1"/>
    ${yTickLabels}
    ${xTickLabels}
    ${linesSvg}
    ${labelsSvg}
  </svg>`;
}

/* ---- spec diff panel ------------------------------------------------- */

const iterPicker = document.getElementById("iter-picker");
const diffRationale = document.getElementById("diff-rationale");
const diffList = document.getElementById("diff-list");

function renderIterPicker(iters) {
  iterPicker.innerHTML = "";
  iters.forEach((it, i) => {
    const btn = document.createElement("button");
    btn.className = "iter-dot";
    btn.type = "button";
    btn.textContent = `#${it.iteration}`;
    btn.addEventListener("click", () => selectIteration(iters, i));
    iterPicker.appendChild(btn);
  });
  if (iters.length) selectIteration(iters, iters.length - 1);
}

function selectIteration(iters, index) {
  iterPicker.querySelectorAll(".iter-dot").forEach((b, i) => b.classList.toggle("active", i === index));
  const curr = iters[index];
  const prev = index > 0 ? iters[index - 1] : null;

  diffRationale.textContent = curr.mutation_rationale
    ? curr.mutation_rationale
    : prev
    ? "No rationale recorded for this iteration."
    : "Baseline iteration -- nothing to compare yet.";

  diffList.innerHTML = "";
  if (!prev) return;

  if (!prev.spec || !curr.spec) {
    const li = document.createElement("li");
    li.textContent = "Spec snapshots weren't included in these iteration records -- showing rationale only.";
    diffList.appendChild(li);
    return;
  }

  const changes = diffSpecs(prev.spec, curr.spec);
  if (!changes.length) {
    const li = document.createElement("li");
    li.textContent = "No structural change detected between these two specs.";
    diffList.appendChild(li);
    return;
  }
  changes.forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="field">${escapeHtml(c.field)}</span>: <span class="from">${escapeHtml(truncate(c.from))}</span> &rarr; <span class="to">${escapeHtml(truncate(c.to))}</span>`;
    diffList.appendChild(li);
  });
}

function truncate(v, n = 64) {
  if (v == null) return String(v);
  const s = String(v);
  return s.length > n ? s.slice(0, n) + "..." : s;
}

function diffSpecs(prev, curr) {
  const changes = [];
  if (prev.design_philosophy !== curr.design_philosophy) {
    changes.push({ field: "design_philosophy", from: prev.design_philosophy, to: curr.design_philosophy });
  }
  const po = prev.orchestration || {}, co = curr.orchestration || {};
  ["topology", "entry_role", "max_steps", "budget_tokens"].forEach((k) => {
    if (po[k] !== co[k]) changes.push({ field: `orchestration.${k}`, from: po[k], to: co[k] });
  });
  const prevRoles = Object.fromEntries((prev.roles || []).map((r) => [r.name, r]));
  const currRoles = Object.fromEntries((curr.roles || []).map((r) => [r.name, r]));
  const allNames = new Set([...Object.keys(prevRoles), ...Object.keys(currRoles)]);
  allNames.forEach((name) => {
    const pr = prevRoles[name], cr = currRoles[name];
    if (!pr) { changes.push({ field: `roles.${name}`, from: "(absent)", to: "added" }); return; }
    if (!cr) { changes.push({ field: `roles.${name}`, from: "present", to: "(removed)" }); return; }
    if (pr.system_prompt !== cr.system_prompt) changes.push({ field: `roles.${name}.system_prompt`, from: pr.system_prompt, to: cr.system_prompt });
    if (JSON.stringify(pr.tools) !== JSON.stringify(cr.tools)) changes.push({ field: `roles.${name}.tools`, from: (pr.tools || []).join(", "), to: (cr.tools || []).join(", ") });
    if (pr.max_tokens !== cr.max_tokens) changes.push({ field: `roles.${name}.max_tokens`, from: pr.max_tokens, to: cr.max_tokens });
    if (pr.temperature !== cr.temperature) changes.push({ field: `roles.${name}.temperature`, from: pr.temperature, to: cr.temperature });
    if ((pr.memory || {}).kind !== (cr.memory || {}).kind) changes.push({ field: `roles.${name}.memory.kind`, from: (pr.memory || {}).kind, to: (cr.memory || {}).kind });
    if ((pr.retry || {}).max_retries !== (cr.retry || {}).max_retries) changes.push({ field: `roles.${name}.retry.max_retries`, from: (pr.retry || {}).max_retries, to: (cr.retry || {}).max_retries });
  });
  return changes;
}

/* ------------------------------------------------------------------ */

loadDomains();
