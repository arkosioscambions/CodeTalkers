const behaviorRows = [
  {
    family: "Qwen2.5-Coder",
    model: "1.5B",
    variant: "base",
    infilling: { ctr: 90.18, nl: 0.66, defcls: 2.1 },
    completion: { ctr: 84.46, nl: 8.46, defcls: 2.8 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "7B",
    variant: "base",
    infilling: { ctr: 92.93, nl: 0.16, defcls: 0.56 },
    completion: { ctr: 89.63, nl: 8.53, defcls: 2.93 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "14B",
    variant: "base",
    infilling: { ctr: 96.96, nl: 0.26, defcls: 0.1 },
    completion: { ctr: 93.97, nl: 5.4, defcls: 7.49 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "32B",
    variant: "base",
    infilling: { ctr: 97.08, nl: 0.15, defcls: 0.12 },
    completion: { ctr: 89.75, nl: 2.93, defcls: 4.36 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "1.5B",
    variant: "instruct",
    infilling: { ctr: 89.74, nl: 0.46, defcls: 1.65 },
    completion: { ctr: 86.51, nl: 8.2, defcls: 9.18 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "7B",
    variant: "instruct",
    infilling: { ctr: 88.72, nl: 2.5, defcls: 2.7 },
    completion: { ctr: 89.03, nl: 9.31, defcls: 8.27 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "14B",
    variant: "instruct",
    infilling: { ctr: 91.78, nl: 2.68, defcls: 1.47 },
    completion: { ctr: 90.73, nl: 6.12, defcls: 2.73 },
  },
  {
    family: "Qwen2.5-Coder",
    model: "32B",
    variant: "instruct",
    infilling: { ctr: 89.48, nl: 2.27, defcls: 1.14 },
    completion: { ctr: 86.21, nl: 14.0, defcls: 9.83 },
  },
  {
    family: "DeepSeek-Coder",
    model: "1.3B",
    variant: "base",
    infilling: { ctr: 90.63, nl: 0.14, defcls: 1.41 },
    completion: { ctr: 79.5, nl: 5.14, defcls: 18.62 },
  },
  {
    family: "DeepSeek-Coder",
    model: "6.7B",
    variant: "base",
    infilling: { ctr: 93.72, nl: 0.15, defcls: 0.79 },
    completion: { ctr: 82.06, nl: 5.66, defcls: 15.04 },
  },
  {
    family: "DeepSeek-Coder",
    model: "33B",
    variant: "base",
    infilling: { ctr: 94.92, nl: 0.15, defcls: 0.72 },
    completion: { ctr: 80.16, nl: 3.32, defcls: 17.38 },
  },
  {
    family: "DeepSeek-Coder",
    model: "1.3B",
    variant: "instruct",
    infilling: { ctr: 62.6, nl: 6.24, defcls: 12.21 },
    completion: { ctr: 85.8, nl: 11.52, defcls: 9.18 },
  },
  {
    family: "DeepSeek-Coder",
    model: "6.7B",
    variant: "instruct",
    infilling: { ctr: 58.38, nl: 2.39, defcls: 11.55 },
    completion: { ctr: 79.68, nl: 4.36, defcls: 24.02 },
  },
  {
    family: "DeepSeek-Coder",
    model: "33B",
    variant: "instruct",
    infilling: { ctr: 54.64, nl: 3.92, defcls: 13.1 },
    completion: { ctr: 79.2, nl: 3.65, defcls: 23.18 },
  },
];

const stageAverages = [
  {
    stage: "Base",
    infilling: 66.1,
    infillingDelta: "0.00%",
    completion: 42.98,
    completionDelta: "0.00%",
    instruction: null,
    instructionDelta: "NA",
  },
  {
    stage: "75k",
    infilling: 64.8,
    infillingDelta: "-1.97%",
    completion: 45.93,
    completionDelta: "+6.86%",
    instruction: 51.19,
    instructionDelta: "0.00%",
  },
  {
    stage: "110k",
    infilling: 64.61,
    infillingDelta: "-2.25%",
    completion: 44.11,
    completionDelta: "+2.65%",
    instruction: 57.17,
    instructionDelta: "+11.68%",
  },
];

function pairRowsByFamily() {
  const families = new Map();

  for (const row of behaviorRows) {
    if (!families.has(row.family)) {
      families.set(row.family, new Map());
    }

    const modelRows = families.get(row.family);
    if (!modelRows.has(row.model)) {
      modelRows.set(row.model, {});
    }

    modelRows.get(row.model)[row.variant] = row;
  }

  return families;
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeFamily(pairMap, family) {
  const deltas = {
    infilling: { ctr: [], nl: [], defcls: [] },
    completion: { ctr: [], nl: [], defcls: [] },
  };

  for (const pair of pairMap.values()) {
    if (!pair.base || !pair.instruct) {
      continue;
    }

    for (const task of ["infilling", "completion"]) {
      for (const metric of ["ctr", "nl", "defcls"]) {
        deltas[task][metric].push(pair.instruct[task][metric] - pair.base[task][metric]);
      }
    }
  }

  return {
    family,
    summary: {
      infilling: {
        ctr: average(deltas.infilling.ctr),
        nl: average(deltas.infilling.nl),
        defcls: average(deltas.infilling.defcls),
      },
      completion: {
        ctr: average(deltas.completion.ctr),
        nl: average(deltas.completion.nl),
        defcls: average(deltas.completion.defcls),
      },
    },
  };
}

function formatDelta(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pts`;
}

function createDeltaRow(label, delta, betterWhenLower, maxDelta) {
  const isGood = betterWhenLower ? delta <= 0 : delta >= 0;
  const width = `${Math.max(8, (Math.abs(delta) / maxDelta) * 100)}%`;
  return `
    <div class="delta-row">
      <div class="delta-head">
        <span>${label}</span>
        <span class="delta-value ${isGood ? "good" : "bad"}">${formatDelta(delta)}</span>
      </div>
      <div class="delta-bar">
        <span class="delta-fill ${isGood ? "good" : "bad"}" style="width:${width};"></span>
      </div>
    </div>
  `;
}

function renderBehaviorGrid() {
  const container = document.getElementById("behavior-grid");
  if (!container) {
    return;
  }

  const familySummaries = [];
  for (const [family, pairMap] of pairRowsByFamily()) {
    familySummaries.push(summarizeFamily(pairMap, family));
  }

  const allDeltaMagnitudes = familySummaries.flatMap(({ summary }) => [
    Math.abs(summary.infilling.ctr),
    Math.abs(summary.infilling.nl),
    Math.abs(summary.infilling.defcls),
    Math.abs(summary.completion.ctr),
    Math.abs(summary.completion.nl),
    Math.abs(summary.completion.defcls),
  ]);
  const maxDelta = Math.max(...allDeltaMagnitudes);

  container.innerHTML = familySummaries
    .map(
      ({ family, summary }) => `
        <article class="family-panel">
          <div class="family-panel-header">
            <h3>${family}</h3>
            <p>Average instruct minus base shift</p>
          </div>
          <div class="task-block">
            <h4>Infilling</h4>
            ${createDeltaRow("CTR", summary.infilling.ctr, false, maxDelta)}
            ${createDeltaRow("NL", summary.infilling.nl, true, maxDelta)}
            ${createDeltaRow("Def/Cls", summary.infilling.defcls, true, maxDelta)}
          </div>
          <div class="task-block">
            <h4>Completion</h4>
            ${createDeltaRow("CTR", summary.completion.ctr, false, maxDelta)}
            ${createDeltaRow("NL", summary.completion.nl, true, maxDelta)}
            ${createDeltaRow("Def/Cls", summary.completion.defcls, true, maxDelta)}
          </div>
        </article>
      `
    )
    .join("");
}

function renderStageGrid() {
  const container = document.getElementById("stage-grid");
  if (!container) {
    return;
  }

  container.innerHTML = stageAverages
    .map(
      (stage) => `
        <article class="stage-card">
          <span class="stage-name">${stage.stage}</span>
          <h3>${stage.stage} stage</h3>
          <p>
            ${
              stage.stage === "Base"
                ? "Reference model before public instruction tuning."
                : "Grouped averages across checkpoints in the replicated Magicoder pipeline."
            }
          </p>
          <div class="stage-metrics">
            <div class="stage-metric">
              <small>Infilling average</small>
              <strong>${stage.infilling.toFixed(2)}</strong>
              <small>Delta: ${stage.infillingDelta}</small>
            </div>
            <div class="stage-metric">
              <small>Completion average</small>
              <strong>${stage.completion.toFixed(2)}</strong>
              <small>Delta: ${stage.completionDelta}</small>
            </div>
            <div class="stage-metric">
              <small>Instruction average</small>
              <strong>${stage.instruction === null ? "NA" : stage.instruction.toFixed(2)}</strong>
              <small>Delta: ${stage.instructionDelta}</small>
            </div>
          </div>
        </article>
      `
    )
    .join("");
}

function enableReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.12 }
  );

  for (const node of document.querySelectorAll(".reveal")) {
    observer.observe(node);
  }
}

renderBehaviorGrid();
renderStageGrid();
enableReveal();
