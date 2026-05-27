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

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizeByFamily() {
  const grouped = new Map();

  for (const row of behaviorRows) {
    if (!grouped.has(row.family)) {
      grouped.set(row.family, new Map());
    }
    const familyMap = grouped.get(row.family);
    if (!familyMap.has(row.model)) {
      familyMap.set(row.model, {});
    }
    familyMap.get(row.model)[row.variant] = row;
  }

  const summaries = [];
  for (const [family, models] of grouped.entries()) {
    const deltas = {
      infilling: { ctr: [], nl: [], defcls: [] },
      completion: { ctr: [], nl: [], defcls: [] },
    };

    for (const pair of models.values()) {
      if (!pair.base || !pair.instruct) {
        continue;
      }

      for (const task of ["infilling", "completion"]) {
        for (const metric of ["ctr", "nl", "defcls"]) {
          deltas[task][metric].push(
            pair.instruct[task][metric] - pair.base[task][metric]
          );
        }
      }
    }

    summaries.push({
      family,
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
    });
  }

  return summaries;
}

function formatDelta(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} pts`;
}

function metricClass(metric, value) {
  if (metric === "ctr") {
    return value >= 0 ? "metric-good" : "metric-bad";
  }
  return value <= 0 ? "metric-good" : "metric-bad";
}

function renderBehaviorGrid() {
  const container = document.getElementById("behavior-grid");
  if (!container) {
    return;
  }

  const summaries = summarizeByFamily();
  container.innerHTML = summaries
    .map(
      (summary) => `
        <section class="behavior-panel">
          <h4>${summary.family}</h4>
          <div class="table-wrap">
            <table class="metric-table">
              <thead>
                <tr>
                  <th>Task / Metric</th>
                  <th>Average delta</th>
                </tr>
              </thead>
              <tbody>
                <tr><td><strong>Infilling CTR</strong></td><td class="${metricClass("ctr", summary.infilling.ctr)}">${formatDelta(summary.infilling.ctr)}</td></tr>
                <tr><td>Infilling NL</td><td class="${metricClass("nl", summary.infilling.nl)}">${formatDelta(summary.infilling.nl)}</td></tr>
                <tr><td>Infilling Def/Cls</td><td class="${metricClass("defcls", summary.infilling.defcls)}">${formatDelta(summary.infilling.defcls)}</td></tr>
                <tr><td><strong>Completion CTR</strong></td><td class="${metricClass("ctr", summary.completion.ctr)}">${formatDelta(summary.completion.ctr)}</td></tr>
                <tr><td>Completion NL</td><td class="${metricClass("nl", summary.completion.nl)}">${formatDelta(summary.completion.nl)}</td></tr>
                <tr><td>Completion Def/Cls</td><td class="${metricClass("defcls", summary.completion.defcls)}">${formatDelta(summary.completion.defcls)}</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      `
    )
    .join("");
}

renderBehaviorGrid();
