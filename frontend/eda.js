const state = {
  charts: {},
};

async function loadEdaSummary() {
  const response = await fetch("./data/eda-summary.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`No se pudo cargar el resumen EDA (${response.status})`);
  }
  return response.json();
}

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("es-CO", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function setHtml(id, html) {
  const node = document.getElementById(id);
  if (node) node.innerHTML = html;
}

function buildHeroStats(summary) {
  const { dataset, invalid_rate_pct_global, weekly_invalid_rate_summary } = summary;
  const cards = [
    ["Filas", formatNumber(dataset.rows)],
    ["Semanas", formatNumber(dataset.weeks)],
    ["SKU", formatNumber(dataset.skus)],
    ["Dato inválido", `${formatNumber(invalid_rate_pct_global, 2)}%`],
    ["Peor semana", weekly_invalid_rate_summary.worst_week],
    ["Pico semanal", `${formatNumber(weekly_invalid_rate_summary.max_pct, 1)}%`],
  ];

  setHtml(
    "hero-stats",
    cards
      .map(
        ([label, value]) => `
          <div class="stat-panel">
            <strong>${label}</strong>
            <span>${value}</span>
          </div>
        `,
      )
      .join(""),
  );

  const invalidRows = Object.entries(summary.raw_note_counts)
    .filter(([note]) => note !== "ok")
    .reduce((acc, [, count]) => acc + count, 0);

  const totalRows = summary.dataset.rows;
  const reliableRows = totalRows - invalidRows;

  const highlight = [
    `Entre ${summary.dataset.date_start} y ${summary.dataset.date_end} se consolidaron ${formatNumber(totalRows)} registros semanales.`,
    `${formatNumber(reliableRows)} entraron como dato confiable y ${formatNumber(invalidRows)} exigieron reglas explícitas de limpieza o imputación.`,
    `La ventana de validación temporal se reservó en ${summary.dataset.holdout_weeks} semanas para no mezclar evaluación con entrenamiento.`,
  ].join(" ");
  setHtml("hero-highlight-text", highlight);
}

function buildInsightGrid(summary) {
  const invalidRows = Object.entries(summary.raw_note_counts)
    .filter(([note]) => note !== "ok")
    .reduce((acc, [, count]) => acc + count, 0);
  const missingRows = (summary.raw_note_counts.missing || 0) + (summary.raw_note_counts.missing_bloque || 0);
  const outlierRows = (summary.raw_note_counts.outlier_alto || 0) + (summary.raw_note_counts.outlier_bajo || 0);
  const worstSku = summary.top_invalid_skus[0];

  const items = [
    {
      label: "Presión de limpieza",
      value: `${formatNumber(invalidRows)}`,
      note: "registros no pudieron usarse de forma directa y requirieron reglas correctivas.",
    },
    {
      label: "Ausencia de dato",
      value: `${formatNumber(missingRows)}`,
      note: "faltantes simples o en bloque, el problema dominante del dataset.",
    },
    {
      label: "Ruido extremo",
      value: `${formatNumber(outlierRows)}`,
      note: "outliers altos o bajos debían aislarse para no sesgar el entrenamiento.",
    },
    {
      label: "SKU más exigente",
      value: worstSku.sku,
      note: `${worstSku.producto} concentró ${formatNumber(worstSku.invalid_rate_pct, 2)}% de registros inválidos.`,
    },
  ];

  setHtml(
    "insight-grid",
    items
      .map(
        (item) => `
          <article class="insight-card">
            <strong>${item.label}</strong>
            <span>${item.value}</span>
            <p>${item.note}</p>
          </article>
        `,
      )
      .join(""),
  );
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
  }
}

function buildNoteChart(summary) {
  const ctx = document.getElementById("note-chart");
  if (!ctx) return;
  destroyChart("notes");

  const labels = Object.keys(summary.raw_note_counts);
  const values = labels.map((label) => summary.raw_note_counts[label]);
  const colors = labels.map((label) => {
    if (label === "ok") return "rgba(13, 138, 116, 0.75)";
    if (label.startsWith("outlier")) return "rgba(255, 107, 53, 0.72)";
    if (label === "negativo") return "rgba(180, 35, 24, 0.72)";
    return "rgba(31, 77, 189, 0.72)";
  });

  state.charts.notes = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Registros",
          data: values,
          backgroundColor: colors,
          borderRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: {
            callback: (value) => formatNumber(value),
          },
        },
      },
    },
  });
}

function buildSkuChart(summary) {
  const ctx = document.getElementById("sku-chart");
  if (!ctx) return;
  destroyChart("sku");

  const top = summary.top_invalid_skus.slice(0, 5);
  state.charts.sku = new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map((item) => item.sku),
      datasets: [
        {
          label: "Tasa inválida %",
          data: top.map((item) => item.invalid_rate_pct),
          backgroundColor: [
            "rgba(255, 107, 53, 0.8)",
            "rgba(250, 135, 74, 0.8)",
            "rgba(31, 77, 189, 0.78)",
            "rgba(13, 138, 116, 0.78)",
            "rgba(138, 43, 226, 0.72)",
          ],
          borderRadius: 10,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: {
            callback: (value) => `${value}%`,
          },
        },
        y: {
          grid: { display: false },
        },
      },
    },
  });

  const topSku = top[0];
  setHtml(
    "sku-chart-caption",
    `${topSku.sku} · ${topSku.producto} lideró la complejidad del proceso con ${formatNumber(topSku.invalid_rate_pct, 2)}% de registros inválidos y una racha máxima de ${topSku.missing_streak_max} semanas faltantes.`,
  );
}

function buildRanking(summary) {
  const rows = summary.top_invalid_skus
    .slice(0, 5)
    .map((item, index) => {
      return `
        <div class="ranking-row">
          <span class="ranking-order">${index + 1}</span>
          <div class="ranking-copy">
            <strong>${item.sku} · ${item.producto}</strong>
            <span>${formatNumber(item.invalid_rows)} registros inválidos sobre ${formatNumber(item.rows)} semanas.</span>
          </div>
          <div class="ranking-value">
            <span>${formatNumber(item.invalid_rate_pct, 2)}%</span>
            <small>racha faltante: ${item.missing_streak_max} semanas</small>
          </div>
        </div>
      `;
    })
    .join("");

  setHtml("ranking-list", rows);
}

function buildClosing(summary) {
  const worstWeek = summary.weekly_invalid_rate_summary.worst_week;
  const invalidRate = formatNumber(summary.invalid_rate_pct_global, 2);
  const modelHoldout = summary.dataset.holdout_weeks;

  setHtml(
    "closing-quote",
    `El reto no se resolvía solo con elegir un modelo: primero había que demostrar criterio sobre un dataset con ${invalidRate}% de registros conflictivos, identificar la semana más crítica (${worstWeek}) y dejar una base limpia, trazable y lista para validar forecasts en un holdout temporal de ${modelHoldout} semanas.`,
  );
}

function render(summary) {
  buildHeroStats(summary);
  buildInsightGrid(summary);
  buildNoteChart(summary);
  buildSkuChart(summary);
  buildRanking(summary);
  buildClosing(summary);
}

async function main() {
  try {
    const summary = await loadEdaSummary();
    render(summary);
  } catch (error) {
    setHtml(
      "hero-highlight-text",
      `No fue posible cargar el resumen EDA. ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

main();
