const state = {
  apiBaseUrl: null,
  catalog: [],
  currentSku: null,
  currentImputation: "seasonal_median",
  currentHorizon: 12,
  cache: new Map(),
  charts: {},
  meta: null,
};

function qs(id) {
  return document.getElementById(id);
}

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("apiBaseUrl");
  const fromConfig = window.APP_CONFIG?.apiBaseUrl;
  return (fromQuery || fromConfig || "").replace(/\/$/, "");
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new Error(
      "La API no devolvió JSON. Verifica que el dev tunnel siga activo y que el puerto esté en visibilidad Public.",
    );
  }
  return response.json();
}

async function loadCatalog() {
  if (state.apiBaseUrl) {
    const payload = await fetchJson(`${state.apiBaseUrl}/api/meta`);
    state.meta = payload.meta;
    state.catalog = payload.catalog;
    return;
  }

  const payload = await fetchJson("./data/catalog.json");
  state.meta = payload.meta;
  state.catalog = payload.catalog;
}

async function loadSkuData(sku) {
  if (state.cache.has(sku)) {
    return state.cache.get(sku);
  }

  const payload = state.apiBaseUrl
    ? await fetchJson(
        `${state.apiBaseUrl}/api/sku/${sku}?imputation=${state.currentImputation}&horizon=${state.currentHorizon}`,
      )
    : await fetchJson(`./data/series/${sku}.json`);

  state.cache.set(sku, payload);
  return payload;
}

function getCurrentAnalysis(payload) {
  if (payload.analysis) {
    return payload.analysis;
  }

  const analysis = payload.analyses[state.currentImputation];
  return {
    ...analysis,
    best_run: {
      ...analysis.best_run,
      forecast: analysis.best_run.forecast.slice(0, state.currentHorizon),
    },
  };
}

function summarizeTopAnomaly(anomalySummary) {
  const entries = Object.entries(anomalySummary || {});
  if (!entries.length) {
    return "sin anomalías";
  }
  const [type, count] = entries.sort((left, right) => right[1] - left[1])[0];
  return `${type} · ${count}`;
}

function metricCard(label, value, note) {
  return `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-note">${note}</div>
    </article>
  `;
}

function renderMetrics(payload, analysis) {
  const metrics = [
    {
      label: "Mejor modelo",
      value: analysis.best_model.replace("_", " "),
      note: `MAPE ${analysis.best_model_metrics.mape}%`,
    },
    {
      label: "Media limpia",
      value: analysis.cleaned_summary.media,
      note: "unidades por semana",
    },
    {
      label: "Desv. estándar",
      value: analysis.cleaned_summary.desviacion_estandar,
      note: "variabilidad de la serie",
    },
    {
      label: "Anomalías",
      value: payload.anomaly_log.length,
      note: summarizeTopAnomaly(payload.anomaly_summary),
    },
    {
      label: "Horizonte",
      value: `${state.currentHorizon}`,
      note: "semanas proyectadas",
    },
  ];

  qs("metrics-grid").innerHTML = metrics
    .map((item) => metricCard(item.label, item.value, item.note))
    .join("");
}

function destroyChart(name) {
  if (state.charts[name]) {
    state.charts[name].destroy();
  }
}

function renderSeriesChart(payload, analysis) {
  destroyChart("series");
  const labels = payload.dates.concat(analysis.best_run.forecast.map((item) => item.fecha));
  const rawValues = payload.raw_series.map((item) => item.valor);
  const cleanedValues = analysis.cleaned_values;
  const fittedValues = analysis.best_run.fitted;
  const forecastValues = Array(payload.dates.length).fill(null).concat(
    analysis.best_run.forecast.map((item) => item.valor_central),
  );
  const lower = Array(payload.dates.length).fill(null).concat(
    analysis.best_run.forecast.map((item) => item.ic_inferior),
  );
  const upper = Array(payload.dates.length).fill(null).concat(
    analysis.best_run.forecast.map((item) => item.ic_superior),
  );

  state.charts.series = new Chart(qs("series-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "observado",
          data: rawValues.concat(Array(state.currentHorizon).fill(null)),
          borderColor: "rgba(23, 32, 42, 0.22)",
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.25,
        },
        {
          label: "limpio",
          data: cleanedValues.concat(Array(state.currentHorizon).fill(null)),
          borderColor: "#0d8a74",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.25,
        },
        {
          label: "ajuste",
          data: fittedValues.concat(Array(state.currentHorizon).fill(null)),
          borderColor: "#1f4dbd",
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [8, 5],
          tension: 0.25,
        },
        {
          label: "IC inferior",
          data: lower,
          borderColor: "rgba(138, 43, 226, 0.08)",
          backgroundColor: "rgba(138, 43, 226, 0.12)",
          pointRadius: 0,
          borderWidth: 0,
        },
        {
          label: "IC superior",
          data: upper,
          borderColor: "rgba(138, 43, 226, 0.08)",
          backgroundColor: "rgba(138, 43, 226, 0.12)",
          pointRadius: 0,
          borderWidth: 0,
          fill: "-1",
        },
        {
          label: "forecast",
          data: forecastValues,
          borderColor: "#8a2be2",
          pointRadius: 0,
          borderWidth: 2.2,
          tension: 0.2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 12,
            color: "#56616f",
          },
          grid: { color: "rgba(23, 32, 42, 0.08)" },
        },
        y: {
          ticks: { color: "#56616f" },
          grid: { color: "rgba(23, 32, 42, 0.08)" },
        },
      },
    },
  });
}

function renderModelTable(analysis) {
  const bestModel = analysis.best_model;
  const rows = analysis.model_comparison
    .map((item) => {
      const isBest = item.modelo === bestModel;
      return `
        <tr class="${isBest ? "row-highlight" : ""}">
          <td>${item.modelo.replace("_", " ")}</td>
          <td>${item.mae ?? "—"}</td>
          <td>${item.mape ?? "—"}</td>
          <td>${item.status}</td>
        </tr>
      `;
    })
    .join("");

  qs("model-table").innerHTML = `
    <thead>
      <tr>
        <th>Modelo</th>
        <th>MAE</th>
        <th>MAPE</th>
        <th>Estado</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function renderForecastTable(analysis) {
  const rows = analysis.best_run.forecast
    .map(
      (item) => `
        <tr>
          <td>${item.fecha}</td>
          <td>${item.valor_central}</td>
          <td>${item.ic_inferior}</td>
          <td>${item.ic_superior}</td>
        </tr>
      `,
    )
    .join("");
  qs("forecast-table").innerHTML = `
    <thead>
      <tr>
        <th>Semana</th>
        <th>Valor central</th>
        <th>IC inferior</th>
        <th>IC superior</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function renderResidualChart(analysis) {
  destroyChart("residuals");
  state.charts.residuals = new Chart(qs("residual-chart"), {
    type: "bar",
    data: {
      labels: analysis.best_run.residual_histogram.bins,
      datasets: [
        {
          label: "residuos",
          data: analysis.best_run.residual_histogram.counts,
          backgroundColor: "rgba(31, 77, 189, 0.22)",
          borderColor: "#1f4dbd",
          borderWidth: 1.2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#56616f", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
          grid: { display: false },
        },
        y: {
          ticks: { color: "#56616f" },
          grid: { color: "rgba(23, 32, 42, 0.08)" },
        },
      },
    },
  });
}

function renderAnomalySummary(payload) {
  const summary = Object.entries(payload.anomaly_summary || {})
    .map(([type, count]) => `<span class="chip">${type}: ${count}</span>`)
    .join("");
  qs("anomaly-summary").innerHTML = summary || '<span class="chip">sin anomalías</span>';
}

function tagClass(type) {
  if (type.includes("missing")) {
    return "warning";
  }
  if (type === "ok") {
    return "good";
  }
  return "";
}

function renderAnomalyLog(payload) {
  const items = payload.anomaly_log
    .slice(0, 80)
    .map(
      (item) => `
        <article class="log-item">
          <span class="log-date">${item.fecha}</span>
          <span>${item.valor_original ?? "sin dato"}</span>
          <span class="tag ${tagClass(item.tipo)}">${item.tipo}</span>
        </article>
      `,
    )
    .join("");
  qs("anomaly-log").innerHTML = items || "<p>No hay anomalías registradas.</p>";
}

async function refreshView() {
  const payload = await loadSkuData(state.currentSku);
  const normalized = {
    sku: payload.sku,
    producto: payload.producto,
    dates: payload.dates,
    raw_series: payload.raw_series || payload.rawSeries || payload.raw_series,
    anomaly_log: payload.anomaly_log,
    anomaly_summary: payload.anomaly_summary,
  };
  const analysis = getCurrentAnalysis(payload);

  qs("chart-title").textContent = `${normalized.producto} · ${normalized.sku}`;
  qs("horizon-label").textContent = `${state.currentHorizon} semanas`;

  renderMetrics(normalized, analysis);
  renderSeriesChart(normalized, analysis);
  renderModelTable(analysis);
  renderResidualChart(analysis);
  renderForecastTable(analysis);
  renderAnomalySummary(normalized);
  renderAnomalyLog(normalized);
}

function populateCatalog() {
  qs("sku-select").innerHTML = state.catalog
    .map((item) => `<option value="${item.sku}">${item.sku} · ${item.producto}</option>`)
    .join("");
  state.currentSku = state.catalog[0]?.sku ?? null;
}

function renderHeaderState() {
  qs("mode-badge").textContent = state.apiBaseUrl ? "live API" : "demo estático";
  qs("coverage-badge").textContent = `${state.meta.weeks} semanas · ${state.meta.sku_count} SKU`;
  qs("dataset-badge").textContent = `${state.meta.date_range.inicio} → ${state.meta.date_range.fin}`;
  qs("api-hint").textContent = state.apiBaseUrl
    ? `Conectado a ${state.apiBaseUrl}`
    : "Leyendo JSON precomputado desde ./data";
}

function attachEvents() {
  qs("sku-select").addEventListener("change", async (event) => {
    state.currentSku = event.target.value;
    if (state.apiBaseUrl) {
      state.cache.delete(state.currentSku);
    }
    await refreshView();
  });

  qs("imputation-select").addEventListener("change", async (event) => {
    state.currentImputation = event.target.value;
    if (state.apiBaseUrl) {
      state.cache.clear();
    }
    await refreshView();
  });

  qs("horizon-range").addEventListener("input", async (event) => {
    state.currentHorizon = Number(event.target.value);
    if (state.apiBaseUrl) {
      state.cache.clear();
    }
    await refreshView();
  });
}

async function bootstrap() {
  try {
    state.apiBaseUrl = resolveApiBaseUrl();
    await loadCatalog();
    populateCatalog();
    renderHeaderState();
    attachEvents();
    await refreshView();
  } catch (error) {
    console.error(error);
    document.body.innerHTML = `
      <main style="padding: 32px; font-family: 'Space Grotesk', sans-serif;">
        <h1>No se pudo cargar el dashboard</h1>
        <p>${error.message}</p>
      </main>
    `;
  }
}

bootstrap();
