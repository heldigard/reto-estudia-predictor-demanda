const state = {
  apiBaseUrl: null,
  catalog: [],
  currentSku: null,
  currentImputation: "seasonal_median",
  currentHorizon: 367,
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

async function probeBackend(baseUrl) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${baseUrl}/api/health`, { signal: controller.signal });
    if (!response.ok) {
      return false;
    }
    const payload = await response.json();
    return payload.status === "ok";
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
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
        `${state.apiBaseUrl}/api/live/sku/${sku}?imputation=${state.currentImputation}&horizon=${state.currentHorizon}`,
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

function formatDateLabel(dateString, mode = "short") {
  const date = new Date(`${dateString}T00:00:00`);
  if (Number.isNaN(date.getTime())) {
    return dateString;
  }
  if (mode === "year") {
    return String(date.getUTCFullYear());
  }
  return new Intl.DateTimeFormat("es-CO", {
    year: "numeric",
    month: "short",
  }).format(date);
}

function findForecastIndexForYear(forecastRows, year) {
  const index = forecastRows.findIndex((row) => row.fecha.startsWith(String(year)));
  return index >= 0 ? index + 1 : null;
}

function renderMetrics(payload, analysis) {
  const forecastRows = analysis.best_run.forecast;
  const forecastEnd = forecastRows.at(-1)?.fecha ?? "—";
  const metrics = [
    {
      label: "Mejor modelo",
      value: prettyModelName(analysis.best_model),
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
      note: `semanas proyectadas · cierre ${forecastEnd}`,
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
            maxTicksLimit: labels.length > 200 ? 16 : 12,
            color: "#56616f",
            callback: (value, index) => {
              const label = labels[index];
              if (!label) {
                return "";
              }
              if (labels.length > 200) {
                const date = new Date(`${label}T00:00:00`);
                const isBoundary =
                  index === 0 ||
                  index === labels.length - 1 ||
                  (date.getUTCMonth() === 0 && date.getUTCDate() <= 7);
                return isBoundary ? String(date.getUTCFullYear()) : "";
              }
              return formatDateLabel(label);
            },
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

  renderModelExplanation(analysis);
}

function prettyModelName(modelName) {
  return modelName.replaceAll("_", " ");
}

function renderModelExplanation(analysis) {
  const explanationNode = qs("model-explanation");
  if (!explanationNode) {
    return;
  }

  const ranked = [...analysis.model_comparison]
    .filter((item) => item.status === "ok" && item.mape !== null)
    .sort((left, right) => left.mape - right.mape);

  if (!ranked.length) {
    explanationNode.textContent = "No fue posible interpretar la comparación de modelos para este producto.";
    return;
  }

  const best = ranked[0];
  const second = ranked[1];
  const marginText = second
    ? ` Mejora el MAPE frente a ${prettyModelName(second.modelo)} por ${Math.abs(second.mape - best.mape).toFixed(2)} puntos.`
    : "";

  const reasonMap = {
    seasonal_naive:
      "Funciona bien cuando el patrón reciente se parece mucho a la misma temporada del año anterior.",
    holt_winters:
      "Captura tendencia y estacionalidad al mismo tiempo, por eso suele responder mejor cuando la serie tiene ciclos marcados.",
    seasonal_regression:
      "Aprovecha la estructura temporal completa y por eso suele adaptarse mejor cuando hay estacionalidad con cambios graduales en el nivel.",
    catboost_gpu_global:
      "Integra señales temporales y patrones históricos más complejos, lo que puede darle ventaja cuando la serie tiene relaciones no lineales.",
  };

  explanationNode.textContent =
    `${prettyModelName(best.modelo)} fue elegido porque obtuvo el menor error para este SKU ` +
    `(MAE ${best.mae}, MAPE ${best.mape}%). ${reasonMap[best.modelo] || ""}${marginText}`;
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

  renderForecastInsights(analysis);
}

function renderForecastInsights(analysis) {
  const forecastRows = analysis.best_run.forecast;
  const container = qs("forecast-insights");
  const chipContainer = qs("forecast-year-chips");
  if (!container || !chipContainer || !forecastRows.length) {
    return;
  }

  const first = forecastRows[0];
  const last = forecastRows.at(-1);
  const middle = forecastRows[Math.floor(forecastRows.length / 2)];
  const lastWidth = (last.ic_superior - last.ic_inferior).toFixed(1);
  const years = [...new Set(forecastRows.map((row) => row.fecha.slice(0, 4)))];

  container.innerHTML = `
    <article class="forecast-card">
      <strong>Arranque forecast</strong>
      <span>${first.valor_central}</span>
      <small>${formatDateLabel(first.fecha)} · primer punto proyectado</small>
    </article>
    <article class="forecast-card">
      <strong>Punto medio</strong>
      <span>${middle.valor_central}</span>
      <small>${formatDateLabel(middle.fecha)} · seguimiento del tramo largo</small>
    </article>
    <article class="forecast-card">
      <strong>Cierre visible</strong>
      <span>${last.valor_central}</span>
      <small>${formatDateLabel(last.fecha)} · último valor visible</small>
    </article>
    <article class="forecast-card">
      <strong>Incertidumbre final</strong>
      <span>${lastWidth}</span>
      <small>Ancho del intervalo en el último punto del forecast</small>
    </article>
  `;

  chipContainer.innerHTML = years
    .map((year) => {
      const horizon = findForecastIndexForYear(forecastRows, Number(year));
      const isActive = forecastRows.at(-1)?.fecha.startsWith(year);
      return `
        <button class="chip year-chip ${isActive ? "active" : ""}" type="button" data-horizon="${horizon}">
          ver ${year}
        </button>
      `;
    })
    .join("");

  chipContainer.querySelectorAll(".year-chip").forEach((button) => {
    button.addEventListener("click", async () => {
      state.currentHorizon = Number(button.dataset.horizon);
      qs("horizon-range").value = String(state.currentHorizon);
      if (state.apiBaseUrl) {
        state.cache.clear();
      }
      await refreshView();
    });
  });
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
  const forecastEnd = analysis.best_run.forecast.at(-1)?.fecha;
  qs("horizon-label").textContent = forecastEnd
    ? `${state.currentHorizon} semanas · hasta ${forecastEnd}`
    : `${state.currentHorizon} semanas`;

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
  const stillExists = state.catalog.some((item) => item.sku === state.currentSku);
  state.currentSku = stillExists ? state.currentSku : (state.catalog[0]?.sku ?? null);
}

function renderHeaderState() {
  qs("mode-badge").textContent = state.apiBaseUrl ? "modelo en vivo" : "demo estático";
  qs("coverage-badge").textContent = `${state.meta.weeks} semanas · ${state.meta.sku_count} SKU`;
  qs("dataset-badge").textContent = `${state.meta.date_range.inicio} → ${state.meta.date_range.fin}`;
  qs("api-hint").textContent = state.apiBaseUrl
    ? "Forecast consultado al backend"
    : "Forecast servido desde artefactos estáticos";
  const livePanel = qs("live-panel");
  if (livePanel) {
    livePanel.hidden = !state.apiBaseUrl;
  }
  const liveStatus = qs("live-status");
  if (liveStatus && state.apiBaseUrl) {
    liveStatus.textContent = "La proyección visible puede recalcularse en tiempo real para el producto seleccionado.";
  }
  const horizonRange = qs("horizon-range");
  horizonRange.max = String(state.meta.max_horizon);
  if (state.currentHorizon > state.meta.max_horizon) {
    state.currentHorizon = state.meta.max_horizon;
  }
  horizonRange.value = String(state.currentHorizon);
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

  const refreshButton = qs("refresh-live-btn");
  if (refreshButton) {
    refreshButton.addEventListener("click", async () => {
      try {
        state.cache.clear();
        const liveStatus = qs("live-status");
        if (liveStatus) {
          liveStatus.textContent = "Actualizando forecast con el modelo activo…";
        }
        await refreshView();
        if (liveStatus) {
          liveStatus.textContent = "Proyección actualizada correctamente.";
        }
      } catch (error) {
        console.error(error);
        const liveStatus = qs("live-status");
        if (liveStatus) {
          liveStatus.textContent =
            error instanceof Error ? error.message : "No fue posible actualizar la proyección.";
        }
      }
    });
  }
}

async function bootstrap() {
  try {
    const detectedApiBaseUrl = resolveApiBaseUrl();
    state.apiBaseUrl = detectedApiBaseUrl && (await probeBackend(detectedApiBaseUrl))
      ? detectedApiBaseUrl
      : "";
    await loadCatalog();
    state.currentHorizon = state.meta.max_horizon;
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
