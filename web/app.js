"use strict";

const state = {
  data: null,
  selectedGames: new Set(),
  selectedGroups: new Set(),
  backendAvailable: false,
  statusPollTimerId: null,
  thumbnailCache: new Map(),
  imageMapViews: {
    umap: null,
    tsne: null,
  },
  latestCanvasRender: {
    samples: [],
    colorMap: {},
  },
  analysisParams: {
    umapNNeighbors: 10,
    umapMinDist: 0.1,
    tsnePerplexity: 10,
  },
  radiusParams: {
    extentPct: 0,
    nNearest: 5,
    useTsne: false,
  },
};

const els = {
  status: document.getElementById("status"),
  runAnalysisBtn: document.getElementById("runAnalysisBtn"),
  loadDefaultBtn: document.getElementById("loadDefaultBtn"),
  jsonFileInput: document.getElementById("jsonFileInput"),
  gameFilter: document.getElementById("gameFilter"),
  groupFilter: document.getElementById("groupFilter"),
  resetFiltersBtn: document.getElementById("resetFiltersBtn"),
  umapNeighborsInput: document.getElementById("umapNeighborsInput"),
  umapMinDistInput: document.getElementById("umapMinDistInput"),
  tsnePerplexityInput: document.getElementById("tsnePerplexityInput"),
  umapNeighborsValue: document.getElementById("umapNeighborsValue"),
  umapMinDistValue: document.getElementById("umapMinDistValue"),
  tsnePerplexityValue: document.getElementById("tsnePerplexityValue"),
  presetLocalBtn: document.getElementById("presetLocalBtn"),
  presetBalancedBtn: document.getElementById("presetBalancedBtn"),
  presetGlobalBtn: document.getElementById("presetGlobalBtn"),
  radiusCanvas: document.getElementById("radiusCanvas"),
  umapImage3d: document.getElementById("umapImage3d"),
  tsneImage3d: document.getElementById("tsneImage3d"),
  umapImage3dLegend: document.getElementById("umapImage3dLegend"),
  tsneImage3dLegend: document.getElementById("tsneImage3dLegend"),
  radiusExtentInput: document.getElementById("radiusExtentInput"),
  radiusNNearestInput: document.getElementById("radiusNNearestInput"),
  radiusUseTsneCheck: document.getElementById("radiusUseTsneCheck"),
  radiusExtentValue: document.getElementById("radiusExtentValue"),
  radiusNNearestValue: document.getElementById("radiusNNearestValue"),
  tsnePlot: document.getElementById("tsnePlot"),
  umapPlot: document.getElementById("umapPlot"),
  centroidHeatmap: document.getElementById("centroidHeatmap"),
  groupCentroidHeatmap: document.getElementById("groupCentroidHeatmap"),
  promptHeatmap: document.getElementById("promptHeatmap"),
  promptGroupHeatmap: document.getElementById("promptGroupHeatmap"),
  clusterTableHead: document.querySelector("#clusterTable thead"),
  clusterTableBody: document.querySelector("#clusterTable tbody"),
  skippedTableBody: document.querySelector("#skippedTable tbody"),
};

function setStatus(message, isError = false) {
  if (!els.status) return;
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

function formatNetworkError(err, actionLabel) {
  const base = err && err.message ? String(err.message) : "Unknown error.";
  const lowered = base.toLowerCase();
  if (lowered.includes("failed to fetch") || lowered.includes("networkerror") || lowered.includes("network error")) {
    return (
      `${actionLabel} failed due to network/server issue. ` +
      "Start local server: python src/local_app_server.py --host 127.0.0.1 --port 8000"
    );
  }
  return base;
}

function debounce(fn, delayMs) {
  let timeoutId = null;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delayMs);
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function forceHorizontalScrollReset() {
  try {
    if (window.scrollX !== 0) {
      window.scrollTo({ left: 0, top: window.scrollY });
    }
    if (document.documentElement) {
      document.documentElement.scrollLeft = 0;
    }
    if (document.body) {
      document.body.scrollLeft = 0;
    }
  } catch (_) {
    // No-op: this is only a layout guard for browsers that restore stale horizontal offsets.
  }
}

function clearThumbnailCache() {
  state.thumbnailCache.clear();
}

function ensureCanvasScale(canvas) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.floor(rect.width || canvas.width || 1));
  const cssHeight = Math.max(1, Math.floor(rect.height || canvas.height || 1));
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  const targetWidth = Math.floor(cssWidth * dpr);
  const targetHeight = Math.floor(cssHeight * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function get2DPoints(samples, key) {
  return samples.map((sample) => ({
    x: Number(sample[key][0]),
    y: Number(sample[key][1]),
    label: sample.label,
    thumbnail: sample.thumbnail || null,
  }));
}

function projectPoints(points, width, height, padding = 24) {
  if (points.length === 0) return [];

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const innerW = Math.max(1, width - padding * 2);
  const innerH = Math.max(1, height - padding * 2);

  return points.map((p) => ({
    ...p,
    px: padding + ((p.x - minX) / spanX) * innerW,
    py: padding + (1 - (p.y - minY) / spanY) * innerH,
  }));
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getThumbnailImageRecord(src, onLoaded) {
  if (!src) return null;
  const cached = state.thumbnailCache.get(src);
  if (cached) return cached;

  const record = {
    img: null,
    loaded: false,
    failed: false,
    promise: null,
  };
  const img = new Image();
  img.decoding = "async";
  record.img = img;
  record.promise = new Promise((resolve) => {
    img.onload = () => {
      record.loaded = true;
      resolve(img);
      if (typeof onLoaded === "function") onLoaded();
    };
    img.onerror = () => {
      record.failed = true;
      resolve(null);
    };
  });
  img.src = src;
  state.thumbnailCache.set(src, record);
  return record;
}

function getColorMap(games) {
  const map = {};
  const total = Math.max(games.length, 1);
  games.forEach((game, idx) => {
    const hue = Math.round((idx / total) * 330);
    map[game] = `hsl(${hue}, 70%, 45%)`;
  });
  return map;
}

const GROUP_LEGEND_ORDER = ["indie", "aaa", "unassigned"];
const GROUP_COLORS = {
  indie: "#10b981",
  aaa: "#f97316",
  unassigned: "#64748b",
};

function normalizeGroupName(group) {
  if (typeof group !== "string") return "unassigned";
  const value = group.trim().toLowerCase();
  return value || "unassigned";
}

function formatGroupLabel(group) {
  const key = normalizeGroupName(group);
  if (key === "aaa") return "AAA";
  if (key === "indie") return "Indie";
  if (key === "unassigned") return "Unassigned";
  return typeof group === "string" && group.trim() ? group.trim() : "Unassigned";
}

function colorFromString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 65%, 48%)`;
}

function getGroupColor(group) {
  const key = normalizeGroupName(group);
  return GROUP_COLORS[key] || colorFromString(key);
}

function getGroupLegendEntries(samples) {
  const seen = new Map();
  for (const sample of samples || []) {
    const rawGroup = typeof sample.group === "string" && sample.group.trim() ? sample.group.trim() : "unassigned";
    const key = normalizeGroupName(rawGroup);
    if (seen.has(key)) continue;
    seen.set(key, {
      key,
      label: formatGroupLabel(rawGroup),
      color: getGroupColor(rawGroup),
    });
  }

  return [...seen.values()].sort((a, b) => {
    const ia = GROUP_LEGEND_ORDER.indexOf(a.key);
    const ib = GROUP_LEGEND_ORDER.indexOf(b.key);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a.label.localeCompare(b.label);
  });
}

function renderGroupLegend(container, samples) {
  if (!container) return;
  container.innerHTML = "";

  const entries = getGroupLegendEntries(samples);
  if (!entries.length) return;

  for (const entry of entries) {
    const item = document.createElement("span");
    item.className = "legend-item";

    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = entry.color;
    swatch.title = `${entry.label} frame color`;

    const text = document.createElement("span");
    text.textContent = entry.label;

    item.appendChild(swatch);
    item.appendChild(text);
    container.appendChild(item);
  }
}

function validateData(data) {
  if (!data || typeof data !== "object") {
    throw new Error("Invalid JSON: expected an object at root level.");
  }
  if (!Array.isArray(data.samples) || data.samples.length === 0) {
    throw new Error("Invalid JSON: 'samples' must be a non-empty array.");
  }

  for (const sample of data.samples) {
    if (!Array.isArray(sample.tsne) || sample.tsne.length !== 3) {
      throw new Error("Invalid JSON: each sample must include a 3D 'tsne' array.");
    }
    if (!Array.isArray(sample.umap) || sample.umap.length !== 3) {
      throw new Error("Invalid JSON: each sample must include a 3D 'umap' array.");
    }
    if (typeof sample.label !== "string") {
      throw new Error("Invalid JSON: each sample must include a string 'label'.");
    }
  }

  return data;
}

function parseJsonFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        resolve(JSON.parse(String(reader.result)));
      } catch (err) {
        reject(new Error(`Could not parse JSON file: ${err.message}`));
      }
    };
    reader.onerror = () => reject(new Error("Failed to read selected file."));
    reader.readAsText(file);
  });
}

async function loadDefaultData() {
  setStatus("Loading default JSON (data/analysis_results.json)...");
  const candidates = ["data/analysis_results.json", "/data/analysis_results.json"];
  let lastHttpError = null;
  let lastParseError = null;

  for (const url of candidates) {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) {
          lastHttpError = response.status;
          continue;
        }

        const body = await response.text();
        try {
          return JSON.parse(body);
        } catch (parseErr) {
          lastParseError = parseErr;
          // Analysis file may be in-flight; wait briefly and retry.
          await sleep(180);
        }
      } catch (err) {
        // Continue to next candidate URL.
        if (err instanceof TypeError) {
          await sleep(120);
          continue;
        }
        throw err;
      }
    }
  }

  if (window.location.protocol === "file:") {
    throw new Error(
      "Cannot fetch local JSON from file://. Start server: python src/local_app_server.py --host 127.0.0.1 --port 8000"
    );
  }
  if (lastHttpError !== null) {
    throw new Error(`Could not load default JSON: HTTP ${lastHttpError}`);
  }
  if (lastParseError) {
    throw new Error(
      "Default JSON is being updated and was briefly unreadable. Please click Load Default Data again."
    );
  }
  throw new Error(
    "Network error while loading default JSON. Start server: python src/local_app_server.py --host 127.0.0.1 --port 8000"
  );
}

async function fetchBackendStatus() {
  const response = await fetch("/api/status", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Backend status request failed: HTTP ${response.status}`);
  }
  return response.json();
}

async function runAnalysisFromBackend(runPayload) {
  const response = await fetch("/api/run-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(runPayload || {}),
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = { ok: false, message: "Invalid server response." };
  }

  if (!response.ok) {
    throw new Error(payload.message || `Run request failed: HTTP ${response.status}`);
  }
  return payload;
}

function stopStatusPolling() {
  if (state.statusPollTimerId !== null) {
    clearInterval(state.statusPollTimerId);
    state.statusPollTimerId = null;
  }
}

function setRunButtonBusy(isBusy) {
  if (!els.runAnalysisBtn) return;
  els.runAnalysisBtn.disabled = isBusy;
  els.runAnalysisBtn.textContent = isBusy ? "Running..." : "Run Analysis";
}

function formatBackendTail(statusPayload) {
  const tail = Array.isArray(statusPayload.output_tail) ? statusPayload.output_tail : [];
  if (tail.length === 0) return "";
  return tail[tail.length - 1];
}

function renderParamValues() {
  if (els.umapNeighborsValue) {
    els.umapNeighborsValue.textContent = String(state.analysisParams.umapNNeighbors);
  }
  if (els.umapMinDistValue) {
    els.umapMinDistValue.textContent = state.analysisParams.umapMinDist.toFixed(2);
  }
  if (els.tsnePerplexityValue) {
    els.tsnePerplexityValue.textContent = String(state.analysisParams.tsnePerplexity);
  }
}

function renderRadiusValues() {
  if (els.radiusExtentValue) {
    els.radiusExtentValue.textContent = `${state.radiusParams.extentPct}%`;
  }
  if (els.radiusNNearestValue) {
    els.radiusNNearestValue.textContent = String(state.radiusParams.nNearest);
  }
}

function syncRadiusInputsFromState() {
  if (els.radiusExtentInput) {
    els.radiusExtentInput.value = String(state.radiusParams.extentPct);
  }
  if (els.radiusNNearestInput) {
    els.radiusNNearestInput.value = String(state.radiusParams.nNearest);
  }
  if (els.radiusUseTsneCheck) {
    els.radiusUseTsneCheck.checked = Boolean(state.radiusParams.useTsne);
  }
  renderRadiusValues();
}

function syncParamInputsFromState() {
  if (els.umapNeighborsInput) {
    els.umapNeighborsInput.value = String(state.analysisParams.umapNNeighbors);
  }
  if (els.umapMinDistInput) {
    els.umapMinDistInput.value = state.analysisParams.umapMinDist.toFixed(2);
  }
  if (els.tsnePerplexityInput) {
    els.tsnePerplexityInput.value = String(state.analysisParams.tsnePerplexity);
  }
  renderParamValues();
}

function getRunPayloadFromState() {
  return {
    umap_n_neighbors: state.analysisParams.umapNNeighbors,
    umap_min_dist: state.analysisParams.umapMinDist,
    tsne_perplexity: state.analysisParams.tsnePerplexity,
  };
}

function applyPreset(presetName) {
  if (presetName === "local") {
    state.analysisParams.umapNNeighbors = 5;
    state.analysisParams.umapMinDist = 0.01;
    state.analysisParams.tsnePerplexity = 8;
  } else if (presetName === "global") {
    state.analysisParams.umapNNeighbors = 40;
    state.analysisParams.umapMinDist = 0.6;
    state.analysisParams.tsnePerplexity = 30;
  } else {
    state.analysisParams.umapNNeighbors = 10;
    state.analysisParams.umapMinDist = 0.1;
    state.analysisParams.tsnePerplexity = 10;
  }
  syncParamInputsFromState();
}

async function pollAnalysisStatus() {
  try {
    const status = await fetchBackendStatus();
    const tailLine = formatBackendTail(status);

    if (status.running) {
      const text = tailLine
        ? `Analysis running... ${tailLine}`
        : "Analysis running... processing screenshots and computing embeddings.";
      setStatus(text);
      setRunButtonBusy(true);
      return;
    }

    // Job finished
    stopStatusPolling();
    setRunButtonBusy(false);

    if (typeof status.last_exit_code === "number" && status.last_exit_code === 0) {
      setStatus("Analysis finished. Reloading visualizations...");
      await onLoadDefaultClick();
      return;
    }

    if (typeof status.last_exit_code === "number") {
      const errorText = tailLine || status.last_error || "Unknown pipeline error.";
      setStatus(`Analysis failed (exit ${status.last_exit_code}): ${errorText}`, true);
      return;
    }
  } catch (err) {
    stopStatusPolling();
    setRunButtonBusy(false);
    setStatus(formatNetworkError(err, "Status polling"), true);
  }
}

function populateGameFilter(games) {
  els.gameFilter.innerHTML = "";
  for (const game of games) {
    const option = document.createElement("option");
    option.value = game;
    option.textContent = game;
    option.selected = true;
    els.gameFilter.appendChild(option);
  }
  state.selectedGames = new Set(games);
}

function populateGroupFilter(groups) {
  if (!els.groupFilter) return;
  els.groupFilter.innerHTML = "";
  for (const group of groups) {
    const option = document.createElement("option");
    option.value = group;
    option.textContent = group;
    option.selected = true;
    els.groupFilter.appendChild(option);
  }
  state.selectedGroups = new Set(groups);
}

function getSelectedGamesFromUi() {
  return new Set([...els.gameFilter.selectedOptions].map((opt) => opt.value));
}

function getSelectedGroupsFromUi() {
  if (!els.groupFilter) return new Set();
  return new Set([...els.groupFilter.selectedOptions].map((opt) => opt.value));
}

function filteredSamples() {
  if (!state.data) return [];
  const selected = state.selectedGames;
  const selectedGroups = state.selectedGroups;
  if (selected.size === 0) return [];
  if (selectedGroups.size === 0) return [];
  return state.data.samples.filter((sample) => {
    const group = typeof sample.group === "string" && sample.group ? sample.group : "unassigned";
    return selected.has(sample.label) && selectedGroups.has(group);
  });
}

function render3DScatter(container, samples, vectorKey, title, colorMap) {
  const grouped = new Map();
  for (const sample of samples) {
    if (!grouped.has(sample.label)) grouped.set(sample.label, []);
    grouped.get(sample.label).push(sample);
  }

  const traces = [];
  for (const [label, points] of grouped.entries()) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: label,
      x: points.map((p) => p[vectorKey][0]),
      y: points.map((p) => p[vectorKey][1]),
      z: points.map((p) => p[vectorKey][2]),
      customdata: points.map((p) => [p.cluster_id ?? "N/A"]),
      hovertemplate:
        "<b>%{fullData.name}</b><br>" +
        `${vectorKey.toUpperCase()}: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>` +
        "Cluster: %{customdata[0]}<extra></extra>",
      marker: {
        size: 4,
        opacity: 0.82,
        color: colorMap[label] || "#334155",
      },
    });
  }

  const manyGames = grouped.size > 12;
  const layout = {
    title,
    margin: { l: 0, r: 0, t: 48, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: { title: `${vectorKey.toUpperCase()}-1` },
      yaxis: { title: `${vectorKey.toUpperCase()}-2` },
      zaxis: { title: `${vectorKey.toUpperCase()}-3` },
      aspectmode: "data",
    },
    showlegend: !manyGames,
    legend: { orientation: "h", y: -0.12 },
  };

  Plotly.react(container, traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  });
}

function renderHeatmap(container, xLabels, yLabels, matrix, title) {
  const trace = {
    type: "heatmap",
    x: xLabels,
    y: yLabels,
    z: matrix,
    colorscale: "Viridis",
    hoverongaps: false,
    colorbar: { title: "Score" },
  };

  const layout = {
    title,
    margin: { l: 90, r: 16, t: 42, b: 120 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { tickangle: -35, automargin: true },
    yaxis: { automargin: true },
  };

  Plotly.react(container, [trace], layout, { responsive: true, displaylogo: false });
}

function renderClusterTable(crosstab) {
  const games = Object.keys(crosstab || {}).sort((a, b) => a.localeCompare(b));
  const clusterIds = uniqueSorted(games.flatMap((g) => Object.keys(crosstab[g] || {}))).sort(
    (a, b) => Number(a) - Number(b)
  );

  if (games.length === 0 || clusterIds.length === 0) {
    els.clusterTableHead.innerHTML = "";
    els.clusterTableBody.innerHTML = "";
    return;
  }

  els.clusterTableHead.innerHTML = "";
  const headRow = document.createElement("tr");
  const gameTh = document.createElement("th");
  gameTh.textContent = "Game";
  headRow.appendChild(gameTh);
  for (const clusterId of clusterIds) {
    const th = document.createElement("th");
    th.textContent = `Cluster ${clusterId}`;
    headRow.appendChild(th);
  }
  els.clusterTableHead.appendChild(headRow);

  els.clusterTableBody.innerHTML = "";
  for (const game of games) {
    const row = document.createElement("tr");
    const gameCell = document.createElement("td");
    gameCell.textContent = game;
    row.appendChild(gameCell);

    for (const clusterId of clusterIds) {
      const td = document.createElement("td");
      td.textContent = String(crosstab[game][clusterId] || 0);
      row.appendChild(td);
    }

    els.clusterTableBody.appendChild(row);
  }
}

function renderSkippedTable(skippedImages) {
  const rows = Array.isArray(skippedImages) ? skippedImages : [];
  els.skippedTableBody.innerHTML = "";

  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "No skipped images.";
    tr.appendChild(td);
    els.skippedTableBody.appendChild(tr);
    return;
  }

  const maxRows = 1000;
  for (const rowData of rows.slice(0, maxRows)) {
    const tr = document.createElement("tr");
    for (const value of [rowData.image_id ?? "", rowData.label ?? "", rowData.path ?? "", rowData.reason ?? ""]) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.appendChild(td);
    }
    els.skippedTableBody.appendChild(tr);
  }

  if (rows.length > maxRows) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = `Showing ${maxRows} of ${rows.length} skipped images.`;
    tr.appendChild(td);
    els.skippedTableBody.appendChild(tr);
  }
}

function updateRadiusControlBounds(sampleCount) {
  if (!els.radiusNNearestInput) return;
  const maxK = Math.max(2, Math.min(40, sampleCount - 1));
  els.radiusNNearestInput.max = String(maxK);
  if (state.radiusParams.nNearest > maxK) {
    state.radiusParams.nNearest = maxK;
  }
  syncRadiusInputsFromState();
}

function drawNeighborhoodExplorer(samples, colorMap) {
  const scaled = ensureCanvasScale(els.radiusCanvas);
  if (!scaled) return;
  const { ctx, width, height } = scaled;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f3f4f6";
  ctx.fillRect(0, 0, width, height);

  if (!Array.isArray(samples) || samples.length < 2) {
    ctx.fillStyle = "#475569";
    ctx.font = "16px 'Space Grotesk', sans-serif";
    ctx.fillText("Need at least 2 samples to render neighborhood explorer.", 20, 36);
    return;
  }

  const key = state.radiusParams.useTsne ? "tsne" : "umap";
  const points2D = get2DPoints(samples, key);
  const projected = projectPoints(points2D, width, height, 28);
  const n = projected.length;
  const k = Math.max(1, Math.min(state.radiusParams.nNearest, n - 1));
  const extentFactor = state.radiusParams.extentPct / 100;

  // Compute pairwise distances in projected space to visualize neighborhood radii.
  const distances = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dx = projected[i].px - projected[j].px;
      const dy = projected[i].py - projected[j].py;
      const d = Math.hypot(dx, dy);
      distances[i].push({ j, d });
      distances[j].push({ j: i, d });
    }
  }
  for (let i = 0; i < n; i += 1) {
    distances[i].sort((a, b) => a.d - b.d);
  }

  if (extentFactor > 0) {
    for (let i = 0; i < n; i += 1) {
      const p = projected[i];
      const kth = distances[i][Math.max(0, k - 1)]?.d || 0;
      const radius = kth * extentFactor;
      if (radius <= 0) continue;

      ctx.beginPath();
      ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
      ctx.fill();
    }

    // Draw fuzzy weighted edges to first k neighbors.
    for (let i = 0; i < n; i += 1) {
      const p = projected[i];
      const kth = distances[i][Math.max(0, k - 1)]?.d || 1e-6;
      for (let t = 0; t < k && t < distances[i].length; t += 1) {
        const { j, d } = distances[i][t];
        if (j < i) continue;
        const q = projected[j];
        const weight = Math.max(0.04, 1 - d / Math.max(1e-6, kth));
        ctx.beginPath();
        ctx.moveTo(p.px, p.py);
        ctx.lineTo(q.px, q.py);
        ctx.strokeStyle = `rgba(14, 116, 144, ${0.25 * weight})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  for (const p of projected) {
    ctx.beginPath();
    ctx.arc(p.px, p.py, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = colorMap[p.label] || "#1d4ed8";
    ctx.fill();
  }

  ctx.fillStyle = "#334155";
  ctx.font = "14px 'Space Grotesk', sans-serif";
  ctx.fillText(
    `Basis: ${key.toUpperCase()} | extent=${state.radiusParams.extentPct}% | n_nearest=${k} | points=${n}`,
    20,
    24
  );
}

class Thumbnail3DMap {
  constructor(container, methodKey, title) {
    this.container = container;
    this.methodKey = methodKey;
    this.title = title;
    this.canvas = null;
    this.ctx = null;
    this.overlay = null;
    this.currentSamples = [];
    this.currentColorMap = {};
    this.basePoints = [];
    this.isPointerDown = false;
    this.pointerMode = "rotate";
    this.lastPointerX = 0;
    this.lastPointerY = 0;
    this.needsDraw = false;
    this.frameId = null;
    this.view = {
      yaw: 0.56,
      pitch: -0.24,
      zoom: 20,
      panX: 0,
      panY: 0,
      camDist: 60,
    };
  }

  init() {
    if (!this.container) return false;
    if (this.canvas && this.ctx) return true;

    this.container.innerHTML = "";
    this.canvas = document.createElement("canvas");
    this.canvas.className = "image3d-canvas";
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    if (!this.ctx) {
      this.showOverlay("Could not create canvas context.");
      return false;
    }

    this.overlay = document.createElement("div");
    this.overlay.className = "image3d-overlay";
    this.container.appendChild(this.overlay);
    this.bindPointerEvents();
    this.resize();
    this.showOverlay("Ready.");
    return true;
  }

  getViewportSize() {
    const width = Math.max(260, Math.floor(this.container.clientWidth || 700));
    const height = Math.max(260, Math.floor(this.container.clientHeight || 620));
    return { width, height };
  }

  showOverlay(message) {
    if (!this.overlay && this.container) {
      this.overlay = document.createElement("div");
      this.overlay.className = "image3d-overlay";
      this.container.appendChild(this.overlay);
    }
    if (!this.overlay) return;
    this.overlay.textContent = message;
    this.overlay.style.display = message ? "block" : "none";
  }

  resize() {
    if (!this.canvas || !this.ctx) return;
    const { width, height } = this.getViewportSize();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.requestDraw();
  }

  bindPointerEvents() {
    if (!this.canvas) return;
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());

    this.canvas.addEventListener("pointerdown", (event) => {
      this.isPointerDown = true;
      this.lastPointerX = event.clientX;
      this.lastPointerY = event.clientY;
      this.pointerMode = event.button === 2 || event.shiftKey ? "pan" : "rotate";
      this.canvas.setPointerCapture(event.pointerId);
    });

    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.isPointerDown) return;
      const dx = event.clientX - this.lastPointerX;
      const dy = event.clientY - this.lastPointerY;
      this.lastPointerX = event.clientX;
      this.lastPointerY = event.clientY;

      if (this.pointerMode === "pan") {
        this.view.panX += dx;
        this.view.panY += dy;
      } else {
        this.view.yaw += dx * 0.0095;
        this.view.pitch = clamp(this.view.pitch + dy * 0.0085, -1.45, 1.45);
      }
      this.requestDraw();
    });

    const stopPointer = (event) => {
      this.isPointerDown = false;
      if (this.canvas.hasPointerCapture(event.pointerId)) {
        this.canvas.releasePointerCapture(event.pointerId);
      }
    };
    this.canvas.addEventListener("pointerup", stopPointer);
    this.canvas.addEventListener("pointercancel", stopPointer);
    this.canvas.addEventListener("pointerleave", stopPointer);

    this.canvas.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0012);
        this.view.zoom = clamp(this.view.zoom * factor, 6, 140);
        this.requestDraw();
      },
      { passive: false }
    );

    this.canvas.addEventListener("dblclick", () => {
      this.view = {
        yaw: 0.56,
        pitch: -0.24,
        zoom: 20,
        panX: 0,
        panY: 0,
        camDist: 60,
      };
      this.requestDraw();
    });
  }

  normalizedPositions(samples) {
    const vectors = samples.map((sample) => sample[this.methodKey]).filter((v) => Array.isArray(v) && v.length === 3);
    if (vectors.length === 0) return [];

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (const v of vectors) {
      minX = Math.min(minX, Number(v[0]));
      minY = Math.min(minY, Number(v[1]));
      minZ = Math.min(minZ, Number(v[2]));
      maxX = Math.max(maxX, Number(v[0]));
      maxY = Math.max(maxY, Number(v[1]));
      maxZ = Math.max(maxZ, Number(v[2]));
    }

    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    const maxSpan = Math.max(1e-6, maxX - minX, maxY - minY, maxZ - minZ);
    const scale = 38 / maxSpan;

    return samples.map((sample) => {
      const v = sample[this.methodKey];
      const x = (Number(v[0]) - centerX) * scale;
      const y = (Number(v[1]) - centerY) * scale;
      const z = (Number(v[2]) - centerZ) * scale;
      return { sample, x, y, z };
    });
  }

  projectPoint(point, width, height) {
    const cosYaw = Math.cos(this.view.yaw);
    const sinYaw = Math.sin(this.view.yaw);
    const cosPitch = Math.cos(this.view.pitch);
    const sinPitch = Math.sin(this.view.pitch);

    // Rotate around Y then X.
    const x1 = point.x * cosYaw + point.z * sinYaw;
    const z1 = -point.x * sinYaw + point.z * cosYaw;
    const y2 = point.y * cosPitch - z1 * sinPitch;
    const z2 = point.y * sinPitch + z1 * cosPitch;

    const denom = this.view.camDist + z2;
    const persp = this.view.zoom / Math.max(8, denom);
    const sx = width * 0.5 + x1 * persp * 28 + this.view.panX;
    const sy = height * 0.5 - y2 * persp * 28 + this.view.panY;
    const depth = z2;
    const densityScale =
      this.basePoints.length > 180 ? 0.68 : this.basePoints.length > 120 ? 0.8 : this.basePoints.length > 80 ? 0.9 : 1;
    const size = clamp((10 + persp * 36) * densityScale, 12, 76);

    return { sx, sy, depth, size };
  }

  requestDraw() {
    if (this.needsDraw) return;
    this.needsDraw = true;
    this.frameId = window.requestAnimationFrame(() => {
      this.needsDraw = false;
      this.draw();
    });
  }

  draw() {
    if (!this.canvas || !this.ctx) return;
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f3f4f6";
    ctx.fillRect(0, 0, width, height);

    if (!this.basePoints.length) {
      this.showOverlay("No samples available for this filter.");
      return;
    }

    // Axis guide.
    ctx.strokeStyle = "rgba(22, 163, 74, 0.38)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width * 0.5 + this.view.panX, 8);
    ctx.lineTo(width * 0.5 + this.view.panX, height - 8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(8, height * 0.5 + this.view.panY);
    ctx.lineTo(width - 8, height * 0.5 + this.view.panY);
    ctx.stroke();

    const drawQueue = this.basePoints.map((p) => {
      const proj = this.projectPoint(p, width, height);
      return { ...p, ...proj };
    });
    drawQueue.sort((a, b) => a.depth - b.depth);

    let thumbnailCount = 0;
    for (const p of drawQueue) {
      const x = p.sx - p.size * 0.5;
      const y = p.sy - p.size * 0.5;
      const gameColor = this.currentColorMap[p.sample.label] || "#0f766e";
      const groupColor = getGroupColor(p.sample.group);
      ctx.fillStyle = gameColor;
      ctx.globalAlpha = 0.52;
      ctx.fillRect(x, y, p.size, p.size);
      ctx.globalAlpha = 1;

      const thumbRec = getThumbnailImageRecord(p.sample.thumbnail, () => this.requestDraw());
      if (thumbRec && thumbRec.loaded && thumbRec.img) {
        thumbnailCount += 1;
        ctx.drawImage(thumbRec.img, x, y, p.size, p.size);
      } else {
        ctx.fillStyle = "#111827";
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, Math.max(2.2, p.size * 0.22), 0, Math.PI * 2);
        ctx.fill();
      }

      // Border color indicates dataset group (Indie/AAA).
      ctx.strokeStyle = groupColor;
      ctx.lineWidth = Math.max(1.2, p.size * 0.08);
      ctx.strokeRect(x, y, p.size, p.size);
    }

    this.showOverlay(`${this.title}: loaded ${thumbnailCount}/${drawQueue.length} thumbnail paths`);
  }

  render(samples, colorMap) {
    if (!this.init()) return;
    if (!this.canvas || !this.ctx) return;

    if (!Array.isArray(samples) || samples.length === 0) {
      this.basePoints = [];
      this.showOverlay("No samples available for this filter.");
      this.requestDraw();
      return;
    }

    this.currentSamples = samples;
    this.currentColorMap = colorMap || {};
    this.basePoints = this.normalizedPositions(samples);
    if (this.basePoints.length === 0) {
      this.showOverlay("Could not map 3D coordinates.");
      this.requestDraw();
      return;
    }
    this.requestDraw();
  }
}

function renderImage3DMaps(samples, colorMap) {
  if (state.imageMapViews.umap) {
    state.imageMapViews.umap.render(samples, colorMap);
  }
  if (state.imageMapViews.tsne) {
    state.imageMapViews.tsne.render(samples, colorMap);
  }
  renderGroupLegend(els.umapImage3dLegend, samples);
  renderGroupLegend(els.tsneImage3dLegend, samples);
}

function renderCanvasVisuals(samples, colorMap) {
  state.latestCanvasRender.samples = samples;
  state.latestCanvasRender.colorMap = colorMap;
  drawNeighborhoodExplorer(samples, colorMap);
  renderImage3DMaps(samples, colorMap);
}

function renderAll() {
  forceHorizontalScrollReset();
  if (!state.data) return;

  const selectedSamples = filteredSamples();
  const total = state.data.samples.length;
  updateRadiusControlBounds(selectedSamples.length);

  if (selectedSamples.length === 0) {
    setStatus("No samples match current game/group filters. Select at least one game and one group.", true);
    Plotly.purge(els.tsnePlot);
    Plotly.purge(els.umapPlot);
    Plotly.purge(els.centroidHeatmap);
    Plotly.purge(els.groupCentroidHeatmap);
    Plotly.purge(els.promptHeatmap);
    Plotly.purge(els.promptGroupHeatmap);
    renderCanvasVisuals([], {});
    return;
  }

  const allGames = uniqueSorted(state.data.samples.map((s) => s.label));
  const colorMap = getColorMap(allGames);

  render3DScatter(els.tsnePlot, selectedSamples, "tsne", `3D t-SNE (${selectedSamples.length}/${total} samples)`, colorMap);
  render3DScatter(els.umapPlot, selectedSamples, "umap", `3D UMAP (${selectedSamples.length}/${total} samples)`, colorMap);

  const centroid = state.data.centroid_similarity;
  if (centroid && Array.isArray(centroid.labels) && Array.isArray(centroid.matrix)) {
    renderHeatmap(els.centroidHeatmap, centroid.labels, centroid.labels, centroid.matrix, "Game Centroid Cosine Similarity");
  } else {
    Plotly.purge(els.centroidHeatmap);
  }
  const groupCentroid = state.data.group_centroid_similarity;
  if (groupCentroid && Array.isArray(groupCentroid.labels) && Array.isArray(groupCentroid.matrix)) {
    renderHeatmap(
      els.groupCentroidHeatmap,
      groupCentroid.labels,
      groupCentroid.labels,
      groupCentroid.matrix,
      "Group Centroid Cosine Similarity"
    );
  } else {
    Plotly.purge(els.groupCentroidHeatmap);
  }

  const prompts = state.data.prompt_similarity;
  if (prompts && Array.isArray(prompts.prompts) && Array.isArray(prompts.games) && Array.isArray(prompts.matrix)) {
    const source = typeof prompts.source === "string" ? prompts.source : "clip_text_prompts";
    const title =
      source === "style_adapter"
        ? "Fine-Tuned Style Adapter Scores by Game"
        : "CLIP Prompt Similarity (Average by Game)";
    renderHeatmap(els.promptHeatmap, prompts.prompts, prompts.games, prompts.matrix, title);
  } else {
    Plotly.purge(els.promptHeatmap);
  }
  const promptByGroup = state.data.prompt_similarity_by_group;
  if (
    promptByGroup &&
    Array.isArray(promptByGroup.prompts) &&
    Array.isArray(promptByGroup.groups) &&
    Array.isArray(promptByGroup.matrix)
  ) {
    const source = typeof promptByGroup.source === "string" ? promptByGroup.source : "clip_text_prompts";
    const title =
      source === "style_adapter"
        ? "Fine-Tuned Style Adapter Scores by Group"
        : "CLIP Prompt Similarity by Group";
    renderHeatmap(els.promptGroupHeatmap, promptByGroup.prompts, promptByGroup.groups, promptByGroup.matrix, title);
  } else {
    Plotly.purge(els.promptGroupHeatmap);
  }

  renderClusterTable(state.data.clusters?.crosstab || {});
  renderSkippedTable(state.data.skipped_images || []);
  renderCanvasVisuals(selectedSamples, colorMap);

  const meta = state.data.meta || {};
  const runtime = state.data.runtime_parameters || {};
  const groupCount = new Set(
    selectedSamples.map((sample) =>
      typeof sample.group === "string" && sample.group ? sample.group : "unassigned"
    )
  ).size;
  setStatus(
    `Loaded ${meta.valid_images ?? total} samples across ${meta.num_games ?? allGames.length} games and ${meta.num_groups ?? groupCount} groups. ` +
      `t-SNE=${runtime.tsne?.method || "unknown"}, UMAP=${runtime.umap?.method || "unknown"}.`
  );
}

async function applyData(rawData) {
  state.data = validateData(rawData);
  clearThumbnailCache();
  const runtime = state.data.runtime_parameters || {};
  if (runtime.umap?.n_neighbors) {
    state.analysisParams.umapNNeighbors = Number(runtime.umap.n_neighbors);
  }
  if (typeof runtime.umap?.min_dist === "number") {
    state.analysisParams.umapMinDist = Number(runtime.umap.min_dist);
  }
  if (runtime.tsne?.perplexity) {
    state.analysisParams.tsnePerplexity = Number(runtime.tsne.perplexity);
  }
  syncParamInputsFromState();

  const games = uniqueSorted(state.data.samples.map((s) => s.label));
  populateGameFilter(games);
  const groups = uniqueSorted(
    state.data.samples.map((s) => (typeof s.group === "string" && s.group ? s.group : "unassigned"))
  );
  populateGroupFilter(groups);
  renderAll();
}

async function onRunAnalysisClick() {
  if (!state.backendAvailable) {
    setStatus(
      "Run Analysis is available only when using the backend server: python src/local_app_server.py",
      true
    );
    return;
  }

  try {
    setRunButtonBusy(true);
    const runPayload = getRunPayloadFromState();
    setStatus(
      `Starting analysis with n_neighbors=${runPayload.umap_n_neighbors}, ` +
        `min_dist=${Number(runPayload.umap_min_dist).toFixed(2)}, ` +
        `perplexity=${runPayload.tsne_perplexity}...`
    );
    const payload = await runAnalysisFromBackend(runPayload);
    setStatus(payload.message || "Analysis started.");

    stopStatusPolling();
    state.statusPollTimerId = setInterval(() => {
      pollAnalysisStatus();
    }, 1500);
    // Run once immediately for faster feedback.
    pollAnalysisStatus();
  } catch (err) {
    setRunButtonBusy(false);
    setStatus(formatNetworkError(err, "Run analysis"), true);
  }
}

async function onLoadDefaultClick() {
  try {
    const json = await loadDefaultData();
    await applyData(json);
  } catch (err) {
    setStatus(formatNetworkError(err, "Loading default data"), true);
  }
}

async function onFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  try {
    setStatus(`Loading file: ${file.name} ...`);
    const json = await parseJsonFromFile(file);
    await applyData(json);
  } catch (err) {
    setStatus(err.message || "Failed to load selected JSON.", true);
  } finally {
    event.target.value = "";
  }
}

function onFilterChanged() {
  state.selectedGames = getSelectedGamesFromUi();
  state.selectedGroups = getSelectedGroupsFromUi();
  renderAll();
}

function onResetFilters() {
  for (const opt of els.gameFilter.options) opt.selected = true;
  if (els.groupFilter) {
    for (const opt of els.groupFilter.options) opt.selected = true;
  }
  state.selectedGames = getSelectedGamesFromUi();
  state.selectedGroups = getSelectedGroupsFromUi();
  renderAll();
}

function onParamInputChange() {
  if (els.umapNeighborsInput) {
    state.analysisParams.umapNNeighbors = Number.parseInt(els.umapNeighborsInput.value, 10) || 10;
  }
  if (els.umapMinDistInput) {
    state.analysisParams.umapMinDist = Number.parseFloat(els.umapMinDistInput.value) || 0.1;
  }
  if (els.tsnePerplexityInput) {
    state.analysisParams.tsnePerplexity = Number.parseInt(els.tsnePerplexityInput.value, 10) || 10;
  }
  renderParamValues();
}

function onRadiusInputChange() {
  if (els.radiusExtentInput) {
    state.radiusParams.extentPct = Number.parseInt(els.radiusExtentInput.value, 10) || 0;
  }
  if (els.radiusNNearestInput) {
    state.radiusParams.nNearest = Number.parseInt(els.radiusNNearestInput.value, 10) || 5;
  }
  if (els.radiusUseTsneCheck) {
    state.radiusParams.useTsne = Boolean(els.radiusUseTsneCheck.checked);
  }
  renderRadiusValues();
  if (!state.data) return;
  const allGames = uniqueSorted(state.data.samples.map((s) => s.label));
  const colorMap = getColorMap(allGames);
  renderCanvasVisuals(filteredSamples(), colorMap);
}

function onWindowResize() {
  forceHorizontalScrollReset();
  if (state.imageMapViews.umap) {
    state.imageMapViews.umap.resize();
  }
  if (state.imageMapViews.tsne) {
    state.imageMapViews.tsne.resize();
  }
  if (!state.data) return;
  const allGames = uniqueSorted(state.data.samples.map((s) => s.label));
  const colorMap = getColorMap(allGames);
  renderCanvasVisuals(filteredSamples(), colorMap);
}

function initImageMapViews() {
  state.imageMapViews.umap = new Thumbnail3DMap(els.umapImage3d, "umap", "UMAP");
  state.imageMapViews.tsne = new Thumbnail3DMap(els.tsneImage3d, "tsne", "t-SNE");
}

async function initBackendAvailability() {
  if (!els.runAnalysisBtn) return;
  try {
    const status = await fetchBackendStatus();
    state.backendAvailable = true;
    els.runAnalysisBtn.disabled = false;
    els.runAnalysisBtn.title = "Run CLIP pipeline and regenerate web/data outputs.";
    if (status.running) {
      setRunButtonBusy(true);
      stopStatusPolling();
      state.statusPollTimerId = setInterval(() => {
        pollAnalysisStatus();
      }, 1500);
      pollAnalysisStatus();
    }
  } catch (_) {
    state.backendAvailable = false;
    els.runAnalysisBtn.disabled = true;
    els.runAnalysisBtn.title = "Start backend server with: python src/local_app_server.py";
  }
}

function bindEvents() {
  if (els.runAnalysisBtn) {
    els.runAnalysisBtn.addEventListener("click", onRunAnalysisClick);
  }
  if (els.umapNeighborsInput) {
    els.umapNeighborsInput.addEventListener("input", onParamInputChange);
  }
  if (els.umapMinDistInput) {
    els.umapMinDistInput.addEventListener("input", onParamInputChange);
  }
  if (els.tsnePerplexityInput) {
    els.tsnePerplexityInput.addEventListener("input", onParamInputChange);
  }
  if (els.radiusExtentInput) {
    els.radiusExtentInput.addEventListener("input", onRadiusInputChange);
  }
  if (els.radiusNNearestInput) {
    els.radiusNNearestInput.addEventListener("input", onRadiusInputChange);
  }
  if (els.radiusUseTsneCheck) {
    els.radiusUseTsneCheck.addEventListener("change", onRadiusInputChange);
  }
  if (els.presetLocalBtn) {
    els.presetLocalBtn.addEventListener("click", () => applyPreset("local"));
  }
  if (els.presetBalancedBtn) {
    els.presetBalancedBtn.addEventListener("click", () => applyPreset("balanced"));
  }
  if (els.presetGlobalBtn) {
    els.presetGlobalBtn.addEventListener("click", () => applyPreset("global"));
  }
  els.loadDefaultBtn.addEventListener("click", onLoadDefaultClick);
  els.jsonFileInput.addEventListener("change", onFileSelected);
  els.resetFiltersBtn.addEventListener("click", onResetFilters);
  els.gameFilter.addEventListener("change", debounce(onFilterChanged, 100));
  if (els.groupFilter) {
    els.groupFilter.addEventListener("change", debounce(onFilterChanged, 100));
  }
  window.addEventListener("resize", debounce(onWindowResize, 120));
}

function boot() {
  try {
    forceHorizontalScrollReset();
    setStatus("Initializing app...");
    if (typeof window.Plotly === "undefined") {
      setStatus("Plotly failed to load. Check network access or include Plotly locally.", true);
      return;
    }
    syncParamInputsFromState();
    syncRadiusInputsFromState();
    initImageMapViews();
    bindEvents();
    initBackendAvailability();
    onLoadDefaultClick();
    setTimeout(forceHorizontalScrollReset, 120);
    setTimeout(forceHorizontalScrollReset, 450);
  } catch (err) {
    setStatus(`App init failed: ${err.message || err}`, true);
  }
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
