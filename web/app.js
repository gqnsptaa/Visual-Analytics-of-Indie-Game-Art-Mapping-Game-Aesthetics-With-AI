"use strict";

const state = {
  data: null,
  selectedGames: new Set(),
  selectedGroups: new Set(),
  backendAvailable: false,
  statusPollTimerId: null,
  phase3StatusPollTimerId: null,
  igdbAvailable: false,
  igdbStatusPollTimerId: null,
  igdbSearchResults: [],
  existingFolderKeys: new Set(),
  thumbnailCache: new Map(),
  imageMapViews: {
    umap: null,
    tsne: null,
  },
  latestCanvasRender: {
    samples: [],
    colorMap: {},
  },
  latestRadiusProjection: [],
  thesisData: null,
  datasetMode: "full",
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
  groupFocus: "both",
  vizSampleSize: "all",
  phase3LastData: null,
};

const els = {
  status: document.getElementById("status"),
  runAnalysisBtn: document.getElementById("runAnalysisBtn"),
  datasetModeSelect: document.getElementById("datasetModeSelect"),
  datasetModeHint: document.getElementById("datasetModeHint"),
  groupFocusSelect: document.getElementById("groupFocusSelect"),
  vizSampleSizeSelect: document.getElementById("vizSampleSizeSelect"),
  igdbDryRunBtn: document.getElementById("igdbDryRunBtn"),
  igdbFetchBtn: document.getElementById("igdbFetchBtn"),
  igdbSeedIndieBtn: document.getElementById("igdbSeedIndieBtn"),
  igdbSeedAaaBtn: document.getElementById("igdbSeedAaaBtn"),
  igdbSearchInput: document.getElementById("igdbSearchInput"),
  igdbCompanyInput: document.getElementById("igdbCompanyInput"),
  igdbSearchLimit: document.getElementById("igdbSearchLimit"),
  igdbHideExistingCheck: document.getElementById("igdbHideExistingCheck"),
  igdbSearchBtn: document.getElementById("igdbSearchBtn"),
  igdbSelectAllBtn: document.getElementById("igdbSelectAllBtn"),
  igdbSearchResults: document.getElementById("igdbSearchResults"),
  igdbSearchGroup: document.getElementById("igdbSearchGroup"),
  igdbAddSelectedBtn: document.getElementById("igdbAddSelectedBtn"),
  igdbBulkListInput: document.getElementById("igdbBulkListInput"),
  igdbAddListBtn: document.getElementById("igdbAddListBtn"),
  igdbPdfInput: document.getElementById("igdbPdfInput"),
  igdbImportPdfBtn: document.getElementById("igdbImportPdfBtn"),
  loadDefaultBtn: document.getElementById("loadDefaultBtn"),
  loadThesisBtn: document.getElementById("loadThesisBtn"),
  jsonFileInput: document.getElementById("jsonFileInput"),
  gameFilter: document.getElementById("gameFilter"),
  groupFilter: document.getElementById("groupFilter"),
  indieCounter: document.getElementById("indieCounter"),
  aaaCounter: document.getElementById("aaaCounter"),
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
  radiusHover: document.getElementById("radiusHover"),
  radiusLegend: document.getElementById("radiusLegend"),
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
  pcaPlot: document.getElementById("pcaPlot"),
  swissRollOriginalPlot: document.getElementById("swissRollOriginalPlot"),
  swissRollLlePlot: document.getElementById("swissRollLlePlot"),
  swissRollPca2dPlot: document.getElementById("swissRollPca2dPlot"),
  centroidHeatmap: document.getElementById("centroidHeatmap"),
  groupCentroidHeatmap: document.getElementById("groupCentroidHeatmap"),
  promptHeatmap: document.getElementById("promptHeatmap"),
  promptGroupHeatmap: document.getElementById("promptGroupHeatmap"),
  qualityTableBody: document.querySelector("#qualityTable tbody"),
  thesisStatus: document.getElementById("thesisStatus"),
  thesisMetrics: document.getElementById("thesisMetrics"),
  thesisImportancePlot: document.getElementById("thesisImportancePlot"),
  thesisStatsPlot: document.getElementById("thesisStatsPlot"),
  thesisTopFeaturesBody: document.querySelector("#thesisTopFeaturesTable tbody"),
  loadPhase2Btn: document.getElementById("loadPhase2Btn"),
  phase2Status: document.getElementById("phase2Status"),
  phase2SummaryBody: document.querySelector("#phase2SummaryTable tbody"),
  phase2UmapFrame: document.getElementById("phase2UmapFrame"),
  phase2DensmapFrame: document.getElementById("phase2DensmapFrame"),
  phase2HistFrame: document.getElementById("phase2HistFrame"),
  phase2UmapLink: document.getElementById("phase2UmapLink"),
  phase2DensmapLink: document.getElementById("phase2DensmapLink"),
  phase2HistLink: document.getElementById("phase2HistLink"),
  phase2UmapTitle: document.getElementById("phase2UmapTitle"),
  phase2DensmapTitle: document.getElementById("phase2DensmapTitle"),
  phase2HistTitle: document.getElementById("phase2HistTitle"),
  runPhase3Btn: document.getElementById("runPhase3Btn"),
  loadPhase3Btn: document.getElementById("loadPhase3Btn"),
  phase3Status: document.getElementById("phase3Status"),
  phase3Metrics: document.getElementById("phase3Metrics"),
  phase3LevelsBody: document.querySelector("#phase3LevelsTable tbody"),
  phase3PcaFrame: document.getElementById("phase3PcaFrame"),
  phase3KdeFrame: document.getElementById("phase3KdeFrame"),
  phase3ResidualFrame: document.getElementById("phase3ResidualFrame"),
  phase3CosineFrame: document.getElementById("phase3CosineFrame"),
  phase3PromptFrame: document.getElementById("phase3PromptFrame"),
  phase3PcaLink: document.getElementById("phase3PcaLink"),
  phase3KdeLink: document.getElementById("phase3KdeLink"),
  phase3ResidualLink: document.getElementById("phase3ResidualLink"),
  phase3CosineLink: document.getElementById("phase3CosineLink"),
  phase3PromptLink: document.getElementById("phase3PromptLink"),
  skippedTableBody: document.querySelector("#skippedTable tbody"),
  outlierTableBody: document.querySelector("#outlierTable tbody"),
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
    group: normalizeGroupName(sample.group),
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
  indie: "#00a86b",
  aaa: "#ff6b00",
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

function normalizeGroupFocus(value) {
  const key = String(value || "").trim().toLowerCase();
  if (key === "indie" || key === "aaa") return key;
  return "both";
}

function normalizeVizSampleSize(value) {
  const key = String(value || "").trim().toLowerCase();
  if (key === "500" || key === "1000") return key;
  return "all";
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

function renderGameLegend(container, samples, colorMap) {
  if (!container) return;
  container.innerHTML = "";
  if (!Array.isArray(samples) || samples.length === 0) return;

  const labels = uniqueSorted(samples.map((sample) => sample.label));
  for (const label of labels) {
    const item = document.createElement("span");
    item.className = "legend-item";

    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = colorMap[label] || "#1d4ed8";

    const text = document.createElement("span");
    text.textContent = label;

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

function setThesisStatus(message, isError = false) {
  if (!els.thesisStatus) return;
  els.thesisStatus.textContent = message;
  els.thesisStatus.classList.toggle("error", isError);
}

function splitCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      const next = i + 1 < line.length ? line[i + 1] : "";
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      result.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  result.push(current);
  return result;
}

function parseCsvText(text) {
  const normalized = String(text || "")
    .replace(/\uFEFF/g, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
  if (!normalized) return [];

  const lines = normalized.split("\n").filter((line) => line.length > 0);
  if (lines.length < 2) return [];

  const headers = splitCsvLine(lines[0]).map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = splitCsvLine(lines[i]);
    const row = {};
    for (let c = 0; c < headers.length; c += 1) {
      row[headers[c]] = cols[c] ?? "";
    }
    rows.push(row);
  }
  return rows;
}

async function fetchOptionalJson(candidates) {
  let sawResponse = false;
  let lastError = null;
  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (response.status === 404) continue;
      sawResponse = true;
      if (!response.ok) {
        lastError = new Error(`HTTP ${response.status} while loading ${url}`);
        continue;
      }
      const text = await response.text();
      return JSON.parse(text);
    } catch (err) {
      lastError = err;
    }
  }
  if (!sawResponse && !lastError) return null;
  if (lastError) throw lastError;
  return null;
}

async function fetchOptionalCsv(candidates) {
  let sawResponse = false;
  let lastError = null;
  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (response.status === 404) continue;
      sawResponse = true;
      if (!response.ok) {
        lastError = new Error(`HTTP ${response.status} while loading ${url}`);
        continue;
      }
      const text = await response.text();
      return parseCsvText(text);
    } catch (err) {
      lastError = err;
    }
  }
  if (!sawResponse && !lastError) return null;
  if (lastError) throw lastError;
  return null;
}

function toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatMetricValue(value, digits = 3) {
  const n = toFiniteNumber(value);
  if (n === null) return "n/a";
  return n.toFixed(digits);
}

function clearThesisViews() {
  if (els.thesisMetrics) {
    els.thesisMetrics.innerHTML = "";
  }
  if (els.thesisTopFeaturesBody) {
    els.thesisTopFeaturesBody.innerHTML = "";
  }
  if (els.thesisImportancePlot && typeof window.Plotly !== "undefined") {
    Plotly.purge(els.thesisImportancePlot);
  }
  if (els.thesisStatsPlot && typeof window.Plotly !== "undefined") {
    Plotly.purge(els.thesisStatsPlot);
  }
}

function appendMetricCardTo(container, label, value) {
  if (!container) return;
  const card = document.createElement("div");
  card.className = "metric-card";
  const labelEl = document.createElement("span");
  labelEl.className = "metric-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("span");
  valueEl.className = "metric-value";
  valueEl.textContent = value;
  card.appendChild(labelEl);
  card.appendChild(valueEl);
  container.appendChild(card);
}

function appendMetricCard(label, value) {
  if (!els.thesisMetrics) return;
  appendMetricCardTo(els.thesisMetrics, label, value);
}

function renderThesisTopTable(statsRows) {
  if (!els.thesisTopFeaturesBody) return;
  els.thesisTopFeaturesBody.innerHTML = "";
  if (!Array.isArray(statsRows) || statsRows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No thesis stats rows available.";
    tr.appendChild(td);
    els.thesisTopFeaturesBody.appendChild(tr);
    return;
  }

  const sorted = [...statsRows].sort((a, b) => {
    const qa = toFiniteNumber(a.q_value_bh) ?? Infinity;
    const qb = toFiniteNumber(b.q_value_bh) ?? Infinity;
    return qa - qb;
  });
  for (const row of sorted.slice(0, 20)) {
    const tr = document.createElement("tr");
    const cells = [
      String(row.feature ?? ""),
      formatMetricValue(row.mean_indie, 4),
      formatMetricValue(row.mean_aaa, 4),
      formatMetricValue(row.mean_diff_aaa_minus_indie, 4),
      formatMetricValue(row.q_value_bh, 6),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    els.thesisTopFeaturesBody.appendChild(tr);
  }
}

function renderThesisPlots(importanceRows, statsRows) {
  if (typeof window.Plotly === "undefined") return;

  if (els.thesisImportancePlot) {
    const rows = Array.isArray(importanceRows) ? importanceRows : [];
    const top = rows
      .map((row) => ({
        feature: String(row.feature ?? ""),
        value: toFiniteNumber(row.perm_importance_mean),
      }))
      .filter((row) => row.feature && row.value !== null)
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
      .slice(0, 15);
    if (top.length > 0) {
      const ordered = [...top].reverse();
      Plotly.react(
        els.thesisImportancePlot,
        [
          {
            type: "bar",
            orientation: "h",
            y: ordered.map((r) => r.feature),
            x: ordered.map((r) => r.value),
            marker: { color: "#0f766e" },
            hovertemplate: "%{y}<br>perm importance=%{x:.4f}<extra></extra>",
          },
        ],
        {
          title: "Top Thesis Features by Permutation Importance",
          margin: { l: 230, r: 24, t: 46, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          xaxis: { title: "Importance (balanced accuracy drop)" },
        },
        { responsive: true, displaylogo: false }
      );
    } else {
      Plotly.purge(els.thesisImportancePlot);
    }
  }

  if (els.thesisStatsPlot) {
    const rows = Array.isArray(statsRows) ? statsRows : [];
    const top = rows
      .map((row) => {
        const q = toFiniteNumber(row.q_value_bh);
        const diff = toFiniteNumber(row.mean_diff_aaa_minus_indie);
        return {
          feature: String(row.feature ?? ""),
          q,
          score: q !== null ? -Math.log10(Math.max(q, 1e-12)) : null,
          diff,
        };
      })
      .filter((row) => row.feature && row.q !== null && row.score !== null)
      .sort((a, b) => (a.q ?? Infinity) - (b.q ?? Infinity))
      .slice(0, 15);
    if (top.length > 0) {
      const ordered = [...top].reverse();
      Plotly.react(
        els.thesisStatsPlot,
        [
          {
            type: "bar",
            orientation: "h",
            y: ordered.map((r) => r.feature),
            x: ordered.map((r) => r.score),
            marker: {
              color: ordered.map((r) => ((r.diff ?? 0) >= 0 ? "#ef4444" : "#2563eb")),
            },
            hovertemplate:
              "%{y}<br>-log10(q)=%{x:.3f}<br>" +
              "Color: red=AAA higher, blue=Indie higher<extra></extra>",
          },
        ],
        {
          title: "Most Significant Thesis Features (Permutation + BH)",
          margin: { l: 230, r: 24, t: 46, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          xaxis: { title: "-log10(q-value)" },
        },
        { responsive: true, displaylogo: false }
      );
    } else {
      Plotly.purge(els.thesisStatsPlot);
    }
  }
}

function renderThesisSection(thesisData) {
  if (!els.thesisStatus) return;
  if (!thesisData || !thesisData.report) {
    clearThesisViews();
    setThesisStatus(
      "No thesis output found yet. Run: python src/thesis_attribute_analysis.py --output-dir web/data/thesis"
    );
    return;
  }

  const report = thesisData.report;
  const metrics = report.classifier_metrics || {};
  const classCounts = report.class_counts || {};

  if (els.thesisMetrics) {
    els.thesisMetrics.innerHTML = "";
    appendMetricCard("Unit of Analysis", String(report.unit_of_analysis || "game"));
    appendMetricCard("Rows Used", String(report.num_rows_modeling ?? "n/a"));
    appendMetricCard("Indie Rows", String(classCounts.indie ?? "n/a"));
    appendMetricCard("AAA Rows", String(classCounts.aaa ?? "n/a"));
    appendMetricCard("Balanced Accuracy", formatMetricValue(metrics.balanced_accuracy, 3));
    appendMetricCard("F1 Score", formatMetricValue(metrics.f1, 3));
    appendMetricCard("ROC AUC", formatMetricValue(metrics.roc_auc, 3));
  }

  renderThesisPlots(thesisData.importanceRows, thesisData.statsRows);
  renderThesisTopTable(thesisData.statsRows);

  const createdAt = report.created_at_utc ? ` (updated: ${report.created_at_utc})` : "";
  setThesisStatus(`Loaded thesis attribute outputs${createdAt}`);
}

async function loadThesisData() {
  const report = await fetchOptionalJson([
    "data/thesis/attribute_analysis_report.json",
    "/data/thesis/attribute_analysis_report.json",
  ]);
  if (!report) return null;

  const [importanceRows, statsRows] = await Promise.all([
    fetchOptionalCsv([
      "data/thesis/attribute_feature_importance.csv",
      "/data/thesis/attribute_feature_importance.csv",
    ]),
    fetchOptionalCsv([
      "data/thesis/attribute_feature_stats.csv",
      "/data/thesis/attribute_feature_stats.csv",
    ]),
  ]);

  return {
    report,
    importanceRows: Array.isArray(importanceRows) ? importanceRows : [],
    statsRows: Array.isArray(statsRows) ? statsRows : [],
  };
}

async function refreshThesisSection(showError = false) {
  try {
    const thesisData = await loadThesisData();
    state.thesisData = thesisData;
    renderThesisSection(thesisData);
  } catch (err) {
    clearThesisViews();
    const message = formatNetworkError(err, "Loading thesis data");
    setThesisStatus(showError ? message : "Thesis data unavailable right now.", showError);
  }
}

function setPhase2Status(message, isError = false) {
  if (!els.phase2Status) return;
  els.phase2Status.textContent = message;
  els.phase2Status.classList.toggle("error", isError);
}

function clearPhase2Section() {
  if (els.phase2SummaryBody) {
    els.phase2SummaryBody.innerHTML = "";
  }
  for (const frame of [els.phase2UmapFrame, els.phase2DensmapFrame, els.phase2HistFrame]) {
    if (frame) {
      frame.removeAttribute("src");
    }
  }
  setPhase2Titles([]);
}

function normalizePhase2WebPath(value, fallbackPath) {
  const raw = String(value || "").trim();
  if (!raw) return fallbackPath;
  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("/")) return raw;
  if (raw.startsWith("web/")) return raw.slice(4);
  return raw;
}

function setPhase2Frame(frameEl, linkEl, basePath, cacheBust = "") {
  if (!frameEl || !linkEl) return;
  const bust = cacheBust ? `?v=${encodeURIComponent(cacheBust)}` : "";
  frameEl.src = `${basePath}${bust}`;
  linkEl.href = basePath;
}

function setPhase2Titles(sampleSizes) {
  const suffix = Array.isArray(sampleSizes) && sampleSizes.length > 0 ? ` (${sampleSizes.join(" / ")})` : "";
  if (els.phase2UmapTitle) {
    els.phase2UmapTitle.textContent = `UMAP Density Contours${suffix}`;
  }
  if (els.phase2DensmapTitle) {
    els.phase2DensmapTitle.textContent = `densMAP Density Contours${suffix}`;
  }
  if (els.phase2HistTitle) {
    els.phase2HistTitle.textContent = `Distance-to-Centroid Histograms${suffix}`;
  }
}

function renderPhase2SummaryRows(rows) {
  if (!els.phase2SummaryBody) return;
  els.phase2SummaryBody.innerHTML = "";
  const dataRows = Array.isArray(rows) ? rows : [];
  if (!dataRows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No phase-2 summary rows available.";
    tr.appendChild(td);
    els.phase2SummaryBody.appendChild(tr);
    return;
  }

  for (const row of dataRows) {
    const tr = document.createElement("tr");
    const values = [
      String(row.sample_size ?? "n/a"),
      formatMetricValue(row.aaa_mean, 4),
      formatMetricValue(row.indie_mean, 4),
      formatMetricValue(row.indie_vs_aaa_mean_ratio, 4),
      formatMetricValue(row.mean_gap_indie_minus_aaa, 4),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    els.phase2SummaryBody.appendChild(tr);
  }
}

async function loadPhase2Data() {
  const report = await fetchOptionalJson([
    "data/phase2_overlap/phase2_overlap_report.json",
    "/data/phase2_overlap/phase2_overlap_report.json",
  ]);
  if (!report) return null;
  const summaryRowsFromCsv = await fetchOptionalCsv([
    "data/phase2_overlap/distance_to_centroid_summary.csv",
    "/data/phase2_overlap/distance_to_centroid_summary.csv",
  ]);
  const summaryRows =
    Array.isArray(summaryRowsFromCsv) && summaryRowsFromCsv.length > 0
      ? summaryRowsFromCsv
      : Array.isArray(report.summary_rows)
        ? report.summary_rows
        : [];

  return {
    report,
    summaryRows,
  };
}

function renderPhase2Section(phase2Data) {
  if (!phase2Data || !phase2Data.report) {
    clearPhase2Section();
    setPhase2Status(
      "Phase-2 outputs not found yet. Run: python src/phase2_overlap_density_analysis.py ...",
      false
    );
    return;
  }

  const report = phase2Data.report;
  renderPhase2SummaryRows(phase2Data.summaryRows);

  const outputs = report.outputs || {};
  const cacheToken = report.created_at_utc || String(Date.now());

  setPhase2Frame(
    els.phase2UmapFrame,
    els.phase2UmapLink,
    normalizePhase2WebPath(
      outputs.umap_contour_html,
      "data/phase2_overlap/umap_density_contours_500_1000_2000.html"
    ),
    cacheToken
  );
  setPhase2Frame(
    els.phase2DensmapFrame,
    els.phase2DensmapLink,
    normalizePhase2WebPath(
      outputs.densmap_contour_html,
      "data/phase2_overlap/densmap_density_contours_500_1000_2000.html"
    ),
    cacheToken
  );
  setPhase2Frame(
    els.phase2HistFrame,
    els.phase2HistLink,
    normalizePhase2WebPath(
      outputs.distance_hist_html,
      "data/phase2_overlap/distance_to_centroid_hist_500_1000_2000.html"
    ),
    cacheToken
  );

  const sampleSizesArray = Array.isArray(report.sample_sizes) ? report.sample_sizes : [];
  setPhase2Titles(sampleSizesArray);
  const sampleSizes = sampleSizesArray.length > 0 ? sampleSizesArray.join(", ") : "n/a";
  setPhase2Status(`Loaded phase-2 overlap outputs for sample sizes: ${sampleSizes}.`);
}

async function refreshPhase2Section(showError = false) {
  try {
    const phase2Data = await loadPhase2Data();
    renderPhase2Section(phase2Data);
  } catch (err) {
    clearPhase2Section();
    const message = formatNetworkError(err, "Loading phase-2 outputs");
    setPhase2Status(showError ? message : "Phase-2 outputs unavailable right now.", showError);
  }
}

function setPhase3Status(message, isError = false) {
  if (!els.phase3Status) return;
  els.phase3Status.textContent = message;
  els.phase3Status.classList.toggle("error", isError);
}

function clearPhase3Section() {
  if (els.phase3Metrics) {
    els.phase3Metrics.innerHTML = "";
  }
  if (els.phase3LevelsBody) {
    els.phase3LevelsBody.innerHTML = "";
  }
  for (const frame of [
    els.phase3PcaFrame,
    els.phase3KdeFrame,
    els.phase3ResidualFrame,
    els.phase3CosineFrame,
    els.phase3PromptFrame,
  ]) {
    if (frame) {
      frame.removeAttribute("src");
    }
  }
}

function normalizePhase3WebPath(value, fallbackPath) {
  const raw = String(value || "").trim();
  if (!raw) return fallbackPath;
  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("/data/")) return raw;
  if (raw.startsWith("data/")) return raw;
  if (raw.startsWith("web/")) return raw.slice(4);
  const webMarker = "/web/";
  const idx = raw.lastIndexOf(webMarker);
  if (idx !== -1) return raw.slice(idx + webMarker.length);
  return fallbackPath;
}

function setPhase3Frame(frameEl, linkEl, basePath, cacheBust = "") {
  if (!frameEl || !linkEl || !basePath) return;
  const bust = cacheBust ? `?v=${encodeURIComponent(cacheBust)}` : "";
  frameEl.src = `${basePath}${bust}`;
  linkEl.href = basePath;
}

function renderPhase3LevelRows(rows) {
  if (!els.phase3LevelsBody) return;
  els.phase3LevelsBody.innerHTML = "";
  const dataRows = Array.isArray(rows) ? rows : [];
  if (!dataRows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "No phase-3 PCA-level rows available.";
    tr.appendChild(td);
    els.phase3LevelsBody.appendChild(tr);
    return;
  }

  for (const row of dataRows) {
    const tr = document.createElement("tr");
    const values = [
      String(row.level ?? "n/a"),
      String(row.n_components ?? "n/a"),
      formatMetricValue(row.logreg_auc, 4),
      formatMetricValue(row.ari_mean, 4),
      formatMetricValue(row.ari_std, 4),
      formatMetricValue(row.cumulative_explained_variance, 4),
      formatMetricValue(row.ovl_predicted_probability_by_group, 4),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    els.phase3LevelsBody.appendChild(tr);
  }
}

function renderPhase3Metrics(report) {
  if (!els.phase3Metrics) return;
  els.phase3Metrics.innerHTML = "";
  if (!report || typeof report !== "object") return;

  const centroid = report.centroid_distance_summary || {};
  const cosine = report.cosine_similarity_summary || {};
  const prompt = report.prompt_style_summary || {};

  appendMetricCardTo(els.phase3Metrics, "Sample Size", String(report.sample_size ?? "n/a"));
  appendMetricCardTo(els.phase3Metrics, "Best Level by AUC", String(report.best_level_by_auc ?? "n/a"));
  appendMetricCardTo(
    els.phase3Metrics,
    "Centroid OVL",
    formatMetricValue(centroid.ovl_centroid_distance, 4)
  );
  appendMetricCardTo(
    els.phase3Metrics,
    "Indie/AAA Mean Dist Ratio",
    formatMetricValue(centroid.indie_vs_aaa_ratio, 4)
  );
  appendMetricCardTo(
    els.phase3Metrics,
    "Mean Gap (Indie - AAA)",
    formatMetricValue(centroid.mean_gap_indie_minus_aaa, 4)
  );
  appendMetricCardTo(
    els.phase3Metrics,
    "Cosine OVL (AAA vs Indie Within)",
    formatMetricValue(cosine.ovl_aaa_vs_indie_within, 4)
  );
  appendMetricCardTo(
    els.phase3Metrics,
    "Prompt Styles",
    `${String(prompt.style_a_name || "style_a")} vs ${String(prompt.style_b_name || "style_b")}`
  );
  appendMetricCardTo(
    els.phase3Metrics,
    "Prompt Delta Mean OVL",
    formatMetricValue(prompt.delta_mean_styleA_minus_styleB?.ovl_aaa_vs_indie, 4)
  );
}

async function loadPhase3Data() {
  const report = await fetchOptionalJson([
    "data/phase3_advanced/phase3_advanced_report.json",
    "/data/phase3_advanced/phase3_advanced_report.json",
  ]);
  if (!report) return null;
  const pcaRows = await fetchOptionalCsv([
    "data/phase3_advanced/pca_level_metrics.csv",
    "/data/phase3_advanced/pca_level_metrics.csv",
  ]);
  return {
    report,
    pcaRows: Array.isArray(pcaRows) ? pcaRows : [],
  };
}

function renderPhase3Section(phase3Data) {
  if (!phase3Data || !phase3Data.report) {
    clearPhase3Section();
    state.phase3LastData = null;
    setPhase3Status(
      "Phase-3 outputs not found yet. Run Phase 3 from this page or run src/phase3_advanced_separability_analysis.py.",
      false
    );
    return;
  }

  const report = phase3Data.report;
  state.phase3LastData = phase3Data;
  renderPhase3Metrics(report);
  renderPhase3LevelRows(phase3Data.pcaRows);

  const outputs = report.outputs || {};
  const cacheToken = report.created_at_utc || String(Date.now());

  setPhase3Frame(
    els.phase3PcaFrame,
    els.phase3PcaLink,
    normalizePhase3WebPath(outputs.pca_metrics_html, "data/phase3_advanced/pca_level_metrics.html"),
    cacheToken
  );
  setPhase3Frame(
    els.phase3KdeFrame,
    els.phase3KdeLink,
    normalizePhase3WebPath(outputs.kde_heatmap_html, "data/phase3_advanced/kde_heatmap_pca2.html"),
    cacheToken
  );
  setPhase3Frame(
    els.phase3ResidualFrame,
    els.phase3ResidualLink,
    normalizePhase3WebPath(outputs.residual_html, "data/phase3_advanced/residual_analysis.html"),
    cacheToken
  );
  setPhase3Frame(
    els.phase3CosineFrame,
    els.phase3CosineLink,
    normalizePhase3WebPath(outputs.cosine_html, "data/phase3_advanced/cosine_similarity_distributions.html"),
    cacheToken
  );
  setPhase3Frame(
    els.phase3PromptFrame,
    els.phase3PromptLink,
    normalizePhase3WebPath(outputs.prompt_html, "data/phase3_advanced/prompt_style_similarity_distributions.html"),
    cacheToken
  );

  const updated = report.created_at_utc ? ` (updated: ${report.created_at_utc})` : "";
  setPhase3Status(
    `Loaded phase-3 outputs for sample size ${report.sample_size ?? "n/a"}; best level by AUC: ${report.best_level_by_auc ?? "n/a"}${updated}.`
  );
}

async function refreshPhase3Section(showError = false) {
  try {
    const phase3Data = await loadPhase3Data();
    renderPhase3Section(phase3Data);
  } catch (err) {
    clearPhase3Section();
    state.phase3LastData = null;
    const message = formatNetworkError(err, "Loading phase-3 outputs");
    setPhase3Status(showError ? message : "Phase-3 outputs unavailable right now.", showError);
  }
}

function getPhase3RunPayloadFromState() {
  return {
    dataset_mode: normalizeDatasetMode(state.datasetMode),
    sample_size: 0,
    batch_size: 32,
    device: "auto",
    clip_backend: "open_clip",
    model_name: "ViT-B/32",
    pca_levels: "2,5,10,25,50,100,200",
    ari_seeds: "42,43,44",
    max_pairs_per_bucket: 200000,
  };
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

async function fetchPhase3BackendStatus() {
  const response = await fetch("/api/phase3-status", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Phase-3 status request failed: HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchIgdbBackendStatus() {
  const response = await fetch("/api/igdb-status", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`IGDB status request failed: HTTP ${response.status}`);
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

async function runPhase3FromBackend(runPayload) {
  const response = await fetch("/api/run-phase3", {
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
    throw new Error(payload.message || `Phase-3 run request failed: HTTP ${response.status}`);
  }
  return payload;
}

async function runIgdbFetchFromBackend(runPayload) {
  const response = await fetch("/api/fetch-igdb-covers", {
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
    throw new Error(payload.message || `IGDB fetch request failed: HTTP ${response.status}`);
  }
  return payload;
}

async function searchIgdbGamesFromBackend(query, limit = 100, company = "") {
  const trimmedQuery = String(query || "").trim();
  const trimmedCompany = String(company || "").trim();
  const params = new URLSearchParams();
  if (trimmedQuery) params.set("q", trimmedQuery);
  if (trimmedCompany) params.set("company", trimmedCompany);
  if (![...params.keys()].length) {
    throw new Error("Provide a game title or company to search.");
  }
  const parsedLimit = Number.parseInt(String(limit), 10);
  const boundedLimit = Number.isFinite(parsedLimit) ? clamp(parsedLimit, 1, 500) : 100;
  const encodedLimit = encodeURIComponent(String(boundedLimit));
  const response = await fetch(`/api/igdb-search-games?${params.toString()}&limit=${encodedLimit}`, {
    cache: "no-store",
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = { ok: false, message: "Invalid server response." };
  }
  if (!response.ok || !payload.ok) {
    throw new Error(payload.message || `IGDB search failed: HTTP ${response.status}`);
  }
  return payload.data || { results: [] };
}

async function fetchGameFoldersFromBackend() {
  const response = await fetch("/api/game-folders", { cache: "no-store" });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = { ok: false, folders: [] };
  }
  if (!response.ok || !payload.ok) {
    throw new Error(payload.message || `Folder list request failed: HTTP ${response.status}`);
  }
  return Array.isArray(payload.folders) ? payload.folders : [];
}

function stopStatusPolling() {
  if (state.statusPollTimerId !== null) {
    clearInterval(state.statusPollTimerId);
    state.statusPollTimerId = null;
  }
}

function stopPhase3StatusPolling() {
  if (state.phase3StatusPollTimerId !== null) {
    clearInterval(state.phase3StatusPollTimerId);
    state.phase3StatusPollTimerId = null;
  }
}

function stopIgdbStatusPolling() {
  if (state.igdbStatusPollTimerId !== null) {
    clearInterval(state.igdbStatusPollTimerId);
    state.igdbStatusPollTimerId = null;
  }
}

function setRunButtonBusy(isBusy) {
  if (!els.runAnalysisBtn) return;
  els.runAnalysisBtn.disabled = isBusy;
  els.runAnalysisBtn.textContent = isBusy ? "Running..." : "Run Analysis";
}

function setPhase3ButtonBusy(isBusy) {
  if (!els.runPhase3Btn) return;
  els.runPhase3Btn.disabled = isBusy || !state.backendAvailable;
  els.runPhase3Btn.textContent = isBusy ? "Running Phase 3..." : "Run Phase 3";
  if (els.loadPhase3Btn) {
    els.loadPhase3Btn.disabled = isBusy;
  }
}

function setIgdbButtonsBusy(isBusy) {
  if (els.igdbDryRunBtn) {
    els.igdbDryRunBtn.disabled = isBusy || !state.igdbAvailable;
    els.igdbDryRunBtn.textContent = isBusy ? "IGDB Dry Run..." : "IGDB Dry Run";
  }
  if (els.igdbFetchBtn) {
    els.igdbFetchBtn.disabled = isBusy || !state.igdbAvailable;
    els.igdbFetchBtn.textContent = isBusy ? "Fetching Covers..." : "Fetch IGDB Covers";
  }
  if (els.igdbSeedIndieBtn) {
    els.igdbSeedIndieBtn.disabled = isBusy || !state.igdbAvailable;
    els.igdbSeedIndieBtn.textContent = isBusy ? "Adding Indie Games..." : "Add Indie Games";
  }
  if (els.igdbSeedAaaBtn) {
    els.igdbSeedAaaBtn.disabled = isBusy || !state.igdbAvailable;
    els.igdbSeedAaaBtn.textContent = isBusy ? "Adding AAA Games..." : "Add AAA 2010-2020";
  }
  if (els.igdbSearchBtn) {
    els.igdbSearchBtn.disabled = isBusy || !state.igdbAvailable;
  }
  if (els.igdbSelectAllBtn) {
    const hasResults = Boolean(els.igdbSearchResults && els.igdbSearchResults.options.length > 0);
    els.igdbSelectAllBtn.disabled = isBusy || !state.igdbAvailable || !hasResults;
  }
  if (els.igdbAddSelectedBtn) {
    const hasSelection = getSelectedIgdbIds().length > 0;
    els.igdbAddSelectedBtn.disabled = isBusy || !state.igdbAvailable || !hasSelection;
  }
  if (els.igdbAddListBtn) {
    const hasList = Boolean(
      els.igdbBulkListInput &&
      parseGameNameList(els.igdbBulkListInput.value).length > 0
    );
    els.igdbAddListBtn.disabled = isBusy || !state.igdbAvailable || !hasList;
    els.igdbAddListBtn.textContent = isBusy ? "Adding Game List..." : "Add Full List";
  }
  if (els.igdbImportPdfBtn) {
    const hasPdf =
      Boolean(els.igdbPdfInput) &&
      Boolean(els.igdbPdfInput.files) &&
      els.igdbPdfInput.files.length > 0;
    els.igdbImportPdfBtn.disabled = isBusy || !state.igdbAvailable || !hasPdf;
    els.igdbImportPdfBtn.textContent = isBusy ? "Importing PDF..." : "Import PDF";
  }
}

function renderIgdbSearchResults(results) {
  state.igdbSearchResults = Array.isArray(results) ? results : [];
  if (!els.igdbSearchResults) return;
  els.igdbSearchResults.innerHTML = "";
  const hideExisting = Boolean(els.igdbHideExistingCheck && els.igdbHideExistingCheck.checked);
  for (const row of state.igdbSearchResults) {
    if (!row || !row.id || !row.name) continue;
    const alreadyAdded = isGameAlreadyAdded(row.name);
    if (hideExisting && alreadyAdded) continue;
    const yearPart = row.release_year ? ` (${row.release_year})` : "";
    const votePart = row.total_rating_count ? ` | votes ${row.total_rating_count}` : "";
    const existingPart = alreadyAdded ? " | already added" : "";
    const option = document.createElement("option");
    option.value = String(row.id);
    option.textContent = `${row.name}${yearPart}${votePart}${existingPart} [${row.id}]`;
    els.igdbSearchResults.appendChild(option);
  }
  if (els.igdbSearchResults.options.length > 0) {
    els.igdbSearchResults.options[0].selected = true;
  }
  if (els.igdbSelectAllBtn) {
    els.igdbSelectAllBtn.disabled =
      !state.igdbAvailable || els.igdbSearchResults.options.length === 0;
  }
  if (els.igdbAddSelectedBtn) {
    els.igdbAddSelectedBtn.disabled =
      !state.igdbAvailable || els.igdbSearchResults.options.length === 0;
  }
}

function getSelectedIgdbIds() {
  if (!els.igdbSearchResults) return [];
  const selected = Array.from(els.igdbSearchResults.selectedOptions || []);
  const ids = [];
  for (const option of selected) {
    const parsed = Number.parseInt(String(option.value || ""), 10);
    if (Number.isFinite(parsed) && parsed > 0 && !ids.includes(parsed)) {
      ids.push(parsed);
    }
  }
  return ids;
}

function parseGameNameList(rawText) {
  const text = String(rawText || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const tokens =
    text.includes("\n") ? text.split("\n") : text.includes(";") ? text.split(";") : text.split(",");
  const names = [];
  for (const token of tokens) {
    const value = String(token || "").trim();
    if (!value) continue;
    if (value.startsWith("#")) continue;
    if (!names.includes(value)) names.push(value);
  }
  return names;
}

function normalizeGameKey(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function rebuildExistingFolderKeys(folders) {
  const keys = new Set();
  for (const folder of Array.isArray(folders) ? folders : []) {
    const key = normalizeGameKey(folder);
    if (key) keys.add(key);
  }
  state.existingFolderKeys = keys;
}

function isGameAlreadyAdded(gameName) {
  const key = normalizeGameKey(gameName);
  return Boolean(key) && state.existingFolderKeys.has(key);
}

async function refreshExistingFolderKeys() {
  try {
    const folders = await fetchGameFoldersFromBackend();
    rebuildExistingFolderKeys(folders);
  } catch (_) {
    // Non-fatal: skip duplicate pre-check if endpoint is unavailable.
    state.existingFolderKeys = new Set();
  }
}

async function extractGameNamesFromPdfFile(file) {
  if (!file) return [];
  const pdfjsLib = window.pdfjsLib;
  if (!pdfjsLib || typeof pdfjsLib.getDocument !== "function") {
    throw new Error("PDF library failed to load. Refresh page and try again.");
  }
  if (pdfjsLib.GlobalWorkerOptions) {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js";
  }
  const data = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;
  const collectedLines = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum += 1) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    let pageText = "";
    let lastY = null;
    for (const item of content.items || []) {
      const str = item && typeof item.str === "string" ? item.str : "";
      if (!str.trim()) continue;
      const y = item && item.transform && Number.isFinite(item.transform[5]) ? item.transform[5] : null;
      if (pageText.length > 0) {
        if (lastY !== null && y !== null && Math.abs(y - lastY) > 4) {
          pageText += "\n";
        } else {
          pageText += " ";
        }
      }
      pageText += str.trim();
      lastY = y;
    }
    if (pageText.trim()) {
      collectedLines.push(pageText);
    }
  }
  const rawNames = parseGameNameList(collectedLines.join("\n"));
  const cleaned = rawNames
    .map((name) => name.replace(/^\s*(?:[-*•]|\d+[\.\)\-])\s*/, "").trim())
    .filter((name) => name.length >= 2 && name.length <= 120)
    .filter((name) => /[A-Za-z]/.test(name));
  return uniqueSorted(cleaned);
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

function normalizeDatasetMode(value) {
  const mode = String(value || "").trim().toLowerCase();
  return mode === "demo" ? "demo" : "full";
}

function renderDatasetModeUi(statusPayload = null) {
  const modes = statusPayload && statusPayload.dataset_modes ? statusPayload.dataset_modes : null;
  const hasAvailabilityInfo = Boolean(modes) && typeof modes.demo_available === "boolean";
  const demoAvailable = hasAvailabilityInfo ? Boolean(modes.demo_available) : true;
  if (els.datasetModeSelect) {
    const demoOption = Array.from(els.datasetModeSelect.options).find((opt) => opt.value === "demo");
    if (demoOption) {
      // Only disable when backend explicitly reports availability=false.
      // If availability info is missing (older server / stale response), keep option selectable.
      demoOption.disabled = hasAvailabilityInfo ? !demoAvailable : false;
    }
    els.datasetModeSelect.value = state.datasetMode;
  }
  if (els.datasetModeHint) {
    if (!modes) {
      els.datasetModeHint.textContent = "Choose dataset for next analysis run (availability unknown).";
      return;
    }
    if (state.datasetMode === "demo") {
      els.datasetModeHint.textContent = demoAvailable
        ? "Using demo dataset (100 AAA + 100 Indie)."
        : "Demo dataset not found on server. Create it first or switch to full dataset.";
      return;
    }
    els.datasetModeHint.textContent = "Using full dataset.";
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
    dataset_mode: normalizeDatasetMode(state.datasetMode),
  };
}

function getIgdbPayload(dryRun) {
  return {
    dry_run: Boolean(dryRun),
    image_size: "cover_big",
    max_games: 0,
    min_match_score: 0.72,
    min_token_overlap: 0.34,
    strict_match_mode: true,
    auto_clean_local_names: true,
    allow_low_confidence: false,
    skip_if_any_image: true,
    overwrite: false,
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
    renderDatasetModeUi(status);
    const runningMode = normalizeDatasetMode(status?.last_params?.dataset_mode);
    const tailLine = formatBackendTail(status);

    if (status.running) {
      const text = tailLine
        ? `Analysis running (${runningMode})... ${tailLine}`
        : `Analysis running (${runningMode})... processing screenshots and computing embeddings.`;
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

async function pollPhase3Status() {
  try {
    const status = await fetchPhase3BackendStatus();
    const runningMode = normalizeDatasetMode(status?.last_params?.dataset_mode);
    const tailLine = formatBackendTail(status);

    if (status.running) {
      const text = tailLine
        ? `Phase-3 running (${runningMode})... ${tailLine}`
        : `Phase-3 running (${runningMode})... computing advanced separability outputs.`;
      setStatus(text);
      setPhase3Status(text);
      setPhase3ButtonBusy(true);
      return;
    }

    stopPhase3StatusPolling();
    setPhase3ButtonBusy(false);

    if (typeof status.last_exit_code === "number" && status.last_exit_code === 0) {
      setPhase3Status("Phase-3 finished. Reloading phase-3 visualizations...");
      await refreshPhase3Section(true);
      setStatus("Phase-3 analysis finished and section refreshed.");
      return;
    }

    if (typeof status.last_exit_code === "number") {
      const errorText = tailLine || status.last_error || "Unknown phase-3 pipeline error.";
      const msg = `Phase-3 failed (exit ${status.last_exit_code}): ${errorText}`;
      setPhase3Status(msg, true);
      setStatus(msg, true);
      return;
    }
  } catch (err) {
    stopPhase3StatusPolling();
    setPhase3ButtonBusy(false);
    const message = formatNetworkError(err, "Phase-3 status polling");
    setPhase3Status(message, true);
    setStatus(message, true);
  }
}

async function pollIgdbStatus() {
  try {
    const status = await fetchIgdbBackendStatus();
    const tailLine = formatBackendTail(status);

    if (status.running) {
      const text = tailLine
        ? `IGDB fetch running... ${tailLine}`
        : "IGDB fetch running... matching games and downloading covers.";
      setStatus(text);
      setIgdbButtonsBusy(true);
      return;
    }

    stopIgdbStatusPolling();
    setIgdbButtonsBusy(false);

    if (typeof status.last_exit_code === "number" && status.last_exit_code === 0) {
      await refreshExistingFolderKeys();
      const wasDryRun = Boolean(status.last_params && status.last_params.dry_run);
      if (wasDryRun) {
        setStatus("IGDB dry run finished. Check web/data/igdb_cover_fetch_report.json.");
      } else {
        setStatus(
          "IGDB cover fetch finished. Re-run analysis to include new covers in embeddings and visualizations."
        );
      }
      return;
    }

    if (typeof status.last_exit_code === "number") {
      const errorText = tailLine || status.last_error || "Unknown IGDB fetch error.";
      setStatus(`IGDB fetch failed (exit ${status.last_exit_code}): ${errorText}`, true);
      return;
    }
  } catch (err) {
    stopIgdbStatusPolling();
    setIgdbButtonsBusy(false);
    setStatus(formatNetworkError(err, "IGDB status polling"), true);
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

function countUniqueGamesByGroup(samples) {
  const groups = new Map();
  for (const sample of Array.isArray(samples) ? samples : []) {
    const label = typeof sample?.label === "string" ? sample.label.trim() : "";
    if (!label) continue;
    const groupKey = normalizeGroupName(sample?.group);
    if (!groups.has(groupKey)) {
      groups.set(groupKey, new Set());
    }
    groups.get(groupKey).add(label);
  }
  return groups;
}

function updateGroupCounters(allSamples, visibleSamples) {
  const totalByGroup = countUniqueGamesByGroup(allSamples);
  const shownByGroup = countUniqueGamesByGroup(visibleSamples);

  const indieTotal = totalByGroup.get("indie")?.size || 0;
  const indieShown = shownByGroup.get("indie")?.size || 0;
  const aaaTotal = totalByGroup.get("aaa")?.size || 0;
  const aaaShown = shownByGroup.get("aaa")?.size || 0;

  if (els.indieCounter) {
    els.indieCounter.textContent = `Indie: ${indieTotal} (${indieShown} shown)`;
  }
  if (els.aaaCounter) {
    els.aaaCounter.textContent = `AAA: ${aaaTotal} (${aaaShown} shown)`;
  }
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

function applyVizSampleLimit(samples) {
  const mode = normalizeVizSampleSize(state.vizSampleSize);
  if (mode === "all") {
    return samples;
  }

  const targetTotal = Number.parseInt(mode, 10);
  if (!Number.isFinite(targetTotal) || targetTotal <= 0) {
    return samples;
  }
  const targetPerGroup = Math.floor(targetTotal / 2);
  if (targetPerGroup <= 0) {
    return samples;
  }

  const indieRows = [];
  const aaaRows = [];
  for (const sample of samples) {
    const group = normalizeGroupName(sample.group);
    if (group === "indie") {
      indieRows.push(sample);
    } else if (group === "aaa") {
      aaaRows.push(sample);
    }
  }
  if (indieRows.length === 0 || aaaRows.length === 0) {
    return samples;
  }

  const stableSortById = (a, b) => {
    const ai = Number.isFinite(Number(a?.image_id)) ? Number(a.image_id) : Number.MAX_SAFE_INTEGER;
    const bi = Number.isFinite(Number(b?.image_id)) ? Number(b.image_id) : Number.MAX_SAFE_INTEGER;
    if (ai !== bi) return ai - bi;
    return String(a?.label || "").localeCompare(String(b?.label || ""));
  };
  indieRows.sort(stableSortById);
  aaaRows.sort(stableSortById);

  const k = Math.min(targetPerGroup, indieRows.length, aaaRows.length);
  if (k <= 0) return samples;

  const subset = [...indieRows.slice(0, k), ...aaaRows.slice(0, k)];
  subset.sort(stableSortById);
  return subset;
}

function render3DScatter(container, samples, vectorKey, title, colorMap, clusterKey = "cluster_id", clusterLabel = "Cluster") {
  if (!container || typeof window.Plotly === "undefined") return;
  const activeFocus = normalizeGroupFocus(state.groupFocus);
  const grouped = new Map();
  for (const sample of samples) {
    const vector = sample && sample[vectorKey];
    if (!Array.isArray(vector) || vector.length !== 3) continue;
    const groupKey = normalizeGroupName(sample.group);
    if (!grouped.has(groupKey)) grouped.set(groupKey, []);
    grouped.get(groupKey).push(sample);
  }
  if (grouped.size === 0) {
    Plotly.purge(container);
    return;
  }

  const traces = [];
  const orderedGroups = ["indie", "aaa", "unassigned"];
  const sortedKeys = [...grouped.keys()].sort((a, b) => {
    const ia = orderedGroups.indexOf(a);
    const ib = orderedGroups.indexOf(b);
    if (ia !== -1 && ib !== -1) return ia - ib;
    if (ia !== -1) return -1;
    if (ib !== -1) return 1;
    return a.localeCompare(b);
  });
  if (activeFocus !== "both") {
    sortedKeys.sort((a, b) => {
      const aw = a === activeFocus ? 1 : 0;
      const bw = b === activeFocus ? 1 : 0;
      return aw - bw;
    });
  }

  for (const groupKey of sortedKeys) {
    const points = grouped.get(groupKey) || [];
    const groupLabel = formatGroupLabel(groupKey);
    const markerColor = getGroupColor(groupKey);
    const isFocused = activeFocus === "both" || groupKey === activeFocus;
    const markerOpacity = isFocused ? 0.9 : 0.14;
    const markerSize = isFocused ? 5.8 : 5.0;
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: groupLabel,
      x: points.map((p) => p[vectorKey][0]),
      y: points.map((p) => p[vectorKey][1]),
      z: points.map((p) => p[vectorKey][2]),
      customdata: points.map((p) => [
        p.label ?? "N/A",
        formatGroupLabel(p.group),
        p[clusterKey] ?? "N/A",
        p.outlier_flag ? "yes" : "no",
        Number.isFinite(Number(p.outlier_score)) ? Number(p.outlier_score) : null,
      ]),
      hovertemplate:
        "<b>Game: %{customdata[0]}</b><br>" +
        "Group: %{customdata[1]}<br>" +
        `${vectorKey.toUpperCase()}: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>` +
        `${clusterLabel}: %{customdata[2]}<br>` +
        "Outlier Flag: %{customdata[3]}<br>" +
        "Outlier Score: %{customdata[4]:.4f}<extra></extra>",
      marker: {
        size: markerSize,
        opacity: markerOpacity,
        color: markerColor,
        line: {
          color: "#0f172a",
          width: 0.45,
        },
      },
    });
  }

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
    showlegend: true,
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

function purgeSwissRollPlots() {
  if (typeof window.Plotly === "undefined") return;
  for (const el of [els.swissRollOriginalPlot, els.swissRollLlePlot, els.swissRollPca2dPlot]) {
    if (el) Plotly.purge(el);
  }
}

function renderSwissRollScatter(container, samples, vectorKey, title) {
  if (!container || typeof window.Plotly === "undefined") return;
  if (!Array.isArray(samples) || samples.length === 0) {
    Plotly.purge(container);
    return;
  }

  const raw = samples
    .map((sample) => ({
      v: sample?.[vectorKey],
      c: Number(sample?.color_value),
    }))
    .filter((p) => Array.isArray(p.v) && (p.v.length === 2 || p.v.length === 3) && Number.isFinite(p.c));
  if (raw.length === 0) {
    Plotly.purge(container);
    return;
  }
  const dim = raw[0].v.length;
  const points = raw
    .map((p) => ({
      x: Number(p.v[0]),
      y: Number(p.v[1]),
      z: dim === 3 ? Number(p.v[2]) : null,
      c: p.c,
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (dim === 2 || Number.isFinite(p.z)));

  if (points.length === 0) {
    Plotly.purge(container);
    return;
  }

  const commonMarker = {
    size: dim === 3 ? 2.8 : 5.0,
    opacity: 0.82,
    color: points.map((p) => p.c),
    colorscale: "Turbo",
    colorbar: { title: "roll t" },
  };

  if (dim === 3) {
    Plotly.react(
      container,
      [
        {
          type: "scatter3d",
          mode: "markers",
          x: points.map((p) => p.x),
          y: points.map((p) => p.y),
          z: points.map((p) => p.z),
          marker: commonMarker,
          hovertemplate: `${vectorKey.toUpperCase()}: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>`,
        },
      ],
      {
        title,
        margin: { l: 0, r: 0, t: 42, b: 0 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        scene: {
          xaxis: { title: `${vectorKey.toUpperCase()}-1` },
          yaxis: { title: `${vectorKey.toUpperCase()}-2` },
          zaxis: { title: `${vectorKey.toUpperCase()}-3` },
          aspectmode: "data",
        },
        showlegend: false,
      },
      { responsive: true, displaylogo: false }
    );
    return;
  }

  Plotly.react(
    container,
    [
      {
        type: "scattergl",
        mode: "markers",
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        marker: commonMarker,
        hovertemplate: `${vectorKey.toUpperCase()}: (%{x:.2f}, %{y:.2f})<extra></extra>`,
      },
    ],
    {
      title,
      margin: { l: 42, r: 20, t: 42, b: 42 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: `${vectorKey.toUpperCase()}-1` },
      yaxis: { title: `${vectorKey.toUpperCase()}-2` },
      showlegend: false,
    },
    { responsive: true, displaylogo: false }
  );
}

function renderSwissRollSection(swissRollDemo) {
  const isEnabled = Boolean(swissRollDemo && swissRollDemo.enabled);
  const samples = isEnabled && Array.isArray(swissRollDemo.samples) ? swissRollDemo.samples : [];
  if (!isEnabled || samples.length === 0) {
    purgeSwissRollPlots();
    return;
  }

  const lleMeta = swissRollDemo?.meta?.lle || {};
  const pcaMeta = swissRollDemo?.meta?.pca || {};
  const lleErr =
    Number.isFinite(Number(lleMeta.reconstruction_error))
      ? Number(lleMeta.reconstruction_error).toFixed(4)
      : "n/a";
  const pcaErr =
    Number.isFinite(Number(pcaMeta.error))
      ? Number(pcaMeta.error).toFixed(4)
      : "n/a";

  renderSwissRollScatter(els.swissRollOriginalPlot, samples, "original", "Swiss Roll Original");
  renderSwissRollScatter(
    els.swissRollLlePlot,
    samples,
    "lle_2d",
    `Swiss Roll + LLE (2D) | reconstruction error=${lleErr}`
  );
  renderSwissRollScatter(
    els.swissRollPca2dPlot,
    samples,
    "pca_2d",
    `Swiss Roll + PCA (2D) | error=${pcaErr}`
  );
}

function subsetSymmetricMatrix(labels, matrix, subsetLabels) {
  const indexMap = new Map(labels.map((label, idx) => [label, idx]));
  return subsetLabels.map((rowLabel) => {
    const rowIdx = indexMap.get(rowLabel);
    return subsetLabels.map((colLabel) => {
      const colIdx = indexMap.get(colLabel);
      if (rowIdx == null || colIdx == null) return 0;
      const row = Array.isArray(matrix[rowIdx]) ? matrix[rowIdx] : [];
      const value = row[colIdx];
      return Number.isFinite(value) ? Number(value) : 0;
    });
  });
}

function pickReadableCentroidGames(labels, samples, maxPerGroup = 5) {
  const groupByGame = new Map();
  for (const sample of samples) {
    if (!groupByGame.has(sample.label)) {
      groupByGame.set(sample.label, normalizeGroupName(sample.group));
    }
  }

  const indie = [];
  const aaa = [];
  const other = [];

  for (const label of labels) {
    const group = groupByGame.get(label) || "unassigned";
    if (group === "indie") {
      if (indie.length < maxPerGroup) indie.push(label);
      continue;
    }
    if (group === "aaa") {
      if (aaa.length < maxPerGroup) aaa.push(label);
      continue;
    }
    other.push(label);
  }

  const chosen = [...indie, ...aaa];
  const target = Math.min(labels.length, maxPerGroup * 2);
  if (chosen.length < target) {
    for (const label of other) {
      if (chosen.length >= target) break;
      if (!chosen.includes(label)) chosen.push(label);
    }
  }
  if (chosen.length < target) {
    for (const label of labels) {
      if (chosen.length >= target) break;
      if (!chosen.includes(label)) chosen.push(label);
    }
  }
  return chosen;
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

function formatMetricNumber(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "n/a";
  return n.toFixed(digits);
}

function renderQualityTable(quality) {
  if (!els.qualityTableBody) return;
  els.qualityTableBody.innerHTML = "";

  const methods = quality && quality.methods && typeof quality.methods === "object" ? quality.methods : {};
  const order = ["pca", "umap", "tsne"];
  const rows = order.filter((name) => methods[name]).concat(Object.keys(methods).filter((name) => !order.includes(name)));

  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No projection quality metrics available.";
    tr.appendChild(td);
    els.qualityTableBody.appendChild(tr);
    return;
  }

  for (const methodName of rows) {
    const row = methods[methodName] || {};
    const tr = document.createElement("tr");
    const values = [
      methodName.toUpperCase(),
      formatMetricNumber(row.trustworthiness),
      formatMetricNumber(row.continuity_proxy),
      formatMetricNumber(row.knn_overlap),
      String(row.status || "n/a"),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    els.qualityTableBody.appendChild(tr);
  }
}

function renderOutlierTable(outliersPayload) {
  if (!els.outlierTableBody) return;
  els.outlierTableBody.innerHTML = "";

  const rows =
    outliersPayload &&
    Array.isArray(outliersPayload.rows)
      ? outliersPayload.rows
      : [];

  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 8;
    td.textContent = "No outlier rows available.";
    tr.appendChild(td);
    els.outlierTableBody.appendChild(tr);
    return;
  }

  const maxRows = 120;
  for (const rowData of rows.slice(0, maxRows)) {
    const tr = document.createElement("tr");
    const values = [
      rowData.rank ?? "",
      rowData.image_id ?? "",
      rowData.label ?? "",
      rowData.group ?? "",
      formatMetricNumber(rowData.outlier_score, 5),
      formatMetricNumber(rowData.outlier_percentile, 4),
      rowData.dbscan_noise ? "yes" : "no",
      rowData.flagged ? "yes" : "no",
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.appendChild(td);
    }
    els.outlierTableBody.appendChild(tr);
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
    state.latestRadiusProjection = [];
    if (els.radiusHover) {
      els.radiusHover.textContent = "Need at least 2 samples to render neighborhood explorer.";
    }
    ctx.fillStyle = "#475569";
    ctx.font = "16px 'Space Grotesk', sans-serif";
    ctx.fillText("Need at least 2 samples to render neighborhood explorer.", 20, 36);
    return;
  }

  const key = state.radiusParams.useTsne ? "tsne" : "umap";
  const points2D = get2DPoints(samples, key);
  const projected = projectPoints(points2D, width, height, 28);
  state.latestRadiusProjection = projected.map((p) => ({ px: p.px, py: p.py, label: p.label }));
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
    ctx.fillStyle = getGroupColor(p.group);
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

function onRadiusCanvasPointerMove(event) {
  if (!els.radiusCanvas || !els.radiusHover) return;
  const points = state.latestRadiusProjection;
  if (!Array.isArray(points) || points.length === 0) {
    els.radiusHover.textContent = "Hover a dot to see its game name.";
    return;
  }

  const rect = els.radiusCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  let bestPoint = null;
  let bestDist = Infinity;
  for (const point of points) {
    const dx = point.px - x;
    const dy = point.py - y;
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist) {
      bestDist = dist;
      bestPoint = point;
    }
  }

  if (bestPoint && bestDist <= 14) {
    els.radiusHover.textContent = `Hover: ${bestPoint.label}`;
  } else {
    els.radiusHover.textContent = "Hover a dot to see its game name.";
  }
}

function onRadiusCanvasPointerLeave() {
  if (!els.radiusHover) return;
  els.radiusHover.textContent = "Hover a dot to see its game name.";
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
  renderGroupLegend(els.radiusLegend, samples);
  renderImage3DMaps(samples, colorMap);
}

function renderAll() {
  forceHorizontalScrollReset();
  if (!state.data) return;

  const filtered = filteredSamples();
  const selectedSamples = applyVizSampleLimit(filtered);
  updateGroupCounters(state.data.samples, selectedSamples);
  const total = state.data.samples.length;
  updateRadiusControlBounds(selectedSamples.length);

  if (selectedSamples.length === 0) {
    setStatus("No samples match current game/group filters. Select at least one game and one group.", true);
    Plotly.purge(els.tsnePlot);
    Plotly.purge(els.umapPlot);
    if (els.pcaPlot) {
      Plotly.purge(els.pcaPlot);
    }
    Plotly.purge(els.centroidHeatmap);
    Plotly.purge(els.groupCentroidHeatmap);
    Plotly.purge(els.promptHeatmap);
    Plotly.purge(els.promptGroupHeatmap);
    purgeSwissRollPlots();
    renderQualityTable(null);
    renderOutlierTable(null);
    renderCanvasVisuals([], {});
    return;
  }

  const allGames = uniqueSorted(state.data.samples.map((s) => s.label));
  const colorMap = getColorMap(allGames);

  render3DScatter(els.tsnePlot, selectedSamples, "tsne", `3D t-SNE (${selectedSamples.length}/${total} samples)`, colorMap);
  render3DScatter(els.umapPlot, selectedSamples, "umap", `3D UMAP (${selectedSamples.length}/${total} samples)`, colorMap);
  render3DScatter(
    els.pcaPlot,
    selectedSamples,
    "pca",
    `3D PCA (${selectedSamples.length}/${total} samples)`,
    colorMap,
    "dbscan_cluster_id",
    "DBSCAN"
  );

  const centroid = state.data.centroid_similarity;
  if (centroid && Array.isArray(centroid.labels) && Array.isArray(centroid.matrix)) {
    const selectedGames = new Set(selectedSamples.map((sample) => sample.label));
    const visibleLabels = centroid.labels.filter((label) => selectedGames.has(label));
    const readableLabels = pickReadableCentroidGames(visibleLabels, selectedSamples, 5);
    if (readableLabels.length >= 2) {
      const readableMatrix = subsetSymmetricMatrix(centroid.labels, centroid.matrix, readableLabels);
      const hasBothGroups =
        readableLabels.some((label) =>
          selectedSamples.some((sample) => sample.label === label && normalizeGroupName(sample.group) === "indie")
        ) &&
        readableLabels.some((label) =>
          selectedSamples.some((sample) => sample.label === label && normalizeGroupName(sample.group) === "aaa")
        );
      const subtitle = hasBothGroups ? "(up to 5 indie + 5 aaa)" : "(up to 10 selected games)";
      renderHeatmap(
        els.centroidHeatmap,
        readableLabels,
        readableLabels,
        readableMatrix,
        `Game Centroid Cosine Similarity ${subtitle}`
      );
    } else {
      Plotly.purge(els.centroidHeatmap);
    }
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
  const promptFocusMeta = state.data.runtime_parameters?.prompt_focus || null;
  const promptFocusSuffix =
    promptFocusMeta &&
    Number.isFinite(Number(promptFocusMeta.display_prompt_count)) &&
    Number.isFinite(Number(promptFocusMeta.full_prompt_count)) &&
    Number(promptFocusMeta.display_prompt_count) < Number(promptFocusMeta.full_prompt_count)
      ? ` (focused ${Number(promptFocusMeta.display_prompt_count)}/${Number(promptFocusMeta.full_prompt_count)} prompts)`
      : "";
  if (prompts && Array.isArray(prompts.prompts) && Array.isArray(prompts.games) && Array.isArray(prompts.matrix)) {
    const source = typeof prompts.source === "string" ? prompts.source : "clip_text_prompts";
    const title =
      source === "style_adapter"
        ? `Fine-Tuned Style Adapter Scores by Game${promptFocusSuffix}`
        : `CLIP Prompt Similarity (Average by Game)${promptFocusSuffix}`;
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
        ? `Fine-Tuned Style Adapter Scores by Group${promptFocusSuffix}`
        : `CLIP Prompt Similarity by Group${promptFocusSuffix}`;
    renderHeatmap(els.promptGroupHeatmap, promptByGroup.prompts, promptByGroup.groups, promptByGroup.matrix, title);
  } else {
    Plotly.purge(els.promptGroupHeatmap);
  }

  renderSkippedTable(state.data.skipped_images || []);
  renderQualityTable(state.data.projection_quality || null);
  renderOutlierTable(state.data.outliers || null);
  renderSwissRollSection(state.data.swiss_roll_demo || null);
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
      `3D view using ${selectedSamples.length}/${filtered.length} filtered samples (mode=${normalizeVizSampleSize(state.vizSampleSize)}). ` +
      `t-SNE=${runtime.tsne?.method || "unknown"}, UMAP=${runtime.umap?.method || "unknown"}, ` +
      `PCA=${runtime.pca?.method || "unknown"}, DBSCAN=${runtime.dbscan?.status || "unknown"}.`
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
  await refreshThesisSection(false);
  await refreshPhase2Section(false);
  await refreshPhase3Section(false);
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
    const modeLabel = normalizeDatasetMode(runPayload.dataset_mode);
    setStatus(
      `Starting ${modeLabel} analysis with n_neighbors=${runPayload.umap_n_neighbors}, ` +
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

async function onRunPhase3Click() {
  if (!state.backendAvailable) {
    const message =
      "Run Phase 3 is available only when using the backend server: python src/local_app_server.py";
    setStatus(message, true);
    setPhase3Status(message, true);
    return;
  }

  try {
    setPhase3ButtonBusy(true);
    const runPayload = getPhase3RunPayloadFromState();
    const modeLabel = normalizeDatasetMode(runPayload.dataset_mode);
    const kickoffMessage = `Starting Phase-3 analysis (${modeLabel}) with full advanced metrics...`;
    setStatus(kickoffMessage);
    setPhase3Status(kickoffMessage);
    const payload = await runPhase3FromBackend(runPayload);
    const responseMessage = payload.message || "Phase-3 analysis started.";
    setStatus(responseMessage);
    setPhase3Status(responseMessage);

    stopPhase3StatusPolling();
    state.phase3StatusPollTimerId = setInterval(() => {
      pollPhase3Status();
    }, 1600);
    pollPhase3Status();
  } catch (err) {
    setPhase3ButtonBusy(false);
    const message = formatNetworkError(err, "Run phase-3 analysis");
    setStatus(message, true);
    setPhase3Status(message, true);
  }
}

function onDatasetModeChange() {
  if (!els.datasetModeSelect) return;
  state.datasetMode = normalizeDatasetMode(els.datasetModeSelect.value);
  renderDatasetModeUi();
}

async function startIgdbFetch(dryRun, extraPayload = null) {
  if (!state.igdbAvailable) {
    setStatus(
      "IGDB fetch is available only when using backend server: python src/local_app_server.py",
      true
    );
    return;
  }

  try {
    setIgdbButtonsBusy(true);
    const payload = {
      ...getIgdbPayload(dryRun),
      ...(extraPayload && typeof extraPayload === "object" ? extraPayload : {}),
    };
    setStatus(
      dryRun
        ? "Starting IGDB dry run..."
        : "Starting IGDB cover fetch..."
    );
    const response = await runIgdbFetchFromBackend(payload);
    setStatus(response.message || "IGDB fetch started.");
    stopIgdbStatusPolling();
    state.igdbStatusPollTimerId = setInterval(() => {
      pollIgdbStatus();
    }, 1600);
    pollIgdbStatus();
  } catch (err) {
    setIgdbButtonsBusy(false);
    setStatus(formatNetworkError(err, "IGDB fetch"), true);
  }
}

async function onIgdbDryRunClick() {
  await startIgdbFetch(true);
}

async function onIgdbFetchClick() {
  await startIgdbFetch(false);
}

async function onIgdbSeedIndieClick() {
  await startIgdbFetch(false, {
    seed_mode: "genre",
    seed_genre_name: "Indie",
    seed_count: 30,
    seed_only: false,
    seed_group_label: "indie",
    seed_update_groups: true,
    seed_write_mappings: true,
    skip_if_any_image: true,
    strict_match_mode: true,
    auto_clean_local_names: true,
  });
}

async function onIgdbSeedAaaClick() {
  await startIgdbFetch(false, {
    seed_mode: "popular_year_range",
    seed_count: 30,
    seed_only: false,
    seed_group_label: "aaa",
    seed_update_groups: true,
    seed_write_mappings: true,
    seed_year_start: 2010,
    seed_year_end: 2020,
    seed_min_total_rating_count: 120,
    seed_exclude_genre_name: "Indie",
    skip_if_any_image: true,
    strict_match_mode: true,
    auto_clean_local_names: true,
  });
}

async function onIgdbSearchClick() {
  if (!state.igdbAvailable) {
    setStatus("IGDB search is unavailable. Start backend server first.", true);
    return;
  }
  const query = els.igdbSearchInput ? String(els.igdbSearchInput.value || "").trim() : "";
  const company = els.igdbCompanyInput ? String(els.igdbCompanyInput.value || "").trim() : "";
  const limit = els.igdbSearchLimit
    ? clamp(Number.parseInt(String(els.igdbSearchLimit.value || "100"), 10) || 100, 1, 500)
    : 100;
  if (!query && !company) {
    setStatus("Type a game title or company (or both) to search IGDB.", true);
    return;
  }
  if (query && query.length < 2) {
    setStatus("Game title search must be at least 2 characters.", true);
    return;
  }
  if (company && company.length < 2) {
    setStatus("Company search must be at least 2 characters.", true);
    return;
  }
  try {
    await refreshExistingFolderKeys();
    const searchLabel =
      query && company
        ? `title "${query}" in company "${company}"`
        : query
          ? `title "${query}"`
          : `company "${company}"`;
    setStatus(`Searching IGDB for ${searchLabel}...`);
    if (els.igdbSearchBtn) {
      els.igdbSearchBtn.disabled = true;
      els.igdbSearchBtn.textContent = "Searching...";
    }
    if (els.igdbSelectAllBtn) {
      els.igdbSelectAllBtn.disabled = true;
    }
    const payload = await searchIgdbGamesFromBackend(query, limit, company);
    const results = Array.isArray(payload.results) ? payload.results : [];
    renderIgdbSearchResults(results);
    const existingCount = results.filter((row) => row && row.name && isGameAlreadyAdded(row.name)).length;
    const shownCount = els.igdbSearchResults ? els.igdbSearchResults.options.length : results.length;
    const hideExisting = Boolean(els.igdbHideExistingCheck && els.igdbHideExistingCheck.checked);
    if (results.length === 0) {
      setStatus(`Found 0 IGDB matches for ${searchLabel}. Try broader terms.`, true);
    } else {
      const hiddenPart =
        hideExisting && existingCount > 0 ? ` (${shownCount} shown, ${existingCount} hidden as already added)` : "";
      const existingPart = existingCount > 0 ? ` (${existingCount} already added)` : "";
      setStatus(`Found ${results.length} IGDB matches for ${searchLabel}.${hiddenPart || existingPart}`);
    }
  } catch (err) {
    setStatus(formatNetworkError(err, "IGDB search"), true);
  } finally {
    if (els.igdbSearchBtn) {
      els.igdbSearchBtn.disabled = !state.igdbAvailable;
      els.igdbSearchBtn.textContent = "Search";
    }
    if (els.igdbSelectAllBtn && els.igdbSearchResults) {
      els.igdbSelectAllBtn.disabled =
        !state.igdbAvailable || els.igdbSearchResults.options.length === 0;
    }
  }
}

function onIgdbSearchSelectionChange() {
  if (!els.igdbAddSelectedBtn || !els.igdbSearchResults) return;
  const hasSelection = getSelectedIgdbIds().length > 0;
  els.igdbAddSelectedBtn.disabled = !state.igdbAvailable || !hasSelection;
}

function onIgdbHideExistingToggle() {
  renderIgdbSearchResults(state.igdbSearchResults);
  onIgdbSearchSelectionChange();
}

function onIgdbSelectAllClick() {
  if (!els.igdbSearchResults) return;
  const options = Array.from(els.igdbSearchResults.options || []);
  if (options.length === 0) {
    setStatus("No IGDB results to select yet.", true);
    return;
  }
  for (const option of options) {
    option.selected = true;
  }
  onIgdbSearchSelectionChange();
  setStatus(`Selected all ${options.length} visible search results.`);
}

function onIgdbBulkListInputChange() {
  if (!els.igdbAddListBtn) return;
  const count = parseGameNameList(els.igdbBulkListInput ? els.igdbBulkListInput.value : "").length;
  els.igdbAddListBtn.disabled = !state.igdbAvailable || count === 0;
}

function onIgdbPdfInputChange() {
  if (!els.igdbImportPdfBtn) return;
  const hasPdf =
    Boolean(els.igdbPdfInput) &&
    Boolean(els.igdbPdfInput.files) &&
    els.igdbPdfInput.files.length > 0;
  els.igdbImportPdfBtn.disabled = !state.igdbAvailable || !hasPdf;
}

async function onIgdbAddSelectedClick() {
  const selectedIds = getSelectedIgdbIds();
  if (selectedIds.length === 0) {
    setStatus("Select one or more games from search results first.", true);
    return;
  }
  const selectedGroup =
    els.igdbSearchGroup && els.igdbSearchGroup.value
      ? String(els.igdbSearchGroup.value)
      : "aaa";
  const selectedRows = state.igdbSearchResults.filter((row) => selectedIds.includes(Number(row.id)));
  const idsToAdd = [];
  let skippedAlreadyAdded = 0;
  for (const row of selectedRows) {
    if (row && row.name && isGameAlreadyAdded(row.name)) {
      skippedAlreadyAdded += 1;
      continue;
    }
    const parsed = Number.parseInt(String(row && row.id ? row.id : ""), 10);
    if (Number.isFinite(parsed) && parsed > 0 && !idsToAdd.includes(parsed)) {
      idsToAdd.push(parsed);
    }
  }
  if (idsToAdd.length === 0) {
    if (skippedAlreadyAdded > 0) {
      setStatus(`All ${skippedAlreadyAdded} selected games are already added.`);
    } else {
      setStatus("No valid games selected.", true);
    }
    return;
  }
  if (skippedAlreadyAdded > 0) {
    setStatus(`Adding ${idsToAdd.length} selected game(s); skipped ${skippedAlreadyAdded} already added title(s).`);
  } else {
    setStatus(`Adding ${idsToAdd.length} selected game(s)...`);
  }

  await startIgdbFetch(false, {
    seed_mode: "id_list",
    seed_game_ids: idsToAdd,
    seed_count: idsToAdd.length,
    seed_only: true,
    seed_group_label: selectedGroup,
    seed_update_groups: true,
    seed_write_mappings: true,
    skip_if_any_image: true,
    strict_match_mode: true,
    auto_clean_local_names: true,
  });
}

async function onIgdbAddListClick() {
  const raw = els.igdbBulkListInput ? els.igdbBulkListInput.value : "";
  const names = parseGameNameList(raw);
  if (names.length === 0) {
    setStatus("Paste at least one game name in Add Game List.", true);
    return;
  }
  const namesToAdd = names.filter((name) => !isGameAlreadyAdded(name));
  const skippedAlreadyAdded = names.length - namesToAdd.length;
  if (namesToAdd.length === 0) {
    setStatus("All titles from this list are already added.");
    return;
  }
  const selectedGroup =
    els.igdbSearchGroup && els.igdbSearchGroup.value
      ? String(els.igdbSearchGroup.value)
      : "aaa";

  if (skippedAlreadyAdded > 0) {
    setStatus(
      `Starting list import for ${namesToAdd.length} new game(s); ${skippedAlreadyAdded} already added title(s) were skipped.`
    );
  } else {
    setStatus(`Starting list import for ${namesToAdd.length} game names...`);
  }
  await startIgdbFetch(false, {
    seed_mode: "name_list",
    seed_game_names: namesToAdd,
    seed_count: namesToAdd.length,
    seed_only: true,
    seed_group_label: selectedGroup,
    seed_update_groups: true,
    seed_write_mappings: true,
    skip_if_any_image: true,
    strict_match_mode: true,
    auto_clean_local_names: true,
  });
}

async function onIgdbImportPdfClick() {
  if (!state.igdbAvailable) {
    setStatus("IGDB import is unavailable. Start backend server first.", true);
    return;
  }
  const file =
    els.igdbPdfInput &&
    els.igdbPdfInput.files &&
    els.igdbPdfInput.files.length > 0
      ? els.igdbPdfInput.files[0]
      : null;
  if (!file) {
    setStatus("Select a PDF file first.", true);
    return;
  }
  try {
    if (els.igdbImportPdfBtn) {
      els.igdbImportPdfBtn.disabled = true;
      els.igdbImportPdfBtn.textContent = "Reading PDF...";
    }
    setStatus(`Reading PDF: ${file.name} ...`);
    const names = await extractGameNamesFromPdfFile(file);
    if (names.length === 0) {
      setStatus("No game names were detected in this PDF.", true);
      return;
    }
    const namesToAdd = names.filter((name) => !isGameAlreadyAdded(name));
    const skippedAlreadyAdded = names.length - namesToAdd.length;
    if (els.igdbBulkListInput) {
      els.igdbBulkListInput.value = namesToAdd.join("\n");
    }
    onIgdbBulkListInputChange();
    if (namesToAdd.length === 0) {
      setStatus(`All ${names.length} titles parsed from PDF are already added.`);
      return;
    }
    const selectedGroup =
      els.igdbSearchGroup && els.igdbSearchGroup.value
        ? String(els.igdbSearchGroup.value)
        : "aaa";
    if (skippedAlreadyAdded > 0) {
      setStatus(
        `Parsed ${names.length} names from PDF. Importing ${namesToAdd.length} new game(s); ${skippedAlreadyAdded} already added title(s) skipped.`
      );
    } else {
      setStatus(`Parsed ${namesToAdd.length} game names from PDF. Starting import...`);
    }
    await startIgdbFetch(false, {
      seed_mode: "name_list",
      seed_game_names: namesToAdd,
      seed_count: namesToAdd.length,
      seed_only: true,
      seed_group_label: selectedGroup,
      seed_update_groups: true,
      seed_write_mappings: true,
      skip_if_any_image: true,
      strict_match_mode: true,
      auto_clean_local_names: true,
    });
  } catch (err) {
    setStatus(formatNetworkError(err, "PDF import"), true);
  } finally {
    if (els.igdbImportPdfBtn) {
      const hasPdf =
        Boolean(els.igdbPdfInput) &&
        Boolean(els.igdbPdfInput.files) &&
        els.igdbPdfInput.files.length > 0;
      els.igdbImportPdfBtn.disabled = !state.igdbAvailable || !hasPdf;
      els.igdbImportPdfBtn.textContent = "Import PDF";
    }
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

function onGroupFocusChanged() {
  state.groupFocus = normalizeGroupFocus(els.groupFocusSelect ? els.groupFocusSelect.value : "both");
  renderAll();
}

function onVizSampleSizeChanged() {
  state.vizSampleSize = normalizeVizSampleSize(
    els.vizSampleSizeSelect ? els.vizSampleSizeSelect.value : "all"
  );
  renderAll();
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
  if (
    !els.runAnalysisBtn &&
    !els.runPhase3Btn &&
    !els.igdbDryRunBtn &&
    !els.igdbFetchBtn &&
    !els.igdbSeedIndieBtn &&
    !els.igdbSeedAaaBtn &&
    !els.igdbSearchBtn &&
    !els.igdbAddListBtn
  ) return;
  try {
    const [status, phase3Status, igdbStatus] = await Promise.all([
      fetchBackendStatus(),
      fetchPhase3BackendStatus(),
      fetchIgdbBackendStatus(),
    ]);
    state.backendAvailable = true;
    state.igdbAvailable = true;
    if (status && status.last_params && status.last_params.dataset_mode) {
      state.datasetMode = normalizeDatasetMode(status.last_params.dataset_mode);
    }
    renderDatasetModeUi(status);
    await refreshExistingFolderKeys();
    els.runAnalysisBtn.disabled = false;
    els.runAnalysisBtn.title = "Run CLIP pipeline and regenerate web/data outputs.";
    if (els.runPhase3Btn) {
      els.runPhase3Btn.disabled = false;
      els.runPhase3Btn.title =
        "Run advanced phase-3 analysis (OVL, ARI, KDE, residuals, cosine distributions, prompt-style comparisons).";
    }
    if (els.loadPhase3Btn) {
      els.loadPhase3Btn.disabled = false;
      els.loadPhase3Btn.title = "Load existing phase-3 advanced outputs from web/data/phase3_advanced.";
    }
    if (els.igdbDryRunBtn) {
      els.igdbDryRunBtn.disabled = false;
      els.igdbDryRunBtn.title = "Resolve IGDB matches without downloading covers.";
    }
    if (els.igdbFetchBtn) {
      els.igdbFetchBtn.disabled = false;
      els.igdbFetchBtn.title = "Fetch and save IGDB covers into game folders.";
    }
    if (els.igdbSeedIndieBtn) {
      els.igdbSeedIndieBtn.disabled = false;
      els.igdbSeedIndieBtn.title = "Create 30 Indie game folders from IGDB and fetch their covers.";
    }
    if (els.igdbSeedAaaBtn) {
      els.igdbSeedAaaBtn.disabled = false;
      els.igdbSeedAaaBtn.title = "Create 30 AAA-like game folders (2010-2020, popular, non-Indie) and fetch covers.";
    }
    if (els.igdbSearchBtn) {
      els.igdbSearchBtn.disabled = false;
      els.igdbSearchBtn.title = "Search IGDB by game title.";
    }
    if (els.igdbSelectAllBtn) {
      const hasResults = Boolean(els.igdbSearchResults && els.igdbSearchResults.options.length > 0);
      els.igdbSelectAllBtn.disabled = !hasResults;
      els.igdbSelectAllBtn.title = "Select all visible IGDB search results.";
    }
    if (els.igdbAddSelectedBtn) {
      els.igdbAddSelectedBtn.disabled = true;
      els.igdbAddSelectedBtn.title = "Add selected search result into local dataset.";
    }
    if (els.igdbAddListBtn) {
      const hasList = parseGameNameList(els.igdbBulkListInput ? els.igdbBulkListInput.value : "").length > 0;
      els.igdbAddListBtn.disabled = !hasList;
      els.igdbAddListBtn.title = "Add all games from the pasted list into local dataset.";
    }
    if (els.igdbImportPdfBtn) {
      const hasPdf =
        Boolean(els.igdbPdfInput) &&
        Boolean(els.igdbPdfInput.files) &&
        els.igdbPdfInput.files.length > 0;
      els.igdbImportPdfBtn.disabled = !hasPdf;
      els.igdbImportPdfBtn.title = "Read game names from a PDF and import them.";
    }
    if (status.running) {
      setRunButtonBusy(true);
      stopStatusPolling();
      state.statusPollTimerId = setInterval(() => {
        pollAnalysisStatus();
      }, 1500);
      pollAnalysisStatus();
    }
    if (igdbStatus.running) {
      setIgdbButtonsBusy(true);
      stopIgdbStatusPolling();
      state.igdbStatusPollTimerId = setInterval(() => {
        pollIgdbStatus();
      }, 1600);
      pollIgdbStatus();
    }
    if (phase3Status.running) {
      setPhase3ButtonBusy(true);
      stopPhase3StatusPolling();
      state.phase3StatusPollTimerId = setInterval(() => {
        pollPhase3Status();
      }, 1600);
      pollPhase3Status();
    } else {
      setPhase3ButtonBusy(false);
    }
  } catch (_) {
    state.backendAvailable = false;
    state.igdbAvailable = false;
    els.runAnalysisBtn.disabled = true;
    els.runAnalysisBtn.title = "Start backend server with: python src/local_app_server.py";
    if (els.runPhase3Btn) {
      els.runPhase3Btn.disabled = true;
      els.runPhase3Btn.title = "Start backend server with: python src/local_app_server.py";
      els.runPhase3Btn.textContent = "Run Phase 3";
    }
    if (els.loadPhase3Btn) {
      els.loadPhase3Btn.disabled = false;
      els.loadPhase3Btn.title = "Load existing phase-3 advanced outputs from web/data/phase3_advanced.";
    }
    if (els.igdbDryRunBtn) {
      els.igdbDryRunBtn.disabled = true;
      els.igdbDryRunBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbFetchBtn) {
      els.igdbFetchBtn.disabled = true;
      els.igdbFetchBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbSeedIndieBtn) {
      els.igdbSeedIndieBtn.disabled = true;
      els.igdbSeedIndieBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbSeedAaaBtn) {
      els.igdbSeedAaaBtn.disabled = true;
      els.igdbSeedAaaBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbSearchBtn) {
      els.igdbSearchBtn.disabled = true;
      els.igdbSearchBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbSelectAllBtn) {
      els.igdbSelectAllBtn.disabled = true;
      els.igdbSelectAllBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbAddSelectedBtn) {
      els.igdbAddSelectedBtn.disabled = true;
      els.igdbAddSelectedBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbAddListBtn) {
      els.igdbAddListBtn.disabled = true;
      els.igdbAddListBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    if (els.igdbImportPdfBtn) {
      els.igdbImportPdfBtn.disabled = true;
      els.igdbImportPdfBtn.title = "Start backend server with: python src/local_app_server.py";
    }
    stopPhase3StatusPolling();
    renderDatasetModeUi();
  }
}

function bindEvents() {
  if (els.runAnalysisBtn) {
    els.runAnalysisBtn.addEventListener("click", onRunAnalysisClick);
  }
  if (els.runPhase3Btn) {
    els.runPhase3Btn.addEventListener("click", onRunPhase3Click);
  }
  if (els.datasetModeSelect) {
    els.datasetModeSelect.addEventListener("change", onDatasetModeChange);
  }
  if (els.groupFocusSelect) {
    els.groupFocusSelect.addEventListener("change", onGroupFocusChanged);
  }
  if (els.vizSampleSizeSelect) {
    els.vizSampleSizeSelect.addEventListener("change", onVizSampleSizeChanged);
  }
  if (els.igdbDryRunBtn) {
    els.igdbDryRunBtn.addEventListener("click", onIgdbDryRunClick);
  }
  if (els.igdbFetchBtn) {
    els.igdbFetchBtn.addEventListener("click", onIgdbFetchClick);
  }
  if (els.igdbSeedIndieBtn) {
    els.igdbSeedIndieBtn.addEventListener("click", onIgdbSeedIndieClick);
  }
  if (els.igdbSeedAaaBtn) {
    els.igdbSeedAaaBtn.addEventListener("click", onIgdbSeedAaaClick);
  }
  if (els.igdbSearchBtn) {
    els.igdbSearchBtn.addEventListener("click", onIgdbSearchClick);
  }
  if (els.igdbSelectAllBtn) {
    els.igdbSelectAllBtn.addEventListener("click", onIgdbSelectAllClick);
  }
  if (els.igdbSearchInput) {
    els.igdbSearchInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        onIgdbSearchClick();
      }
    });
  }
  if (els.igdbCompanyInput) {
    els.igdbCompanyInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        onIgdbSearchClick();
      }
    });
  }
  if (els.igdbSearchResults) {
    els.igdbSearchResults.addEventListener("change", onIgdbSearchSelectionChange);
    els.igdbSearchResults.addEventListener("dblclick", onIgdbAddSelectedClick);
  }
  if (els.igdbHideExistingCheck) {
    els.igdbHideExistingCheck.addEventListener("change", onIgdbHideExistingToggle);
  }
  if (els.igdbAddSelectedBtn) {
    els.igdbAddSelectedBtn.addEventListener("click", onIgdbAddSelectedClick);
  }
  if (els.igdbBulkListInput) {
    els.igdbBulkListInput.addEventListener("input", onIgdbBulkListInputChange);
  }
  if (els.igdbAddListBtn) {
    els.igdbAddListBtn.addEventListener("click", onIgdbAddListClick);
  }
  if (els.igdbPdfInput) {
    els.igdbPdfInput.addEventListener("change", onIgdbPdfInputChange);
  }
  if (els.igdbImportPdfBtn) {
    els.igdbImportPdfBtn.addEventListener("click", onIgdbImportPdfClick);
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
  if (els.loadThesisBtn) {
    els.loadThesisBtn.addEventListener("click", () => {
      refreshThesisSection(true);
    });
  }
  if (els.loadPhase2Btn) {
    els.loadPhase2Btn.addEventListener("click", () => {
      refreshPhase2Section(true);
    });
  }
  if (els.loadPhase3Btn) {
    els.loadPhase3Btn.addEventListener("click", () => {
      refreshPhase3Section(true);
    });
  }
  if (els.radiusCanvas) {
    els.radiusCanvas.addEventListener("pointermove", onRadiusCanvasPointerMove);
    els.radiusCanvas.addEventListener("pointerleave", onRadiusCanvasPointerLeave);
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
    renderDatasetModeUi();
    if (els.groupFocusSelect) {
      els.groupFocusSelect.value = normalizeGroupFocus(state.groupFocus);
    }
    if (els.vizSampleSizeSelect) {
      els.vizSampleSizeSelect.value = normalizeVizSampleSize(state.vizSampleSize);
    }
    initImageMapViews();
    bindEvents();
    initBackendAvailability();
    refreshPhase3Section(false);
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
