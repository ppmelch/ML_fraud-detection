/* =========================
   CONFIG
========================= */

// Backend API base URL. Override by loading this file with a
// `data-api-base` attribute on the <script> tag, e.g.
//   <script src="js/app.js" data-api-base="https://api.example.com"></script>
const API_BASE_URL =
    (document.currentScript && document.currentScript.dataset.apiBase) ||
    "http://127.0.0.1:8000";

// GeoJSON property holding the state name (must match the backend's
// `geojson_name`, produced by StateGeoMapper).
const GEOJSON_NAME_PROPERTY = "name";


/* =========================
   STATE
========================= */

let stateMetrics = {};       // geojson_name -> state record from /api/states
let stateRecords = [];       // raw array from /api/states
let selectedState = null;    // geojson_name of the pinned state, or null
let colorScale = { min: 0, max: 1 };
let leafletMap = null;
let stateLayer = null;
let geojsonStateNames = [];  // sorted feature names, for the navbar search


/* =========================
   API CLIENT
========================= */

/**
 * GET a JSON endpoint from the fraud detection API.
 *
 * @param {string} path - API path, e.g. "/api/fraud/summary".
 * @returns {Promise<any>} Parsed JSON body.
 * @throws {Error} When the network request fails or the response is not ok.
 */
function apiGet(path) {

    return fetch(`${API_BASE_URL}${path}`)
        .then(response => {

            if (!response.ok) {

                throw new Error(
                    `${path} responded ${response.status} ${response.statusText}`
                );
            }

            return response.json();
        })
        .catch(error => {

            throw new Error(
                `Could not reach backend at ${API_BASE_URL}${path} (${error.message}). ` +
                `Is the API running? Start it with: uvicorn backend.src.api.main:app --port 8000`
            );
        });
}


/* =========================
   FORMAT HELPERS
========================= */

function formatInt(value) {
    if (value == null || Number.isNaN(value)) return "N/A";
    return Number(value).toLocaleString("en-IN");
}

function formatPercent(value, decimals = 2) {
    if (value == null || Number.isNaN(value)) return "N/A";
    return `${(value * 100).toFixed(decimals)}%`;
}

function formatCurrency(value) {
    if (value == null || Number.isNaN(value)) return "N/A";
    return `₹${Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

function formatAuc(value) {
    if (value == null || Number.isNaN(value)) return "N/A";
    return Number(value).toFixed(3);
}

function formatHour(hour) {
    if (hour == null) return "N/A";
    return `${String(hour).padStart(2, "0")}:00`;
}


/* =========================
   SECTION STATUS HELPERS
   (loading / error / empty banners reused across sections)
========================= */

function setSectionStatus(elementId, message, kind) {

    const el = document.getElementById(elementId);

    if (!el) return;

    if (!message) {

        el.classList.remove("visible", "loading");
        el.textContent = "";
        return;
    }

    el.textContent = message;
    el.classList.add("visible");
    el.classList.toggle("loading", kind === "loading");
}

function setChartState(cardId, state, message) {

    const card = document.getElementById(cardId);

    if (!card) return;

    card.classList.remove("is-loading", "is-error");

    const placeholder = card.querySelector(".chart-placeholder");

    if (state === "loading") {

        card.classList.add("is-loading");
        if (placeholder) placeholder.textContent = message || "Loading…";

    } else if (state === "error") {

        card.classList.add("is-error");
        if (placeholder) placeholder.textContent = message || "Failed to load.";
    }
    // state === "ready": both classes already removed above, chart shows.
}


/* =========================
   PLOTLY THEME (reused across all charts)
========================= */

const plotTheme = {
    paper_bgcolor: "rgba(239, 239, 239, 0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
        color: "#F6EFE3",
        family: "Inter, sans-serif"
    },
    xaxis: {
        showgrid: false,
        zeroline: false,
    },
    yaxis: {
        showgrid: true,
        gridcolor: "rgba(246,239,227,0.16)",
        zeroline: false
    }
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };


/* =========================
   CHART PALETTE
   Mirrors the CSS custom properties in css/style.css — India Fraud
   Analytics: saffron accent, India green for legitimate, deep crimson
   for fraud, marigold for baselines, warm sand for neutral series.
========================= */

const CHART_FRAUD = "#C1272D";       // deep crimson — fraud / danger
const CHART_FRAUD_FILL = "rgba(193,39,45,0.22)";
const CHART_LEGIT = "#138808";       // India green — legitimate
const CHART_LEGIT_FILL = "rgba(19,136,8,0.22)";
const CHART_ACCENT = "#FF9933";      // saffron — primary model series
const CHART_BASELINE = "#F6B21B";    // marigold — reference / baseline lines
const CHART_MODEL_LINE = "#F6EFE3";  // warm white — model ROC curve
const CHART_NEUTRAL = "#C9BBA8";     // warm sand — neutral series
const CHART_NEUTRAL_FILL = "rgba(201,187,168,0.3)";


/* =========================
   COLOR SCALE (fraud rate, computed from real data — no hardcoded cutoffs)
========================= */

/**
 * Compute the min/max fraud rate across all states, for a relative scale.
 * Fraud rates in this dataset sit in a narrow band, so a fixed absolute
 * scale (like a 0-1 PD scale) would paint everything the same color.
 *
 * @param {Array<Object>} records - State records with a `fraud_rate` field.
 * @returns {{min: number, max: number}}
 */
function computeColorScale(records) {

    const rates = records
        .map(r => r.fraud_rate)
        .filter(r => r != null && !Number.isNaN(r));

    if (rates.length === 0) return { min: 0, max: 1 };

    return { min: Math.min(...rates), max: Math.max(...rates) };
}

function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]) {
    return "#" + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, "0")).join("");
}

function lerpColor(hexA, hexB, t) {
    const a = hexToRgb(hexA);
    const b = hexToRgb(hexB);
    return rgbToHex(a.map((v, i) => v + (b[i] - v) * t));
}

// Choropleth ramp — coherent Indian-warm sequence, matching the
// --ramp-* custom properties and the legend gradient in css/style.css:
// pale marigold (low) -> saffron (mid) -> deep crimson (high).
const SCALE_LOW = "#FBE3B0";
const SCALE_MID = "#FF9933";
const SCALE_HIGH = "#C1272D";
const SCALE_NO_DATA = "#5a5346";

/**
 * Map a fraud rate to a color on the project's green -> amber -> red scale,
 * relative to the observed min/max across all states.
 *
 * @param {?number} rate - State fraud rate, or null/undefined for no data.
 * @returns {string} Hex color.
 */
function getStateColor(rate) {

    if (rate == null || Number.isNaN(rate)) return SCALE_NO_DATA;

    const { min, max } = colorScale;

    if (max <= min) return SCALE_MID;

    const t = Math.min(1, Math.max(0, (rate - min) / (max - min)));

    return t <= 0.5
        ? lerpColor(SCALE_LOW, SCALE_MID, t / 0.5)
        : lerpColor(SCALE_MID, SCALE_HIGH, (t - 0.5) / 0.5);
}


/* =========================
   STATE CARD (hover + click / pin)
========================= */

function renderStateCard(record, { pinned }) {

    const card = document.getElementById("state-card");
    const kicker = document.getElementById("state-card-kicker");
    const nameEl = document.getElementById("state-name");
    const infoEl = document.getElementById("state-info");

    kicker.textContent = pinned ? "Selected State" : "Hovering";
    nameEl.textContent = record.state;

    if (!record.matched && record.fraud_rate == null) {

        infoEl.innerHTML = `<p class="no-fraud-data">No fraud data available for this state.</p>`;

    } else {

        const rows = [
            ["Fraud Rate", formatPercent(record.fraud_rate)],
            ["Fraud Cases", formatInt(record.fraud_cases)],
            ["Transactions", formatInt(record.total_transactions)],
            ["Peak Fraud Hour", formatHour(record.peak_fraud_hour)],
            ["Peak Hour Fraud Rate", formatPercent(record.peak_fraud_hour_rate)],
            ["Peak Fraud Day", record.peak_fraud_day || "N/A"],
            ["Peak Day Fraud Rate", formatPercent(record.peak_fraud_day_rate)],
            ["Top Fraud Device", record.most_common_fraud_device || "N/A"],
            ["Top Fraud Merchant", record.top_fraud_merchant_category || "N/A"],
            ["Average Transaction", formatCurrency(record.avg_transaction_amount)],
            ["Average Balance", formatCurrency(record.avg_account_balance)],
        ];

        infoEl.innerHTML = rows
            .map(([label, value]) => `<p>${label}: <span>${value}</span></p>`)
            .join("");
    }

    card.classList.add("active");
    card.classList.toggle("pinned", pinned);
}

function hideStateCard() {

    const card = document.getElementById("state-card");

    if (selectedState && stateMetrics[selectedState]) {

        // A state is pinned: keep showing its profile instead of hiding.
        renderStateCard(stateMetrics[selectedState], { pinned: true });
        return;
    }

    card.classList.remove("active", "pinned");
}

function clearSelection() {

    if (selectedState && stateLayer) {
        restyleState(selectedState);
    }

    selectedState = null;

    document.getElementById("state-card").classList.remove("active", "pinned");
}

/**
 * Pin a state by its GeoJSON name: styles it as selected and shows its
 * profile card. Shared by the map click handler and the navbar search.
 *
 * @param {string} geojsonName - feature name (GEOJSON_NAME_PROPERTY value).
 * @returns {boolean} true if a fraud record existed and was pinned.
 */
function selectState(geojsonName) {

    const record = stateMetrics[geojsonName];

    if (!record) return false; // no data to pin for this state

    if (selectedState && selectedState !== geojsonName) {
        restyleState(selectedState);
    }

    selectedState = geojsonName;

    renderStateCard(record, { pinned: true });
    restyleState(geojsonName);

    return true;
}

document.addEventListener("DOMContentLoaded", () => {

    const closeBtn = document.getElementById("state-close-btn");

    if (closeBtn) closeBtn.addEventListener("click", clearSelection);
});


/* =========================
   MAP STATUS OVERLAY
========================= */

function setMapStatus(kind, message) {

    const status = document.getElementById("map-status");
    const text = document.getElementById("map-status-text");
    const box = status.querySelector(".status-box");

    if (kind === "hidden") {
        status.classList.add("hidden");
        return;
    }

    status.classList.remove("hidden");
    box.classList.toggle("error", kind === "error");
    text.innerHTML = kind === "loading"
        ? `<div class="spinner"></div>${message}`
        : message;
}


/* =========================
   MAP CONFIGURATION
========================= */

leafletMap = L.map('map', {

    scrollWheelZoom: false,
    dragging: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    zoomControl: false,
    touchZoom: false

});


/**
 * Restyle a single feature layer back to its resting (non-hover) fill.
 * @param {string} geojsonName
 */
function restyleState(geojsonName) {

    if (!stateLayer) return;

    stateLayer.eachLayer(layer => {

        if (layer.feature.properties[GEOJSON_NAME_PROPERTY] !== geojsonName) return;

        const record = stateMetrics[geojsonName];
        const isSelected = geojsonName === selectedState;

        layer.setStyle({
            fillColor: getStateColor(record ? record.fraud_rate : null),
            fillOpacity: isSelected ? 1 : 0.8,
            weight: isSelected ? 3 : 1,
            color: isSelected ? "#FFFFFF" : "#3A2F22",
        });
    });
}


/**
 * Padding to bias fitBounds so the map stays clear of the hero title
 * (left) and the state card (right), without ever eating so much of a
 * narrower viewport that the map collapses to a sliver.
 *
 * @returns {{paddingTopLeft: number[], paddingBottomRight: number[]}}
 */
function getMapFitPadding() {

    const width = window.innerWidth;

    // Below this, the hero title and state card already switch to a
    // stacked/full-width layout (see the 900px media query) — the map
    // should use nearly the full viewport.
    if (width < 900) {
        return { paddingTopLeft: [30, 30], paddingBottomRight: [30, 50] };
    }

    // Scale the desktop padding down for mid-width (tablet) screens instead
    // of applying it verbatim, which would leave little to no room for the
    // map itself once both sides are subtracted.
    const left = Math.min(620, width * 0.32);
    const right = Math.min(380, width * 0.22);

    return {
        paddingTopLeft: [left, 40],
        paddingBottomRight: [right, 60],
    };
}


/* =========================
   LOAD MAP + STATE METRICS
========================= */

function loadMap() {

    setMapStatus("loading", "Loading India fraud map…");

    Promise.all([
        fetch("data/in.json").then(r => {
            if (!r.ok) throw new Error(`data/in.json responded ${r.status}`);
            return r.json();
        }),
        apiGet("/api/states"),
        apiGet("/api/states/validation").catch(() => null),
    ])
        .then(([geojson, states, validation]) => {

            stateRecords = states;

            stateMetrics = {};
            states.forEach(record => {
                if (record.geojson_name) {
                    stateMetrics[record.geojson_name] = record;
                }
            });

            colorScale = computeColorScale(states);

            // --- runtime join validation: surface mismatches, never hide them ---
            const geojsonNames = new Set(
                geojson.features.map(f => f.properties[GEOJSON_NAME_PROPERTY])
            );
            const unmatchedLookupKeys = Object.keys(stateMetrics)
                .filter(name => !geojsonNames.has(name));

            // Feed the navbar search / autocomplete from the geometry itself,
            // never a hardcoded list of Indian states.
            geojsonStateNames = [...geojsonNames].filter(Boolean).sort();
            populateStateSearchList(geojsonStateNames);

            if (unmatchedLookupKeys.length > 0) {
                console.warn(
                    "[India Fraud Map] Backend state names not found in data/in.json:",
                    unmatchedLookupKeys
                );
            }

            if (validation && validation.ok === false) {
                console.warn(
                    "[India Fraud Map] Backend reports an unresolved state/GeoJSON join:",
                    validation
                );
            }

            updateLegend(colorScale);

            stateLayer = L.geoJSON(geojson, {

                style: feature => {

                    const name = feature.properties[GEOJSON_NAME_PROPERTY];
                    const record = stateMetrics[name];

                    return {
                        color: "#3A2F22",
                        weight: 1,
                        fillColor: getStateColor(record ? record.fraud_rate : null),
                        fillOpacity: 0.8
                    };
                },

                onEachFeature: (feature, layer) => {

                    const name = feature.properties[GEOJSON_NAME_PROPERTY];

                    layer.on({

                        mouseover: () => {

                            const record = stateMetrics[name] || {
                                state: name,
                                matched: false,
                                fraud_rate: null,
                            };

                            renderStateCard(record, { pinned: name === selectedState });

                            layer.setStyle({
                                fillColor: "#1A3C8C",
                                fillOpacity: 1,
                                weight: 2,
                            });
                        },

                        mouseout: () => {

                            hideStateCard();
                            restyleState(name);
                        },

                        click: () => {

                            selectState(name);
                        }
                    });
                }

            }).addTo(leafletMap);

            // India's landmass is much wider than Jalisco's, so a plain
            // fitBounds centers it under the hero title and the state card.
            // Bias the fit with padding on both sides to keep the map clear
            // of those overlays — scaled to the viewport so the padding
            // itself never squeezes the map into a sliver on a narrower
            // (e.g. tablet) screen.
            leafletMap.fitBounds(stateLayer.getBounds(), getMapFitPadding());

            setMapStatus("hidden");
        })
        .catch(error => {

            console.error("[India Fraud Map] Failed to load:", error);
            setMapStatus("error", error.message);
        });
}

// Re-fit on resize (debounced) so rotating a tablet or resizing the window
// doesn't leave the map padded for a viewport size it no longer has.
let resizeTimer = null;
window.addEventListener("resize", () => {

    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {

        if (!leafletMap || !stateLayer) return;

        leafletMap.invalidateSize();
        leafletMap.fitBounds(stateLayer.getBounds(), getMapFitPadding());

    }, 250);
});

function updateLegend({ min, max }) {

    document.getElementById("legend-min").textContent = formatPercent(min, 1);
    document.getElementById("legend-max").textContent = formatPercent(max, 1);
    document.getElementById("legend-mid").textContent = formatPercent((min + max) / 2, 1);
}

loadMap();


/* =========================
   NAVBAR STATE SEARCH + RESET VIEW
   The search box filters the map to a named state (reusing selectState);
   the "Reset view" button fits the whole country and clears the selection.
========================= */

const searchForm = document.querySelector(".search-box");
const searchInput = searchForm ? searchForm.querySelector("input") : null;
const searchButton = searchForm ? searchForm.querySelector(".search-btn") : null;
const resetViewButton = document.querySelector(".location-btn");
const heroSection = document.querySelector(".hero-section");

/**
 * Fill the <datalist> that backs the search input's autocomplete.
 *
 * @param {string[]} names - state names, already sorted.
 */
function populateStateSearchList(names) {

    const list = document.getElementById("state-list");

    if (!list) return;

    list.innerHTML = names
        .map(name => `<option value="${name}"></option>`)
        .join("");
}

/**
 * Resolve a free-text query to a state name: exact, then prefix, then
 * substring (case-insensitive). Returns null when nothing matches.
 *
 * @param {string} query
 * @returns {?string}
 */
function findStateMatch(query) {

    const q = query.trim().toLowerCase();

    if (!q) return null;

    return geojsonStateNames.find(n => n.toLowerCase() === q)
        || geojsonStateNames.find(n => n.toLowerCase().startsWith(q))
        || geojsonStateNames.find(n => n.toLowerCase().includes(q))
        || null;
}

/**
 * Swap the icon button between "search" (empty field) and "clear"
 * (field has text) by toggling the .has-text class the CSS keys off.
 */
function syncSearchButtonMode() {

    if (!searchForm || !searchInput || !searchButton) return;

    const hasText = searchInput.value.trim().length > 0;

    searchForm.classList.toggle("has-text", hasText);
    searchButton.setAttribute(
        "aria-label", hasText ? "Clear search" : "Search states"
    );
    searchButton.setAttribute(
        "title", hasText ? "Clear search" : "Search states"
    );
}

function bringMapIntoView() {

    if (heroSection) {
        heroSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

if (searchForm && searchInput && searchButton) {

    searchInput.addEventListener("input", () => {
        searchForm.classList.remove("no-match");
        syncSearchButtonMode();
    });

    // Enter / submit: jump the map to the matched state.
    searchForm.addEventListener("submit", event => {

        event.preventDefault();

        const match = findStateMatch(searchInput.value);

        if (!match) {
            searchForm.classList.add("no-match");
            return;
        }

        searchForm.classList.remove("no-match");
        searchInput.value = match;
        syncSearchButtonMode();

        const pinned = selectState(match);

        if (!pinned) {
            // State is on the map but has no fraud record — show the same
            // "no data" profile the map hover would.
            renderStateCard(
                { state: match, matched: false, fraud_rate: null },
                { pinned: false }
            );
        }

        bringMapIntoView();
        searchInput.blur();
    });

    // Icon button: clear when there's text, otherwise act as "search".
    searchButton.addEventListener("click", () => {

        if (searchForm.classList.contains("has-text")) {
            searchInput.value = "";
            searchForm.classList.remove("no-match");
            syncSearchButtonMode();
            clearSelection();
            searchInput.focus();
            return;
        }

        if (searchInput.value.trim()) {
            searchForm.requestSubmit();
        } else {
            searchInput.focus();
        }
    });

    syncSearchButtonMode();
}

if (resetViewButton) {

    resetViewButton.addEventListener("click", () => {

        clearSelection();

        if (searchInput) {
            searchInput.value = "";
            syncSearchButtonMode();
        }

        if (leafletMap && stateLayer) {
            leafletMap.invalidateSize();
            leafletMap.fitBounds(stateLayer.getBounds(), getMapFitPadding());
        }

        bringMapIntoView();
    });
}


/* =========================
   SCROLL ANIMATION
========================= */

const heroContent = document.querySelector(".hero-content");
const stateCardEl = document.getElementById("state-card");
const mapLegendEl = document.getElementById("map-legend");

window.addEventListener("scroll", () => {

    const scrollY = window.scrollY;
    const opacity = 1 - scrollY / 50;

    heroContent.style.opacity = Math.max(opacity, 0);
    heroContent.style.transform = `translateY(calc(-50% - ${scrollY * 0.2}px))`;

    // The state card and legend are fixed to the viewport so they stay put
    // while hovering inside the hero. Once the hero scrolls out of view they
    // must not linger on top of the sections below (e.g. a hover that never
    // received its mouseout, or a pinned state left open).
    const pastHero = scrollY > window.innerHeight * 0.85;

    stateCardEl.classList.toggle("scroll-hidden", pastHero);
    mapLegendEl.classList.toggle("scroll-hidden", pastHero);

});


/* =========================
   OVERVIEW SECTION
========================= */

function loadOverview() {

    setSectionStatus("overview-status", "Loading overview metrics…", "loading");

    Promise.all([
        apiGet("/api/fraud/summary"),
        apiGet("/api/model/metrics"),
    ])
        .then(([fraudSummary, modelMetrics]) => {

            const overview = fraudSummary.overview;
            const test = modelMetrics.metrics.test;

            document.getElementById("kpi-total-transactions").textContent =
                formatInt(overview.total_transactions);

            document.getElementById("kpi-fraud-cases").textContent =
                formatInt(overview.fraud_cases);

            document.getElementById("kpi-fraud-rate").textContent =
                formatPercent(overview.fraud_rate);

            document.getElementById("kpi-roc-auc").textContent =
                formatAuc(test.roc_auc);

            document.getElementById("kpi-pr-auc").textContent =
                formatAuc(test.pr_auc);

            setSectionStatus("overview-status", null);
        })
        .catch(error => {

            console.error("[Overview] Failed to load:", error);
            setSectionStatus("overview-status", error.message, "error");
        });
}


/* =========================
   FRAUD ANALYSIS SECTION
========================= */

function loadFraudAnalysis() {

    const cards = [
        "card-fraud-split", "card-fraud-amount",
        "card-fraud-hour", "card-fraud-day",
        "card-fraud-merchant", "card-fraud-device",
    ];
    cards.forEach(id => setChartState(id, "loading", "Loading…"));

    apiGet("/api/fraud/summary")
        .then(summary => {

            renderFraudSplit(summary.overview);
            renderAmountDistribution(summary.amount_distribution);
            renderRateByHour(summary.by_hour);
            renderRateByDay(summary.by_day_of_week);
            renderRateByCategory("fraud-by-merchant", summary.by_merchant_category, "Merchant Category");
            renderRateByCategory("fraud-by-device", summary.by_device_type, "Device Type");

            cards.forEach(id => setChartState(id, "ready"));
        })
        .catch(error => {

            console.error("[Fraud Analysis] Failed to load:", error);
            setSectionStatus("fraud-analysis-status", error.message, "error");
            cards.forEach(id => setChartState(id, "error", "No data — backend unavailable."));
        });
}

function renderFraudSplit(overview) {

    Plotly.newPlot("fraud-split", [{
        type: "pie",
        labels: ["Legitimate", "Fraud"],
        values: [overview.legitimate_cases, overview.fraud_cases],
        hole: 0.55,
        marker: { colors: [CHART_LEGIT, CHART_FRAUD] },
        textinfo: "label+percent",
        textfont: { color: "#ffffff" },
    }], {
        ...plotTheme,
        title: "",
        autosize: true,
        height: 460,
        showlegend: false,
    }, PLOTLY_CONFIG);
}

function renderAmountDistribution(dist) {

    Plotly.newPlot("fraud-amount-distribution", [
        {
            x: dist.bin_centers,
            y: dist.legit_counts,
            type: "bar",
            name: "Legitimate",
            marker: { color: CHART_NEUTRAL_FILL },
        },
        {
            x: dist.bin_centers,
            y: dist.fraud_counts,
            type: "bar",
            name: "Fraud",
            marker: { color: CHART_FRAUD },
        },
    ], {
        ...plotTheme,
        autosize: true,
        height: 460,
        barmode: "overlay",
        xaxis: { ...plotTheme.xaxis, title: "Transaction Amount" },
        yaxis: { ...plotTheme.yaxis, title: "Count" },
        legend: { font: { color: "#ffffff" } },
    }, PLOTLY_CONFIG);
}

function renderRateByHour(byHour) {

    Plotly.newPlot("fraud-by-hour", [{
        x: byHour.map(r => r.hour),
        y: byHour.map(r => r.fraud_rate * 100),
        type: "scatter",
        mode: "lines+markers",
        line: { color: CHART_FRAUD, width: 3, shape: "spline" },
        marker: { color: "#ffffff", size: 6 },
        fill: "tozeroy",
        fillcolor: CHART_FRAUD_FILL,
    }], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: "Hour of Day", dtick: 2 },
        yaxis: { ...plotTheme.yaxis, title: "Fraud Rate (%)" },
    }, PLOTLY_CONFIG);
}

function renderRateByDay(byDay) {

    Plotly.newPlot("fraud-by-day", [{
        x: byDay.map(r => r.day),
        y: byDay.map(r => r.fraud_rate * 100),
        type: "bar",
        marker: { color: CHART_ACCENT },
    }], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: "" },
        yaxis: { ...plotTheme.yaxis, title: "Fraud Rate (%)" },
    }, PLOTLY_CONFIG);
}

function renderRateByCategory(targetId, records, axisTitle) {

    const sorted = [...records].sort((a, b) => b.fraud_rate - a.fraud_rate);

    Plotly.newPlot(targetId, [{
        x: sorted.map(r => r.category),
        y: sorted.map(r => r.fraud_rate * 100),
        type: "bar",
        marker: { color: CHART_FRAUD },
    }], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: axisTitle },
        yaxis: { ...plotTheme.yaxis, title: "Fraud Rate (%)" },
    }, PLOTLY_CONFIG);
}


/* =========================
   MODEL PERFORMANCE SECTION
========================= */

function loadModelPerformance() {

    const cards = ["card-roc", "card-pr", "card-cm", "card-density", "card-importance"];
    cards.forEach(id => setChartState(id, "loading", "Loading…"));

    Promise.all([
        apiGet("/api/model/metrics"),
        apiGet("/api/model/explainability"),
    ])
        .then(([modelMetrics, explainability]) => {

            const test = modelMetrics.metrics.test;

            renderModelKpis(test);
            renderCaveatBanner(explainability);
            renderRocCurve(test);
            renderPrCurve(test);
            renderConfusionMatrix(test.confusion_matrix);
            renderDensity(test.probability_density);
            renderFeatureImportance(explainability.feature_impact, explainability.feature_importance);

            cards.forEach(id => setChartState(id, "ready"));
        })
        .catch(error => {

            console.error("[Model Performance] Failed to load:", error);
            setSectionStatus("model-status", error.message, "error");
            cards.forEach(id => setChartState(id, "error", "No data — backend unavailable."));
        });
}

function renderModelKpis(test) {

    document.getElementById("kpi-model-roc-auc").textContent = formatAuc(test.roc_auc);
    document.getElementById("kpi-model-pr-auc").textContent = formatAuc(test.pr_auc);
    document.getElementById("kpi-model-precision").textContent = formatPercent(test.precision);
    document.getElementById("kpi-model-recall").textContent = formatPercent(test.recall);
    document.getElementById("kpi-model-f1").textContent = formatAuc(test.f1_score);
    document.getElementById("kpi-model-accuracy").textContent = formatPercent(test.accuracy);
}

function renderCaveatBanner(explainability) {

    const banner = document.getElementById("model-caveat-banner");
    const text = document.getElementById("model-caveat-text");
    const generalizes = explainability.model_generalization.generalizes;

    banner.style.display = "flex";
    banner.classList.toggle("ok", generalizes);
    text.innerHTML =
        `<strong>${generalizes ? "Model generalizes." : "Model does not generalize."}</strong> ` +
        explainability.caveat;
}

function renderRocCurve(test) {

    const traces = [
        {
            x: [0, 1], y: [0, 1],
            mode: "lines", name: "Random baseline",
            line: { color: CHART_BASELINE, dash: "dash" },
        },
    ];

    if (test.roc_curve) {
        traces.unshift({
            x: test.roc_curve.fpr,
            y: test.roc_curve.tpr,
            mode: "lines",
            name: `Model (AUC = ${formatAuc(test.roc_auc)})`,
            line: { color: CHART_ACCENT, width: 4 },
        });
    }

    Plotly.newPlot("roc-test", traces, {
        ...plotTheme,
        autosize: true,
        height: 480,
        xaxis: { ...plotTheme.xaxis, title: "False Positive Rate" },
        yaxis: { ...plotTheme.yaxis, title: "True Positive Rate" },
        legend: { font: { color: "#ffffff" }, x: 0.35, y: 0.05 },
    }, PLOTLY_CONFIG);
}

function renderPrCurve(test) {

    const traces = [
        {
            x: [0, 1], y: [test.pr_auc_baseline, test.pr_auc_baseline],
            mode: "lines", name: `Baseline (${formatPercent(test.pr_auc_baseline)})`,
            line: { color: CHART_BASELINE, dash: "dash" },
        },
    ];

    if (test.pr_curve) {
        traces.unshift({
            x: test.pr_curve.recall,
            y: test.pr_curve.precision,
            mode: "lines",
            name: `Model (PR-AUC = ${formatAuc(test.pr_auc)})`,
            line: { color: CHART_ACCENT, width: 4 },
        });
    }

    Plotly.newPlot("pr-test", traces, {
        ...plotTheme,
        autosize: true,
        height: 480,
        xaxis: { ...plotTheme.xaxis, title: "Recall" },
        yaxis: { ...plotTheme.yaxis, title: "Precision" },
        legend: { font: { color: "#ffffff" }, x: 0.3, y: 0.95 },
    }, PLOTLY_CONFIG);
}

function renderConfusionMatrix(cm) {

    Plotly.newPlot("cm-test", [{
        z: cm,
        type: "heatmap",
        colorscale: [[0, "#F3E4C6"], [1, "#FF9933"]],
        showscale: false,
        hoverinfo: "skip",
    }], {
        ...plotTheme,
        autosize: true,
        height: 480,
        annotations: [
            { x: 0, y: 0, text: `TN<br>${formatInt(cm[0][0])}`, showarrow: false, font: { color: "#17130F", size: 22 } },
            { x: 1, y: 0, text: `FP<br>${formatInt(cm[0][1])}`, showarrow: false, font: { color: "#17130F", size: 22 } },
            { x: 0, y: 1, text: `FN<br>${formatInt(cm[1][0])}`, showarrow: false, font: { color: "#17130F", size: 22 } },
            { x: 1, y: 1, text: `TP<br>${formatInt(cm[1][1])}`, showarrow: false, font: { color: "#17130F", size: 22 } },
        ],
        xaxis: { title: "Predicted Label", tickvals: [0, 1], ticktext: ["Legit", "Fraud"], showgrid: false, zeroline: false },
        yaxis: { title: "True Label", tickvals: [0, 1], ticktext: ["Legit", "Fraud"], autorange: "reversed", showgrid: false, zeroline: false },
    }, PLOTLY_CONFIG);
}

function renderDensity(density) {

    if (!density) {
        document.getElementById("density-test").innerHTML =
            '<p style="color:rgba(255,255,255,0.5)">No density data available.</p>';
        return;
    }

    Plotly.newPlot("density-test", [
        {
            x: density.legit_x, y: density.legit_y,
            mode: "lines", name: "Legitimate",
            line: { color: CHART_LEGIT, width: 4, shape: "spline" },
            fill: "tozeroy", fillcolor: CHART_LEGIT_FILL,
        },
        {
            x: density.fraud_x, y: density.fraud_y,
            mode: "lines", name: "Fraud",
            line: { color: CHART_FRAUD, width: 4, shape: "spline" },
            fill: "tozeroy", fillcolor: CHART_FRAUD_FILL,
        },
    ], {
        ...plotTheme,
        autosize: true,
        height: 480,
        xaxis: { ...plotTheme.xaxis, title: "Predicted Fraud Probability" },
        yaxis: { ...plotTheme.yaxis, title: "Density" },
        legend: { font: { color: "#ffffff" } },
    }, PLOTLY_CONFIG);
}

function renderFeatureImportance(impact, importance) {

    // feature_impact carries direction; fall back to feature_importance
    // (native/SHAP magnitude only) if impact is unavailable.
    const source = (impact && impact.length ? impact : importance || [])
        .slice(0, 15)
        .slice()
        .reverse(); // ascending, so the top feature renders at the top of a horizontal bar

    if (source.length === 0) {
        document.getElementById("feature-importance").innerHTML =
            '<p style="color:rgba(255,255,255,0.5)">No feature importance data available.</p>';
        return;
    }

    const hasDirection = source.every(r => "direction" in r);

    Plotly.newPlot("feature-importance", [{
        x: source.map(r => r.mean_abs_shap != null ? r.mean_abs_shap : r.importance),
        y: source.map(r => r.feature),
        type: "bar",
        orientation: "h",
        marker: {
            color: hasDirection
                ? source.map(r => r.direction === "increases_fraud_risk" ? CHART_FRAUD : CHART_LEGIT)
                : CHART_FRAUD,
        },
    }], {
        ...plotTheme,
        autosize: true,
        height: 560,
        margin: { l: 220 },
        xaxis: { ...plotTheme.xaxis, title: "Mean |SHAP value|" },
        yaxis: { ...plotTheme.yaxis, showgrid: false, automargin: true },
    }, PLOTLY_CONFIG);
}


/* =========================
   BOOTSTRAP
========================= */

loadOverview();
loadFraudAnalysis();
loadModelPerformance();
