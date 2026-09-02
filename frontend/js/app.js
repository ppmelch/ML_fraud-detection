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

// Bounding box of the contiguous United States. The GeoJSON also carries
// Alaska, Hawaii and Puerto Rico, whose geometry (Alaska crosses the
// antimeridian) would blow up a plain `getBounds()` fit — so every fit
// targets this fixed box instead.
const US_CONTIGUOUS_BOUNDS = L.latLngBounds([[24.4, -125.0], [49.4, -66.9]]);


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

let topAnomalyRows = [];                       // raw /api/model/metrics test.top_anomalies
let topAnomalySort = { key: "anomaly_score", dir: "desc" };


/* =========================
   API CLIENT
========================= */

/**
 * GET a JSON endpoint from the anomaly detection API.
 *
 * @param {string} path - API path, e.g. "/api/anomaly/summary".
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
    if (value == null || Number.isNaN(value)) return "—";
    return Number(value).toLocaleString("en-US");
}

function formatPercent(value, decimals = 2) {
    if (value == null || Number.isNaN(value)) return "—";
    return `${(value * 100).toFixed(decimals)}%`;
}

function formatCurrency(value) {
    if (value == null || Number.isNaN(value)) return "—";
    return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

function formatScore(value) {
    if (value == null || Number.isNaN(value)) return "—";
    return Number(value).toFixed(3);
}

function formatHour(hour) {
    if (hour == null || Number.isNaN(hour)) return "—";
    return `${String(hour).padStart(2, "0")}:00`;
}

function formatTimestamp(value) {
    if (!value) return "—";
    return String(value).replace("T", " ").slice(0, 16);
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
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
        color: "#ECE4D2",
        family: "Inter, sans-serif"
    },
    xaxis: {
        showgrid: false,
        zeroline: false,
    },
    yaxis: {
        showgrid: true,
        gridcolor: "rgba(236,228,210,0.16)",
        zeroline: false
    }
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };


/* =========================
   CHART PALETTE
   Mirrors the CSS custom properties in css/style.css. Vintage Americana:
   oxblood / barn red for flagged / anomalous series, dusty slate-blue for
   the normal population, dusty denim for the primary model series,
   antique brass for reference lines, aged taupe for neutral (train) series.
========================= */

const CHART_FLAGGED = "#9E3B32";       // oxblood / barn red — flagged / anomalous
const CHART_FLAGGED_FILL = "rgba(158,59,50,0.22)";
const CHART_NORMAL = "#7E8CA3";        // dusty slate-blue — normal population
const CHART_NORMAL_FILL = "rgba(126,140,163,0.22)";
const CHART_ACCENT = "#47679B";        // dusty denim — primary model series
const CHART_BASELINE = "#C9A24B";      // antique brass — reference / threshold lines
const CHART_NEUTRAL = "#9A8F79";       // aged taupe — neutral series
const CHART_NEUTRAL_FILL = "rgba(154,143,121,0.3)";


/* =========================
   COLOR SCALE (anomaly rate, computed by the backend — no hardcoded cutoffs)
========================= */

/**
 * Compute the min/max anomaly rate across all states, for a relative scale.
 * Anomaly rates in this dataset sit in a narrow band, so a fixed absolute
 * scale (like a 0-1 score scale) would paint every state the same color.
 *
 * @param {Array<Object>} records - State records with an `anomaly_rate` field.
 * @returns {{min: number, max: number}}
 */
function computeColorScale(records) {

    const rates = records
        .map(r => r.anomaly_rate)
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

// Choropleth ramp — matching the --ramp-* custom properties and the legend
// gradient in css/style.css: pale bone (low) -> antique brass (mid) ->
// oxblood (high).
const SCALE_LOW = "#E6DCC6";
const SCALE_MID = "#B8894B";
const SCALE_HIGH = "#7E2B22";
const SCALE_NO_DATA = "#2A3140";

/**
 * Map an anomaly rate to a color on the project's pale bone -> antique
 * brass -> oxblood scale, relative to the observed min/max across all
 * states.
 *
 * @param {?number} rate - State anomaly rate, or null/undefined for no data.
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

    if (!record.matched) {

        infoEl.innerHTML = `<p class="no-fraud-data">No anomaly data for this state.</p>`;

    } else {

        const rows = [
            ["Anomaly Rate", formatPercent(record.anomaly_rate)],
            ["Mean Anomaly Score", formatScore(record.mean_anomaly_score)],
            ["Flagged", formatInt(record.flagged)],
            ["Transactions", formatInt(record.total_transactions)],
            ["Peak Anomaly Hour", formatHour(record.peak_anomaly_hour)],
            ["Peak Anomaly Day", record.peak_anomaly_day || "—"],
            ["Top Flagged Channel", record.most_common_flagged_channel || "—"],
            ["Top Flagged Occupation", record.top_flagged_occupation || "—"],
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
 * @returns {boolean} true if an anomaly record existed and was pinned.
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
    touchZoom: false,

    // Allow fractional zoom so fitBounds fills the wide US aspect ratio
    // instead of snapping down a whole level and leaving the map tiny.
    zoomSnap: 0

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
            fillColor: getStateColor(record ? record.anomaly_rate : null),
            fillOpacity: isSelected ? 1 : 0.8,
            weight: isSelected ? 3 : 1,
            color: isSelected ? "#ECE4D2" : "#586A82",
        });
    });
}


/**
 * Padding to bias fitBounds so the map stays clear of the hero title
 * (left) and the state card (right), without ever eating so much of a
 * narrower viewport that the map collapses to a sliver.
 *
 * The contiguous US is far wider than it is tall, so the fit is
 * width-bound: heavy left/right padding would only zoom the map further
 * out and shrink the visible band. Keep the sides light and bias
 * vertically instead — clear of the fixed navbar at the top.
 *
 * @returns {{paddingTopLeft: number[], paddingBottomRight: number[]}}
 */
function getMapFitPadding() {

    const width = window.innerWidth;

    if (width < 900) {
        return { paddingTopLeft: [20, 70], paddingBottomRight: [20, 30] };
    }

    const side = Math.min(90, width * 0.05);

    return {
        paddingTopLeft: [side, 96],
        paddingBottomRight: [side, 48],
    };
}


/* =========================
   LOAD MAP + STATE METRICS
========================= */

function loadMap() {

    setMapStatus("loading", "Loading US anomaly map…");

    Promise.all([
        fetch("data/us-states.json").then(r => {
            if (!r.ok) throw new Error(`data/us-states.json responded ${r.status}`);
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
            // never a hardcoded list of states.
            geojsonStateNames = [...geojsonNames].filter(Boolean).sort();
            populateStateSearchList(geojsonStateNames);

            if (unmatchedLookupKeys.length > 0) {
                console.warn(
                    "[US Anomaly Map] Backend state names not found in data/us-states.json:",
                    unmatchedLookupKeys
                );
            }

            if (validation && validation.ok === false) {
                console.warn(
                    "[US Anomaly Map] Backend reports an unresolved state/GeoJSON join:",
                    validation
                );
            }

            updateLegend(colorScale);

            stateLayer = L.geoJSON(geojson, {

                style: feature => {

                    const name = feature.properties[GEOJSON_NAME_PROPERTY];
                    const record = stateMetrics[name];

                    return {
                        color: "#586A82",
                        weight: 1,
                        fillColor: getStateColor(record ? record.anomaly_rate : null),
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
                                anomaly_rate: null,
                            };

                            renderStateCard(record, { pinned: name === selectedState });

                            layer.setStyle({
                                fillColor: "#5878AE",
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

            // Fit the contiguous-US box rather than the layer bounds: the
            // GeoJSON also carries Alaska / Hawaii / Puerto Rico, and
            // Alaska's antimeridian-crossing polygon would otherwise stretch
            // the fit across the whole globe.
            leafletMap.fitBounds(US_CONTIGUOUS_BOUNDS, getMapFitPadding());

            setMapStatus("hidden");
        })
        .catch(error => {

            console.error("[US Anomaly Map] Failed to load:", error);
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
        leafletMap.fitBounds(US_CONTIGUOUS_BOUNDS, getMapFitPadding());

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
            // State is on the map but has no anomaly record — show the same
            // "no data" profile the map hover would.
            renderStateCard(
                { state: match, matched: false, anomaly_rate: null },
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
            leafletMap.fitBounds(US_CONTIGUOUS_BOUNDS, getMapFitPadding());
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

    apiGet("/api/anomaly/summary")
        .then(summary => {

            const overview = summary.overview;

            document.getElementById("kpi-total-transactions").textContent =
                formatInt(overview.total_transactions);

            document.getElementById("kpi-accounts").textContent =
                formatInt(overview.total_accounts);

            document.getElementById("kpi-flagged").textContent =
                formatInt(overview.flagged_transactions);

            document.getElementById("kpi-flagged-rate").textContent =
                formatPercent(overview.flagged_rate);

            document.getElementById("kpi-mean-score").textContent =
                formatScore(overview.mean_anomaly_score);

            setSectionStatus("overview-status", null);
        })
        .catch(error => {

            console.error("[Overview] Failed to load:", error);
            setSectionStatus("overview-status", error.message, "error");
        });
}


/* =========================
   ANOMALY ANALYSIS SECTION
========================= */

function loadAnomalyAnalysis() {

    const cards = [
        "card-score-distribution", "card-amount-distribution",
        "card-rate-hour", "card-rate-day",
        "card-rate-channel", "card-rate-type",
        "card-rate-occupation",
    ];
    cards.forEach(id => setChartState(id, "loading", "Loading…"));

    apiGet("/api/anomaly/summary")
        .then(summary => {

            renderScoreDistribution(summary.score_distribution);
            renderAmountDistribution(summary.amount_distribution);
            renderRateByBucket("rate-by-hour", summary.by_hour, "Hour of Day", "scatter");
            renderRateByBucket("rate-by-day", summary.by_day_of_week, "", "bar");
            renderRateByCategory("rate-by-channel", summary.by_channel, "Channel");
            renderRateByCategory("rate-by-type", summary.by_transaction_type, "Transaction Type");
            renderRateByCategory("rate-by-occupation", summary.by_occupation, "Customer Occupation");

            cards.forEach(id => setChartState(id, "ready"));
        })
        .catch(error => {

            console.error("[Anomaly Analysis] Failed to load:", error);
            setSectionStatus("anomaly-analysis-status", error.message, "error");
            cards.forEach(id => setChartState(id, "error", "No data — backend unavailable."));
        });
}

function renderScoreDistribution(dist) {

    Plotly.newPlot("score-distribution", [{
        x: dist.bin_centers,
        y: dist.counts,
        type: "bar",
        marker: { color: CHART_FLAGGED },
    }], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: "Anomaly Score" },
        yaxis: { ...plotTheme.yaxis, title: "Transactions" },
    }, PLOTLY_CONFIG);
}

function renderAmountDistribution(dist) {

    Plotly.newPlot("amount-distribution", [
        {
            x: dist.bin_centers,
            y: dist.normal_counts,
            type: "bar",
            name: "Normal",
            marker: { color: CHART_NEUTRAL_FILL },
        },
        {
            x: dist.bin_centers,
            y: dist.flagged_counts,
            type: "bar",
            name: "Flagged",
            marker: { color: CHART_FLAGGED },
        },
    ], {
        ...plotTheme,
        autosize: true,
        height: 460,
        barmode: "overlay",
        xaxis: { ...plotTheme.xaxis, title: "Transaction Amount" },
        yaxis: { ...plotTheme.yaxis, title: "Count" },
        legend: { font: { color: "#ECE4D2" } },
    }, PLOTLY_CONFIG);
}

function renderRateByBucket(targetId, records, axisTitle, mode) {

    const base = {
        x: records.map(r => r.bucket),
        y: records.map(r => r.anomaly_rate * 100),
    };

    const trace = mode === "scatter"
        ? {
            ...base,
            type: "scatter",
            mode: "lines+markers",
            line: { color: CHART_FLAGGED, width: 3, shape: "spline" },
            marker: { color: "#ffffff", size: 7 },
            fill: "tozeroy",
            fillcolor: CHART_FLAGGED_FILL,
        }
        : {
            ...base,
            type: "bar",
            marker: { color: CHART_ACCENT },
        };

    Plotly.newPlot(targetId, [trace], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: axisTitle, dtick: mode === "scatter" ? 1 : undefined },
        yaxis: { ...plotTheme.yaxis, title: "Anomaly Rate (%)" },
    }, PLOTLY_CONFIG);
}

function renderRateByCategory(targetId, records, axisTitle) {

    const sorted = [...records].sort((a, b) => b.anomaly_rate - a.anomaly_rate);

    Plotly.newPlot(targetId, [{
        x: sorted.map(r => r.category),
        y: sorted.map(r => r.anomaly_rate * 100),
        type: "bar",
        marker: { color: CHART_FLAGGED },
    }], {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: axisTitle },
        yaxis: { ...plotTheme.yaxis, title: "Anomaly Rate (%)" },
    }, PLOTLY_CONFIG);
}


/* =========================
   MODEL DIAGNOSTICS SECTION
========================= */

function loadModelDiagnostics() {

    const cards = [
        "card-score-histogram", "card-score-rank",
        "card-feature-contrast", "card-importance",
        "card-top-anomalies",
    ];
    cards.forEach(id => setChartState(id, "loading", "Loading…"));

    Promise.all([
        apiGet("/api/model/metrics"),
        apiGet("/api/model/explainability"),
    ])
        .then(([modelMetrics, explainability]) => {

            const test = modelMetrics.metrics.test;

            renderModelKpis(modelMetrics);
            renderScoreHistogram(modelMetrics.metrics.train, test);
            renderScoreRankCurve(test.score_rank_curve, modelMetrics.threshold);
            renderFeatureContrast(test.feature_contrast);
            renderFeatureImportance(explainability.feature_impact, explainability.feature_importance);
            renderTopAnomalies(test.top_anomalies);
            renderDiagnosticsNote(test.heuristic_alignment, explainability.caveat);

            cards.forEach(id => setChartState(id, "ready"));
        })
        .catch(error => {

            console.error("[Model Diagnostics] Failed to load:", error);
            setSectionStatus("model-status", error.message, "error");
            cards.forEach(id => setChartState(id, "error", "No data — backend unavailable."));
        });
}

function renderModelKpis(modelMetrics) {

    const test = modelMetrics.metrics.test;
    const pct = test.score_percentiles || {};

    document.getElementById("kpi-model-name").textContent =
        modelMetrics.model_name ? modelMetrics.model_name.replace(/_/g, " ") : "—";
    document.getElementById("kpi-threshold").textContent = formatScore(modelMetrics.threshold);
    document.getElementById("kpi-contamination").textContent = formatPercent(modelMetrics.contamination);
    document.getElementById("kpi-test-flagged-rate").textContent = formatPercent(test.flagged_rate);
    document.getElementById("kpi-test-p95").textContent = formatScore(pct.p95);
    document.getElementById("kpi-test-p99").textContent = formatScore(pct.p99);
}

function renderDiagnosticsNote(alignment, caveat) {

    const el = document.getElementById("diagnostics-note");

    if (!el) return;

    const parts = [];

    if (alignment) {
        parts.push(
            `<strong>Heuristic alignment (sanity check, not ground truth):</strong> ` +
            `Spearman ${formatScore(alignment.spearman)}, ` +
            `overlap at flagged ${formatPercent(alignment.overlap_at_flagged)}.` +
            (alignment.note ? ` ${alignment.note}` : "")
        );
    }

    if (caveat) parts.push(caveat);

    el.innerHTML = parts.map(p => `<span>${p}</span>`).join("<br><br>");
}

function renderScoreHistogram(train, test) {

    Plotly.newPlot("score-histogram", [
        {
            x: train.score_histogram.bin_centers,
            y: train.score_histogram.counts,
            type: "bar",
            name: "Train",
            marker: { color: CHART_NEUTRAL },
            opacity: 0.55,
        },
        {
            x: test.score_histogram.bin_centers,
            y: test.score_histogram.counts,
            type: "bar",
            name: "Test",
            marker: { color: CHART_FLAGGED },
            opacity: 0.65,
            yaxis: "y2",
        },
    ], {
        ...plotTheme,
        autosize: true,
        height: 460,
        barmode: "overlay",
        xaxis: { ...plotTheme.xaxis, title: "Anomaly Score" },
        yaxis: { ...plotTheme.yaxis, title: "Train count" },
        yaxis2: {
            title: "Test count",
            overlaying: "y",
            side: "right",
            showgrid: false,
            zeroline: false,
            color: "#ECE4D2",
        },
        legend: { font: { color: "#ECE4D2" } },
    }, PLOTLY_CONFIG);
}

function renderScoreRankCurve(curve, threshold) {

    if (!curve) {
        document.getElementById("score-rank").innerHTML =
            '<p style="color:rgba(255,255,255,0.5)">No score-rank curve available.</p>';
        return;
    }

    const traces = [
        {
            x: curve.rank,
            y: curve.score,
            mode: "lines",
            name: "Test scores (ranked)",
            line: { color: CHART_ACCENT, width: 4 },
        },
    ];

    if (threshold != null) {
        traces.push({
            x: [curve.rank[0], curve.rank[curve.rank.length - 1]],
            y: [threshold, threshold],
            mode: "lines",
            name: `Flag threshold (${formatScore(threshold)})`,
            line: { color: CHART_BASELINE, dash: "dash", width: 2 },
        });
    }

    Plotly.newPlot("score-rank", traces, {
        ...plotTheme,
        autosize: true,
        height: 460,
        xaxis: { ...plotTheme.xaxis, title: "Rank (most anomalous first)" },
        yaxis: { ...plotTheme.yaxis, title: "Anomaly Score" },
        legend: { font: { color: "#ECE4D2" }, x: 0.35, y: 0.95 },
    }, PLOTLY_CONFIG);
}

function renderFeatureContrast(contrast) {

    if (!contrast || contrast.length === 0) {
        document.getElementById("feature-contrast").innerHTML =
            '<p style="color:rgba(255,255,255,0.5)">No feature-contrast data available.</p>';
        return;
    }

    const source = [...contrast]
        .sort((a, b) => Math.abs(b.std_gap) - Math.abs(a.std_gap))
        .slice(0, 15)
        .reverse(); // largest gap at the top of the horizontal bar

    Plotly.newPlot("feature-contrast", [{
        x: source.map(r => r.std_gap),
        y: source.map(r => r.feature),
        type: "bar",
        orientation: "h",
        marker: {
            color: source.map(r => r.std_gap >= 0 ? CHART_FLAGGED : CHART_NORMAL),
        },
    }], {
        ...plotTheme,
        autosize: true,
        height: 520,
        margin: { l: 220 },
        xaxis: { ...plotTheme.xaxis, title: "Standardized mean gap (flagged − normal)" },
        yaxis: { ...plotTheme.yaxis, showgrid: false, automargin: true },
    }, PLOTLY_CONFIG);
}

function renderFeatureImportance(impact, importance) {

    // feature_impact carries direction; fall back to feature_importance
    // (magnitude only) if impact is unavailable.
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
        x: source.map(r => r.mean_abs_impact != null ? r.mean_abs_impact : r.importance),
        y: source.map(r => r.feature),
        type: "bar",
        orientation: "h",
        marker: {
            color: hasDirection
                ? source.map(r => r.direction === "increases_anomaly_score" ? CHART_FLAGGED : CHART_NORMAL)
                : CHART_FLAGGED,
        },
    }], {
        ...plotTheme,
        autosize: true,
        height: 520,
        margin: { l: 220 },
        xaxis: { ...plotTheme.xaxis, title: "Mean |impact on anomaly score|" },
        yaxis: { ...plotTheme.yaxis, showgrid: false, automargin: true },
    }, PLOTLY_CONFIG);
}


/* =========================
   TOP ANOMALIES TABLE
   Backend-provided rows only; the browser sorts and formats, it never
   scores. Clicking a header re-sorts the same rows.
========================= */

const TOP_ANOMALY_COLUMNS = [
    { key: "anomaly_score", label: "Score", numeric: true, format: formatScore },
    { key: "TransactionID", label: "Txn ID" },
    { key: "Transaction_Timestamp", label: "Timestamp", format: formatTimestamp },
    { key: "USState", label: "State" },
    { key: "TransactionAmount", label: "Amount", numeric: true, format: formatCurrency },
    { key: "TransactionType", label: "Type" },
    { key: "Channel", label: "Channel" },
    { key: "CustomerOccupation", label: "Occupation" },
    { key: "CustomerAge", label: "Age", numeric: true, format: formatInt },
    { key: "TransactionDuration", label: "Duration", numeric: true, format: formatInt },
    { key: "LoginAttempts", label: "Logins", numeric: true, format: formatInt },
    { key: "AccountBalance", label: "Balance", numeric: true, format: formatCurrency },
];

function renderTopAnomalies(rows) {

    topAnomalyRows = Array.isArray(rows) ? rows.slice() : [];

    if (topAnomalyRows.length === 0) {
        document.getElementById("top-anomalies").innerHTML =
            '<p style="color:rgba(255,255,255,0.5)">No top-anomaly rows available.</p>';
        return;
    }

    drawTopAnomalies();
}

function drawTopAnomalies() {

    const { key, dir } = topAnomalySort;
    const sign = dir === "asc" ? 1 : -1;

    const sorted = [...topAnomalyRows].sort((a, b) => {

        const av = a[key];
        const bv = b[key];

        if (av == null) return 1;
        if (bv == null) return -1;

        if (typeof av === "number" && typeof bv === "number") {
            return (av - bv) * sign;
        }

        return String(av).localeCompare(String(bv)) * sign;
    });

    const head = TOP_ANOMALY_COLUMNS.map(col => {
        const ariaSort = col.key === key
            ? (dir === "asc" ? "ascending" : "descending")
            : "none";
        return `<th data-key="${col.key}" aria-sort="${ariaSort}">${col.label}</th>`;
    }).join("");

    const body = sorted.map(row => {
        const cells = TOP_ANOMALY_COLUMNS.map(col => {
            const raw = row[col.key];
            const value = col.format ? col.format(raw) : (raw == null ? "—" : raw);
            return `<td class="${col.numeric ? "num" : ""}">${value}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
    }).join("");

    const target = document.getElementById("top-anomalies");

    target.innerHTML =
        `<div class="anomaly-table-wrap">` +
        `<table class="anomaly-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>` +
        `</div>`;

    target.querySelectorAll("th").forEach(th => {
        th.addEventListener("click", () => {

            const nextKey = th.dataset.key;

            if (topAnomalySort.key === nextKey) {
                topAnomalySort.dir = topAnomalySort.dir === "asc" ? "desc" : "asc";
            } else {
                topAnomalySort.key = nextKey;
                topAnomalySort.dir = "desc";
            }

            drawTopAnomalies();
        });
    });
}


/* =========================
   BOOTSTRAP
========================= */

loadOverview();
loadAnomalyAnalysis();
loadModelDiagnostics();
