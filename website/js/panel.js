// panel.js — shared utilities: glow filters, pulse rings, count-up, district panel

// ── Count-up animation ────────────────────────────────────────────────────────
function countUp(element, target, duration) {
    duration = duration || 1200;
    var start = performance.now();
    function step(now) {
        var t = Math.min((now - start) / duration, 1);
        var eased = 1 - Math.pow(1 - t, 3); // cubic ease-out
        element.textContent = Math.round(eased * target);
        if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ── SVG glow filter definitions ───────────────────────────────────────────────
function addGlowFilters(svg) {
    var defs = svg.append("defs");

    var filters = [
        { id: "glow-hi", color: "#F85149", blur: 5, opacity: 0.6 },
        { id: "glow-md", color: "#D29922", blur: 4, opacity: 0.5 },
        { id: "glow-lo", color: "#3FB950", blur: 3, opacity: 0.35 }
    ];

    filters.forEach(function(f) {
        var filter = defs.append("filter")
            .attr("id", f.id)
            .attr("x", "-30%").attr("y", "-30%")
            .attr("width", "160%").attr("height", "160%");

        filter.append("feGaussianBlur")
            .attr("in", "SourceGraphic")
            .attr("stdDeviation", f.blur)
            .attr("result", "blur");

        var merge = filter.append("feMerge");
        merge.append("feMergeNode").attr("in", "blur");
        merge.append("feMergeNode").attr("in", "SourceGraphic");
    });
}

function glowFilter(level) {
    var map = { high: "url(#glow-hi)", medium: "url(#glow-md)", low: "url(#glow-lo)" };
    return map[level] || null;
}

// ── Pulse rings at district centroids ────────────────────────────────────────
function drawPulseRings(svg, centroids, predictions) {
    Object.keys(centroids).forEach(function(name) {
        var c = centroids[name];
        if (!c || c.some(isNaN)) return;
        var pred = predictions[name];
        var level = pred ? pred.level : "low";

        // Two rings, staggered for depth
        [0, 0.5].forEach(function(delay, i) {
            svg.append("circle")
                .attr("class", "pulse-ring " + level + (i === 1 ? " pulse-ring-2" : ""))
                .attr("cx", c[0])
                .attr("cy", c[1])
                .attr("r", 8)
                .style("animation-delay", delay + "s");
        });

        // Static center dot
        svg.append("circle")
            .attr("class", "centroid-dot")
            .attr("cx", c[0])
            .attr("cy", c[1])
            .attr("r", 4)
            .attr("fill", level === "high" ? "var(--risk-hi)" :
                          level === "medium" ? "var(--risk-md)" : "var(--risk-lo)")
            .attr("opacity", 0.9);
    });
}

// ── District floating panel ───────────────────────────────────────────────────
var _panelPredictions = {};
var _panelWeather = {};

function initPanel(predictions, weather) {
    _panelPredictions = predictions || {};
    _panelWeather = weather || {};

    var panel   = document.getElementById("districtPanel");
    var overlay = document.getElementById("panelOverlay");
    var closeBtn = document.getElementById("panelClose");

    if (!panel) return;

    function close() {
        panel.classList.remove("open");
        overlay.classList.remove("show");
    }

    if (closeBtn)  closeBtn.addEventListener("click", close);
    if (overlay)   overlay.addEventListener("click", close);

    document.addEventListener("keydown", function(e) {
        if (e.key === "Escape") close();
    });
}

function openPanel(name) {
    var panel   = document.getElementById("districtPanel");
    var overlay = document.getElementById("panelOverlay");
    var body    = document.getElementById("panelBody");
    var cta     = document.getElementById("panelCta");

    if (!panel || !body) return;

    var pred    = _panelPredictions[name] || {};
    var weather = _panelWeather[name] || _panelWeather || {};
    var level   = pred.level || "low";
    var riskHex = { high: "#F85149", medium: "#D29922", low: "#3FB950" };
    var riskColor = riskHex[level] || "#888";
    var levelLabel = { high: "HIGH RISK", medium: "MEDIUM RISK", low: "LOW RISK" };

    body.innerHTML = [
        '<div class="panel-district-name">' + name + '</div>',
        '<span class="risk-badge ' + level + '" style="margin-bottom:1.25rem;display:inline-flex">' + (levelLabel[level] || level.toUpperCase()) + '</span>',

        '<div class="panel-section-label">AI Forecast</div>',
        '<div class="panel-stat-row">',
          '<span class="panel-stat-lbl">Predicted cases</span>',
          '<span class="panel-stat-num" id="ps-predicted">0</span>' +
          (pred.uncertainty !== undefined
              ? '<span style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.7rem;margin-left:4px">± ' + pred.uncertainty + '</span>'
              : ''),
        '</div>',
        '<div class="panel-stat-row">',
          '<span class="panel-stat-lbl">4-week avg</span>',
          '<span class="panel-stat-num" id="ps-avg">0</span>',
        '</div>',
        pred.trend ? [
          '<div class="panel-stat-row">',
            '<span class="panel-stat-lbl">Week-on-week</span>',
            '<span class="panel-stat-num trend-' + pred.trend + '">' +
              (pred.trend === "up" ? "↑" : pred.trend === "down" ? "↓" : "→") +
              (pred.change_pct !== undefined ? " " + Math.abs(pred.change_pct) + "%" : "") +
            '</span>',
          '</div>'
        ].join("") : "",

        '<div class="panel-section-label" style="margin-top:1rem">Weather</div>',
        '<div class="panel-stat-row">',
          '<span class="panel-stat-lbl">Max / Min Temp</span>',
          '<span class="panel-stat-num">' + (weather.tmax || '—') + '° / ' + (weather.tmin || '—') + '°C</span>',
        '</div>',
        '<div class="panel-stat-row">',
          '<span class="panel-stat-lbl">Rainfall</span>',
          '<span class="panel-stat-num">' + (weather.rain || '0') + ' mm</span>',
        '</div>',
        '<div class="panel-stat-row">',
          '<span class="panel-stat-lbl">Humidity AM / PM</span>',
          '<span class="panel-stat-num">' + (weather.rh_am || '—') + '% / ' + (weather.rh_pm || '—') + '%</span>',
        '</div>'
    ].join("");

    // Count-up for the numbers
    var elPred = document.getElementById("ps-predicted");
    var elAvg  = document.getElementById("ps-avg");
    if (elPred) countUp(elPred, Math.round(pred.risk || 0), 900);
    if (elAvg)  countUp(elAvg,  Math.round(pred.avg4  || pred.last_cases || 0), 1100);

    if (cta) cta.href = "district.html?name=" + encodeURIComponent(name);

    panel.classList.add("open");
    overlay.classList.add("show");
}
