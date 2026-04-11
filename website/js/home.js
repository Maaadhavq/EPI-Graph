// home.js — India map hero: click Gujarat to explore, other states dimmed

document.addEventListener("DOMContentLoaded", async function() {
    initScrollNav();

    var tooltip = document.getElementById("homeTooltip");

    try {
        var results = await Promise.all([
            apiFetch("/api/predictions"),
            d3.json("assets/geo/india_states.geojson")
        ]);

        var predictions = results[0];
        var geoData     = results[1];

        drawIndiaMap(geoData, predictions, tooltip);
        buildTicker(predictions);
        initPanel({}, {});  // sets up close / ESC handlers

    } catch(err) {
        console.error("Home init failed:", err);
        var hero = document.getElementById("mapHero");
        if (hero) hero.innerHTML += '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.9rem;pointer-events:none">Map unavailable</div>';
    }
});

// ── Scroll-aware navbar ───────────────────────────────────────────────────────
function initScrollNav() {
    var nav = document.getElementById("mainNav");
    if (!nav) return;
    window.addEventListener("scroll", function() {
        nav.classList.toggle("scrolled", window.scrollY > 60);
    }, { passive: true });
}

// ── India map ─────────────────────────────────────────────────────────────────
function drawIndiaMap(geoData, predictions, tooltip) {
    var svgEl = document.getElementById("gujaratMap");
    if (!svgEl) return;

    var W = window.innerWidth;
    var H = window.innerHeight;

    var svg = d3.select("#gujaratMap")
        .attr("viewBox", "0 0 " + W + " " + H)
        .attr("preserveAspectRatio", "xMidYMid meet");

    // Glow filter for Gujarat
    addGlowFilters(svg);

    var features  = geoData.features;
    var projection = d3.geoMercator().fitSize([W * 0.85, H * 0.90], geoData);
    var pathGen    = d3.geoPath().projection(projection);

    // ── All states (muted background) ────────────────────────────────────────
    var otherFeatures = features.filter(function(f) {
        return (f.properties.ST_NM || "").trim() !== "Gujarat";
    });

    svg.selectAll("path.india-state-other")
        .data(otherFeatures)
        .join("path")
        .attr("class", "india-state-other")
        .attr("d", pathGen)
        .on("mousemove", function(event, f) {
            var name = (f.properties.ST_NM || "").trim();
            if (!tooltip) return;
            tooltip.style.display = "block";
            tooltip.style.left = (event.clientX + 16) + "px";
            tooltip.style.top  = (event.clientY + 16) + "px";
            tooltip.innerHTML =
                "<strong>" + name + "</strong>" +
                "<div style='color:var(--text-muted);font-size:0.75rem;margin-top:4px'>Surveillance coming soon</div>";
        })
        .on("mouseleave", function() {
            if (tooltip) tooltip.style.display = "none";
        });

    // ── Gujarat — active, glowing, clickable ─────────────────────────────────
    var gujaratFeatures = features.filter(function(f) {
        return (f.properties.ST_NM || "").trim() === "Gujarat";
    });

    // Determine overall Gujarat risk from predictions
    var riskCounts = { high: 0, medium: 0, low: 0 };
    Object.values(predictions).forEach(function(p) {
        if (p.level) riskCounts[p.level] = (riskCounts[p.level] || 0) + 1;
    });
    var gujaratLevel = riskCounts.high > 0 ? "high" : riskCounts.medium > 0 ? "medium" : "low";
    var riskHex = { high: "#F85149", medium: "#D29922", low: "#3FB950" };

    svg.selectAll("path.india-state-active")
        .data(gujaratFeatures)
        .join("path")
        .attr("class", "india-state-active india-gujarat " + gujaratLevel)
        .attr("filter", glowFilter(gujaratLevel))
        .attr("d", pathGen)
        .on("mousemove", function(event) {
            if (!tooltip) return;
            var highCount = riskCounts.high || 0;
            tooltip.style.display = "block";
            tooltip.style.left = (event.clientX + 16) + "px";
            tooltip.style.top  = (event.clientY + 16) + "px";
            tooltip.innerHTML =
                "<strong>Gujarat</strong>" +
                "<span class='tt-level'><span class='tt-dot' style='background:" + (riskHex[gujaratLevel]) + "'></span>" +
                (highCount + " high-risk district" + (highCount !== 1 ? "s" : "")) + "</span>" +
                "<div style='color:var(--text-sub);font-size:0.78rem;margin-top:3px'>5 districts · Active surveillance</div>" +
                "<div style='color:var(--amber);font-size:0.7rem;margin-top:5px;font-family:var(--font-mono)'>Click to explore →</div>";
        })
        .on("mouseleave", function() {
            if (tooltip) tooltip.style.display = "none";
        })
        .on("click", function() {
            window.location.href = "map.html";
        });

    // ── "ACTIVE" label on Gujarat centroid ───────────────────────────────────
    gujaratFeatures.forEach(function(f) {
        var c = pathGen.centroid(f);
        if (c.some(isNaN)) return;

        // State name
        svg.append("text")
            .attr("class", "india-state-label")
            .attr("x", c[0]).attr("y", c[1] - 8)
            .text("Gujarat");

        // Active badge
        svg.append("text")
            .attr("class", "india-state-sublabel")
            .attr("x", c[0]).attr("y", c[1] + 10)
            .text("ACTIVE ▸");
    });

    // ── Pulse rings on Gujarat centroid ──────────────────────────────────────
    var gujaratCentroids = {};
    gujaratFeatures.forEach(function(f) {
        var c = pathGen.centroid(f);
        if (!c.some(isNaN)) gujaratCentroids["Gujarat"] = c;
    });
    var fakePreds = { "Gujarat": { level: gujaratLevel } };
    drawPulseRings(svg, gujaratCentroids, fakePreds);
}

// ── Gujarat state summary panel ───────────────────────────────────────────────
function openGujaratPanel(predictions) {
    var panel   = document.getElementById("districtPanel");
    var overlay = document.getElementById("panelOverlay");
    var body    = document.getElementById("panelBody");
    var cta     = document.getElementById("panelCta");
    if (!panel || !body) return;

    var riskHex   = { high: "#F85149", medium: "#D29922", low: "#3FB950" };
    var riskLabel = { high: "HIGH RISK", medium: "MEDIUM RISK", low: "LOW RISK" };
    var districts = ["Ahmedabad", "Gandhinagar", "Rajkot", "Surat", "Vadodara"];

    var levels = districts.map(function(d) { return (predictions[d] || {}).level || "low"; });
    var overallLevel = levels.includes("high") ? "high" : levels.includes("medium") ? "medium" : "low";

    var rows = districts.map(function(d) {
        var p = predictions[d] || {};
        var color = riskHex[p.level] || "#888";
        var arrow = p.trend === "up" ? "↑" : p.trend === "down" ? "↓" : "→";
        var trendCls = "trend-" + (p.trend || "stable");
        return '<div class="panel-district-row">' +
            '<span class="pdr-dot" style="background:' + color + ';box-shadow:0 0 5px ' + color + '66"></span>' +
            '<span class="pdr-name">' + d + '</span>' +
            '<span class="' + trendCls + '" style="font-size:0.72rem;margin-right:0.3rem">' + arrow + '</span>' +
            '<span class="pdr-lvl" style="color:' + color + '">' + (p.level || "—").toUpperCase() + '</span>' +
        '</div>';
    }).join("");

    body.innerHTML =
        '<div class="panel-district-name">Gujarat</div>' +
        '<span class="risk-badge ' + overallLevel + '" style="margin-bottom:1.25rem;display:inline-flex">' + riskLabel[overallLevel] + '</span>' +
        '<div class="panel-section-label">District Risk Levels</div>' +
        rows +
        '<div style="font-size:0.7rem;color:var(--text-muted);margin-top:1rem;font-family:var(--font-mono)">5 districts · Active surveillance</div>';

    if (cta) {
        cta.href = "map.html";
        cta.textContent = "Open Gujarat District Map \u2192";
    }

    panel.classList.add("open");
    if (overlay) overlay.classList.add("show");
}

// ── Bottom risk ticker ────────────────────────────────────────────────────────
function buildTicker(predictions) {
    var inner = document.getElementById("tickerInner");
    if (!inner) return;

    var riskHex   = { high: "#F85149", medium: "#D29922", low: "#3FB950" };
    var riskLabel = { high: "HIGH",    medium: "MED",     low: "LOW" };
    var districts = ["Ahmedabad", "Gandhinagar", "Rajkot", "Surat", "Vadodara"];

    function makeItems() {
        return districts.map(function(d) {
            var p = predictions[d] || {};
            var color = riskHex[p.level] || "#888";
            var label = riskLabel[p.level] || "—";
            var arrow = p.trend === "up" ? "↑" : p.trend === "down" ? "↓" : "→";
            var trendCls = "trend-" + (p.trend || "stable");
            var changeTxt = p.change_pct !== undefined ? Math.abs(p.change_pct) + "%" : "";
            return '<span class="ticker-item">' +
                '<span class="ticker-dot" style="background:' + color + '"></span>' +
                '<span class="ticker-name">' + d + '</span>' +
                '<span class="ticker-pill" style="color:' + color + ';font-family:var(--font-mono);font-size:0.65rem;font-weight:700;letter-spacing:0.06em;">' + label + '</span>' +
                '<span class="ticker-cases" style="color:var(--text-muted);font-family:var(--font-mono);font-size:0.7rem;">' + Math.round(p.risk || 0) + ' pred</span>' +
                '<span class="' + trendCls + '" style="margin-left:2px">' + arrow + changeTxt + '</span>' +
            '</span>';
        }).join('<span class="ticker-sep">·</span>');
    }

    inner.innerHTML = makeItems() + makeItems();
}
