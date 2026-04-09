// map.js — Gujarat district map using D3 + local GeoJSON

// Our 5 target model districts
const TARGET_DISTRICTS = ["Ahmedabad", "Gandhinagar", "Rajkot", "Surat", "Vadodara"];

// Map from GeoJSON district names → our internal names
const DISTRICT_NAME_MAP = {
    "Ahmadabad": "Ahmedabad",
    "Ahmedabad": "Ahmedabad",
    "Gandhi Nagar": "Gandhinagar",
    "Gandhinagar": "Gandhinagar",
    "Rajkot": "Rajkot",
    "Surat": "Surat",
    "Vadodara": "Vadodara",
    "Baroda": "Vadodara"
};

function normalizeName(raw) {
    if (!raw) return null;
    const trimmed = raw.trim();
    return DISTRICT_NAME_MAP[trimmed] || null;
}

function riskClass(level) {
    return { high: "risk-h", medium: "risk-m", low: "risk-l" }[level] || null;
}

function levelColor(level) {
    return { high: "var(--risk-high)", medium: "var(--risk-medium)", low: "var(--risk-low)" }[level] || "#888";
}

document.addEventListener("DOMContentLoaded", async () => {
    const tooltip = document.getElementById("mapTooltip");

    try {
        const [predictions, connectivity, geoData] = await Promise.all([
            apiFetch("/api/predictions"),
            apiFetch("/api/connectivity"),
            d3.json("assets/geo/gujarat_districts.geojson")
        ]);

        const features = geoData.features;
        const svg = d3.select("#gujaratMap");
        const width = 500, height = 560;
        svg.attr("viewBox", `0 0 ${width} ${height}`);

        const projection = d3.geoMercator().fitSize([width - 20, height - 20], geoData);
        const pathGen = d3.geoPath().projection(projection);

        // Draw all Gujarat districts (non-model ones as muted BG)
        svg.selectAll("path.guj-other")
            .data(features)
            .join("path")
            .attr("class", d => {
                const name = normalizeName(d.properties.district || d.properties.DISTRICT || "");
                return (name && TARGET_DISTRICTS.includes(name)) ? null : "guj-other";
            })
            .filter(d => {
                const name = normalizeName(d.properties.district || d.properties.DISTRICT || "");
                return !(name && TARGET_DISTRICTS.includes(name));
            })
            .attr("d", pathGen);

        // Draw connectivity lines between target district centroids
        const centroids = {};
        features.forEach(f => {
            const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
            if (name && TARGET_DISTRICTS.includes(name)) {
                centroids[name] = pathGen.centroid(f);
            }
        });

        connectivity.edges.forEach(edge => {
            const s = centroids[edge.source];
            const t = centroids[edge.target];
            if (s && t && !s.some(isNaN) && !t.some(isNaN)) {
                svg.append("line")
                    .attr("class", "conn-line")
                    .attr("x1", s[0]).attr("y1", s[1])
                    .attr("x2", t[0]).attr("y2", t[1]);
            }
        });

        // Draw target district polygons
        const targetFeatures = features.filter(f => {
            const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
            return name && TARGET_DISTRICTS.includes(name);
        });

        svg.selectAll("path.guj-district")
            .data(targetFeatures)
            .join("path")
            .attr("class", f => {
                const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
                const pred = predictions[name];
                const rc = pred ? riskClass(pred.level) : "";
                return `guj-district ${rc}`;
            })
            .attr("d", pathGen)
            .on("mousemove", function (event, f) {
                const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
                const pred = predictions[name];
                if (!pred) return;
                tooltip.style.display = "block";
                tooltip.style.left = (event.clientX + 16) + "px";
                tooltip.style.top = (event.clientY + 16) + "px";
                tooltip.innerHTML = `
                    <strong>${name}</strong>
                    <span class="tooltip-badge" style="background:${levelColor(pred.level)}22;color:${levelColor(pred.level)}">${pred.level.toUpperCase()}</span>
                    <div style="color:var(--text-secondary);font-size:0.8rem;margin-top:4px">${Math.round(pred.risk)} predicted cases</div>
                `;
            })
            .on("mouseleave", () => { tooltip.style.display = "none"; })
            .on("click", function (event, f) {
                const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
                if (name) window.location.href = `district.html?name=${name}`;
            });

        // Labels on target districts
        targetFeatures.forEach(f => {
            const name = normalizeName(f.properties.district || f.properties.DISTRICT || "");
            if (!name) return;
            const c = pathGen.centroid(f);
            if (c.some(isNaN)) return;
            const pred = predictions[name];

            svg.append("text")
                .attr("class", "guj-label")
                .attr("x", c[0]).attr("y", c[1] - 5)
                .text(name);

            if (pred) {
                svg.append("text")
                    .attr("class", "guj-sublabel")
                    .attr("x", c[0]).attr("y", c[1] + 11)
                    .text(`${Math.round(pred.risk)} cases`);
            }
        });

        // Sidebar — district list
        const districtList = document.getElementById("districtList");
        TARGET_DISTRICTS.forEach(name => {
            const pred = predictions[name];
            if (!pred) return;
            const levelLabel = { high: "High", medium: "Medium", low: "Low" };
            const row = document.createElement("div");
            row.className = "district-row";
            row.onclick = () => window.location.href = `district.html?name=${name}`;
            row.innerHTML = `
                <div class="district-dot ${pred.level}"></div>
                <span class="district-row-name">${name}</span>
                <span class="district-row-cases">${Math.round(pred.risk)}</span>
            `;
            districtList.appendChild(row);
        });

        // Sidebar — connections
        const connList = document.getElementById("connList");
        connectivity.edges.forEach(edge => {
            const row = document.createElement("div");
            row.className = "conn-row";
            row.innerHTML = `<span>${edge.source} ↔ ${edge.target}</span><span class="conn-weight">${edge.weight.toFixed(1)}</span>`;
            connList.appendChild(row);
        });

        // Risk Overview Cards
        const cardContainer = document.getElementById("riskCards");
        if (cardContainer && predictions) {
            const levelLabel = { high: "High Risk", medium: "Medium Risk", low: "Low Risk" };
            TARGET_DISTRICTS.forEach(district => {
                const data = predictions[district];
                if (!data) return;
                const card = document.createElement("div");
                card.className = `risk-card ${data.level}`;
                card.onclick = () => window.location.href = `district.html?name=${district}`;
                card.innerHTML = `
                    <div class="rc-district">${district}</div>
                    <span class="risk-badge ${data.level}">${levelLabel[data.level]}</span>
                    <div class="rc-cases">${Math.round(data.risk)}</div>
                    <div class="rc-label">predicted cases</div>
                `;
                cardContainer.appendChild(card);
            });
        }

    } catch (err) {
        console.error("Map load failed:", err);
        document.querySelector(".map-main").innerHTML = `
            <div style="text-align:center;padding:4rem 2rem;color:var(--text-muted)">
                <p>Map could not be loaded.</p>
                <small style="font-size:0.75rem">${err.message}</small>
            </div>`;
    }
});
