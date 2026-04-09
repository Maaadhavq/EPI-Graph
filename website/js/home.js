// home.js — India map with D3 + local GeoJSON, risk cards

document.addEventListener("DOMContentLoaded", async () => {
    drawIndiaMap();
});

async function drawIndiaMap() {
    const svgEl = document.getElementById("indiaMap");
    const width = 460, height = 520;

    try {
        const indiaData = await d3.json("assets/geo/india_states.geojson");
        const svg = d3.select("#indiaMap").attr("viewBox", `0 0 ${width} ${height}`);

        const projection = d3.geoMercator().fitSize([width - 10, height - 10], indiaData);
        const pathGen = d3.geoPath().projection(projection);

        svg.selectAll("path.state")
            .data(indiaData.features)
            .join("path")
            .attr("class", d => {
                const name = (d.properties.ST_NM || "").toLowerCase();
                return name.includes("gujarat") ? "india-gujarat" : "india-state";
            })
            .attr("d", pathGen)
            .on("click", (event, d) => {
                if ((d.properties.ST_NM || "").toLowerCase().includes("gujarat")) {
                    window.location.href = "map.html";
                }
            });

        // Gujarat label at centroid
        const gujFeature = indiaData.features.find(d =>
            (d.properties.ST_NM || "").toLowerCase().includes("gujarat")
        );
        if (gujFeature) {
            const [cx, cy] = pathGen.centroid(gujFeature);
            svg.append("text")
                .attr("x", cx).attr("y", cy + 4)
                .attr("text-anchor", "middle")
                .attr("fill", "rgba(255,255,255,0.9)")
                .attr("font-size", "11px")
                .attr("font-weight", "700")
                .attr("font-family", "Inter, sans-serif")
                .attr("pointer-events", "none")
                .text("Gujarat");
        }

    } catch (err) {
        console.warn("India map load failed:", err);
        d3.select("#indiaMap")
            .attr("viewBox", `0 0 ${width} ${height}`)
            .append("text")
            .attr("x", width / 2).attr("y", height / 2)
            .attr("text-anchor", "middle")
            .attr("fill", "var(--text-muted)")
            .attr("font-size", "13px")
            .text("Map unavailable offline");
    }
}


