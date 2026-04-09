const USE_MOCK = false;

const RISK_COLORS = { high: "#ef4444", medium: "#f59e0b", low: "#10b981" };

const MOCK_DATA = {
    predictions: {
        Ahmedabad:   { risk: 20.2, level: "high" },
        Gandhinagar: { risk: 19.9, level: "high" },
        Rajkot:      { risk: 8.5,  level: "low" },
        Surat:       { risk: 17.6, level: "medium" },
        Vadodara:    { risk: 27.3, level: "high" }
    },
    districts: {
        districts: [
            { name: "Ahmedabad", color: "#3b82f6" },
            { name: "Gandhinagar", color: "#8b5cf6" },
            { name: "Rajkot", color: "#06b6d4" },
            { name: "Surat", color: "#ec4899" },
            { name: "Vadodara", color: "#10b981" }
        ]
    },
    weather: {
        Ahmedabad:   { district: "Ahmedabad",   tmax: 34.2, tmin: 21.5, rain: 52.3, rh_am: 68, rh_pm: 42 },
        Gandhinagar: { district: "Gandhinagar", tmax: 33.8, tmin: 20.9, rain: 48.7, rh_am: 70, rh_pm: 44 },
        Rajkot:      { district: "Rajkot",      tmax: 33.5, tmin: 20.2, rain: 35.4, rh_am: 65, rh_pm: 38 },
        Surat:       { district: "Surat",       tmax: 32.8, tmin: 22.1, rain: 78.5, rh_am: 75, rh_pm: 55 },
        Vadodara:    { district: "Vadodara",    tmax: 33.1, tmin: 21.3, rain: 62.1, rh_am: 72, rh_pm: 48 }
    },
    connectivity: {
        edges: [
            { source: "Ahmedabad", target: "Gandhinagar", weight: 1.0 },
            { source: "Ahmedabad", target: "Vadodara", weight: 0.9 },
            { source: "Ahmedabad", target: "Surat", weight: 0.7 },
            { source: "Surat", target: "Vadodara", weight: 0.8 },
            { source: "Rajkot", target: "Ahmedabad", weight: 0.5 },
            { source: "Rajkot", target: "Surat", weight: 0.4 }
        ]
    },
    cases: {
        Ahmedabad: { district: "Ahmedabad", cases: [{ date: "2023-01-01", value: 5 }, { date: "2023-01-08", value: 12 }, { date: "2023-01-15", value: 25 }] },
        Gandhinagar: { district: "Gandhinagar", cases: [{ date: "2023-01-01", value: 4 }, { date: "2023-01-08", value: 9 }, { date: "2023-01-15", value: 20 }] },
        Rajkot: { district: "Rajkot", cases: [{ date: "2023-01-01", value: 2 }, { date: "2023-01-08", value: 3 }, { date: "2023-01-15", value: 5 }] },
        Surat: { district: "Surat", cases: [{ date: "2023-01-01", value: 6 }, { date: "2023-01-08", value: 10 }, { date: "2023-01-15", value: 18 }] },
        Vadodara: { district: "Vadodara", cases: [{ date: "2023-01-01", value: 8 }, { date: "2023-01-08", value: 15 }, { date: "2023-01-15", value: 30 }] }
    },
    news: {
        Ahmedabad: { district: "Ahmedabad", alerts: [{ date: "2023-01-14", headline: "Hospitals report surge in viral fever.", type: "Medical_Alert" }] },
        Gandhinagar: { district: "Gandhinagar", alerts: [{ date: "2023-01-12", headline: "Municipal corp starts fogging drive.", type: "Medical_Alert" }] },
        Rajkot: { district: "Rajkot", alerts: [{ date: "2023-01-10", headline: "Weather alert: Unseasonal rain expected.", type: "Weather_Alert" }] },
        Surat: { district: "Surat", alerts: [{ date: "2023-01-15", headline: "5 new dengue cases detected in city.", type: "Medical_Alert" }] },
        Vadodara: { district: "Vadodara", alerts: [{ date: "2023-01-13", headline: "Waterlogging in low-lying areas after heavy rain.", type: "Weather_Alert" }] }
    },
    explanations: {
        Ahmedabad: {
            district: "Ahmedabad", total_risk: 20.2,
            factors: [
                { name: "Recent Case Trajectory", contribution_pct: 45, type: "temporal" },
                { name: "High Rainfall Pattern", contribution_pct: 25, type: "weather" },
                { name: "Inflow from Neighbors (Surat)", contribution_pct: 20, type: "spatial" },
                { name: "Baseline Susceptibility", contribution_pct: 10, type: "demographic" }
            ]
        },
        Gandhinagar: {
            district: "Gandhinagar", total_risk: 19.9,
            factors: [
                { name: "Recent Case Trajectory", contribution_pct: 40, type: "temporal" },
                { name: "Inflow from Neighbors (Ahmedabad)", contribution_pct: 35, type: "spatial" },
                { name: "High Rainfall Pattern", contribution_pct: 15, type: "weather" },
                { name: "Baseline Susceptibility", contribution_pct: 10, type: "demographic" }
            ]
        },
        Rajkot: {
            district: "Rajkot", total_risk: 8.5,
            factors: [
                { name: "Recent Case Trajectory", contribution_pct: 50, type: "temporal" },
                { name: "Baseline Susceptibility", contribution_pct: 30, type: "demographic" },
                { name: "Moderate Rainfall Pattern", contribution_pct: 15, type: "weather" },
                { name: "Inflow from Neighbors", contribution_pct: 5, type: "spatial" }
            ]
        },
        Surat: {
            district: "Surat", total_risk: 17.6,
            factors: [
                { name: "High Rainfall Pattern", contribution_pct: 55, type: "weather" },
                { name: "Recent Case Trajectory", contribution_pct: 25, type: "temporal" },
                { name: "Baseline Susceptibility", contribution_pct: 15, type: "demographic" },
                { name: "Inflow from Neighbors", contribution_pct: 5, type: "spatial" }
            ]
        },
        Vadodara: {
            district: "Vadodara", total_risk: 27.3,
            factors: [
                { name: "Recent Case Trajectory", contribution_pct: 40, type: "temporal" },
                { name: "Inflow from Neighbors", contribution_pct: 30, type: "spatial" },
                { name: "High Rainfall Pattern", contribution_pct: 20, type: "weather" },
                { name: "Baseline Susceptibility", contribution_pct: 10, type: "demographic" }
            ]
        }
    }
};

async function apiFetch(endpoint) {
    if (USE_MOCK) {
        await new Promise(r => setTimeout(r, 200));

        if (endpoint.startsWith("/api/predictions")) return MOCK_DATA.predictions;
        if (endpoint.startsWith("/api/districts"))   return MOCK_DATA.districts;
        if (endpoint.startsWith("/api/connectivity")) return MOCK_DATA.connectivity;

        const params = new URLSearchParams(endpoint.split("?")[1] || "");
        const district = params.get("district");
        
        if (endpoint.startsWith("/api/weather"))
            return district ? MOCK_DATA.weather[district] : MOCK_DATA.weather;
        if (endpoint.startsWith("/api/news"))
            return district ? MOCK_DATA.news[district] : { alerts: [] };
        if (endpoint.startsWith("/api/cases"))
            return district ? MOCK_DATA.cases[district] : { cases: [] };
        if (endpoint.startsWith("/api/explain"))
            return district && MOCK_DATA.explanations[district] 
                ? MOCK_DATA.explanations[district] 
                : { factors: [] };

        return {};
    }

    const res = await fetch(endpoint);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

function initBreadcrumb() {
    const bc = document.getElementById("breadcrumb");
    if (!bc) return;

    const path = window.location.pathname;
    const params = new URLSearchParams(window.location.search);

    let crumbs = [{ label: "India", href: "/" }];

    if (path.includes("map.html") || path.includes("district.html")) {
        crumbs.push({ label: "Gujarat", href: "map.html" });
    }
    
    if (path.includes("district.html")) {
        const name = params.get("name") || "District";
        crumbs.push({ label: name, href: null });
    } else if (path.includes("prevention.html")) {
        crumbs.push({ label: "Prevention Guide", href: null });
    }

    bc.innerHTML = crumbs.map((c, i) => {
        const isLast = i === crumbs.length - 1;
        if (isLast) return `<span class="bc-current">${c.label}</span>`;
        return `<a href="${c.href}" class="bc-link">${c.label}</a><span class="bc-sep">›</span>`;
    }).join("");
}

document.addEventListener("DOMContentLoaded", initBreadcrumb);
