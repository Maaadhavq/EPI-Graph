// district.js — District detail page logic

document.addEventListener("DOMContentLoaded", async () => {
    Chart.defaults.color = "#8B949E";
    Chart.defaults.borderColor = "#21262D";
    Chart.defaults.font.family = "Inter, system-ui, sans-serif";

    const params = new URLSearchParams(window.location.search);
    const district = params.get("name") || "Ahmedabad";

    document.getElementById("districtName").textContent = district;

    // Apply skeleton loaders while data fetches
    ["tempMax","tempMin","rainfall","humidityAM","humidityPM"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add("skeleton");
    });
    ["xaiChart","caseChart","compareChart"].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.style.minHeight = "120px"; el.classList.add("skeleton"); }
    });

    try {
        const [predictions, casesData, weatherData, newsData, xaiData] = await Promise.all([
            apiFetch("/api/predictions"),
            apiFetch(`/api/cases?district=${district}`),
            apiFetch(`/api/weather?district=${district}`),
            apiFetch(`/api/news?district=${district}`),
            apiFetch(`/api/explain?district=${district}`)
        ]);

        const pred = predictions[district];

        // Remove skeletons
        ["tempMax","tempMin","rainfall","humidityAM","humidityPM","xaiChart","caseChart","compareChart"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove("skeleton");
        });

        // Header
        if (pred) {
            const badge = document.getElementById("riskBadge");
            const labelMap = { high: "High Risk", medium: "Medium Risk", low: "Low Risk" };
            badge.textContent = labelMap[pred.level];
            badge.className = `risk-badge ${pred.level}`;
            const caseEl = document.getElementById("caseCount");
            caseEl.textContent = Math.round(pred.risk);
            if (pred.uncertainty !== undefined) {
                caseEl.insertAdjacentHTML("afterend",
                    `<span class="case-uncertainty">± ${pred.uncertainty}</span>`);
            }

            // Trend subtext
            if (pred.trend) {
                const arrow = pred.trend === "up" ? "↑" : pred.trend === "down" ? "↓" : "→";
                const trendCls = "trend-" + pred.trend;
                const pct = pred.change_pct !== undefined ? " " + Math.abs(pred.change_pct) + "%" : "";
                badge.insertAdjacentHTML("afterend",
                    `<span class="district-trend-text ${trendCls}">${arrow}${pct} week-on-week</span>`);
            }
        }

        // Weather
        if (weatherData) {
            document.getElementById("tempMax").textContent = `${weatherData.tmax}°C`;
            document.getElementById("tempMin").textContent = `${weatherData.tmin}°C`;
            document.getElementById("rainfall").textContent = `${weatherData.rain}mm`;
            document.getElementById("humidityAM").textContent = `${weatherData.rh_am}%`;
            document.getElementById("humidityPM").textContent = `${weatherData.rh_pm}%`;
        }

        // Advice
        const adviceCard = document.getElementById("adviceCard");
        const level = pred ? pred.level : "low";
        adviceCard.className = `advice-card ${level}`;
        const advice = {
            high: {
                title: "Take Immediate Precautions",
                items: [
                    "Use mosquito nets while sleeping — even during daytime naps",
                    "Drain ALL standing water near your home (pots, tires, gutters)",
                    "Apply DEET-based mosquito repellent on exposed skin",
                    "Wear long sleeves and full-length trousers, especially evenings",
                    "Seek medical attention immediately if fever persists for 2+ days",
                    "Keep windows and doors screened or closed at dusk"
                ]
            },
            medium: {
                title: "Stay Alert",
                items: [
                    "Remove stagnant water from pots, containers, and drains weekly",
                    "Apply mosquito repellent before going outdoors",
                    "Use mosquito coils or electric repellents inside the home",
                    "Monitor for symptoms: fever, severe headache, body ache"
                ]
            },
            low: {
                title: "Continue Preventive Measures",
                items: [
                    "Maintain clean surroundings and dispose of waste properly",
                    "Inspect and empty any water-holding containers weekly",
                    "Use mosquito repellent as a routine precaution",
                    "Stay informed about local health updates"
                ]
            }
        };
        const a = advice[level];
        adviceCard.innerHTML = 
            '<h3>' + a.title + '</h3>' +
            '<ul>' + a.items.map(i => '<li>' + i + '</li>').join("") + '</ul>';


        // Explainable AI (XAI) Chart
        if (xaiData && xaiData.factors && xaiData.factors.length > 0) {
            const ctxXai = document.getElementById("xaiChart").getContext("2d");
            const factors = xaiData.factors;
            new Chart(ctxXai, {
                type: 'bar',
                data: {
                    labels: factors.map(f => f.name),
                    datasets: [{
                        label: 'Risk Contribution (%)',
                        data: factors.map(f => f.contribution_pct),
                        backgroundColor: factors.map(f =>
                            f.type === 'temporal'    ? 'rgba(230,135,58,0.75)' :
                            f.type === 'weather'     ? 'rgba(63,185,80,0.75)'  :
                            f.type === 'spatial'     ? 'rgba(248,81,73,0.75)'  :
                                                       'rgba(139,148,158,0.6)'
                        ),
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => ctx.raw + "% contribution"
                            }
                        }
                    },
                    scales: {
                        x: { beginAtZero: true, max: 100, title: { display: true, text: "Contribution %" } }
                    }
                }
            });
        } else {
            const chartElem = document.getElementById("xaiChart");
            if(chartElem) chartElem.parentElement.innerHTML += '<p class="no-data">Explanation data not available.</p>';
        }

        // Case Chart
        if (casesData && casesData.cases && casesData.cases.length > 0) {
            const ctx = document.getElementById("caseChart").getContext("2d");
            new Chart(ctx, {
                type: "line",
                data: {
                    labels: casesData.cases.map(c => c.date),
                    datasets: [{
                        label: "Dengue Cases",
                        data: casesData.cases.map(c => c.value),
                        borderColor: "#E6873A",
                        backgroundColor: "rgba(230,135,58,0.08)",
                        fill: true,
                        tension: 0.35,
                        pointRadius: 2,
                        pointBackgroundColor: "#E6873A"
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { maxTicksLimit: 8 } },
                        y: { beginAtZero: true, title: { display: true, text: "Cases" } }
                    }
                }
            });
        } else {
            document.getElementById("caseChart").parentElement.innerHTML += '<p class="no-data">No case history available.</p>';
        }

        // News Feed
        const newsFeed = document.getElementById("newsFeed");
        if (newsData && newsData.alerts && newsData.alerts.length > 0) {
            newsFeed.innerHTML = newsData.alerts.map(n => 
                '<div class="news-item">' +
                    '<span class="news-badge ' + (n.type === "Medical_Alert" ? "medical" : "weather") + '">' +
                        (n.type === "Medical_Alert" ? "Medical" : "Weather") +
                    '</span>' +
                    '<div class="news-content">' +
                        '<div class="news-headline">' + n.headline + '</div>' +
                        '<div class="news-date">' + n.date + '</div>' +
                    '</div>' +
                '</div>'
            ).join("");
        } else {
            newsFeed.innerHTML = '<p class="no-data">No recent alerts for this district.</p>';
        }

        // Comparison Chart — use model risk score (actual differs per district)
        if (predictions) {
            const ctx2 = document.getElementById("compareChart").getContext("2d");
            const colorHex = { high: "#F85149", medium: "#D29922", low: "#3FB950" };
            const labels = Object.keys(predictions);

            // Build a composite risk index (0-100) so bars show meaningful differences
            const maxRisk = Math.max(...Object.values(predictions).map(p => p.risk), 1);
            const maxCases = Math.max(...Object.values(predictions).map(p => p.last_cases ?? 1), 1);
            const values = labels.map(l => {
                const p = predictions[l];
                const modelPct  = (p.risk / maxRisk) * 70;
                const casesPct  = ((p.last_cases ?? 0) / maxCases) * 30;
                return Math.round(modelPct + casesPct);
            });

            new Chart(ctx2, {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        data: values,
                        backgroundColor: labels.map(l => {
                            const p = predictions[l];
                            const hex = colorHex[p.level] || "#888";
                            return l === district ? hex + "dd" : hex + "44";
                        }),
                        borderColor: labels.map(l => colorHex[predictions[l].level] || "#888"),
                        borderWidth: l => labels[l] === district ? 2 : 1,
                        borderRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    const l = labels[ctx.dataIndex];
                                    const p = predictions[l];
                                    return `Risk Index: ${ctx.raw}  |  Last: ${p.last_cases ?? p.risk} cases`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { color: "#21262D" }, ticks: { color: "#8B949E" } },
                        y: { beginAtZero: true, max: 105, grid: { color: "#21262D" },
                             ticks: { color: "#8B949E" },
                             title: { display: true, text: "Risk Index (AI)", color: "#484F58", font: { size: 11 } } }
                    }
                }
            });
        }

    } catch (e) {
        console.error("District page error:", e);
    }
});
