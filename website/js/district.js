// district.js — District detail page logic

document.addEventListener("DOMContentLoaded", async () => {
    Chart.defaults.color = "#8888aa";
    Chart.defaults.borderColor = "#1e293b";

    const params = new URLSearchParams(window.location.search);
    const district = params.get("name") || "Ahmedabad";

    document.getElementById("districtName").textContent = district;

    try {
        const [predictions, casesData, weatherData, newsData, xaiData] = await Promise.all([
            apiFetch("/api/predictions"),
            apiFetch(`/api/cases?district=${district}`),
            apiFetch(`/api/weather?district=${district}`),
            apiFetch(`/api/news?district=${district}`),
            apiFetch(`/api/explain?district=${district}`)
        ]);

        const pred = predictions[district];

        // Header
        if (pred) {
            const badge = document.getElementById("riskBadge");
            const labelMap = { high: "High Risk", medium: "Medium Risk", low: "Low Risk" };
            badge.textContent = labelMap[pred.level];
            badge.className = `risk-badge ${pred.level}`;
            document.getElementById("caseCount").textContent = Math.round(pred.risk);
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
                            f.type === 'temporal' ? 'rgba(59, 130, 246, 0.7)' :
                            f.type === 'weather' ? 'rgba(6, 182, 212, 0.7)' :
                            f.type === 'spatial' ? 'rgba(239, 68, 68, 0.7)' :
                            'rgba(16, 185, 129, 0.7)'
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
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59,130,246,0.08)",
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointBackgroundColor: "#3b82f6"
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

        // Comparison Chart
        if (predictions) {
            const ctx2 = document.getElementById("compareChart").getContext("2d");
            const labels = Object.keys(predictions);
            const values = Object.values(predictions).map(p => p.risk);
            const colorMap = { high: "#f43f5e", medium: "#f59e0b", low: "#10b981" };

            new Chart(ctx2, {
                type: "bar",
                data: {
                    labels,
                    datasets: [{
                        data: values,
                        backgroundColor: labels.map(l => {
                            const p = predictions[l];
                            const base = colorMap[p.level] || "#3b82f6";
                            return l === district ? base + "cc" : base + "33";
                        }),
                        borderColor: labels.map(l => {
                            const p = predictions[l];
                            return colorMap[p.level] || "#3b82f6";
                        }),
                        borderWidth: 1.5,
                        borderRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: "#1e293b" } },
                        y: { beginAtZero: true, max: 100, grid: { color: "#1e293b" }, title: { display: true, text: "Predicted Cases" } }
                    }
                }
            });
        }

    } catch (e) {
        console.error("District page error:", e);
    }
});
