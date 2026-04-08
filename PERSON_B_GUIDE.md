# Person B Guide: Frontend Website

## Your Scope
You own everything inside the `website/` folder. You will:
1. Build a public-facing dengue awareness website with 4 pages
2. Use a 3D perspective India map as the main interaction
3. Fetch all data from Person A's API endpoints
4. Never touch `.py` files, `.ipynb` files, or anything outside `website/`

## Getting Started

```bash
cd EpiGraph-AI
git checkout -b frontend-website
```

You will only edit files inside:
```
website/
├── index.html          ← Home page (3D India map + risk overview)
├── map.html            ← Gujarat district map
├── district.html       ← District detail page
├── prevention.html     ← Dengue prevention guide
├── css/
│   └── styles.css      ← All styles
├── js/
│   ├── app.js          ← Shared nav, API helper, utilities
│   ├── home.js         ← Home page logic
│   ├── map.js          ← Gujarat map logic
│   ├── district.js     ← District detail logic
│   └── data.js         ← SVG map paths (no model data)
└── assets/
    └── (icons, images)
```

---

## Design Guidelines

- **Audience**: Common people, NOT data scientists
- **Tone**: Friendly, clear, actionable — "Stay safe", "Take precautions"
- **No jargon**: No R², RMSE, loss curves. Say "risk level", "predicted cases"
- **Visual style**: Dark theme, 3D perspective maps, warm accents (red/amber = danger, green = safe)
- **Typography**: Large, readable fonts. Minimum 16px body text
- **Icons**: Use emoji or simple SVG icons for weather, symptoms, etc.

---

## Working Without the Backend (Mock Data)

Person A is building the backend in parallel. Until it's ready, use mock data.
Edit `js/app.js` to include a mock mode:

```javascript
// Set to true while developing without backend, false after merge
const USE_MOCK = true;

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
    }
};

// Universal fetch helper — uses mock when backend isn't running
async function apiFetch(endpoint) {
    if (USE_MOCK) {
        // Simulate network delay
        await new Promise(r => setTimeout(r, 200));

        if (endpoint.startsWith("/api/predictions")) return MOCK_DATA.predictions;
        if (endpoint.startsWith("/api/districts"))   return MOCK_DATA.districts;
        if (endpoint.startsWith("/api/connectivity")) return MOCK_DATA.connectivity;

        const params = new URLSearchParams(endpoint.split("?")[1] || "");
        const district = params.get("district");
        if (endpoint.startsWith("/api/weather"))
            return district ? MOCK_DATA.weather[district] : MOCK_DATA.weather;
        if (endpoint.startsWith("/api/news"))
            return { district, alerts: [] };  // Empty for mock
        if (endpoint.startsWith("/api/cases"))
            return { district, cases: [] };   // Empty for mock

        return {};
    }

    const res = await fetch(endpoint);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}
```

**Before your final commit, set `USE_MOCK = false`** so it uses the real API after merge.

---

## PAGE 1: Home (`index.html` + `js/home.js`)

### What Users See

```
┌─────────────────────────────────────────────────┐
│  NAVBAR: [Logo] EpiGraph-AI     [Map] [Guide]   │
├─────────────────────────────────────────────────┤
│                                                   │
│          ┌─────────────────────┐                 │
│          │                     │   "AI-Powered    │
│          │   3D INDIA MAP      │    Dengue        │
│          │   (Gujarat glows)   │    Early         │
│          │                     │    Warning"      │
│          │   "Click Gujarat    │                  │
│          │    to explore"      │   [5 Districts]  │
│          │                     │   [Weekly Data]  │
│          └─────────────────────┘   [AI Powered]   │
│                                                   │
├─────────────────────────────────────────────────┤
│  RISK OVERVIEW: 5 district cards                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
│  │Ahmedab.│ │Gandhin.│ │ Rajkot │ │ Surat  │    │
│  │ 🔴 HIGH│ │ 🔴 HIGH│ │ 🟢 LOW │ │ 🟡 MED │    │
│  │ 20 cases│ │ 20 cases│ │ 9 cases│ │ 18 cases│   │
│  └────────┘ └────────┘ └────────┘ └────────┘    │
├─────────────────────────────────────────────────┤
│  HOW IT WORKS                                     │
│  [1. Data Collection] → [2. AI Analysis] → [3.]  │
├─────────────────────────────────────────────────┤
│  FOOTER                                           │
└─────────────────────────────────────────────────┘
```

### 3D India Map

The map is an inline SVG with CSS perspective transforms:

```html
<div class="map-container">
    <svg class="india-map" id="indiaMap" viewBox="0 0 800 900">
        <!-- All other states (muted, decorative) -->
        <path class="state-other" d="..." />
        <!-- Gujarat (highlighted, clickable) -->
        <path class="state-gujarat" d="..." onclick="window.location.href='/map'" />
        <!-- Gujarat label -->
        <text x="145" y="480" class="state-label">Gujarat</text>
    </svg>
</div>
```

CSS for 3D effect:
```css
.map-container {
    perspective: 1200px;
    display: flex;
    justify-content: center;
    padding: 40px;
}
.india-map {
    width: 500px;
    transform: rotateX(15deg) rotateY(-10deg);
    transition: transform 0.6s ease;
    filter: drop-shadow(0 20px 40px rgba(0,0,0,0.5));
}
.india-map:hover {
    transform: rotateX(10deg) rotateY(-5deg) scale(1.02);
}
.state-other {
    fill: #1a1a2e;
    stroke: #2a2a3e;
    stroke-width: 0.5;
}
.state-gujarat {
    fill: #3b82f6;
    stroke: #60a5fa;
    stroke-width: 1.5;
    cursor: pointer;
    filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.4));
    animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 8px rgba(59,130,246,0.4)); }
    50% { filter: drop-shadow(0 0 16px rgba(59,130,246,0.7)); }
}
```

### Risk Cards

Fetch from API and render:
```javascript
// home.js
document.addEventListener("DOMContentLoaded", async () => {
    const predictions = await apiFetch("/api/predictions");
    const container = document.getElementById("riskCards");

    for (const [district, data] of Object.entries(predictions)) {
        const colors = { high: "#ef4444", medium: "#f59e0b", low: "#10b981" };
        const card = document.createElement("div");
        card.className = `risk-card risk-${data.level}`;
        card.innerHTML = `
            <h3>${district}</h3>
            <span class="risk-badge ${data.level}">${data.level.toUpperCase()}</span>
            <p class="risk-score">${Math.round(data.risk)} predicted cases</p>
        `;
        card.onclick = () => window.location.href = `/district?name=${district}`;
        container.appendChild(card);
    }
});
```

### Where to Get SVG Paths

You need SVG path data for India. Options:

1. **Simplest**: Search GitHub for "india svg map" — many open-source options exist
2. **Best quality**: Go to https://mapshaper.org, upload India shapefile from Natural Earth (naturalearthdata.com), simplify to ~5%, export as SVG, copy the path `d="..."` attributes
3. **Quick and dirty**: Use a simplified hand-drawn outline — geographic precision doesn't matter much for this use case

Put the path strings in `js/data.js`:
```javascript
const INDIA_MAP = {
    viewBox: "0 0 800 900",
    gujaratPath: "M 143 432 L 156 418 ...",  // Actual SVG path here
    otherStatesPath: "M 200 100 L ...",       // All other states combined
    gujaratLabelPos: { x: 145, y: 480 }
};
```

---

## PAGE 2: Gujarat Map (`map.html` + `js/map.js`)

### What Users See

```
┌─────────────────────────────────────────────────┐
│  NAVBAR: [Logo]   India > Gujarat      [Guide]   │
├────────────────────────────┬────────────────────┤
│                            │  DISTRICT SUMMARY  │
│    3D GUJARAT MAP          │                    │
│                            │  Ahmedabad  🔴 20  │
│    [Ahmedabad]  [Gandhin.] │  Gandhinagar🔴 20  │
│                            │  Rajkot     🟢  9  │
│    [Rajkot]                │  Surat      🟡 18  │
│                            │  Vadodara   🔴 27  │
│    [Surat]    [Vadodara]   │                    │
│                            │  ── Connections ── │
│    (colored by risk level) │  Ahm↔Gan: 1.0     │
│                            │  Ahm↔Vad: 0.9     │
└────────────────────────────┴────────────────────┘
```

### District SVG

Same 3D approach as India map. Each district is a separate `<path>`:

```html
<svg class="gujarat-map" viewBox="0 0 600 700">
    <path class="district-path risk-high" data-district="Ahmedabad" d="..." />
    <path class="district-path risk-high" data-district="Gandhinagar" d="..." />
    <path class="district-path risk-low"  data-district="Rajkot" d="..." />
    <!-- etc -->
</svg>
```

```css
.district-path.risk-high   { fill: #ef444488; stroke: #ef4444; }
.district-path.risk-medium { fill: #f59e0b88; stroke: #f59e0b; }
.district-path.risk-low    { fill: #10b98188; stroke: #10b981; }
.district-path:hover {
    filter: brightness(1.3);
    stroke-width: 2.5;
    cursor: pointer;
}
```

### Tooltips on Hover

```javascript
// map.js
document.querySelectorAll(".district-path").forEach(path => {
    path.addEventListener("mouseenter", (e) => {
        const name = path.dataset.district;
        const pred = predictions[name];
        tooltip.innerHTML = `<strong>${name}</strong><br>${pred.level.toUpperCase()} risk<br>${Math.round(pred.risk)} cases`;
        tooltip.style.display = "block";
    });
    path.addEventListener("click", () => {
        window.location.href = `/district?name=${path.dataset.district}`;
    });
});
```

### Where to Get Gujarat District SVGs

Same process as India map but for Gujarat Level 2 administrative boundaries:
1. Download Gujarat district shapefile from GADM (gadm.org) or DataMeet GitHub
2. Open in mapshaper.org → filter to the 5 districts → simplify → export SVG
3. Copy each district's path `d="..."` into `js/data.js`

For the remaining ~28 Gujarat districts not in the model, combine them into one background `<path>` with muted styling.

---

## PAGE 3: District Detail (`district.html` + `js/district.js`)

### What Users See

```
┌─────────────────────────────────────────────────┐
│  NAVBAR: [Logo]  India > Gujarat > Ahmedabad     │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌─ AHMEDABAD ────────────────────────────────┐  │
│  │  🔴 HIGH RISK          20 predicted cases   │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─ WEATHER ─────────────────────────────────┐   │
│  │  🌡 34°C max  ❄ 22°C min  🌧 52mm rain    │   │
│  │  💧 68% AM humidity   🌅 42% PM humidity   │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌─ WEEKLY CASES (2010-2013) ────────────────┐   │
│  │  📈 Line chart (Chart.js)                  │   │
│  │  Shows dengue cases over time              │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌─ WHAT YOU SHOULD DO ──────────────────────┐   │
│  │  ⚠ Take immediate precautions:            │   │
│  │  • Use mosquito nets while sleeping        │   │
│  │  • Drain all standing water near home      │   │
│  │  • Seek medical help if fever persists     │   │
│  │  • Use mosquito repellent                  │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌─ NEWS & ALERTS ───────────────────────────┐   │
│  │  🏥 Medical: Hospitals report fever surge  │   │
│  │  🌧 Weather: Heavy monsoon and waterlog.  │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
│  ┌─ COMPARE WITH OTHER DISTRICTS ────────────┐   │
│  │  Bar chart (Chart.js) — all 5 districts    │   │
│  └────────────────────────────────────────────┘   │
│                                                   │
└─────────────────────────────────────────────────┘
```

### Key Logic (`district.js`)

```javascript
document.addEventListener("DOMContentLoaded", async () => {
    const params = new URLSearchParams(window.location.search);
    const district = params.get("name") || "Ahmedabad";

    // Fetch all data in parallel
    const [predictions, cases, weather, news] = await Promise.all([
        apiFetch("/api/predictions"),
        apiFetch(`/api/cases?district=${district}`),
        apiFetch(`/api/weather?district=${district}`),
        apiFetch(`/api/news?district=${district}`)
    ]);

    const pred = predictions[district];

    // 1. Render header
    document.getElementById("districtName").textContent = district;
    document.getElementById("riskBadge").textContent = pred.level.toUpperCase();
    document.getElementById("riskBadge").className = `risk-badge ${pred.level}`;
    document.getElementById("caseCount").textContent = `${Math.round(pred.risk)} predicted cases`;

    // 2. Render weather
    document.getElementById("tempMax").textContent = `${weather.tmax}°C`;
    document.getElementById("tempMin").textContent = `${weather.tmin}°C`;
    document.getElementById("rainfall").textContent = `${weather.rain}mm`;
    document.getElementById("humidityAM").textContent = `${weather.rh_am}%`;
    document.getElementById("humidityPM").textContent = `${weather.rh_pm}%`;

    // 3. Render case chart
    if (cases.cases && cases.cases.length > 0) {
        new Chart(document.getElementById("caseChart"), {
            type: "line",
            data: {
                labels: cases.cases.map(c => c.date),
                datasets: [{
                    label: "Dengue Cases",
                    data: cases.cases.map(c => c.value),
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59,130,246,0.1)",
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { maxTicksLimit: 12 } },
                    y: { beginAtZero: true, title: { display: true, text: "Cases" } }
                }
            }
        });
    }

    // 4. Render advice based on risk level
    const adviceBox = document.getElementById("advice");
    const advice = {
        high: {
            title: "Take Immediate Precautions",
            items: [
                "Use mosquito nets while sleeping",
                "Drain ALL standing water near your home",
                "Use mosquito repellent on exposed skin",
                "Wear long sleeves and pants, especially in the evening",
                "Seek medical help immediately if you have fever for 2+ days",
                "Keep windows and doors closed or use screens"
            ]
        },
        medium: {
            title: "Stay Alert",
            items: [
                "Remove stagnant water from pots, tires, and drains",
                "Apply mosquito repellent before going outdoors",
                "Use mosquito coils or electric repellents at home",
                "Monitor for symptoms: fever, headache, body pain"
            ]
        },
        low: {
            title: "Continue Preventive Measures",
            items: [
                "Maintain clean surroundings",
                "Check for any water accumulation weekly",
                "Use mosquito repellent as a precaution",
                "Stay informed about local health updates"
            ]
        }
    };

    const a = advice[pred.level];
    adviceBox.innerHTML = `
        <h3>${a.title}</h3>
        <ul>${a.items.map(i => `<li>${i}</li>`).join("")}</ul>
    `;

    // 5. Render news feed
    const newsFeed = document.getElementById("newsFeed");
    if (news.alerts && news.alerts.length > 0) {
        newsFeed.innerHTML = news.alerts.map(n => `
            <div class="news-item ${n.type === 'Medical_Alert' ? 'medical' : 'weather'}">
                <span class="news-type">${n.type === 'Medical_Alert' ? '🏥 Medical' : '🌧 Weather'}</span>
                <p>${n.headline}</p>
                <small>${n.date}</small>
            </div>
        `).join("");
    } else {
        newsFeed.innerHTML = '<p class="no-data">No recent alerts for this district.</p>';
    }

    // 6. Comparison bar chart
    new Chart(document.getElementById("compareChart"), {
        type: "bar",
        data: {
            labels: Object.keys(predictions),
            datasets: [{
                data: Object.values(predictions).map(p => p.risk),
                backgroundColor: Object.keys(predictions).map(d =>
                    d === district ? "#3b82f6" : "#1a1a2e"
                ),
                borderColor: Object.keys(predictions).map(d =>
                    d === district ? "#60a5fa" : "#2a2a3e"
                ),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, title: { display: true, text: "Predicted Cases" } } }
        }
    });
});
```

---

## PAGE 4: Prevention Guide (`prevention.html`)

This is **fully static** — no API calls needed. Just HTML + CSS.

### Sections to Build

**1. Hero Banner**
```
"Protect Yourself from Dengue"
Brief: Dengue is a mosquito-borne disease that affects millions in India each year.
```

**2. What is Dengue?**
- Spread by Aedes mosquitoes (active during day)
- Symptoms appear 4-10 days after bite
- Common during and after monsoon season

**3. Symptoms** (use icons/emoji)
- 🤒 High fever (40°C / 104°F)
- 🤕 Severe headache
- 😣 Pain behind the eyes
- 💪 Muscle and joint pain
- 🤢 Nausea and vomiting
- 🔴 Skin rash (appears 2-5 days after fever)

**4. Prevention Tips** (illustrated cards)
- 🚰 Drain stagnant water — empty pots, tires, gutters
- 🛏 Sleep under mosquito nets
- 🧴 Apply mosquito repellent (DEET-based)
- 👕 Wear long sleeves and pants
- 🪟 Install window/door screens
- 🧹 Keep surroundings clean
- 🕐 Be extra careful at dawn and dusk

**5. When to See a Doctor** (warning box with red accent)
Seek IMMEDIATE medical attention if you have:
- Severe stomach pain
- Persistent vomiting
- Bleeding from nose or gums
- Blood in vomit or stool
- Extreme fatigue or restlessness
- Difficulty breathing

**6. Emergency Contacts**
- Gujarat State Health Helpline: 104
- National Health Helpline: 1800-180-1104
- Nearest Government Hospital emergency number

---

## Shared Navigation Bar

Every page should have the same navbar. Put this in each HTML file:

```html
<nav class="navbar">
    <a href="/" class="nav-logo">
        <span class="logo-icon">🧬</span>
        <span class="logo-text">EpiGraph<span class="logo-ai">AI</span></span>
    </a>

    <div class="nav-breadcrumb" id="breadcrumb">
        <!-- Populated by JS based on current page -->
    </div>

    <div class="nav-links">
        <a href="/map" class="nav-link">District Map</a>
        <a href="/prevention" class="nav-link">Prevention Guide</a>
    </div>
</nav>
```

Breadcrumb logic in `app.js`:
```javascript
function initBreadcrumb() {
    const bc = document.getElementById("breadcrumb");
    const path = window.location.pathname;
    const params = new URLSearchParams(window.location.search);

    let crumbs = [{ label: "India", href: "/" }];

    if (path === "/map" || path === "/district") {
        crumbs.push({ label: "Gujarat", href: "/map" });
    }
    if (path === "/district") {
        const name = params.get("name") || "District";
        crumbs.push({ label: name, href: null });
    }

    bc.innerHTML = crumbs.map((c, i) => {
        const isLast = i === crumbs.length - 1;
        if (isLast) return `<span class="bc-current">${c.label}</span>`;
        return `<a href="${c.href}" class="bc-link">${c.label}</a><span class="bc-sep">›</span>`;
    }).join("");
}

document.addEventListener("DOMContentLoaded", initBreadcrumb);
```

---

## CSS Structure (`css/styles.css`)

Organize your stylesheet in this order:

```css
/* 1. CSS Variables */
:root {
    --bg-primary: #09090b;
    --bg-card: #111113;
    --bg-card-hover: #1a1a1f;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --border: #27272a;
    --accent-blue: #3b82f6;
    --risk-high: #ef4444;
    --risk-medium: #f59e0b;
    --risk-low: #10b981;
    --font-main: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* 2. Reset + Base */
/* 3. Typography */
/* 4. Navbar + Breadcrumb */
/* 5. 3D Map Styles */
/* 6. Risk Cards */
/* 7. District Detail Sections */
/* 8. Prevention Page */
/* 9. Footer */
/* 10. Responsive (@media queries) */
/* 11. Animations */
```

---

## Before Committing

```bash
# Set mock mode off before final commit
# In js/app.js, change:
const USE_MOCK = false;
```

## Commit and Push

```bash
cd EpiGraph-AI
git add website/
git commit -m "Build frontend: 3D India map, Gujarat districts, detail pages, prevention guide"
git push -u origin frontend-website
```

Then create a Pull Request on GitHub: `frontend-website` → `main`.

---

## Checklist Before PR

- [ ] Home page: 3D India map renders, Gujarat glows and is clickable
- [ ] Home page: 5 risk cards show correct data
- [ ] Map page: Gujarat districts colored by risk, clickable
- [ ] District page: Chart, weather, news, advice all render
- [ ] District page: Breadcrumb shows India > Gujarat > DistrictName
- [ ] Prevention page: All sections readable and well-formatted
- [ ] Navbar consistent across all 4 pages
- [ ] `USE_MOCK = false` in app.js
- [ ] No files outside `website/` were touched
- [ ] Looks good on desktop (1280px+) and tablet (768px)
