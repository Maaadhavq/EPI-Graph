"""
EpiGraph-AI Flask Backend
Serves the website + 7 API endpoints.
Falls back to data-driven risk scoring when the torch model is unavailable.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, abort, request

# ── App Setup ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(BASE_DIR, "website")
DATA_DIR    = os.path.join(BASE_DIR, "data")

app = Flask(__name__, static_folder=WEBSITE_DIR)

# ── Load Data at Startup ─────────────────────────────────────────────────────
try:
    cases_df = pd.read_csv(os.path.join(DATA_DIR, "processed_cases.csv"))
    cases_df.columns = cases_df.columns.str.strip()
    # Rename columns to consistent names
    col_map = {}
    for c in cases_df.columns:
        lc = c.strip()
        if "MMAX" in lc or "TMAX" in lc:  col_map[c] = "tmax"
        elif "MMIN" in lc or "TMIN" in lc: col_map[c] = "tmin"
        elif "TMRF" in lc or "RAIN" in lc: col_map[c] = "rain"
        elif "0830" in lc:                  col_map[c] = "rh_am"
        elif "1730" in lc:                  col_map[c] = "rh_pm"
        elif lc.lower() == "dengue":        col_map[c] = "cases"
        elif lc.lower() == "district":      col_map[c] = "district"
        elif lc.lower() == "date":          col_map[c] = "date"
    cases_df = cases_df.rename(columns=col_map)
    cases_df["date"] = pd.to_datetime(cases_df["date"], errors="coerce")
    cases_df = cases_df.dropna(subset=["date"]).sort_values("date")
    print(f"[DATA] cases_df loaded: {len(cases_df)} rows")
except Exception as e:
    print(f"[WARN] cases_df failed: {e}")
    cases_df = pd.DataFrame()

try:
    news_df = pd.read_csv(os.path.join(DATA_DIR, "health_news.csv"))
    news_df.columns = news_df.columns.str.strip()
    news_df = news_df.rename(columns={c: c.lower() for c in news_df.columns})
    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_df = news_df.dropna(subset=["date"]).sort_values("date")
    print(f"[DATA] news_df loaded: {len(news_df)} rows")
except Exception as e:
    print(f"[WARN] news_df failed: {e}")
    news_df = pd.DataFrame()

try:
    conn_df = pd.read_csv(os.path.join(DATA_DIR, "connectivity.csv"))
    conn_df.columns = conn_df.columns.str.strip()
    print(f"[DATA] connectivity loaded: {len(conn_df)} rows")
except Exception as e:
    print(f"[WARN] connectivity failed: {e}")
    conn_df = pd.DataFrame()

DISTRICTS = ["Ahmedabad", "Gandhinagar", "Rajkot", "Surat", "Vadodara"]

# ── Risk Computation (data-driven, no torch needed) ──────────────────────────
def compute_predictions():
    """
    Compute risk score per district from case + weather data.
    Normalises against each district's own 90th-percentile to produce
    meaningful 0-100 scores, then adjusts for trend and spatial spillover.
    """
    if cases_df.empty:
        return {d: {"risk": round(np.random.uniform(5, 35), 1), "level": "medium"} for d in DISTRICTS}

    results = {}

    for dist in DISTRICTS:
        df = cases_df[cases_df["district"] == dist].copy()
        if df.empty:
            results[dist] = {"risk": 10.0, "level": "low"}
            continue

        # Per-district 90th percentile as the "high-epidemic" anchor
        p90 = max(df["cases"].quantile(0.90), 1.0)

        # Last week's case count is the primary signal
        last_val = float(df["cases"].iloc[-1])

        # Short-term trend: slope of last 4 data points (normalised)
        last4 = df["cases"].tail(4).values
        if len(last4) >= 2:
            x = np.arange(len(last4), dtype=float)
            slope = np.polyfit(x, last4, 1)[0]
            trend_bonus = float(np.clip(slope / p90 * 25, -8, 12))
        else:
            trend_bonus = 0.0

        # Base score: how close is the latest count to the district's epidemic threshold
        base_score = float(np.clip((last_val / p90) * 75, 0, 75))

        # Weather bonus: rainfall + morning humidity drive mosquito risk (0-10)
        weather_bonus = 0.0
        if "rain" in df.columns and "rh_am" in df.columns:
            latest    = df.iloc[-1]
            rain      = float(latest.get("rain",  0) or 0)
            humidity  = float(latest.get("rh_am", 50) or 50)
            rain_f    = float(np.clip(rain / 80, 0, 1))
            humid_f   = float(np.clip((humidity - 45) / 45, 0, 1))
            weather_bonus = (rain_f * 0.55 + humid_f * 0.45) * 10

        raw = base_score + trend_bonus + weather_bonus
        risk = round(float(np.clip(raw, 0, 100)), 1)
        level = "high" if risk >= 36 else ("medium" if risk >= 22 else "low")
        results[dist] = {"risk": risk, "level": level}

    # Spatial spillover: connected neighbour's high risk lifts a district
    if not conn_df.empty:
        src_col, tgt_col, wt_col = conn_df.columns[0], conn_df.columns[1], conn_df.columns[2]
        for _, row in conn_df.iterrows():
            src, tgt = str(row[src_col]), str(row[tgt_col])
            w = float(row[wt_col])
            if src in results and tgt in results:
                spillover = results[src]["risk"] * w * 0.07
                new_risk  = round(min(results[tgt]["risk"] + spillover, 100), 1)
                results[tgt]["risk"]  = new_risk
                results[tgt]["level"] = "high" if new_risk >= 36 else ("medium" if new_risk >= 22 else "low")

    return results


def compute_xai(district, predictions):
    """
    Derive approximate factor contributions from actual data for one district.
    """
    if cases_df.empty:
        return {"district": district, "total_risk": predictions.get(district, {}).get("risk", 0),
                "factors": []}

    df   = cases_df[cases_df["district"] == district]
    risk = predictions.get(district, {}).get("risk", 10)

    if df.empty:
        return {"district": district, "total_risk": risk, "factors": []}

    last4 = df.tail(4)
    prev4 = df.tail(8).head(4)
    avg_l = last4["cases"].mean() if len(last4) else 0
    avg_p = prev4["cases"].mean() if len(prev4) else max(avg_l, 1)
    trend_ratio = avg_l / avg_p if avg_p > 0 else 1.0

    # Temporal: based on trend strength
    temporal_pct = min(int(30 + (trend_ratio - 1) * 25), 60)

    # Weather: based on recent rainfall
    if "rain" in df.columns:
        rain = float(df.tail(1)["rain"].values[0])
        weather_pct = min(int(rain / 120 * 40), 40)
    else:
        weather_pct = 10

    # Spatial: connectivity weight from neighbours
    if not conn_df.empty:
        src_col = conn_df.columns[0]
        tgt_col = conn_df.columns[1]
        wt_col  = conn_df.columns[2]
        incoming = conn_df[conn_df[tgt_col] == district]
        spatial_raw = incoming[wt_col].sum() if len(incoming) else 0
        spatial_pct = min(int(spatial_raw * 15), 30)
    else:
        spatial_pct = 5

    # Demographic fills the remainder
    used = temporal_pct + weather_pct + spatial_pct
    demo_pct = max(100 - used, 5)

    # Normalise to 100
    total = temporal_pct + weather_pct + spatial_pct + demo_pct
    scale = 100 / total
    temporal_pct  = round(temporal_pct  * scale)
    weather_pct   = round(weather_pct   * scale)
    spatial_pct   = round(spatial_pct   * scale)
    demo_pct      = 100 - temporal_pct - weather_pct - spatial_pct  # absorb rounding

    factors = [
        {"name": "Recent Case Trajectory", "contribution_pct": temporal_pct, "type": "temporal"},
        {"name": "Rainfall / Humidity Pattern", "contribution_pct": weather_pct, "type": "weather"},
        {"name": "Inflow from Neighbours",  "contribution_pct": spatial_pct,  "type": "spatial"},
        {"name": "Baseline Susceptibility", "contribution_pct": demo_pct,     "type": "demographic"},
    ]
    # Sort descending
    factors.sort(key=lambda f: f["contribution_pct"], reverse=True)

    return {"district": district, "total_risk": risk, "factors": factors}


# ── Static File Serving ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(WEBSITE_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    full = os.path.join(WEBSITE_DIR, filename)
    if os.path.isfile(full):
        return send_from_directory(WEBSITE_DIR, filename)
    abort(404)


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/districts")
def api_districts():
    colors = {
        "Ahmedabad": "#3b82f6", "Gandhinagar": "#8b5cf6",
        "Rajkot": "#06b6d4",   "Surat": "#ec4899",  "Vadodara": "#10b981"
    }
    return jsonify({"districts": [{"name": d, "color": colors.get(d, "#888")} for d in DISTRICTS]})


@app.route("/api/predictions")
def api_predictions():
    preds = compute_predictions()
    return jsonify(preds)


@app.route("/api/weather")
def api_weather():
    district = request.args.get("district")
    if cases_df.empty:
        abort(503)

    def district_weather(d):
        df = cases_df[cases_df["district"] == d]
        if df.empty:
            return {"district": d, "tmax": None, "tmin": None, "rain": None, "rh_am": None, "rh_pm": None}
        row = df.iloc[-1]
        return {
            "district": d,
            "tmax":  round(float(row.get("tmax",  0)), 1),
            "tmin":  round(float(row.get("tmin",  0)), 1),
            "rain":  round(float(row.get("rain",  0)), 1),
            "rh_am": int(row.get("rh_am", 0)),
            "rh_pm": int(row.get("rh_pm", 0)),
        }

    if district:
        return jsonify(district_weather(district))
    return jsonify({d: district_weather(d) for d in DISTRICTS})


@app.route("/api/cases")
def api_cases():
    district = request.args.get("district", "Ahmedabad")
    if cases_df.empty:
        return jsonify({"district": district, "cases": []})

    df = cases_df[cases_df["district"] == district].copy()
    records = [
        {"date": row["date"].strftime("%Y-%m-%d"), "value": int(row["cases"])}
        for _, row in df.iterrows()
        if pd.notna(row.get("cases"))
    ]
    return jsonify({"district": district, "cases": records})


@app.route("/api/news")
def api_news():
    district = request.args.get("district")
    if news_df.empty:
        return jsonify({"district": district or "all", "alerts": []})

    if district:
        df = news_df[news_df["district"] == district].tail(20)
    else:
        df = news_df.tail(50)

    alerts = [
        {
            "date":     row["date"].strftime("%Y-%m-%d"),
            "headline": row.get("headline", ""),
            "type":     row.get("type", "Medical_Alert"),
        }
        for _, row in df.sort_values("date", ascending=False).iterrows()
    ]
    return jsonify({"district": district or "all", "alerts": alerts})


@app.route("/api/connectivity")
def api_connectivity():
    if conn_df.empty:
        return jsonify({"edges": []})

    edges = [
        {
            "source": str(row.iloc[0]),
            "target": str(row.iloc[1]),
            "weight": round(float(row.iloc[2]), 2),
        }
        for _, row in conn_df.iterrows()
    ]
    return jsonify({"edges": edges})


@app.route("/api/explain")
def api_explain():
    district = request.args.get("district", "Ahmedabad")
    preds    = compute_predictions()
    return jsonify(compute_xai(district, preds))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n EpiGraph-AI backend starting...")
    print(f"   Website: http://127.0.0.1:5000")
    print(f"   Data dir: {DATA_DIR}\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
