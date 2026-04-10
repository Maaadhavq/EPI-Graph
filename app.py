"""
EpiGraph-AI Flask Backend
Serves the website + 7 API endpoints.
Uses the trained GATv2+LSTM model for real predictions.
Falls back to data-driven scoring if model files are missing.
"""

import os, warnings
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_from_directory, abort, request
warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(BASE_DIR, "website")
DATA_DIR    = os.path.join(BASE_DIR, "data")

app = Flask(__name__, static_folder=WEBSITE_DIR)

# ── Load Data ─────────────────────────────────────────────────────────────────
cases_df = pd.read_csv(os.path.join(DATA_DIR, "processed_cases.csv"))
news_df  = pd.read_csv(os.path.join(DATA_DIR, "health_news.csv"))
conn_df  = pd.read_csv(os.path.join(DATA_DIR, "connectivity.csv"))

# Consistent column names
cases_df.columns = cases_df.columns.str.strip()
col_map = {}
for c in cases_df.columns:
    lc = c.strip()
    if   "MMAX"  in lc: col_map[c] = "tmax"
    elif "MMIN"  in lc: col_map[c] = "tmin"
    elif "TMRF"  in lc: col_map[c] = "rain"
    elif "0830"  in lc: col_map[c] = "rh_am"
    elif "1730"  in lc: col_map[c] = "rh_pm"
    elif lc.lower() == "dengue":   col_map[c] = "cases"
    elif lc.lower() == "district": col_map[c] = "district"
    elif lc.lower() == "date":     col_map[c] = "date"
cases_df = cases_df.rename(columns=col_map)
cases_df["date"] = pd.to_datetime(cases_df["date"], errors="coerce")
cases_df = cases_df.dropna(subset=["date"]).sort_values("date")

news_df.columns = news_df.columns.str.strip().str.lower()
news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
news_df = news_df.dropna(subset=["date"]).sort_values("date")

# Original district order from notebook (sorted alphabetically)
DISTRICTS = sorted(cases_df["district"].unique().tolist())
print(f"[DATA] Districts: {DISTRICTS}")
print(f"[DATA] cases={len(cases_df)} rows, news={len(news_df)} rows, connectivity={len(conn_df)} rows")

# ── Try to load the real trained model ────────────────────────────────────────
MODEL_PREDICTIONS = None   # cached dict populated at startup

def _try_load_model():
    global MODEL_PREDICTIONS
    try:
        import torch, torch.nn as nn, torch.nn.functional as F, joblib
        from torch_geometric.nn import GATv2Conv

        MODEL_PATH   = os.path.join(BASE_DIR, "epigraph_model_v2.pth")
        EMBED_CACHE  = os.path.join(BASE_DIR, "embeddings_cache_orig_32.pt")
        BASE_SCALER  = os.path.join(BASE_DIR, "base_scaler.pkl")
        TGT_SCALER   = os.path.join(BASE_DIR, "target_scaler_v3.pkl")

        for f in [MODEL_PATH, EMBED_CACHE, BASE_SCALER, TGT_SCALER]:
            if not os.path.exists(f):
                print(f"[MODEL] Missing {f} — using data-driven fallback")
                return

        # ── Config (must match train_original_arch.py) ─────────────────────
        WINDOW_SIZE    = 7
        HIDDEN_DIM     = 128
        BERT_PROJ_DIM  = 16
        NUM_BASE_FEATS = 13
        INPUT_DIM      = NUM_BASE_FEATS + BERT_PROJ_DIM   # 29
        DROPOUT        = 0.4

        node_map  = {d: i for i, d in enumerate(DISTRICTS)}
        num_nodes = len(DISTRICTS)
        dates     = sorted(cases_df["date"].dt.strftime("%Y-%m-%d").unique())
        T         = len(dates)

        # Edge index
        adj = np.zeros((num_nodes, num_nodes))
        for _, row in conn_df.iterrows():
            s = node_map.get(str(row.iloc[0])); t = node_map.get(str(row.iloc[1]))
            if s is not None and t is not None:
                w = float(row.iloc[2]); adj[s,t] = w; adj[t,s] = w
        np.fill_diagonal(adj, 1.0)
        edge_index = torch.tensor(adj, dtype=torch.float32).nonzero().t().contiguous()

        # Raw case features (T, N, 6) — dengue first (original feature_cols order)
        feat6_cols = ["cases", "tmax", "tmin", "rain", "rh_am", "rh_pm"]
        raw_feat = np.zeros((T, num_nodes, 6))
        for di, d in enumerate(DISTRICTS):
            ddf = cases_df[cases_df["district"] == d].sort_values("date")
            for ci, col in enumerate(feat6_cols):
                if col in ddf.columns:
                    raw_feat[:, di, ci] = ddf[col].values[:T]

        # 7 engineered features
        dengue_col = raw_feat[:, :, 0]
        eng = np.zeros((T, num_nodes, 7))
        for ni in range(num_nodes):
            s = pd.Series(dengue_col[:, ni])
            eng[:,ni,0] = s.rolling(4,  min_periods=1).mean()
            eng[:,ni,1] = s.rolling(8,  min_periods=1).mean()
            eng[:,ni,2] = s.rolling(4,  min_periods=1).std().fillna(0)
            eng[:,ni,3] = s.shift(1).fillna(0)
            eng[:,ni,4] = s.shift(2).fillna(0)
            eng[:,ni,5] = s.diff().fillna(0)
            eng[:,ni,6] = np.log1p(dengue_col[:, ni])

        feat13 = np.concatenate([raw_feat, eng], axis=2)   # (T, N, 13)

        # Normalise
        base_scaler   = joblib.load(BASE_SCALER)
        target_scaler = joblib.load(TGT_SCALER)
        feat13_n = base_scaler.transform(feat13.reshape(T*num_nodes, 13)).reshape(T, num_nodes, 13)

        # Model (exact architecture)
        class EpiGraphModelV3(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_dim = HIDDEN_DIM
                self.num_base   = NUM_BASE_FEATS
                self.bert_projection = nn.Sequential(nn.Linear(768, BERT_PROJ_DIM))
                self.gat1 = GATv2Conv(INPUT_DIM, HIDDEN_DIM, heads=2, dropout=DROPOUT)
                self.gat2 = GATv2Conv(HIDDEN_DIM*2, HIDDEN_DIM, heads=1, concat=False, dropout=DROPOUT)
                self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True, num_layers=2, dropout=DROPOUT)
                self.layer_norm = nn.LayerNorm(HIDDEN_DIM)
                self.temporal_attention = nn.Sequential(nn.Linear(HIDDEN_DIM, 1))
                self.skip_fc = nn.Sequential(
                    nn.Linear(NUM_BASE_FEATS, 64), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 32))
                self.fc = nn.Sequential(
                    nn.Linear(HIDDEN_DIM+32, 64), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 1))
            def forward(self, x, edge_index):
                b, t, n, f = x.size()
                seq = []
                for i in range(t):
                    xt = x[:, i]; bs = []
                    for bi in range(b):
                        h = F.elu(self.gat1(xt[bi], edge_index))
                        h = F.elu(self.gat2(h, edge_index))
                        bs.append(h)
                    seq.append(torch.stack(bs))
                sp = torch.stack(seq, dim=1).view(b*n, t, self.hidden_dim)
                lo, _ = self.lstm(sp)
                lo = self.layer_norm(lo)
                ctx = (F.softmax(self.temporal_attention(lo), dim=1) * lo).sum(dim=1)
                sk  = self.skip_fc(x[:, -1, :, :self.num_base].reshape(b*n, self.num_base))
                return self.fc(torch.cat([ctx, sk], dim=1)).view(b, n, 1)

        model = EpiGraphModelV3()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
        model.eval()
        print(f"[MODEL] Loaded epigraph_model_v2.pth ({sum(p.numel() for p in model.parameters()):,} params)")

        # Project BERT
        cache   = torch.load(EMBED_CACHE, map_location="cpu", weights_only=False)
        raw_emb = cache['embeddings']   # (T, N, 768)
        raw_emb_t = torch.tensor(raw_emb.reshape(T*num_nodes, 768), dtype=torch.float32)
        with torch.no_grad():
            bert_proj = model.bert_projection(raw_emb_t).numpy().reshape(T, num_nodes, BERT_PROJ_DIM)

        combined = np.concatenate([feat13_n, bert_proj], axis=2)
        x_tensor = torch.tensor(combined, dtype=torch.float32)

        # Inference on last WINDOW_SIZE timesteps
        input_seq = x_tensor[-WINDOW_SIZE:].unsqueeze(0)
        with torch.no_grad():
            risk_scaled = model(input_seq, edge_index).squeeze().numpy()

        risk_scores = np.clip(
            target_scaler.inverse_transform(risk_scaled.reshape(-1, 1)).flatten(), 0, None
        )
        global_median = float(np.median(risk_scores))

        MODEL_PREDICTIONS = {}
        for di, d in enumerate(DISTRICTS):
            score = float(risk_scores[di])
            rel   = score / max(global_median, 1.0)
            level = "high" if rel >= 1.5 else ("medium" if rel >= 0.8 else "low")
            MODEL_PREDICTIONS[d] = {"risk": round(score, 1), "level": level}

        print(f"[MODEL] Predictions: {MODEL_PREDICTIONS}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[MODEL] Could not load model — using data-driven fallback")
        MODEL_PREDICTIONS = None

_try_load_model()

# ── Data-driven fallback ───────────────────────────────────────────────────────
def _compute_predictions_fallback():
    results = {}
    global_ref = max(cases_df["cases"].quantile(0.75), 1)
    for dist in DISTRICTS:
        df = cases_df[cases_df["district"] == dist]
        if df.empty:
            results[dist] = {"risk": 10.0, "level": "low"}
            continue
        p90       = max(df["cases"].quantile(0.90), 1)
        last_val  = float(df["cases"].iloc[-1])
        last4     = df["cases"].tail(4).values
        slope     = float(np.polyfit(range(len(last4)), last4, 1)[0]) if len(last4) >= 2 else 0
        trend     = float(np.clip(slope / p90 * 25, -8, 12))
        base      = float(np.clip((last_val / p90) * 75, 0, 75))
        weather   = 0.0
        if "rain" in df.columns and "rh_am" in df.columns:
            r = float(df.iloc[-1].get("rain", 0) or 0)
            h = float(df.iloc[-1].get("rh_am", 50) or 50)
            weather = (np.clip(r/80, 0, 1)*0.55 + np.clip((h-45)/45, 0, 1)*0.45) * 10
        risk  = round(float(np.clip(base + trend + weather, 0, 100)), 1)
        level = "high" if risk >= 36 else ("medium" if risk >= 22 else "low")
        results[dist] = {"risk": risk, "level": level}
    if not conn_df.empty:
        for _, row in conn_df.iterrows():
            s, t = str(row.iloc[0]), str(row.iloc[1]); w = float(row.iloc[2])
            if s in results and t in results:
                nr = round(min(results[t]["risk"] + results[s]["risk"] * w * 0.06, 100), 1)
                results[t] = {"risk": nr, "level": "high" if nr>=36 else ("medium" if nr>=22 else "low")}
    return results

def get_predictions():
    return MODEL_PREDICTIONS if MODEL_PREDICTIONS else _compute_predictions_fallback()

def compute_xai(district, predictions):
    df   = cases_df[cases_df["district"] == district]
    risk = predictions.get(district, {}).get("risk", 10)
    if df.empty:
        return {"district": district, "total_risk": risk, "factors": []}
    last4 = df["cases"].tail(4); prev4 = df["cases"].tail(8).head(4)
    avg_l = last4.mean(); avg_p = max(prev4.mean(), 1)
    trend_ratio  = avg_l / avg_p
    temporal_pct = min(int(30 + (trend_ratio - 1) * 25), 60)
    rain = float(df.iloc[-1].get("rain", 0) or 0) if "rain" in df.columns else 0
    weather_pct  = min(int(rain / 120 * 40), 40)
    if not conn_df.empty:
        tgt_col = conn_df.columns[1]; wt_col = conn_df.columns[2]
        inc = conn_df[conn_df[tgt_col] == district]
        spatial_pct = min(int(inc[wt_col].sum() * 15), 30) if len(inc) else 5
    else:
        spatial_pct = 5
    demo_pct = max(100 - temporal_pct - weather_pct - spatial_pct, 5)
    total    = temporal_pct + weather_pct + spatial_pct + demo_pct
    s        = 100 / total
    tp, wp, sp = round(temporal_pct*s), round(weather_pct*s), round(spatial_pct*s)
    dp = 100 - tp - wp - sp
    factors = sorted([
        {"name": "Recent Case Trajectory",    "contribution_pct": tp, "type": "temporal"},
        {"name": "Rainfall / Humidity Pattern","contribution_pct": wp, "type": "weather"},
        {"name": "Inflow from Neighbours",     "contribution_pct": sp, "type": "spatial"},
        {"name": "Baseline Susceptibility",    "contribution_pct": dp, "type": "demographic"},
    ], key=lambda f: f["contribution_pct"], reverse=True)
    return {"district": district, "total_risk": risk, "factors": factors}

# ── Static Serving ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(WEBSITE_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    full = os.path.join(WEBSITE_DIR, filename)
    if os.path.isfile(full):
        return send_from_directory(WEBSITE_DIR, filename)
    abort(404)

# ── API ───────────────────────────────────────────────────────────────────────
COLORS = {"Ahmedabad":"#3b82f6","Gandhinagar":"#8b5cf6",
          "Rajkot":"#06b6d4","Surat":"#ec4899","Vadodara":"#10b981"}

@app.route("/api/districts")
def api_districts():
    return jsonify({"districts": [{"name": d, "color": COLORS.get(d,"#888")} for d in DISTRICTS]})

@app.route("/api/predictions")
def api_predictions():
    return jsonify(get_predictions())

@app.route("/api/weather")
def api_weather():
    district = request.args.get("district")
    def row_for(d):
        df = cases_df[cases_df["district"] == d]
        r  = df.iloc[-1] if not df.empty else {}
        return {"district": d,
                "tmax":  round(float(r.get("tmax",  0) or 0), 1),
                "tmin":  round(float(r.get("tmin",  0) or 0), 1),
                "rain":  round(float(r.get("rain",  0) or 0), 1),
                "rh_am": int(r.get("rh_am", 0) or 0),
                "rh_pm": int(r.get("rh_pm", 0) or 0)}
    return jsonify(row_for(district) if district else {d: row_for(d) for d in DISTRICTS})

@app.route("/api/cases")
def api_cases():
    district = request.args.get("district", "Ahmedabad")
    df = cases_df[cases_df["district"] == district]
    records = [{"date": r["date"].strftime("%Y-%m-%d"), "value": int(r["cases"])}
               for _, r in df.iterrows() if pd.notna(r.get("cases"))]
    return jsonify({"district": district, "cases": records})

@app.route("/api/news")
def api_news():
    district = request.args.get("district")
    df = news_df[news_df["district"] == district].tail(20) if district else news_df.tail(50)
    alerts = [{"date": r["date"].strftime("%Y-%m-%d"),
               "headline": r.get("headline",""),
               "type": r.get("type","Medical_Alert")}
              for _, r in df.sort_values("date", ascending=False).iterrows()]
    return jsonify({"district": district or "all", "alerts": alerts})

@app.route("/api/connectivity")
def api_connectivity():
    edges = [{"source": str(r.iloc[0]), "target": str(r.iloc[1]), "weight": round(float(r.iloc[2]),2)}
             for _, r in conn_df.iterrows()]
    return jsonify({"edges": edges})

@app.route("/api/explain")
def api_explain():
    district = request.args.get("district", "Ahmedabad")
    return jsonify(compute_xai(district, get_predictions()))

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = "GNN model" if MODEL_PREDICTIONS else "data-driven fallback"
    print(f"\n EpiGraph-AI backend [{mode}]")
    print(f"   Open: http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
