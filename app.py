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
MODEL_PREDICTIONS  = None   # cached dict populated at startup
GAT_ATTENTION      = {}     # {district: {neighbor: score}} — real spatial attribution

def _try_load_model():
    global MODEL_PREDICTIONS, GAT_ATTENTION
    try:
        import torch, torch.nn as nn, torch.nn.functional as F, joblib
        from torch_geometric.nn import GATv2Conv

        # v3 model takes priority; fall back to v2 if not yet trained
        MODEL_PATH   = os.path.join(BASE_DIR, "epigraph_model_v3.pth")
        if not os.path.exists(MODEL_PATH):
            MODEL_PATH = os.path.join(BASE_DIR, "epigraph_model_v2.pth")
        EMBED_CACHE  = os.path.join(BASE_DIR, "embeddings_cache_orig_32.pt")
        BASE_SCALER  = os.path.join(BASE_DIR, "base_scaler.pkl")
        TGT_SCALER   = os.path.join(BASE_DIR, "target_scaler_v3.pkl")

        for f in [MODEL_PATH, EMBED_CACHE, BASE_SCALER, TGT_SCALER]:
            if not os.path.exists(f):
                print(f"[MODEL] Missing {f} — using data-driven fallback")
                return
        print(f"[MODEL] Using weights: {os.path.basename(MODEL_PATH)}")

        # ── Config (must match train_original_arch.py v3) ──────────────────
        WINDOW_SIZE    = 7
        HIDDEN_DIM     = 128
        BERT_PROJ_DIM  = 16
        # v3: 8 raw (6 original + 2 seasonal) + 11 engineered = 19 base features
        # Detect model version from scaler shape
        base_scaler_tmp   = joblib.load(BASE_SCALER)
        NUM_BASE_FEATS    = base_scaler_tmp.n_features_in_   # 13 (v2) or 19 (v3)
        INPUT_DIM         = NUM_BASE_FEATS + BERT_PROJ_DIM   # 29 or 35
        DROPOUT           = 0.4
        IS_V3             = (NUM_BASE_FEATS == 19)
        print(f"[MODEL] Detected {'v3 (19-feature)' if IS_V3 else 'v2 (13-feature)'} model  INPUT_DIM={INPUT_DIM}")

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

        # Raw case features (T, N, 6)
        feat6_cols = ["cases", "tmax", "tmin", "rain", "rh_am", "rh_pm"]
        raw_feat = np.zeros((T, num_nodes, 6))
        for di, d in enumerate(DISTRICTS):
            ddf = cases_df[cases_df["district"] == d].sort_values("date")
            for ci, col in enumerate(feat6_cols):
                if col in ddf.columns:
                    raw_feat[:, di, ci] = ddf[col].values[:T]

        # B1 ── Seasonal features (v3 only) → raw_feat (T, N, 8)
        if IS_V3:
            date_idx  = pd.to_datetime(dates)
            week_nums = date_idx.isocalendar().week.values.astype(float)
            sin_week  = np.sin(2 * np.pi * week_nums / 52)
            cos_week  = np.cos(2 * np.pi * week_nums / 52)
            seasonal_feat = np.tile(
                np.stack([sin_week, cos_week], axis=1)[:, np.newaxis, :],
                (1, num_nodes, 1)
            )
            raw_feat = np.concatenate([raw_feat, seasonal_feat], axis=2)  # (T, N, 8)

        # Engineered features: 7 (v2) or 11 (v3)
        dengue_col = raw_feat[:, :, 0]
        rain_col   = raw_feat[:, :, 3]
        rh_col     = raw_feat[:, :, 4]
        num_eng    = 11 if IS_V3 else 7
        eng = np.zeros((T, num_nodes, num_eng))
        for ni in range(num_nodes):
            s    = pd.Series(dengue_col[:, ni])
            eng[:,ni,0] = s.rolling(4,  min_periods=1).mean()
            eng[:,ni,1] = s.rolling(8,  min_periods=1).mean()
            eng[:,ni,2] = s.rolling(4,  min_periods=1).std().fillna(0)
            eng[:,ni,3] = s.shift(1).fillna(0)
            eng[:,ni,4] = s.shift(2).fillna(0)
            eng[:,ni,5] = s.diff().fillna(0)
            eng[:,ni,6] = np.log1p(dengue_col[:, ni])
            if IS_V3:
                rain = pd.Series(rain_col[:, ni])
                rh   = pd.Series(rh_col[:, ni])
                eng[:,ni,7]  = s.shift(3).fillna(0)                  # B2: lag-3
                eng[:,ni,8]  = s.shift(4).fillna(0)                  # B2: lag-4
                eng[:,ni,9]  = rain.rolling(4, min_periods=1).mean() # B3: 4-week rain
                eng[:,ni,10] = rh.rolling(4, min_periods=1).mean()   # B3: 4-week rh

        feat13 = np.concatenate([raw_feat, eng], axis=2)  # (T, N, 13 or 19)

        # Normalise
        base_scaler   = base_scaler_tmp
        target_scaler = joblib.load(TGT_SCALER)
        feat13_n = base_scaler.transform(feat13.reshape(T*num_nodes, NUM_BASE_FEATS)).reshape(T, num_nodes, NUM_BASE_FEATS)

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

        # Monte Carlo Dropout inference (20 passes with dropout ON)
        input_seq = x_tensor[-WINDOW_SIZE:].unsqueeze(0)
        MC_SAMPLES = 20
        mc_preds = []
        model.train()   # keep dropout active for uncertainty estimation
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                mc_preds.append(model(input_seq, edge_index).squeeze().numpy())
        model.eval()

        mc_arr      = np.stack(mc_preds)                          # (20, N)
        risk_mean_s = mc_arr.mean(axis=0)                         # (N,) scaled
        risk_std_s  = mc_arr.std(axis=0)                          # (N,) scaled uncertainty

        def _inv(arr):
            return np.clip(
                target_scaler.inverse_transform(arr.reshape(-1, 1)).flatten(), 0, None
            )

        risk_scores = _inv(risk_mean_s)
        risk_stds   = _inv(risk_std_s)   # uncertainty in original case scale

        # Build per-district info
        district_info = []
        for di, d in enumerate(DISTRICTS):
            score    = float(risk_scores[di])
            unc      = round(float(risk_stds[di]), 1)
            dist_df  = cases_df[cases_df["district"] == d].sort_values("date")
            last_c   = int(dist_df["cases"].iloc[-1]) if not dist_df.empty else 0
            avg4     = round(float(dist_df["cases"].tail(4).mean()), 1) if not dist_df.empty else 0
            display  = int(round(avg4))
            district_info.append((di, d, score, unc, last_c, avg4, display))

        # Rank-based classification: sort ascending by (score, last_cases)
        # → top 2 = high, middle 1 = medium, bottom 2 = low
        ranked = sorted(district_info, key=lambda x: (x[2], x[4]))
        n = len(ranked)
        level_for = {}
        for rank, row in enumerate(ranked):
            d = row[1]
            if rank >= n - 2:    level_for[d] = "high"
            elif rank >= n - 3:  level_for[d] = "medium"
            else:                level_for[d] = "low"

        MODEL_PREDICTIONS = {}
        for di, d, score, unc, last_c, avg4, display in district_info:
            MODEL_PREDICTIONS[d] = {
                "risk":        round(score, 1),
                "uncertainty": unc,
                "level":       level_for[d],
                "last_cases":  display,
                "avg4":        avg4
            }
        _add_trend_data(MODEL_PREDICTIONS)

        # ── Extract real GAT attention weights for XAI ─────────────────────
        try:
            model.eval()
            x_last = x_tensor[-WINDOW_SIZE:].unsqueeze(0)  # (1, T, N, F)
            # Run one forward pass through GAT layers only, collecting attention
            x_t = x_last[:, -1]  # use last timestep: (1, N, F)
            with torch.no_grad():
                _, (ei1, a1) = model.gat1(x_t[0], edge_index, return_attention_weights=True)
                h1 = torch.nn.functional.elu(model.gat1(x_t[0], edge_index))
                _, (ei2, a2) = model.gat2(h1, edge_index, return_attention_weights=True)

            # Average attention across multi-head (gat1 has 2 heads → mean)
            a1_mean = a1.mean(dim=1).cpu().numpy()  # (E,)
            a2_mean = a2.mean(dim=1).cpu().numpy()  # (E,)
            alpha   = (a1_mean + a2_mean) / 2.0     # average both layers

            # Build per-district attention dict: {district → {source_neighbor: score}}
            ei_np = ei2.cpu().numpy()  # (2, E)
            GAT_ATTENTION = {}
            for di, d in enumerate(DISTRICTS):
                incoming = {}
                for edge_i in range(ei_np.shape[1]):
                    tgt = int(ei_np[1, edge_i])
                    src = int(ei_np[0, edge_i])
                    if tgt == di and src != di:    # only incoming edges, no self-loop
                        incoming[DISTRICTS[src]] = round(float(alpha[edge_i]), 4)
                GAT_ATTENTION[d] = incoming
            print(f"[MODEL] GAT attention extracted: {GAT_ATTENTION}")
        except Exception as att_err:
            print(f"[MODEL] GAT attention extraction failed: {att_err}")
            GAT_ATTENTION = {}

        print(f"[MODEL] Predictions: {MODEL_PREDICTIONS}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[MODEL] Could not load model — using data-driven fallback")
        MODEL_PREDICTIONS = None

def _add_trend_data(results):
    """Compute week-over-week trend for each district and add to results in-place."""
    for dist, data in results.items():
        df = cases_df[cases_df["district"] == dist].sort_values("date")
        if len(df) >= 4:
            last2 = float(df["cases"].tail(2).mean())
            prev2 = float(df["cases"].tail(4).head(2).mean())
            prev2 = max(prev2, 1)
            change_pct = round((last2 - prev2) / prev2 * 100, 1)
        elif len(df) >= 2:
            last_val = float(df["cases"].iloc[-1])
            prev_val = max(float(df["cases"].iloc[-2]), 1)
            change_pct = round((last_val - prev_val) / prev_val * 100, 1)
        else:
            change_pct = 0.0
        data["trend"]      = "up" if change_pct > 10 else "down" if change_pct < -10 else "stable"
        data["change_pct"] = change_pct
    return results

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
        dist_df    = cases_df[cases_df["district"] == dist].sort_values("date")
        last_cases = int(dist_df["cases"].iloc[-1]) if not dist_df.empty else 0
        avg4       = round(float(dist_df["cases"].tail(4).mean()), 1) if not dist_df.empty else 0
        display    = int(round(avg4))   # 4-week avg as display value
        results[dist] = {"risk": risk, "last_cases": display, "avg4": avg4}
    if not conn_df.empty:
        for _, row in conn_df.iterrows():
            s, t = str(row.iloc[0]), str(row.iloc[1]); w = float(row.iloc[2])
            if s in results and t in results:
                nr = round(min(results[t]["risk"] + results[s]["risk"] * w * 0.06, 100), 1)
                results[t]["risk"] = nr
    # Rank-based levels: top 2 = high, middle 1 = medium, bottom 2 = low
    ranked = sorted(results.items(), key=lambda x: (x[1]["risk"], x[1]["last_cases"]))
    n = len(ranked)
    for rank, (dist, _) in enumerate(ranked):
        if rank >= n - 2:
            results[dist]["level"] = "high"
        elif rank >= n - 3:
            results[dist]["level"] = "medium"
        else:
            results[dist]["level"] = "low"
    _add_trend_data(results)
    return results

def get_predictions():
    return MODEL_PREDICTIONS if MODEL_PREDICTIONS else _compute_predictions_fallback()

def compute_xai(district, predictions):
    df   = cases_df[cases_df["district"] == district]
    risk = predictions.get(district, {}).get("risk", 10)
    if df.empty:
        return {"district": district, "total_risk": risk, "factors": [], "attention": {}}
    last4 = df["cases"].tail(4); prev4 = df["cases"].tail(8).head(4)
    avg_l = last4.mean(); avg_p = max(prev4.mean(), 1)
    trend_ratio  = avg_l / avg_p
    temporal_pct = min(int(30 + (trend_ratio - 1) * 25), 60)
    rain = float(df.iloc[-1].get("rain", 0) or 0) if "rain" in df.columns else 0
    weather_pct  = min(int(rain / 120 * 40), 40)

    # ── Spatial: use real GAT attention if available, else connectivity heuristic ──
    attention = GAT_ATTENTION.get(district, {})
    if attention:
        # Normalize attention scores to 0-30% range
        total_att   = max(sum(attention.values()), 1e-6)
        spatial_pct = min(int(total_att / (total_att + 0.5) * 30), 30)
        # Build top-neighbour label
        top_neighbour = max(attention, key=attention.get) if attention else None
        spatial_name  = f"Inflow from Neighbours (top: {top_neighbour})" if top_neighbour else "Inflow from Neighbours"
    else:
        if not conn_df.empty:
            tgt_col = conn_df.columns[1]; wt_col = conn_df.columns[2]
            inc = conn_df[conn_df[tgt_col] == district]
            spatial_pct = min(int(inc[wt_col].sum() * 15), 30) if len(inc) else 5
        else:
            spatial_pct = 5
        spatial_name = "Inflow from Neighbours"

    demo_pct = max(100 - temporal_pct - weather_pct - spatial_pct, 5)
    total    = temporal_pct + weather_pct + spatial_pct + demo_pct
    s        = 100 / total
    tp, wp, sp = round(temporal_pct*s), round(weather_pct*s), round(spatial_pct*s)
    dp = 100 - tp - wp - sp
    factors = sorted([
        {"name": "Recent Case Trajectory",     "contribution_pct": tp, "type": "temporal"},
        {"name": "Rainfall / Humidity Pattern", "contribution_pct": wp, "type": "weather"},
        {"name": spatial_name,                  "contribution_pct": sp, "type": "spatial"},
        {"name": "Baseline Susceptibility",     "contribution_pct": dp, "type": "demographic"},
    ], key=lambda f: f["contribution_pct"], reverse=True)
    return {
        "district":    district,
        "total_risk":  risk,
        "factors":     factors,
        "attention":   attention,   # raw per-neighbour GAT scores for frontend use
        "xai_source":  "gat_attention" if attention else "heuristic"
    }

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
    preds = get_predictions()
    result = dict(preds)
    try:
        result["data_as_of"] = cases_df["date"].max().strftime("%Y-%m-%d")
    except Exception:
        result["data_as_of"] = None
    return jsonify(result)

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
