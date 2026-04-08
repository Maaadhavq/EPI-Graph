# Person A Guide: Backend + Model Improvements

## Your Scope
You own all Python files. You will:
1. Improve the ML model in the notebook (better accuracy)
2. Build a Flask backend that serves predictions via API
3. Never touch anything inside `website/` folder

## Getting Started

```bash
cd EpiGraph-AI
git checkout -b backend-model
```

You will only edit these files:
```
notebooks/EpiGraph_AI.ipynb   ← model improvements
app.py                        ← NEW file: Flask server
requirements.txt              ← add flask
```

---

## PART 1: Model Improvements (Notebook)

Open `notebooks/EpiGraph_AI.ipynb` and make the following changes in order.

### Step 1: Update Hyperparameters (Cell 4)

Find the hyperparameters section and change:
```python
# OLD
HIDDEN_DIM = 64
PCA_DIM = 32
BATCH_SIZE = 8
DROPOUT = 0.4
LR = 0.0005
PATIENCE = 15

# NEW
HIDDEN_DIM = 128
PCA_DIM = 16          # Less BioBERT dims = better balance with tabular features
BATCH_SIZE = 16        # More stable gradients
DROPOUT = 0.2          # 0.4 was too aggressive
LR = 0.001
PATIENCE = 30          # Give model more time
NUM_BASE_FEATURES = 13 # 6 raw + 7 engineered (you'll add these next)
```

### Step 2: Add Engineered Temporal Features (NEW cell after Cell 14)

After Cell 14 finishes collecting `raw_case_features` (shape T, N, 6), add a new cell:

```python
# === Engineered Temporal Features ===
# Compute rolling statistics and lag features from raw dengue cases

import pandas as pd

dengue_cases = raw_case_features[:, :, 0]  # (T, N) — first column is dengue

engineered = np.zeros((dengue_cases.shape[0], dengue_cases.shape[1], 7))

for n in range(dengue_cases.shape[1]):  # per district
    series = pd.Series(dengue_cases[:, n])

    engineered[:, n, 0] = series.rolling(4, min_periods=1).mean()     # rolling_mean_4
    engineered[:, n, 1] = series.rolling(8, min_periods=1).mean()     # rolling_mean_8
    engineered[:, n, 2] = series.rolling(4, min_periods=1).std().fillna(0)  # rolling_std_4
    engineered[:, n, 3] = series.shift(1).fillna(0)                   # lag_1
    engineered[:, n, 4] = series.shift(2).fillna(0)                   # lag_2
    engineered[:, n, 5] = series.diff().fillna(0)                     # delta
    engineered[:, n, 6] = np.log1p(dengue_cases[:, n])                # log_cases

print(f"Engineered features shape: {engineered.shape}")  # (T, N, 7)

# Combine: raw (6) + engineered (7) = 13 base features
raw_case_features = np.concatenate([raw_case_features, engineered], axis=2)
print(f"Combined base features shape: {raw_case_features.shape}")  # (T, N, 13)
```

### Step 3: Log-Transform Targets (Cell 16)

In the target normalization section, change:

```python
# OLD
target_flat = raw_targets.reshape(T * N, 1)
target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target_flat).reshape(T, N, 1)

# NEW — log-transform BEFORE scaling
target_flat = np.log1p(raw_targets.reshape(T * N, 1))  # log1p handles zeros safely
target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target_flat).reshape(T, N, 1)
print(f"Target transform: log1p → StandardScaler")
```

### Step 4: Upgrade Model Architecture (Cell 20)

Replace the entire `EpiGraphModel` class with this improved version that adds skip connections and temporal attention:

```python
class EpiGraphModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1,
                 heads=2, dropout=0.2, num_base_features=13):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_base_features = num_base_features

        # Input projection (for residual)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # === DEEP PATH: GAT → LSTM → Temporal Attention ===
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)

        # Temporal attention: learn which timestep matters most
        self.temporal_attn = nn.Linear(hidden_dim, 1)

        # === SKIP PATH: raw base features → MLP ===
        self.skip_fc = nn.Sequential(
            nn.Linear(num_base_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # === COMBINE: deep (hidden_dim) + skip (hidden_dim//4) → prediction ===
        combined_dim = hidden_dim + hidden_dim // 4
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        b, t, n, f = x.size()

        # === DEEP PATH ===
        spatial_features_seq = []
        for i in range(t):
            xt = x[:, i, :, :]  # (B, N, F)
            batch_spatial = []
            for bi in range(b):
                h = self.gat1(xt[bi], edge_index)
                h = self.bn1(h)
                h = F.elu(h)
                h = self.dropout(h)
                h = self.gat2(h, edge_index)
                h = self.bn2(h)
                h = F.elu(h)
                residual = self.input_proj(xt[bi])
                h = h + residual
                batch_spatial.append(h)
            spatial_features_seq.append(torch.stack(batch_spatial))

        spatial_seq = torch.stack(spatial_features_seq, dim=1)  # (B, T, N, H)
        spatial_flat = spatial_seq.view(b * n, t, self.hidden_dim)

        lstm_out, _ = self.lstm(spatial_flat)  # (B*N, T, H)

        # Temporal attention (instead of just last timestep)
        attn_weights = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B*N, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B*N, H)

        # === SKIP PATH ===
        last_base = x[:, -1, :, :self.num_base_features]  # (B, N, 13)
        last_base_flat = last_base.reshape(b * n, self.num_base_features)
        skip_out = self.skip_fc(last_base_flat)  # (B*N, H//4)

        # === COMBINE ===
        combined = torch.cat([context, skip_out], dim=1)  # (B*N, H + H//4)
        out = self.fc(combined)  # (B*N, 1)
        return out.view(b, n, -1)
```

Update the model instantiation below the class:
```python
num_features = x_tensor.shape[2]
model = EpiGraphModel(
    num_nodes, num_features, HIDDEN_DIM,
    dropout=DROPOUT, num_base_features=NUM_BASE_FEATURES
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {total_params:,}")
```

### Step 5: Update Training Loop (Cell 22)

Change the scheduler:
```python
# OLD
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

# NEW
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
```

And in the training loop, change:
```python
# OLD
scheduler.step(avg_val)

# NEW
scheduler.step(epoch)
```

### Step 6: Fix Evaluation Inverse Transform (Cell 26)

Since you added `log1p`, the inverse transform must now also undo it:

```python
# OLD
preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

# NEW — undo StandardScaler, then undo log1p
preds = np.expm1(target_scaler.inverse_transform(preds_scaled.reshape(-1, 1))).flatten()
targets = np.expm1(target_scaler.inverse_transform(targets_scaled.reshape(-1, 1))).flatten()
```

**Do the same in Cell 30** (per-district metrics) and **Cell 32** (risk dashboard):
```python
# Cell 30: change both inverse_transform lines
d_preds = np.expm1(target_scaler.inverse_transform(preds_all[:, i, :])).flatten()
d_targets = np.expm1(target_scaler.inverse_transform(targets_all[:, i, :])).flatten()

# Cell 32: change risk score inverse transform
risk_scores = np.expm1(target_scaler.inverse_transform(risk_scaled.reshape(-1, 1))).flatten()
```

### Step 7: Retrain and Verify

Run the entire notebook top to bottom. Check:
- [ ] R² should be higher than 0.585 (target: >0.65)
- [ ] RMSE should be lower than 55.91
- [ ] Loss curves converge smoothly
- [ ] Per-district R² all positive

Save the model — it will create `epigraph_model_v2.pth`.

---

## PART 2: Flask Backend

Create a new file `app.py` in the project root. This server:
- Serves all website pages
- Loads the trained model on startup
- Provides 6 API endpoints (see `API_CONTRACT.md` for exact response shapes)

### Step 1: Add Flask to requirements.txt

Add this line to `requirements.txt`:
```
flask
```

### Step 2: Create `app.py`

```python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.nn import GATv2Conv

# ─── Configuration ───────────────────────────────────────────
DATA_DIR = "data"
CASES_FILE = os.path.join(DATA_DIR, "processed_cases.csv")
NEWS_FILE = os.path.join(DATA_DIR, "health_news.csv")
CONNECTIVITY_FILE = os.path.join(DATA_DIR, "connectivity.csv")
MODEL_PATH = "epigraph_model_v2.pth"
WINDOW_SIZE = 7
PCA_DIM = 16
HIDDEN_DIM = 128
NUM_BASE_FEATURES = 13

DISTRICT_COLORS = {
    "Ahmedabad": "#3b82f6",
    "Gandhinagar": "#8b5cf6",
    "Rajkot": "#06b6d4",
    "Surat": "#ec4899",
    "Vadodara": "#10b981"
}

# ─── App Setup ───────────────────────────────────────────────
app = Flask(__name__, static_folder="website", static_url_path="")

# ─── Load Data ───────────────────────────────────────────────
cases_df = pd.read_csv(CASES_FILE)
news_df = pd.read_csv(NEWS_FILE)
conn_df = pd.read_csv(CONNECTIVITY_FILE)

districts = sorted(cases_df["District"].unique())
node_mapping = {d: i for i, d in enumerate(districts)}
num_nodes = len(districts)

# ─── Build Edge Index ────────────────────────────────────────
adj_matrix = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    src = node_mapping.get(row["Source"])
    tgt = node_mapping.get(row["Target"])
    if src is not None and tgt is not None:
        adj_matrix[src, tgt] = row["Weight"]
        adj_matrix[tgt, src] = row["Weight"]
np.fill_diagonal(adj_matrix, 1.0)
edge_index = torch.tensor(adj_matrix, dtype=torch.float32).nonzero().t().contiguous()

# ─── Copy the EpiGraphModel class here (same as notebook Cell 20) ───
# IMPORTANT: Paste the exact same EpiGraphModel class from your updated notebook
# so the model weights load correctly.

class EpiGraphModel(nn.Module):
    # ... paste full class from notebook Step 4 ...
    pass  # REPLACE THIS with the actual class

# ─── Load Trained Model ─────────────────────────────────────
device = torch.device("cpu")  # CPU for serving
model = EpiGraphModel(
    num_nodes=num_nodes,
    input_dim=NUM_BASE_FEATURES + PCA_DIM,  # 13 + 16 = 29
    hidden_dim=HIDDEN_DIM,
    num_base_features=NUM_BASE_FEATURES
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ─── Precompute Predictions ─────────────────────────────────
# Replicate the notebook's preprocessing pipeline to get predictions
# You need: feature scaling, PCA, windowing, then inference
# Store results in a dict for fast API responses

# TODO: Implement this function following the notebook pipeline
# cached_predictions = precompute_predictions()

# ─── Page Routes ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("website", "index.html")

@app.route("/map")
def map_page():
    return send_from_directory("website", "map.html")

@app.route("/district")
def district_page():
    return send_from_directory("website", "district.html")

@app.route("/prevention")
def prevention_page():
    return send_from_directory("website", "prevention.html")

# ─── API Routes ──────────────────────────────────────────────

@app.route("/api/districts")
def api_districts():
    return jsonify({
        "districts": [
            {"name": d, "color": DISTRICT_COLORS.get(d, "#888")}
            for d in districts
        ]
    })

@app.route("/api/predictions")
def api_predictions():
    # TODO: Return cached_predictions computed on startup
    # For now, return placeholder structure matching API_CONTRACT.md
    # Replace this with real model inference results
    return jsonify({
        d: {"risk": 0.0, "level": "low"} for d in districts
    })

@app.route("/api/cases")
def api_cases():
    district = request.args.get("district")
    df = cases_df.copy()
    if district:
        df = df[df["District"] == district]

    result = {}
    for d in df["District"].unique():
        d_df = df[df["District"] == d].sort_values("Date")
        result[d] = {
            "district": d,
            "cases": [
                {"date": row["Date"], "value": int(row["dengue"])}
                for _, row in d_df.iterrows()
            ]
        }

    if district and district in result:
        return jsonify(result[district])
    return jsonify(result)

@app.route("/api/weather")
def api_weather():
    district = request.args.get("district")
    df = cases_df.copy()
    if district:
        df = df[df["District"] == district]

    result = {}
    for d in df["District"].unique():
        d_df = df[df["District"] == d]
        result[d] = {
            "district": d,
            "tmax": round(d_df[".MMAX"].mean(), 1),
            "tmin": round(d_df[".MMIN"].mean(), 1),
            "rain": round(d_df["..TMRF"].mean(), 1),
            "rh_am": round(d_df[".RH -0830"].mean()),
            "rh_pm": round(d_df[".RH -1730"].mean())
        }

    if district and district in result:
        return jsonify(result[district])
    return jsonify(result)

@app.route("/api/news")
def api_news():
    district = request.args.get("district")
    df = news_df.copy()
    if district:
        df = df[df["District"] == district]

    alerts = [
        {
            "date": row["Date"],
            "headline": row["Headline"],
            "type": row["Type"],
            "district": row["District"]
        }
        for _, row in df.iterrows()
    ]

    if district:
        return jsonify({"district": district, "alerts": alerts})
    return jsonify({"alerts": alerts})

@app.route("/api/connectivity")
def api_connectivity():
    edges = [
        {
            "source": row["Source"],
            "target": row["Target"],
            "weight": float(row["Weight"])
        }
        for _, row in conn_df.iterrows()
    ]
    return jsonify({"edges": edges})

# ─── Run Server ──────────────────────────────────────────────
if __name__ == "__main__":
    print("EpiGraph-AI server starting...")
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Districts: {districts}")
    print(f"Open http://localhost:5000")
    app.run(debug=True, port=5000)
```

### Step 3: Implement Prediction Inference

The hardest part is replicating the notebook's preprocessing in `app.py` to generate live predictions. You need to:

1. Load raw case features (same as Cell 14)
2. Compute engineered features (same as your new Step 2 cell)
3. Normalize with StandardScaler (same as Cell 16)
4. PCA compress BioBERT embeddings (same as Cell 16) — OR skip BioBERT entirely and use zero embeddings if you want to simplify
5. Build the last 7-day window
6. Run `model(window, edge_index)` → get predictions
7. Inverse transform: `np.expm1(target_scaler.inverse_transform(...))`
8. Assign risk levels: >15 = "high", 10-15 = "medium", <10 = "low"
9. Cache results in a dict

**Tip:** You can save the scalers and PCA from the notebook using `joblib`:
```python
# In notebook, after fitting scalers:
import joblib
joblib.dump(scaler, 'case_scaler.pkl')
joblib.dump(emb_scaler, 'emb_scaler.pkl')
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
```
Then load them in `app.py` instead of re-fitting.

### Step 4: Test

```bash
pip install flask
python app.py
```

Test each endpoint:
- http://localhost:5000/api/districts
- http://localhost:5000/api/predictions
- http://localhost:5000/api/cases?district=Ahmedabad
- http://localhost:5000/api/weather?district=Surat
- http://localhost:5000/api/news?district=Rajkot
- http://localhost:5000/api/connectivity

Verify response shapes match `API_CONTRACT.md`.

---

## PART 3: Commit and Push

```bash
git add app.py requirements.txt notebooks/EpiGraph_AI.ipynb
# Don't add .pth files (they're in .gitignore)
git commit -m "Add Flask backend + model improvements (log-transform, skip connections, temporal attention)"
git push -u origin backend-model
```

Then create a Pull Request on GitHub: `backend-model` → `main`.

---

## Checklist Before PR

- [ ] Notebook runs top-to-bottom without errors
- [ ] R² improved over baseline (0.585)
- [ ] `python app.py` starts without errors
- [ ] All 6 API endpoints return valid JSON
- [ ] Response shapes match API_CONTRACT.md
- [ ] No files inside `website/` were touched
- [ ] requirements.txt has flask added
