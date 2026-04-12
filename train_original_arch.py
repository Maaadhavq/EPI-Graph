"""
EpiGraph-AI: Retrain using the EXACT original architecture decoded from epigraph_model.pth.

Architecture:
  - 19 base features (8 raw + 11 engineered)  [v3: was 13 = 6+7]
  - 16-dim learned BERT projection (bert_projection layer)
  - input_dim = 35, HIDDEN_DIM = 128          [v3: was 29]
  - GATv2(2 heads) -> GATv2(1 head) -> LSTM(2 layer) -> LayerNorm
  - Temporal attention (Linear 128->1)
  - Skip path: Linear(19,64)->ReLU->Dropout->Linear(64,32)
  - Head: Linear(160,64)->ReLU->Dropout->Linear(64,1)

Improvements v3 (over v2):
  B1 - Seasonal encoding: sin/cos of week-of-year (+2 raw features -> 8 total)
  B2 - Extended lags: shift(3) + shift(4) (+2 engineered -> 9 total)
  B3 - Rolling weather: 4-week mean rain + 4-week mean rh_am (+2 engineered -> 11 total)
  B4 - Outbreak-weighted loss: 3× weight on top-quartile outbreak weeks
  A1 - PCA initialization for bert_projection (retained from v2)
  A2 - Periodic BERT re-projection every REPROJECT_INTERVAL epochs (retained from v2)
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, f1_score, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.nn import GATv2Conv
import joblib
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM          = 128
BERT_PROJ_DIM       = 16
NUM_BASE_FEATS      = 19    # v3: 8 raw (6+2 seasonal) + 11 engineered (7+2 lags+2 weather)
INPUT_DIM           = NUM_BASE_FEATS + BERT_PROJ_DIM  # 35
WINDOW_SIZE         = 7
EPOCHS              = 350       # more epochs for better convergence
LR                  = 0.0005    # reduced from 0.001 for more stable convergence
WEIGHT_DECAY        = 1e-4
PATIENCE            = 60        # longer patience to avoid premature stopping
BATCH_SIZE          = 8         # smaller batches for better gradient diversity
DROPOUT             = 0.25      # lower dropout: small dataset benefits from less regularization
REPROJECT_INTERVAL  = 20        # less frequent BERT reprojection for stability

# ── Load Data ──────────────────────────────────────────────────────────────────
cases_df = pd.read_csv(os.path.join(BASE, "data", "processed_cases.csv"))
conn_df  = pd.read_csv(os.path.join(BASE, "data", "connectivity.csv"))

districts = sorted(cases_df['District'].unique())
node_map  = {d: i for i, d in enumerate(districts)}
num_nodes = len(districts)
dates     = sorted(cases_df['Date'].unique())
T         = len(dates)
print(f"Districts: {districts}")
print(f"T={T}, N={num_nodes}")

# Edge index
adj = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    s = node_map.get(row['Source']); t = node_map.get(row['Target'])
    if s is not None and t is not None:
        adj[s,t]=row['Weight']; adj[t,s]=row['Weight']
np.fill_diagonal(adj, 1.0)
edge_index = torch.tensor(adj, dtype=torch.float32).nonzero().t().contiguous().to(device)

# ── Build Raw Features (T,N,6) and targets ─────────────────────────────────────
feature_cols = ['dengue', '.MMAX', '.MMIN', '..TMRF', '.RH -0830', '.RH -1730']
full_idx = pd.MultiIndex.from_product([dates, districts], names=['Date','District'])
cases_idx = cases_df[['Date','District']+feature_cols].set_index(['Date','District'])
cases_filled = cases_idx.reindex(full_idx, fill_value=0)

raw_feat = np.zeros((T, num_nodes, 6))
raw_tgt  = np.zeros((T, num_nodes, 1))
for ti, date in enumerate(dates):
    day = cases_filled.loc[date].reindex(districts, fill_value=0)
    raw_feat[ti] = day[feature_cols].values
    raw_tgt[ti]  = day[['dengue']].values

# B1 ── Seasonal encoding: sin/cos of week-of-year (+2 features -> 8 raw total) ─
date_idx  = pd.to_datetime(dates)
week_nums = date_idx.isocalendar().week.values.astype(float)  # (T,)
sin_week  = np.sin(2 * np.pi * week_nums / 52)
cos_week  = np.cos(2 * np.pi * week_nums / 52)
seasonal  = np.stack([sin_week, cos_week], axis=1)                         # (T, 2)
seasonal_feat = np.tile(seasonal[:, np.newaxis, :], (1, num_nodes, 1))    # (T, N, 2)
raw_feat  = np.concatenate([raw_feat, seasonal_feat], axis=2)             # (T, N, 8)
print(f"[v3] Added seasonal features -> raw_feat shape: {raw_feat.shape}")

# B2+B3 ── Engineered Features (+11) -> total 19 base features ──────────────────
dengue_col = raw_feat[:, :, 0]   # dengue is still col 0
rain_col   = raw_feat[:, :, 3]   # rain (col 3 of original 6)
rh_col     = raw_feat[:, :, 4]   # rh_am (col 4 of original 6)
eng = np.zeros((T, num_nodes, 11))
for ni in range(num_nodes):
    s    = pd.Series(dengue_col[:, ni])
    rain = pd.Series(rain_col[:, ni])
    rh   = pd.Series(rh_col[:, ni])
    eng[:,ni,0]  = s.rolling(4,  min_periods=1).mean()       # 4-week rolling mean
    eng[:,ni,1]  = s.rolling(8,  min_periods=1).mean()       # 8-week rolling mean
    eng[:,ni,2]  = s.rolling(4,  min_periods=1).std().fillna(0)  # 4-week rolling std
    eng[:,ni,3]  = s.shift(1).fillna(0)                      # lag-1
    eng[:,ni,4]  = s.shift(2).fillna(0)                      # lag-2
    eng[:,ni,5]  = s.diff().fillna(0)                        # week-over-week diff
    eng[:,ni,6]  = np.log1p(dengue_col[:, ni])               # log-transform
    eng[:,ni,7]  = s.shift(3).fillna(0)                      # B2: lag-3
    eng[:,ni,8]  = s.shift(4).fillna(0)                      # B2: lag-4
    eng[:,ni,9]  = rain.rolling(4, min_periods=1).mean()     # B3: 4-week mean rainfall
    eng[:,ni,10] = rh.rolling(4, min_periods=1).mean()       # B3: 4-week mean humidity

feat_base = np.concatenate([raw_feat, eng], axis=2)   # (T, N, 19)
print(f"[v3] Feature matrix shape: {feat_base.shape}  (19 base = 8 raw + 11 engineered)")

# ── Normalize 19 base features ─────────────────────────────────────────────────
base_scaler = StandardScaler()
feat13_n = base_scaler.fit_transform(feat_base.reshape(T*num_nodes, NUM_BASE_FEATS)).reshape(T, num_nodes, NUM_BASE_FEATS)
joblib.dump(base_scaler, os.path.join(BASE, 'base_scaler.pkl'))

# ── Load BERT embeddings (raw 768-dim) ─────────────────────────────────────────
EMBED_CACHE = os.path.join(BASE, "embeddings_cache_orig_32.pt")
if os.path.exists(EMBED_CACHE):
    print("Loading cached BERT embeddings...")
    c = torch.load(EMBED_CACHE, map_location='cpu', weights_only=False)
    raw_emb = c['embeddings']   # (T, N, 768)
else:
    print("ERROR: Run train_original.py first to generate embeddings cache.")
    exit(1)

raw_emb_t = torch.tensor(raw_emb.reshape(T*num_nodes, 768), dtype=torch.float32)  # (T*N, 768)
print(f"Base features: {feat_base.shape}, BERT: {raw_emb.shape}")

# ── Targets ────────────────────────────────────────────────────────────────────
target_scaler = StandardScaler()
tgt_n = target_scaler.fit_transform(raw_tgt.reshape(T*num_nodes,1)).reshape(T, num_nodes, 1)
joblib.dump(target_scaler, os.path.join(BASE, 'target_scaler_v3.pkl'))
y_all = torch.tensor(tgt_n, dtype=torch.float32)

# ── Model ──────────────────────────────────────────────────────────────────────
class EpiGraphModelV3(nn.Module):
    """Exact architecture matching epigraph_model.pth state_dict keys."""
    def __init__(self, num_nodes=5, input_dim=29, hidden_dim=128,
                 heads=2, dropout=0.4, num_base_features=13, bert_proj_dim=16):
        super().__init__()
        self.hidden_dim     = hidden_dim
        self.num_base       = num_base_features
        self.bert_proj_dim  = bert_proj_dim

        # Learned BERT projection (replaces external PCA)
        self.bert_projection = nn.Sequential(nn.Linear(768, bert_proj_dim))

        # Spatial encoder
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=1, concat=False, dropout=dropout)

        # Temporal encoder + norm
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Temporal attention
        self.temporal_attention = nn.Sequential(nn.Linear(hidden_dim, 1))

        # Skip path
        self.skip_fc = nn.Sequential(
            nn.Linear(num_base_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        # x: (B, T, N, 29) — already has projected BERT appended
        b, t, n, f = x.size()
        seq = []
        for i in range(t):
            xt = x[:, i]
            bs = []
            for bi in range(b):
                h = F.elu(self.gat1(xt[bi], edge_index))
                h = F.elu(self.gat2(h, edge_index))
                bs.append(h)
            seq.append(torch.stack(bs))
        spatial = torch.stack(seq, dim=1).view(b*n, t, self.hidden_dim)
        lstm_out, _ = self.lstm(spatial)
        lstm_out = self.layer_norm(lstm_out)

        attn = F.softmax(self.temporal_attention(lstm_out), dim=1)
        ctx  = (attn * lstm_out).sum(dim=1)

        skip = self.skip_fc(x[:, -1, :, :self.num_base].reshape(b*n, self.num_base))
        return self.fc(torch.cat([ctx, skip], dim=1)).view(b, n, 1)


model = EpiGraphModelV3(
    num_nodes=num_nodes, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT, num_base_features=NUM_BASE_FEATS, bert_proj_dim=BERT_PROJ_DIM
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Transfer learning: warm-start from v2 weights (compatible layers only) ────
# v2 and v3 share the same HIDDEN_DIM=128. Only gat1 (input_dim changed 29->35)
# and skip_fc (num_base changed 13->19) differ. Copy everything else from v2.
_v2_path = os.path.join(BASE, "epigraph_model_v2.pth")
if os.path.exists(_v2_path):
    print("Transfer learning: loading compatible v2 weights...")
    v2_state = torch.load(_v2_path, map_location="cpu", weights_only=False)
    v3_state  = model.state_dict()
    transferred, skipped = [], []
    for k, v in v2_state.items():
        if k in v3_state and v3_state[k].shape == v.shape:
            v3_state[k] = v
            transferred.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(v3_state)
    print(f"  Transferred {len(transferred)} tensors, re-init {len(skipped)}: {skipped}")
else:
    print("v2 weights not found — training from random init")

# ── Initialize bert_projection with PCA (replaces random init) ────────────────
# Random initialization means BERT features start as noise. PCA gives meaningful
# directions of maximum variance, warm-starting the projection layer.
print("Initializing bert_projection with PCA components...")
pca_init = PCA(n_components=BERT_PROJ_DIM)
pca_init.fit(raw_emb.reshape(T * num_nodes, 768))
W_pca = torch.tensor(pca_init.components_, dtype=torch.float32)      # (16, 768)
b_pca = torch.tensor(
    -(pca_init.mean_ @ pca_init.components_.T), dtype=torch.float32  # (16,)
)
model.bert_projection[0].weight.data = W_pca
model.bert_projection[0].bias.data   = b_pca
print(f"PCA init done. Explained variance: {pca_init.explained_variance_ratio_.sum()*100:.1f}%")

# ── Helper: build windowed dataset from current bert_projection ────────────────
def build_datasets():
    """Project BERT with current model weights, build windows and splits."""
    model.eval()
    with torch.no_grad():
        bp = model.bert_projection(raw_emb_t.to(device)).cpu().numpy()  # (T*N, 16)
    bp3d = bp.reshape(T, num_nodes, BERT_PROJ_DIM)
    comb = np.concatenate([feat13_n, bp3d], axis=2)   # (T, N, 35)
    xa   = torch.tensor(comb, dtype=torch.float32)

    Xw_l, Yw_l = [], []
    for i in range(T - WINDOW_SIZE):
        Xw_l.append(xa[i:i+WINDOW_SIZE])
        Yw_l.append(y_all[i+WINDOW_SIZE:i+WINDOW_SIZE+1])
    Xww = torch.stack(Xw_l); Yww = torch.stack(Yw_l)

    nt = len(Xww); _tr = int(nt * 0.70); _va = int(nt * 0.85)
    _x_tr = Xww[:_tr].to(device);    _y_tr = Yww[:_tr].to(device)
    _x_va = Xww[_tr:_va].to(device); _y_va = Yww[_tr:_va].to(device)
    _x_te = Xww[_va:].to(device);    _y_te = Yww[_va:].to(device)
    _tdl = DataLoader(TensorDataset(_x_tr, _y_tr), batch_size=BATCH_SIZE, shuffle=True)
    _vdl = DataLoader(TensorDataset(_x_va, _y_va), batch_size=BATCH_SIZE, shuffle=False)
    _edl = DataLoader(TensorDataset(_x_te, _y_te), batch_size=BATCH_SIZE, shuffle=False)
    return _tdl, _vdl, _edl

train_dl, val_dl, test_dl = build_datasets()
print(f"Train:{len(train_dl.dataset)}  Val:{len(val_dl.dataset)}  Test:{len(test_dl.dataset)}")

# ── Training ───────────────────────────────────────────────────────────────────
optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
huber_base = nn.HuberLoss(delta=1.0, reduction='none')   # B4: element-wise for weighting
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

# B4 ── Outbreak weight threshold: 75th percentile of scaled targets ────────────
_q75_scaled     = float(np.quantile(tgt_n.flatten(), 0.75))  # in normalized space
OUTBREAK_WEIGHT = 2.0   # 2× weight on outbreak weeks — less aggressive than 3× on small dataset

def weighted_loss(pred, target):
    """B4: Huber loss with 4× weight on outbreak-level weeks.
    pred:   (B, N, 1) — model output
    target: (B, 1, N, 1) — from DataLoader (stacked (1,N,1) slices)
    """
    pred_sq = pred.squeeze(-1)                  # (B, N)
    tgt_sq  = target.squeeze(1).squeeze(-1)     # (B, N) — squeeze both extra dims
    elem    = huber_base(pred_sq, tgt_sq)       # (B, N)
    weights = torch.where(tgt_sq > _q75_scaled,
                          torch.full_like(tgt_sq, OUTBREAK_WEIGHT),
                          torch.ones_like(tgt_sq))
    return (elem * weights).mean()

best_val, patience_counter, best_state = float('inf'), 0, None
train_losses, val_losses = [], []

print(f"\nTraining v3 (HIDDEN={HIDDEN_DIM}, INPUT={INPUT_DIM}, LR={LR}, outbreak_weight={OUTBREAK_WEIGHT})...")
for epoch in range(1, EPOCHS+1):
    # Periodically re-project BERT so bert_projection actually learns
    if epoch > 1 and (epoch - 1) % REPROJECT_INTERVAL == 0:
        train_dl, val_dl, test_dl = build_datasets()

    model.train()
    tl = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = weighted_loss(model(xb, edge_index), yb)   # B4 weighted loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()
    avg_train = tl / len(train_dl)
    train_losses.append(avg_train)

    model.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            vl += weighted_loss(model(xb, edge_index), yb).item()
    avg_val = vl / max(len(val_dl), 1)
    val_losses.append(avg_val)
    scheduler.step(avg_val)   # ReduceLROnPlateau needs val loss

    if avg_val < best_val:
        best_val = avg_val; patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " <- best"
    else:
        patience_counter += 1; marker = ""

    if epoch % 10 == 0 or patience_counter == 0:
        print(f"  ep {epoch:3d}  train={avg_train:.4f}  val={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}{marker}")

    if patience_counter >= PATIENCE:
        print(f"  Early stop at epoch {epoch}"); break

model.load_state_dict(best_state); model.to(device)
# Rebuild test_dl with best model's bert_projection
_, _, test_dl = build_datasets()

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(dl, label):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(model(xb, edge_index).cpu().numpy())
            ts.append(yb.squeeze(1).cpu().numpy())
    p = np.concatenate(ps).reshape(-1, 1)
    t = np.concatenate(ts).reshape(-1, 1)
    p_inv = np.clip(target_scaler.inverse_transform(p).flatten(), 0, None)
    t_inv = target_scaler.inverse_transform(t).flatten()
    mae  = mean_absolute_error(t_inv, p_inv)
    rmse = float(np.sqrt(np.mean((p_inv - t_inv)**2)))
    r2   = r2_score(t_inv, p_inv)
    nz   = t_inv > 0
    mape = np.mean(np.abs((t_inv[nz]-p_inv[nz])/t_inv[nz]))*100 if nz.sum() else float('nan')
    thr  = np.median(t_inv)
    f1   = f1_score((t_inv>thr).astype(int),(p_inv>thr).astype(int), zero_division=0)
    oacc = accuracy_score((t_inv>thr).astype(int),(p_inv>thr).astype(int))*100
    print(f"  {label}  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  MAPE={mape:.1f}%  F1={f1:.4f}  OutbreakAcc={oacc:.1f}%")
    return r2

print("\n" + "="*65)
print("  FINAL RESULTS")
print("="*65)
r2_tr = evaluate(train_dl, "Train:")
r2_va = evaluate(val_dl,   "Val:  ")
r2_te = evaluate(test_dl,  "Test: ")

# ── Save ───────────────────────────────────────────────────────────────────────
# v3 model: new features (19 base = 8 raw + 11 eng) + outbreak-weighted loss
out = os.path.join(BASE, 'epigraph_model_v3.pth')
torch.save(model.state_dict(), out)

# Save scalers (base_scaler now fitted on 19-feature matrix)
joblib.dump(base_scaler,   os.path.join(BASE, 'base_scaler.pkl'))
joblib.dump(target_scaler, os.path.join(BASE, 'target_scaler_v3.pkl'))

print(f"\nSaved: {out}  (v3 — 19 base features, seasonal + extended lags + rolling weather + outbreak loss)")
print(f"Saved: base_scaler.pkl  (fitted on 19 features), target_scaler_v3.pkl")
print(f"Val R2={r2_va:.4f}  Test R2={r2_te:.4f}")
print("\nTo deploy: rename epigraph_model_v3.pth -> epigraph_model_v2.pth  (or update MODEL_PATH in app.py)")
