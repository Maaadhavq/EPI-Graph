"""
EpiGraph-AI: Retrain using the EXACT original architecture decoded from epigraph_model.pth.

Architecture:
  - 13 base features (6 raw + 7 engineered)
  - 16-dim learned BERT projection (bert_projection layer)
  - input_dim = 29, HIDDEN_DIM = 128
  - GATv2(2 heads) -> GATv2(1 head) -> LSTM(2 layer) -> LayerNorm
  - Temporal attention (Linear 128->1)
  - Skip path: Linear(13,64)->ReLU->Dropout->Linear(64,32)
  - Head: Linear(160,64)->ReLU->Dropout->Linear(64,1)
  - LR=0.001, CosineAnnealingWarmRestarts(T_0=30, T_mult=2)
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
from torch_geometric.nn import GATv2Conv
import joblib
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM      = 128
BERT_PROJ_DIM   = 16
NUM_BASE_FEATS  = 13
INPUT_DIM       = NUM_BASE_FEATS + BERT_PROJ_DIM  # 29
WINDOW_SIZE     = 7
EPOCHS          = 100
LR              = 0.001
WEIGHT_DECAY    = 1e-4
PATIENCE        = 20
BATCH_SIZE      = 16
DROPOUT         = 0.4

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

# ── Engineered Features (+7) → total 13 base features ─────────────────────────
dengue_col = raw_feat[:, :, 0]   # dengue is col 0 in feature_cols
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

# ── Normalize 13 base features ─────────────────────────────────────────────────
base_scaler = StandardScaler()
feat13_n = base_scaler.fit_transform(feat13.reshape(T*num_nodes, 13)).reshape(T, num_nodes, 13)
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
print(f"Base features: {feat13.shape}, BERT: {raw_emb.shape}")

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

# ── Build windowed dataset (project BERT inside dataloader for memory) ─────────
# Pre-project all BERT at once using the model's bert_projection
model.eval()
with torch.no_grad():
    bert_proj = model.bert_projection(raw_emb_t.to(device)).cpu().numpy()   # (T*N, 16)
bert_proj_3d = bert_proj.reshape(T, num_nodes, BERT_PROJ_DIM)

combined = np.concatenate([feat13_n, bert_proj_3d], axis=2)   # (T, N, 29)
x_all = torch.tensor(combined, dtype=torch.float32)

Xw, Yw = [], []
for i in range(T - WINDOW_SIZE):
    Xw.append(x_all[i:i+WINDOW_SIZE])
    Yw.append(y_all[i+WINDOW_SIZE:i+WINDOW_SIZE+1])
Xw, Yw = torch.stack(Xw), torch.stack(Yw)

n_total = len(Xw)
tr = int(n_total * 0.70); va = int(n_total * 0.85)
x_tr, y_tr = Xw[:tr].to(device),    Yw[:tr].to(device)
x_va, y_va = Xw[tr:va].to(device),  Yw[tr:va].to(device)
x_te, y_te = Xw[va:].to(device),    Yw[va:].to(device)
print(f"Train:{len(x_tr)}  Val:{len(x_va)}  Test:{len(x_te)}")

train_dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(TensorDataset(x_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(TensorDataset(x_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

# ── Training ───────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.HuberLoss(delta=1.0)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

best_val, patience_counter, best_state = float('inf'), 0, None
train_losses, val_losses = [], []

print(f"\nTraining (HIDDEN={HIDDEN_DIM}, LR={LR}, CosineWarmRestart T0=30)...")
for epoch in range(1, EPOCHS+1):
    model.train()
    tl = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb, edge_index), yb.squeeze(1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()
    avg_train = tl / len(train_dl)
    train_losses.append(avg_train)
    scheduler.step(epoch)

    model.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            vl += criterion(model(xb, edge_index), yb.squeeze(1)).item()
    avg_val = vl / max(len(val_dl), 1)
    val_losses.append(avg_val)

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
out = os.path.join(BASE, 'epigraph_model_v2.pth')
torch.save(model.state_dict(), out)

# Also save base_scaler for app.py
joblib.dump(base_scaler, os.path.join(BASE, 'base_scaler.pkl'))
joblib.dump(target_scaler, os.path.join(BASE, 'target_scaler_v3.pkl'))

print(f"\nSaved: {out}")
print(f"Saved: base_scaler.pkl, target_scaler_v3.pkl")
print(f"Val R2={r2_va:.4f}  Test R2={r2_te:.4f}")
