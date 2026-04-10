"""
EpiGraph-AI: Final Training — Improved model (best config).
Runs one optimised experiment, prints comparison vs baseline, saves model.
"""
import os, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.nn import GATv2Conv
warnings.filterwarnings('ignore')

BASE   = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")

# ── Hyperparameters (chosen after sweep) ────────────────────────────────
HIDDEN_DIM = 64        # small → less overfitting on tiny dataset
PCA_DIM    = 16
DROPOUT    = 0.3
LR         = 0.0008
BATCH_SIZE = 8
WINDOW     = 7
EPOCHS     = 200
PATIENCE   = 50

# ── Load Data ────────────────────────────────────────────────────────────
cases_df = pd.read_csv(os.path.join(BASE, "data", "processed_cases.csv"))
conn_df  = pd.read_csv(os.path.join(BASE, "data", "connectivity.csv"))

districts = list(cases_df["District"].unique())
node_map  = {d: i for i, d in enumerate(districts)}
num_nodes = len(districts)
dates     = sorted(cases_df["Date"].unique())
T         = len(dates)

adj = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    s, t = node_map.get(row["Source"]), node_map.get(row["Target"])
    if s is not None and t is not None:
        adj[s, t] = row["Weight"]; adj[t, s] = row["Weight"]
np.fill_diagonal(adj, 1.0)
edge_index = torch.tensor(adj, dtype=torch.float32).nonzero().t().contiguous()

feature_cols = [".MMAX", ".MMIN", "..TMRF", ".RH -0830", ".RH -1730", "dengue"]
raw_features = np.zeros((T, num_nodes, 6))
raw_targets  = np.zeros((T, num_nodes, 1))
for di, d in enumerate(districts):
    ddf = cases_df[cases_df["District"] == d].sort_values("Date").reset_index(drop=True)
    for ci, col in enumerate(feature_cols):
        raw_features[:, di, ci] = ddf[col].values
    raw_targets[:, di, 0] = ddf["dengue"].values

bert_flat = torch.load(os.path.join(BASE, "bert_cache_181d2c65.pt"),
                        map_location="cpu", weights_only=False).numpy()
bert_emb = np.zeros((T, num_nodes, 768))
for idx, (_, row) in enumerate(cases_df.iterrows()):
    ti = dates.index(row["Date"])
    ni = node_map[row["District"]]
    bert_emb[ti, ni, :] = bert_flat[idx]

print(f"Districts: {districts}")
print(f"T={T}, N={num_nodes}")

# ── Feature Engineering ──────────────────────────────────────────────────
dengue = raw_features[:, :, 5]
eng = np.zeros((T, num_nodes, 7))
for ni in range(num_nodes):
    s = pd.Series(dengue[:, ni])
    eng[:, ni, 0] = s.rolling(4,  min_periods=1).mean()
    eng[:, ni, 1] = s.rolling(8,  min_periods=1).mean()
    eng[:, ni, 2] = s.rolling(4,  min_periods=1).std().fillna(0)
    eng[:, ni, 3] = s.shift(1).fillna(0)
    eng[:, ni, 4] = s.shift(2).fillna(0)
    eng[:, ni, 5] = s.diff().fillna(0)
    eng[:, ni, 6] = np.log1p(dengue[:, ni])

feat = np.concatenate([raw_features, eng], axis=2)  # (T, N, 13)
NUM_BASE = 13
print(f"Base features: {NUM_BASE}, PCA dims: {PCA_DIM}")

# ── Normalise ────────────────────────────────────────────────────────────
feat_s = StandardScaler()
feat_n = feat_s.fit_transform(feat.reshape(T * num_nodes, -1)).reshape(T, num_nodes, -1)

bert_s = StandardScaler()
bert_n = bert_s.fit_transform(bert_emb.reshape(T * num_nodes, 768))
pca    = PCA(n_components=PCA_DIM)
bert_p = pca.fit_transform(bert_n).reshape(T, num_nodes, PCA_DIM)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

combined = np.concatenate([feat_n, bert_p], axis=2)
x_all    = torch.tensor(combined, dtype=torch.float32)
NUM_FEAT = combined.shape[2]

# log1p → StandardScaler on targets
tgt_raw   = np.log1p(raw_targets.reshape(T * num_nodes, 1))
tgt_s     = StandardScaler()
tgt_n     = tgt_s.fit_transform(tgt_raw).reshape(T, num_nodes, 1)
y_all     = torch.tensor(tgt_n, dtype=torch.float32)

# ── Windowing ────────────────────────────────────────────────────────────
Xw, Yw = [], []
for i in range(T - WINDOW):
    Xw.append(x_all[i:i+WINDOW])
    Yw.append(y_all[i+WINDOW:i+WINDOW+1])
Xw, Yw = torch.stack(Xw), torch.stack(Yw)

n = len(Xw); tr = int(n*0.70); va = int(n*0.85)
x_tr, y_tr = Xw[:tr], Yw[:tr]
x_va, y_va = Xw[tr:va], Yw[tr:va]
x_te, y_te = Xw[va:], Yw[va:]
print(f"Windows — train:{len(x_tr)} val:{len(x_va)} test:{len(x_te)}")

# ── Model ────────────────────────────────────────────────────────────────
class ImprovedModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, dropout=0.3, num_base=13):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_base   = num_base
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Spatial
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=2, dropout=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim * 2)
        self.gat2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2  = nn.BatchNorm1d(hidden_dim)
        # Temporal
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        # Temporal attention
        self.t_attn = nn.Linear(hidden_dim, 1)
        # Skip path
        self.skip = nn.Sequential(
            nn.Linear(num_base, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        b, t, n, f = x.size()
        sp_seq = []
        for i in range(t):
            xt = x[:, i]
            batch = []
            for bi in range(b):
                h = F.elu(self.bn1(self.gat1(xt[bi], edge_index)))
                h = self.drop(h)
                h = F.elu(self.bn2(self.gat2(h, edge_index)))
                h = h + self.input_proj(xt[bi])
                batch.append(h)
            sp_seq.append(torch.stack(batch))
        sp = torch.stack(sp_seq, dim=1).view(b * n, t, self.hidden_dim)
        lo, _ = self.lstm(sp)
        # Temporal attention
        aw = F.softmax(self.t_attn(lo), dim=1)
        ctx = (aw * lo).sum(dim=1)
        # Skip
        sk = self.skip(x[:, -1, :, :self.num_base].reshape(b * n, self.num_base))
        return self.head(torch.cat([ctx, sk], dim=1)).view(b, n, 1)


model = ImprovedModel(num_nodes, NUM_FEAT, HIDDEN_DIM, DROPOUT, NUM_BASE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Train ────────────────────────────────────────────────────────────────
opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
crit  = nn.HuberLoss(delta=1.0)
tr_dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
va_dl = DataLoader(TensorDataset(x_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

best_val, no_imp, best_state = float('inf'), 0, None

print("\nTraining...")
for ep in range(1, EPOCHS + 1):
    model.train()
    tl = sum(
        (opt.zero_grad() or True) and
        (loss := crit(model(xb, edge_index), yb.squeeze(1))) and
        loss.backward() or
        (nn.utils.clip_grad_norm_(model.parameters(), 1.0) and False) or
        opt.step() or loss.item()
        for xb, yb in tr_dl
    )
    sched.step(ep)
    model.eval()
    with torch.no_grad():
        vl = sum(crit(model(xb, edge_index), yb.squeeze(1)).item() for xb, yb in va_dl)
    avg_v = vl / max(len(va_dl), 1)

    if avg_v < best_val:
        best_val = avg_v; no_imp = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_imp += 1

    if ep % 25 == 0:
        print(f"  ep {ep:3d}  val={avg_v:.4f}  best={best_val:.4f}")
    if no_imp >= PATIENCE:
        print(f"  Early stop at epoch {ep}"); break

model.load_state_dict(best_state)

# ── Evaluate ─────────────────────────────────────────────────────────────
def get_metrics(model, x_te, y_te):
    model.eval()
    dl = DataLoader(TensorDataset(x_te, y_te), batch_size=32)
    ps, ts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(model(xb, edge_index).cpu().numpy())
            ts.append(yb.squeeze(1).cpu().numpy())
    ps = tgt_s.inverse_transform(np.concatenate(ps).reshape(-1, 1))
    ts = tgt_s.inverse_transform(np.concatenate(ts).reshape(-1, 1))
    p  = np.clip(np.expm1(ps).flatten(), 0, None)
    t  = np.expm1(ts).flatten()

    mae  = mean_absolute_error(t, p)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    r2   = r2_score(t, p)
    thresh = np.median(t)
    po, ao = (p > thresh).astype(int), (t > thresh).astype(int)
    tp = np.sum((po==1)&(ao==1)); fp = np.sum((po==1)&(ao==0)); fn = np.sum((po==0)&(ao==1))
    pr = tp/max(tp+fp,1); re = tp/max(tp+fn,1)
    f1 = 2*pr*re/max(pr+re,1e-8)
    oacc = np.mean(po == ao)

    print(f"\n  MAE:          {mae:.2f}")
    print(f"  RMSE:         {rmse:.2f}")
    print(f"  R²:           {r2:.4f}")
    print(f"  Outbreak F1:  {f1:.4f}")
    print(f"  Outbreak Acc: {oacc:.4f}")
    print(f"\n  Per-district:")
    ps2d = np.concatenate([model(x_te[[i]], edge_index).detach().numpy() for i in range(len(x_te))], axis=0)
    ts2d = y_te.squeeze(1).numpy()
    for di, d in enumerate(districts):
        dp = np.clip(np.expm1(tgt_s.inverse_transform(ps2d[:,di,:].reshape(-1,1))).flatten(), 0, None)
        dt = np.expm1(tgt_s.inverse_transform(ts2d[:,di,:].reshape(-1,1))).flatten()
        print(f"    {d:15s}  MAE={mean_absolute_error(dt,dp):6.1f}  "
              f"RMSE={np.sqrt(np.mean((dp-dt)**2)):6.1f}  R²={r2_score(dt,dp):.4f}")
    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,4),
            "F1": round(f1,4), "OAcc": round(oacc,4)}

print("\n" + "="*55)
print("  BASELINE (from train_improved.py)")
print("="*55)
print("  MAE:          52.03")
print("  RMSE:         103.08")
print("  R²:           -0.1299")
print("  Outbreak F1:  0.6486")
print("  Outbreak Acc: 0.4800")

print("\n" + "="*55)
print("  IMPROVED MODEL (final)")
print("="*55)
m = get_metrics(model, x_te, y_te)

print("\n" + "="*55)
print("  DELTA vs BASELINE")
print("="*55)
bl = {"MAE":52.03, "RMSE":103.08, "R2":-0.1299, "F1":0.6486, "OAcc":0.48}
for k, bv in bl.items():
    iv = m[k]
    if k in ("MAE","RMSE"):
        pct = (bv - iv) / max(abs(bv), 1e-8) * 100
        tag = "BETTER" if pct > 0 else "WORSE"
    else:
        pct = (iv - bv) / max(abs(bv), 1e-8) * 100
        tag = "BETTER" if pct > 0 else "WORSE"
    print(f"  {k:<12}  {bv:>9.4f} → {iv:>9.4f}  {pct:>+7.1f}%  {tag}")

# Save
out = os.path.join(BASE, "epigraph_model_v2.pth")
torch.save(model.state_dict(), out)
print(f"\nModel saved → {out}")
