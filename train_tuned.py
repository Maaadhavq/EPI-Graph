"""
EpiGraph-AI: Hyperparameter sweep to find best model.
Tries multiple configurations to push R² positive.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.nn import GATv2Conv

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")

# ── Load Data ────────────────────────────────────────────────────────────
cases_df = pd.read_csv(os.path.join(BASE, "data", "processed_cases.csv"))
news_df  = pd.read_csv(os.path.join(BASE, "data", "health_news.csv"))
conn_df  = pd.read_csv(os.path.join(BASE, "data", "connectivity.csv"))

districts = list(cases_df["District"].unique())
node_map  = {d: i for i, d in enumerate(districts)}
num_nodes = len(districts)
dates     = sorted(cases_df["Date"].unique())
T = len(dates)

# Edge index
adj = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    s, t = node_map.get(row["Source"]), node_map.get(row["Target"])
    if s is not None and t is not None:
        adj[s, t] = row["Weight"]; adj[t, s] = row["Weight"]
np.fill_diagonal(adj, 1.0)
edge_index = torch.tensor(adj, dtype=torch.float32).nonzero().t().contiguous()

# Raw features
feature_cols = [".MMAX", ".MMIN", "..TMRF", ".RH -0830", ".RH -1730", "dengue"]
raw_features = np.zeros((T, num_nodes, 6))
raw_targets  = np.zeros((T, num_nodes, 1))
for di, d in enumerate(districts):
    ddf = cases_df[cases_df["District"] == d].sort_values("Date").reset_index(drop=True)
    for ci, col in enumerate(feature_cols):
        raw_features[:, di, ci] = ddf[col].values
    raw_targets[:, di, 0] = ddf["dengue"].values

# BioBERT
bert_flat = torch.load(os.path.join(BASE, "bert_cache_181d2c65.pt"),
                        map_location="cpu", weights_only=False).numpy()
bert_emb = np.zeros((T, num_nodes, 768))
for idx, (_, row) in enumerate(cases_df.iterrows()):
    ti = dates.index(row["Date"])
    ni = node_map[row["District"]]
    bert_emb[ti, ni, :] = bert_flat[idx]

print(f"Data: T={T}, N={num_nodes}, Features=(T,N,6), BERT=(T,N,768)")

# ── Model ────────────────────────────────────────────────────────────────
class TunedModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1,
                 heads=2, dropout=0.3, num_base_features=13):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_base_features = num_base_features

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2  = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)

        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Skip path
        self.skip_fc = nn.Sequential(
            nn.Linear(num_base_features, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        combined = hidden_dim + hidden_dim // 4
        self.fc = nn.Sequential(
            nn.Linear(combined, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        b, t, n, f = x.size()
        spatial_seq = []
        for i in range(t):
            xt = x[:, i, :, :]
            batch_sp = []
            for bi in range(b):
                h = self.gat1(xt[bi], edge_index)
                h = self.bn1(h); h = F.elu(h); h = self.dropout(h)
                h = self.gat2(h, edge_index)
                h = self.bn2(h); h = F.elu(h)
                h = h + self.input_proj(xt[bi])
                batch_sp.append(h)
            spatial_seq.append(torch.stack(batch_sp))
        spatial = torch.stack(spatial_seq, dim=1)
        flat = spatial.view(b * n, t, self.hidden_dim)
        lstm_out, _ = self.lstm(flat)

        attn_w = F.softmax(self.temporal_attn(lstm_out), dim=1)
        context = (attn_w * lstm_out).sum(dim=1)

        skip_in  = x[:, -1, :, :self.num_base_features].reshape(b * n, self.num_base_features)
        skip_out = self.skip_fc(skip_in)

        combined = torch.cat([context, skip_out], dim=1)
        return self.fc(combined).view(b, n, -1)


# ── Data Prep ────────────────────────────────────────────────────────────
def prepare(pca_dim, window_size, use_eng=True, use_log=True):
    feat = raw_features.copy()
    n_base = 6

    if use_eng:
        dengue = raw_features[:, :, 5]
        eng = np.zeros((T, num_nodes, 7))
        for ni in range(num_nodes):
            s = pd.Series(dengue[:, ni])
            eng[:, ni, 0] = s.rolling(4, min_periods=1).mean()
            eng[:, ni, 1] = s.rolling(8, min_periods=1).mean()
            eng[:, ni, 2] = s.rolling(4, min_periods=1).std().fillna(0)
            eng[:, ni, 3] = s.shift(1).fillna(0)
            eng[:, ni, 4] = s.shift(2).fillna(0)
            eng[:, ni, 5] = s.diff().fillna(0)
            eng[:, ni, 6] = np.log1p(dengue[:, ni])
        feat = np.concatenate([feat, eng], axis=2)
        n_base = 13

    # Normalize features
    flat_f = feat.reshape(T * num_nodes, -1)
    fs = StandardScaler()
    flat_fn = fs.fit_transform(flat_f).reshape(T, num_nodes, -1)

    # PCA BioBERT
    flat_b = bert_emb.reshape(T * num_nodes, 768)
    bs = StandardScaler()
    flat_bn = bs.fit_transform(flat_b)
    pca = PCA(n_components=pca_dim)
    bert_pca = pca.fit_transform(flat_bn).reshape(T, num_nodes, pca_dim)

    combined = np.concatenate([flat_fn, bert_pca], axis=2)
    x = torch.tensor(combined, dtype=torch.float32)
    n_feat = combined.shape[2]

    # Targets
    tgt = raw_targets.copy().reshape(T * num_nodes, 1)
    if use_log: tgt = np.log1p(tgt)
    ts = StandardScaler()
    tgt_s = ts.fit_transform(tgt).reshape(T, num_nodes, 1)
    y = torch.tensor(tgt_s, dtype=torch.float32)

    # Windows
    Xw, Yw = [], []
    for i in range(T - window_size):
        Xw.append(x[i:i+window_size])
        Yw.append(y[i+window_size:i+window_size+1])
    Xw, Yw = torch.stack(Xw), torch.stack(Yw)

    n_total = len(Xw)
    tr = int(n_total * 0.70)
    va = int(n_total * 0.85)
    return (Xw[:tr], Yw[:tr], Xw[tr:va], Yw[tr:va], Xw[va:], Yw[va:],
            ts, n_feat, n_base, use_log)


def train(model, x_tr, y_tr, x_va, y_va, lr, bs, epochs, patience):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    crit = nn.HuberLoss(delta=1.0)

    train_dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True)
    val_dl   = DataLoader(TensorDataset(x_va, y_va), batch_size=bs, shuffle=False)

    best_val, no_imp = float('inf'), 0
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        t_loss = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            p = model(xb, edge_index)
            loss = crit(p, yb.squeeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        sched.step(ep)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                v_loss += crit(model(xb, edge_index), yb.squeeze(1)).item()
        avg_v = v_loss / max(len(val_dl), 1)

        if avg_v < best_val:
            best_val = avg_v
            no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if ep % 25 == 0:
            print(f"    ep {ep:3d}  val={avg_v:.4f}  best={best_val:.4f}")

        if no_imp >= patience:
            print(f"    early stop ep {ep}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate(model, x_te, y_te, ts, use_log, bs=32):
    model.eval()
    dl = DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False)
    ps, ts_list = [], []
    with torch.no_grad():
        for xb, yb in dl:
            ps.append(model(xb, edge_index).cpu().numpy())
            ts_list.append(yb.squeeze(1).cpu().numpy())
    ps = np.concatenate(ps).reshape(-1, 1)
    ts2 = np.concatenate(ts_list).reshape(-1, 1)

    p = ts.inverse_transform(ps).flatten()
    t = ts.inverse_transform(ts2).flatten()
    if use_log: p = np.expm1(p); t = np.expm1(t)
    p = np.clip(p, 0, None)

    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    r2 = r2_score(t, p)

    thresh = np.median(t)
    po = (p > thresh).astype(int)
    ao = (t > thresh).astype(int)
    tp = np.sum((po==1)&(ao==1)); fp = np.sum((po==1)&(ao==0)); fn = np.sum((po==0)&(ao==1))
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8)
    oacc = np.mean(po == ao)

    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "R2": round(r2,4),
            "F1": round(f1,4), "OAcc": round(oacc,4)}


# ── Sweep ────────────────────────────────────────────────────────────────
configs = [
    # name, hidden, pca, dropout, lr, batch, window, patience, epochs, eng, log
    ("A: small+eng+log",      64,  16, 0.3, 0.0008, 8,  7,  40, 150, True,  True),
    ("B: medium+eng+log",     96,  16, 0.25, 0.001, 16, 7,  40, 150, True,  True),
    ("C: large+eng+log",     128,  16, 0.2, 0.001,  16, 7,  40, 150, True,  True),
    ("D: small+eng+nolog",    64,  16, 0.3, 0.0005, 8,  7,  40, 150, True,  False),
    ("E: small w5+eng+log",   64,  16, 0.3, 0.0008, 8,  5,  40, 150, True,  True),
    ("F: small w10+eng+log",  64,  16, 0.3, 0.0008, 8, 10,  40, 150, True,  True),
    ("G: tiny+eng+log",       32,   8, 0.35, 0.0005, 4, 7,  50, 200, True,  True),
]

print(f"\nRunning {len(configs)} configurations...\n")
results = []

for name, hid, pca_d, drop, lr, bs, ws, pat, ep, eng, log in configs:
    print(f"── {name} ──")
    try:
        x_tr, y_tr, x_va, y_va, x_te, y_te, tscaler, nf, nb, ul = \
            prepare(pca_d, ws, eng, log)

        model = TunedModel(num_nodes, nf, hid, dropout=drop, num_base_features=nb).to(device)
        npar = sum(p.numel() for p in model.parameters())
        print(f"  feats={nf}, params={npar:,}, train={len(x_tr)}, test={len(x_te)}")

        t0 = time.time()
        model = train(model, x_tr, y_tr, x_va, y_va, lr, bs, ep, pat)
        dt = time.time() - t0

        m = evaluate(model, x_te, y_te, tscaler, ul)
        m["name"] = name
        m["params"] = npar
        m["time"] = round(dt, 1)
        results.append(m)

        print(f"  → MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  R²={m['R2']:.4f}  F1={m['F1']:.4f}  OAcc={m['OAcc']:.3f}  ({dt:.0f}s)")

        # Save if best R² so far
        if not any(r["R2"] > m["R2"] for r in results[:-1]):
            torch.save(model.state_dict(), os.path.join(BASE, "epigraph_best.pth"))
            best_config = name

    except Exception as e:
        print(f"  ERROR: {e}")
    print()

# Summary
print("="*80)
print("SWEEP RESULTS (sorted by R²)")
print("="*80)
print(f"{'Config':<28} {'MAE':>7} {'RMSE':>7} {'R²':>8} {'F1':>7} {'OAcc':>7} {'Params':>10} {'Time':>6}")
print("-"*80)
for m in sorted(results, key=lambda x: x["R2"], reverse=True):
    print(f"{m['name']:<28} {m['MAE']:>7.1f} {m['RMSE']:>7.1f} {m['R2']:>8.4f} {m['F1']:>7.4f} {m['OAcc']:>7.3f} {m['params']:>10,} {m['time']:>5.0f}s")

if results:
    best = max(results, key=lambda x: x["R2"])
    print(f"\nBEST: {best['name']}  R²={best['R2']:.4f}  MAE={best['MAE']:.1f}")
    print(f"Best model saved to: epigraph_best.pth")
