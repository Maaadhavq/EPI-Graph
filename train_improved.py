"""
EpiGraph-AI: Improved Training Script
Runs BASELINE vs IMPROVED model side-by-side and compares metrics.

Improvements implemented:
  1. Hyperparameters: HIDDEN_DIM 64→128, PCA_DIM 32→16, DROPOUT 0.4→0.2, LR 0.0005→0.001, BATCH_SIZE 8→16
  2. Engineered temporal features: rolling means, lags, delta, log-cases (7 extra features)
  3. Log-transform on targets before scaling
  4. Architecture: skip connections + temporal attention
  5. CosineAnnealingWarmRestarts scheduler
  6. Fixed inverse transform (expm1 after inverse scaler)
"""

import os
import sys
import time
import hashlib
import warnings
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

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE, "data")
CASES_FILE   = os.path.join(DATA_DIR, "processed_cases.csv")
NEWS_FILE    = os.path.join(DATA_DIR, "health_news.csv")
CONN_FILE    = os.path.join(DATA_DIR, "connectivity.csv")
BERT_CACHE   = os.path.join(BASE, "bert_cache_181d2c65.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ════════════════════════════════════════════════════════════════════════════
#  SHARED DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

cases_df = pd.read_csv(CASES_FILE)
news_df  = pd.read_csv(NEWS_FILE)
conn_df  = pd.read_csv(CONN_FILE)

# Preserve the original order of districts as they appear in the CSV
# (Ahmedabad, Surat, Vadodara, Rajkot, Gandhinagar)
districts = list(cases_df["District"].unique())
node_map  = {d: i for i, d in enumerate(districts)}
num_nodes = len(districts)
dates     = sorted(cases_df["Date"].unique())
T         = len(dates)
print(f"Districts: {districts}")
print(f"Timesteps: {T}, Nodes: {num_nodes}")

# ── Edge Index ──
adj = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    s, t = node_map.get(row["Source"]), node_map.get(row["Target"])
    if s is not None and t is not None:
        adj[s, t] = row["Weight"]
        adj[t, s] = row["Weight"]
np.fill_diagonal(adj, 1.0)
edge_index = torch.tensor(adj, dtype=torch.float32).nonzero().t().contiguous().to(device)

# ── Raw Features (T, N, 6) and Targets (T, N, 1) ──
feature_cols = [".MMAX", ".MMIN", "..TMRF", ".RH -0830", ".RH -1730", "dengue"]
raw_features = np.zeros((T, num_nodes, 6))
raw_targets  = np.zeros((T, num_nodes, 1))

for di, d in enumerate(districts):
    ddf = cases_df[cases_df["District"] == d].copy()
    ddf = ddf.sort_values("Date").reset_index(drop=True)
    for col_i, col in enumerate(feature_cols):
        raw_features[:, di, col_i] = ddf[col].values
    raw_targets[:, di, 0] = ddf["dengue"].values

print(f"Raw features: {raw_features.shape}, Targets: {raw_targets.shape}")

# ── BioBERT Embeddings (T, N, 768) ──
# Use cached embeddings to avoid loading BioBERT each time
if os.path.exists(BERT_CACHE):
    print("Loading cached BioBERT embeddings...")
    bert_flat = torch.load(BERT_CACHE, map_location="cpu", weights_only=False).numpy()  # (860, 768)
    # Cache rows follow cases_df's original row order: grouped by district
    # (172 rows for Ahmedabad, 172 for Surat, 172 for Vadodara, etc.)
    bert_emb = np.zeros((T, num_nodes, 768))
    for idx, (_, row) in enumerate(cases_df.iterrows()):
        ti = dates.index(row["Date"])
        ni = node_map[row["District"]]
        bert_emb[ti, ni, :] = bert_flat[idx]
    print(f"BioBERT embeddings: {bert_emb.shape}")
else:
    print("No BioBERT cache found - encoding from scratch...")
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    bert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)
    bert_model.eval()

    bert_emb = np.zeros((T, num_nodes, 768))
    date_district_news = {}
    for _, row in news_df.iterrows():
        key = (row["Date"], row["District"])
        date_district_news.setdefault(key, []).append(row["Headline"])

    with torch.no_grad():
        for ti, date in enumerate(dates):
            for ni, dist in enumerate(districts):
                headlines = date_district_news.get((date, dist), [])
                if headlines:
                    text = " ".join(headlines)
                    inputs = tok(text, return_tensors="pt", max_length=512,
                                 truncation=True, padding=True).to(device)
                    out = bert_model(**inputs)
                    bert_emb[ti, ni, :] = out.last_hidden_state[:, 0, :].cpu().numpy()
            if (ti + 1) % 20 == 0:
                print(f"  Encoded {ti+1}/{T} timesteps")

    # Save cache
    torch.save(torch.tensor(bert_emb.reshape(-1, 768), dtype=torch.float32), BERT_CACHE)
    print(f"BioBERT embeddings: {bert_emb.shape}")


# ════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

class BaselineModel(nn.Module):
    """Original EpiGraphModel from the notebook (no improvements)."""
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1, heads=2, dropout=0.4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
                h = self.bn1(h)
                h = F.elu(h)
                h = self.dropout(h)
                h = self.gat2(h, edge_index)
                h = self.bn2(h)
                h = F.elu(h)
                residual = self.input_proj(xt[bi])
                h = h + residual
                batch_sp.append(h)
            spatial_seq.append(torch.stack(batch_sp))
        spatial = torch.stack(spatial_seq, dim=1)
        flat = spatial.view(b * n, t, self.hidden_dim)
        lstm_out, _ = self.lstm(flat)
        last = lstm_out[:, -1, :]
        out = self.fc(last)
        return out.view(b, n, -1)


class ImprovedModel(nn.Module):
    """
    Improved EpiGraphModel with:
    - Skip connections (raw base features → MLP bypass)
    - Temporal attention (instead of last-timestep-only)
    - Larger hidden dim
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1,
                 heads=2, dropout=0.2, num_base_features=13):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_base_features = num_base_features

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Spatial: 2-layer GATv2
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2  = nn.BatchNorm1d(hidden_dim)

        # Temporal: LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)

        # Temporal attention
        self.temporal_attn = nn.Linear(hidden_dim, 1)

        # Skip path: raw base features → small MLP
        self.skip_fc = nn.Sequential(
            nn.Linear(num_base_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Combine: deep (hidden_dim) + skip (hidden_dim//4)
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

        # Deep path: GAT per timestep
        spatial_seq = []
        for i in range(t):
            xt = x[:, i, :, :]
            batch_sp = []
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
                batch_sp.append(h)
            spatial_seq.append(torch.stack(batch_sp))

        spatial = torch.stack(spatial_seq, dim=1)  # (B, T, N, H)
        flat = spatial.view(b * n, t, self.hidden_dim)
        lstm_out, _ = self.lstm(flat)  # (B*N, T, H)

        # Temporal attention
        attn_w = F.softmax(self.temporal_attn(lstm_out), dim=1)  # (B*N, T, 1)
        context = (attn_w * lstm_out).sum(dim=1)  # (B*N, H)

        # Skip path: last timestep's base features
        last_base = x[:, -1, :, :self.num_base_features]  # (B, N, base_f)
        skip_in   = last_base.reshape(b * n, self.num_base_features)
        skip_out  = self.skip_fc(skip_in)  # (B*N, H//4)

        # Combine
        combined = torch.cat([context, skip_out], dim=1)
        out = self.fc(combined)
        return out.view(b, n, -1)


# ════════════════════════════════════════════════════════════════════════════
#  HELPER: PREPARE DATA FOR A GIVEN CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(pca_dim, use_engineered=False, use_log_target=False, window_size=7):
    """
    Build tensors from raw data.
    Returns: x_train, y_train, x_val, y_val, x_test, y_test, target_scaler, num_features, num_base
    """
    # --- Case features ---
    case_feat = raw_features.copy()  # (T, N, 6)
    num_base = 6

    if use_engineered:
        dengue = raw_features[:, :, 5]  # dengue is column 5
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
        case_feat = np.concatenate([case_feat, eng], axis=2)  # (T, N, 13)
        num_base = 13
        print(f"  Engineered features added → {case_feat.shape[2]} base features")

    # --- Normalize case features ---
    flat_case = case_feat.reshape(T * num_nodes, -1)
    case_scaler = StandardScaler()
    flat_case_n = case_scaler.fit_transform(flat_case).reshape(T, num_nodes, -1)

    # --- PCA on BioBERT ---
    flat_bert = bert_emb.reshape(T * num_nodes, 768)
    emb_scaler = StandardScaler()
    flat_bert_n = emb_scaler.fit_transform(flat_bert)
    pca_model = PCA(n_components=pca_dim)
    bert_pca = pca_model.fit_transform(flat_bert_n)
    explained = sum(pca_model.explained_variance_ratio_) * 100
    print(f"  PCA: 768 → {pca_dim} dims ({explained:.1f}% variance)")
    bert_pca_n = bert_pca.reshape(T, num_nodes, pca_dim)

    # --- Combine → (T, N, num_base + pca_dim) ---
    combined = np.concatenate([flat_case_n.reshape(T, num_nodes, -1), bert_pca_n], axis=2)
    x_tensor = torch.tensor(combined, dtype=torch.float32)
    total_features = combined.shape[2]

    # --- Targets ---
    tgt = raw_targets.copy().reshape(T * num_nodes, 1)
    if use_log_target:
        tgt = np.log1p(tgt)
        print("  Targets: log1p → StandardScaler")
    tgt_scaler = StandardScaler()
    tgt_scaled = tgt_scaler.fit_transform(tgt).reshape(T, num_nodes, 1)
    y_tensor = torch.tensor(tgt_scaled, dtype=torch.float32)

    # --- Windowing ---
    X_out, Y_out = [], []
    for i in range(T - window_size - 1 + 1):
        X_out.append(x_tensor[i : i + window_size])
        Y_out.append(y_tensor[i + window_size : i + window_size + 1])
    X_out = torch.stack(X_out)
    Y_out = torch.stack(Y_out)

    n_total = len(X_out)
    tr_end  = int(n_total * 0.70)
    va_end  = int(n_total * 0.85)

    x_tr = X_out[:tr_end].to(device)
    y_tr = Y_out[:tr_end].to(device)
    x_va = X_out[tr_end:va_end].to(device)
    y_va = Y_out[tr_end:va_end].to(device)
    x_te = X_out[va_end:].to(device)
    y_te = Y_out[va_end:].to(device)
    print(f"  Windows: train={len(x_tr)}, val={len(x_va)}, test={len(x_te)}")

    return x_tr, y_tr, x_va, y_va, x_te, y_te, tgt_scaler, total_features, num_base, use_log_target


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def train_model(model, x_tr, y_tr, x_va, y_va, epochs, lr, batch_size,
                patience, scheduler_type="plateau", weight_decay=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)

    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb, edge_index)
            loss = criterion(pred, yb.squeeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb, edge_index)
                val_loss += criterion(pred, yb.squeeze(1)).item()
        avg_val = val_loss / max(len(val_loader), 1)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if scheduler_type == "plateau":
            scheduler.step(avg_val)
        else:
            scheduler.step(epoch)

        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train={avg_train:.4f}  val={avg_val:.4f}  best={best_val:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return model, train_losses, val_losses


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, x_te, y_te, target_scaler, log_targets, batch_size=16):
    model.eval()
    loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb, edge_index)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.squeeze(1).cpu().numpy())

    preds_scaled   = np.concatenate(all_preds, axis=0)    # (N_test, 5, 1)
    targets_scaled = np.concatenate(all_targets, axis=0)

    # Inverse transform
    p_flat = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    t_flat = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

    if log_targets:
        p_flat = np.expm1(p_flat)
        t_flat = np.expm1(t_flat)

    p_flat = np.clip(p_flat, 0, None)
    t_flat = np.clip(t_flat, 0, None)

    mae  = mean_absolute_error(t_flat, p_flat)
    rmse = np.sqrt(np.mean((p_flat - t_flat) ** 2))
    r2   = r2_score(t_flat, p_flat)

    # Outbreak detection
    threshold = np.median(t_flat)
    pred_out   = (p_flat > threshold).astype(int)
    actual_out = (t_flat > threshold).astype(int)
    tp = np.sum((pred_out == 1) & (actual_out == 1))
    fp = np.sum((pred_out == 1) & (actual_out == 0))
    fn = np.sum((pred_out == 0) & (actual_out == 1))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    outbreak_acc = np.mean(pred_out == actual_out)

    # Per-district R²
    per_district = {}
    for di, d in enumerate(districts):
        dp = preds_scaled[:, di, :].flatten()
        dt = targets_scaled[:, di, :].flatten()
        dp2 = target_scaler.inverse_transform(dp.reshape(-1, 1)).flatten()
        dt2 = target_scaler.inverse_transform(dt.reshape(-1, 1)).flatten()
        if log_targets:
            dp2 = np.expm1(dp2)
            dt2 = np.expm1(dt2)
        dp2 = np.clip(dp2, 0, None)
        per_district[d] = {
            "MAE":  round(mean_absolute_error(dt2, dp2), 2),
            "RMSE": round(np.sqrt(np.mean((dp2 - dt2) ** 2)), 2),
            "R2":   round(r2_score(dt2, dp2) if len(dt2) > 1 else 0, 4)
        }

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Outbreak_F1": round(f1, 4),
        "Outbreak_Acc": round(outbreak_acc, 4),
        "per_district": per_district
    }


# ════════════════════════════════════════════════════════════════════════════
#  RUN BOTH EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    EPOCHS = 100

    # ── BASELINE ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  BASELINE MODEL (original notebook settings)")
    print("="*70)
    print("Config: HIDDEN_DIM=64, PCA_DIM=32, DROPOUT=0.4, LR=0.0005, BATCH=8")

    x_tr, y_tr, x_va, y_va, x_te, y_te, tgt_scaler_b, n_feat_b, n_base_b, log_b = \
        prepare_data(pca_dim=32, use_engineered=False, use_log_target=False)

    baseline = BaselineModel(num_nodes, n_feat_b, hidden_dim=64, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in baseline.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    baseline, bl_train, bl_val = train_model(
        baseline, x_tr, y_tr, x_va, y_va,
        epochs=EPOCHS, lr=0.0005, batch_size=8, patience=15,
        scheduler_type="plateau"
    )
    bl_time = time.time() - t0
    print(f"  Training time: {bl_time:.1f}s")

    bl_metrics = evaluate(baseline, x_te, y_te, tgt_scaler_b, log_b)
    print(f"\n  BASELINE RESULTS:")
    print(f"    MAE:  {bl_metrics['MAE']}")
    print(f"    RMSE: {bl_metrics['RMSE']}")
    print(f"    R²:   {bl_metrics['R2']}")
    print(f"    Outbreak F1:  {bl_metrics['Outbreak_F1']}")
    print(f"    Outbreak Acc: {bl_metrics['Outbreak_Acc']}")
    print(f"    Per-district:")
    for d, m in bl_metrics["per_district"].items():
        print(f"      {d:15s} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R²={m['R2']:7.4f}")

    # Save baseline
    torch.save(baseline.state_dict(), os.path.join(BASE, "epigraph_baseline.pth"))

    # ── IMPROVED ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  IMPROVED MODEL (all enhancements)")
    print("="*70)
    print("Config: HIDDEN_DIM=128, PCA_DIM=16, DROPOUT=0.2, LR=0.001, BATCH=16")
    print("        + engineered features, log-targets, skip connections, temporal attention")

    x_tr2, y_tr2, x_va2, y_va2, x_te2, y_te2, tgt_scaler_i, n_feat_i, n_base_i, log_i = \
        prepare_data(pca_dim=16, use_engineered=True, use_log_target=True)

    improved = ImprovedModel(
        num_nodes, n_feat_i, hidden_dim=128,
        dropout=0.2, num_base_features=n_base_i
    ).to(device)
    n_params2 = sum(p.numel() for p in improved.parameters())
    print(f"  Parameters: {n_params2:,}")

    t0 = time.time()
    improved, im_train, im_val = train_model(
        improved, x_tr2, y_tr2, x_va2, y_va2,
        epochs=EPOCHS, lr=0.001, batch_size=16, patience=30,
        scheduler_type="cosine"
    )
    im_time = time.time() - t0
    print(f"  Training time: {im_time:.1f}s")

    im_metrics = evaluate(improved, x_te2, y_te2, tgt_scaler_i, log_i)
    print(f"\n  IMPROVED RESULTS:")
    print(f"    MAE:  {im_metrics['MAE']}")
    print(f"    RMSE: {im_metrics['RMSE']}")
    print(f"    R²:   {im_metrics['R2']}")
    print(f"    Outbreak F1:  {im_metrics['Outbreak_F1']}")
    print(f"    Outbreak Acc: {im_metrics['Outbreak_Acc']}")
    print(f"    Per-district:")
    for d, m in im_metrics["per_district"].items():
        print(f"      {d:15s} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  R²={m['R2']:7.4f}")

    # Save improved model
    torch.save(improved.state_dict(), os.path.join(BASE, "epigraph_model_v2.pth"))

    # ── COMPARISON ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  HEAD-TO-HEAD COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
    print("-" * 58)

    for metric in ["MAE", "RMSE", "R2", "Outbreak_F1", "Outbreak_Acc"]:
        bv = bl_metrics[metric]
        iv = im_metrics[metric]
        if metric in ["MAE", "RMSE"]:
            delta = bv - iv  # lower is better
            better = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "SAME")
            pct = (delta / max(abs(bv), 1e-8)) * 100
        else:
            delta = iv - bv  # higher is better
            better = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "SAME")
            pct = (delta / max(abs(bv), 1e-8)) * 100
        print(f"{metric:<20} {bv:>12.4f} {iv:>12.4f} {pct:>+10.1f}% {better}")

    print(f"\nParameters: {n_params:,} → {n_params2:,}")
    print(f"Train time: {bl_time:.0f}s → {im_time:.0f}s")
    print(f"\nModels saved: epigraph_baseline.pth, epigraph_model_v2.pth")
    print("Done!")
