"""
EpiGraph-AI: Original notebook pipeline — exact replica of EpiGraph_AI.ipynb.
Runs the exact same code that produced R²=0.58.
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
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GATv2Conv
import joblib
warnings.filterwarnings('ignore')

# ── Config (exact from notebook Cell 4) ────────────────────────────────────
DATA_DIR          = "data"
CASES_FILE        = os.path.join(DATA_DIR, "processed_cases.csv")
NEWS_FILE         = os.path.join(DATA_DIR, "health_news.csv")
CONNECTIVITY_FILE = os.path.join(DATA_DIR, "connectivity.csv")

WINDOW_SIZE = 7
HORIZON     = 1
HIDDEN_DIM  = 64
EPOCHS      = 100
LR          = 0.0005
WEIGHT_DECAY= 1e-4
PATIENCE    = 15
PCA_DIM     = 32
BATCH_SIZE  = 8
DROPOUT     = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Loading (Cell 6) ───────────────────────────────────────────────────
cases_df = pd.read_csv(CASES_FILE)
news_df  = pd.read_csv(NEWS_FILE)
conn_df  = pd.read_csv(CONNECTIVITY_FILE)
print(f"Cases: {cases_df.shape}, News: {news_df.shape}, Conn: {conn_df.shape}")

# ── BioBERT Encoder (Cell 10) ───────────────────────────────────────────────
class BioBERTEncoder:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    return_tensors="pt", max_length=64)
            with torch.no_grad():
                out = self.model(**inputs)
                all_embeddings.append(out.last_hidden_state[:, 0, :])
        return torch.cat(all_embeddings, dim=0)

# Cache path — keyed to PCA_DIM so it's safe to reuse
EMBED_CACHE = f"embeddings_cache_orig_{PCA_DIM}.pt"

# ── Graph Construction (Cell 12) ────────────────────────────────────────────
districts    = sorted(cases_df['District'].unique())  # ALPHABETICAL — same as notebook
node_mapping = {d: i for i, d in enumerate(districts)}
num_nodes    = len(districts)
print(f"Districts (sorted): {districts}")

adj_matrix = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    s = node_mapping.get(row['Source'])
    t = node_mapping.get(row['Target'])
    if s is not None and t is not None:
        adj_matrix[s, t] = row['Weight']
        adj_matrix[t, s] = row['Weight']
np.fill_diagonal(adj_matrix, 1.0)
adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
edge_index  = adj_tensor.nonzero().t().contiguous().to(device)
print(f"Edge index: {edge_index.shape}")

# ── Feature & Embedding Assembly (Cell 14) ──────────────────────────────────
feature_cols = ['dengue', '.MMAX', '.MMIN', '..TMRF', '.RH -0830', '.RH -1730']
dates        = sorted(cases_df['Date'].unique())
full_idx     = pd.MultiIndex.from_product([dates, districts], names=['Date','District'])

cases_filtered = cases_df[['Date','District'] + feature_cols].copy()
cases_indexed  = cases_filtered.set_index(['Date','District'])
cases_filled   = cases_indexed.reindex(full_idx, fill_value=0)
news_grouped   = news_df.groupby(['Date','District'])['Headline'].apply(lambda x: " ".join(x))

num_timesteps = len(dates)
emb_dim_raw   = 768

if os.path.exists(EMBED_CACHE):
    print(f"Loading cached embeddings from {EMBED_CACHE}...")
    cache = torch.load(EMBED_CACHE, map_location='cpu', weights_only=False)
    raw_case_features = cache['case_features']
    raw_embeddings    = cache['embeddings']
    raw_targets       = cache['targets']
    print(f"Loaded. case_features={raw_case_features.shape}, embeddings={raw_embeddings.shape}")
else:
    print("Loading BioBERT...")
    bert_encoder = BioBERTEncoder()
    print(f"Processing {num_timesteps} timesteps...")

    raw_case_features, raw_embeddings, raw_targets = [], [], []

    for t_i, date in enumerate(dates):
        day_data = cases_filled.loc[date].reindex(districts, fill_value=0)
        feats    = day_data[feature_cols].values          # (N, 6)
        raw_case_features.append(feats)
        raw_targets.append(day_data[['dengue']].values)   # (N, 1)

        day_embs = []
        for district in districts:
            if (date, district) in news_grouped.index:
                headline = news_grouped.loc[(date, district)]
                emb = bert_encoder.encode([headline]).squeeze(0).numpy()
            else:
                emb = np.zeros(emb_dim_raw)
            day_embs.append(emb)
        raw_embeddings.append(np.array(day_embs))        # (N, 768)

        if (t_i + 1) % 50 == 0:
            print(f"  {t_i+1}/{num_timesteps} timesteps")

    raw_case_features = np.array(raw_case_features)  # (T, N, 6)
    raw_embeddings    = np.array(raw_embeddings)      # (T, N, 768)
    raw_targets       = np.array(raw_targets)         # (T, N, 1)

    torch.save({'case_features': raw_case_features,
                'embeddings':    raw_embeddings,
                'targets':       raw_targets}, EMBED_CACHE)
    print(f"Embeddings cached to {EMBED_CACHE}")

print(f"case_features={raw_case_features.shape}, embeddings={raw_embeddings.shape}, targets={raw_targets.shape}")

# ── Normalisation & PCA (Cell 16) ───────────────────────────────────────────
T, N, F_case = raw_case_features.shape
case_flat = raw_case_features.reshape(T * N, F_case)
scaler = StandardScaler()
case_scaled = scaler.fit_transform(case_flat).reshape(T, N, F_case)

emb_flat    = raw_embeddings.reshape(T * N, 768)
pca         = PCA(n_components=PCA_DIM)
emb_pca     = pca.fit_transform(emb_flat).reshape(T, N, PCA_DIM)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

emb_scaler  = StandardScaler()
emb_pca_scaled = emb_scaler.fit_transform(emb_pca.reshape(T*N, PCA_DIM)).reshape(T, N, PCA_DIM)

combined_features = np.concatenate([case_scaled, emb_pca_scaled], axis=2)
print(f"Combined features: {combined_features.shape}  ({F_case} case + {PCA_DIM} PCA)")

target_flat   = raw_targets.reshape(T * N, 1)
target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target_flat).reshape(T, N, 1)

# Save scalers so app.py can reuse them
joblib.dump(scaler,        'case_scaler.pkl')
joblib.dump(emb_scaler,    'emb_scaler.pkl')
joblib.dump(pca,           'pca_model.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl')
print("Scalers saved.")

x_tensor = torch.tensor(combined_features, dtype=torch.float32)
y_tensor = torch.tensor(target_scaled,     dtype=torch.float32)

# ── Windowing & Split (Cell 18) ─────────────────────────────────────────────
def create_windows(x, y, ws=7, horizon=1):
    Xo, Yo = [], []
    for i in range(len(x) - ws - horizon + 1):
        Xo.append(x[i:i+ws])
        Yo.append(y[i+ws:i+ws+horizon])
    return torch.stack(Xo), torch.stack(Yo)

x_windows, y_windows = create_windows(x_tensor, y_tensor, WINDOW_SIZE, HORIZON)
n_total = len(x_windows)
train_end = int(n_total * 0.70)
val_end   = int(n_total * 0.85)

x_train, y_train = x_windows[:train_end].to(device), y_windows[:train_end].to(device)
x_val,   y_val   = x_windows[train_end:val_end].to(device), y_windows[train_end:val_end].to(device)
x_test,  y_test  = x_windows[val_end:].to(device), y_windows[val_end:].to(device)
print(f"Train:{len(x_train)}  Val:{len(x_val)}  Test:{len(x_test)}")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(x_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# ── Model (Cell 20) ─────────────────────────────────────────────────────────
class EpiGraphModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1, heads=2, dropout=0.4):
        super().__init__()
        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2  = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.fc   = nn.Sequential(
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
            xt = x[:, i]
            batch_sp = []
            for bi in range(b):
                h = self.gat1(xt[bi], edge_index)
                h = self.bn1(h); h = F.elu(h); h = self.dropout(h)
                h = self.gat2(h, edge_index)
                h = self.bn2(h); h = F.elu(h)
                h = h + self.input_proj(xt[bi])
                batch_sp.append(h)
            spatial_seq.append(torch.stack(batch_sp))
        spatial_flat = torch.stack(spatial_seq, dim=1).view(b * n, t, self.hidden_dim)
        lstm_out, _  = self.lstm(spatial_flat)
        last_out     = lstm_out[:, -1, :]
        return self.fc(last_out).view(b, n, -1)

num_features = x_tensor.shape[2]
model        = EpiGraphModel(num_nodes, num_features, HIDDEN_DIM, dropout=DROPOUT).to(device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Input features: {num_features}")

# ── Training (Cell 22) ──────────────────────────────────────────────────────
optimizer  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion  = nn.HuberLoss(delta=1.0)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

train_losses, val_losses = [], []
best_val_loss    = float('inf')
best_model_state = None
patience_counter = 0

print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out  = model(bx, edge_index)
        loss = criterion(out, by.squeeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()
    avg_train = total_train_loss / len(train_loader)
    train_losses.append(avg_train)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            out  = model(bx, edge_index)
            loss = criterion(out, by.squeeze(1))
            total_val_loss += loss.item()
    avg_val = total_val_loss / len(val_loader)
    val_losses.append(avg_val)
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss    = avg_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        marker = " ← best"
    else:
        patience_counter += 1
        marker = ""

    if (epoch + 1) % 10 == 0 or patience_counter == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train={avg_train:.4f} | val={avg_val:.4f}{marker}")

    if patience_counter >= PATIENCE:
        print(f"  Early stop at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "epigraph_model_v2.pth")
print("Model saved → epigraph_model_v2.pth")

# ── Evaluation (Cell 26) ────────────────────────────────────────────────────
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for bx, by in test_loader:
        all_preds.append(model(bx, edge_index).cpu())
        all_targets.append(by.squeeze(1).cpu())

preds_scaled   = torch.cat(all_preds).numpy().flatten()
targets_scaled = torch.cat(all_targets).numpy().flatten()

preds   = np.clip(target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten(), 0, None)
targets = target_scaler.inverse_transform(targets_scaled.reshape(-1,1)).flatten()

mae  = mean_absolute_error(targets, preds)
rmse = np.sqrt(np.mean((targets - preds) ** 2))
r2   = r2_score(targets, preds)

nonzero = targets != 0
mape    = np.mean(np.abs((targets[nonzero]-preds[nonzero])/targets[nonzero]))*100 if nonzero.sum() else float('nan')

threshold   = np.median(targets)
pred_out    = (preds > threshold).astype(int)
true_out    = (targets > threshold).astype(int)
f1          = f1_score(true_out, pred_out, zero_division=0)
outbreak_acc= accuracy_score(true_out, pred_out) * 100

print("\n" + "="*55)
print("  TEST SET RESULTS")
print("="*55)
print(f"  MAE:              {mae:.4f}")
print(f"  RMSE:             {rmse:.4f}")
print(f"  R² Score:         {r2:.4f}")
print(f"  MAPE:             {mape:.2f}%")
print(f"  Outbreak F1:      {f1:.4f}")
print(f"  Outbreak Acc:     {outbreak_acc:.1f}%")

# Per-district (Cell 30)
preds_all   = torch.cat(all_preds).numpy()     # (samples, N, 1)
targets_all = torch.cat(all_targets).numpy()   # (samples, N, 1)

print("\n  Per-district:")
for i, d in enumerate(districts):
    dp = np.clip(target_scaler.inverse_transform(preds_all[:,i,:]).flatten(), 0, None)
    dt = target_scaler.inverse_transform(targets_all[:,i,:]).flatten()
    dr2 = r2_score(dt, dp) if len(np.unique(dt)) > 1 else 0
    print(f"    {d:15s}  MAE={mean_absolute_error(dt,dp):.2f}  "
          f"RMSE={np.sqrt(np.mean((dt-dp)**2)):.2f}  R²={dr2:.4f}")

# Risk dashboard (Cell 32)
input_seq  = x_tensor[-WINDOW_SIZE:].unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    risk_scaled = model(input_seq, edge_index).squeeze().cpu().numpy()
risk_scores = np.clip(target_scaler.inverse_transform(risk_scaled.reshape(-1,1)).flatten(), 0, None)

print("\n  Risk Dashboard (next week prediction):")
for d, s in sorted(zip(districts, risk_scores), key=lambda x: -x[1]):
    bar = "█" * int(s / 5)
    print(f"    {d:15s}  {s:6.1f} cases  {bar}")

print(f"\nDone. Model → epigraph_model_v2.pth | Scalers → *.pkl")
