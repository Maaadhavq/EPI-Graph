"""
Script to generate the EpiGraph-AI consolidated Jupyter Notebook (v2 - Improved).
Run: python create_notebook.py
"""
import json
import os

def make_source(text):
    """Convert a multi-line string into a Jupyter-compatible source list."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": make_source(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": make_source(source), "outputs": [], "execution_count": None}

cells = []

# ========== CELL: Title ==========
cells.append(md("""# EpiGraph-AI: Spatiotemporal Multi-Modal Disease Outbreak Forecasting
## Using Graph Neural Networks + BioBERT + LSTM (v2 — Improved)

**Problem**: Integrating semantic risk signals from medical news with temporal case data to predict localized disease outbreaks.

**Architecture**: BioBERT (NLP) → PCA Compression → GAT (Spatial) → LSTM (Temporal) → Prediction

**Key Improvements in v2**:
- ✅ Feature normalization (StandardScaler)
- ✅ BioBERT embeddings compressed via PCA (768 → 32 dims)
- ✅ Huber Loss (robust to outlier spikes)
- ✅ Early stopping on validation loss
- ✅ Weight decay regularization
- ✅ Gradient clipping
- ✅ Proper train/val/test split (70/15/15)"""))

# ========== CELL: Install ==========
cells.append(md("## 0. Install Dependencies (Colab)"))
cells.append(code("""!pip install -q torch-geometric transformers scikit-learn seaborn
print("All dependencies installed!")"""))

# ========== CELL: Imports ==========
cells.append(md("## 1. Imports & Configuration"))
cells.append(code("""import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, f1_score, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GATv2Conv
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
CASES_FILE = os.path.join(DATA_DIR, "processed_cases.csv")
NEWS_FILE = os.path.join(DATA_DIR, "health_news.csv")
CONNECTIVITY_FILE = os.path.join(DATA_DIR, "connectivity.csv")

# Hyperparameters
WINDOW_SIZE = 7
HORIZON = 1
HIDDEN_DIM = 64
EPOCHS = 100
LR = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE = 15        # Early stopping patience
PCA_DIM = 32         # Compress BioBERT 768 → 32
BATCH_SIZE = 8
DROPOUT = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")"""))

# ========== CELL: Load Data ==========
cells.append(md("## 2. Data Loading & Exploration"))
cells.append(code("""cases_df = pd.read_csv(CASES_FILE)
news_df = pd.read_csv(NEWS_FILE)
conn_df = pd.read_csv(CONNECTIVITY_FILE)

print("=== Cases Data ===")
print(f"Shape: {cases_df.shape}")
display(cases_df.head())
print(f"\\nDengue stats:\\n{cases_df['dengue'].describe()}")

print("\\n=== News Data ===")
print(f"Shape: {news_df.shape}")
display(news_df.head())

print("\\n=== Connectivity Data ===")
print(f"Shape: {conn_df.shape}")
display(conn_df)"""))

# ========== CELL: EDA ==========
cells.append(md("### 2.1 Exploratory Data Analysis"))
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Dengue cases by district
for district in cases_df['District'].unique():
    d = cases_df[cases_df['District'] == district]
    axes[0, 0].plot(range(len(d)), d['dengue'].values, label=district, alpha=0.7)
axes[0, 0].set_title('Dengue Cases Over Time by District')
axes[0, 0].set_xlabel('Week Index')
axes[0, 0].set_ylabel('Cases')
axes[0, 0].legend(fontsize=8)

# Plot 2: Distribution of case counts
axes[0, 1].hist(cases_df['dengue'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(cases_df['dengue'].median(), color='red', linestyle='--', label=f"Median: {cases_df['dengue'].median():.0f}")
axes[0, 1].set_title('Distribution of Dengue Cases')
axes[0, 1].set_xlabel('Cases')
axes[0, 1].legend()

# Plot 3: Correlation heatmap
feature_cols_eda = ['dengue', '.MMAX', '.MMIN', '..TMRF', '.RH -0830', '.RH -1730']
corr = cases_df[feature_cols_eda].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], vmin=-1, vmax=1)
axes[1, 0].set_title('Feature Correlation Matrix')

# Plot 4: News type distribution
news_df['Type'].value_counts().plot(kind='bar', ax=axes[1, 1], color=['coral', 'skyblue'])
axes[1, 1].set_title('News Type Distribution')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150)
plt.show()"""))

# ========== CELL: BioBERT ==========
cells.append(md("## 3. BioBERT Encoder (NLP Component)"))
cells.append(code("""class BioBERTEncoder:
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                return_tensors="pt", max_length=64
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

print("Loading BioBERT...")
bert_encoder = BioBERTEncoder()
print("BioBERT loaded successfully!")

# Test encoding
sample = bert_encoder.encode(["Health Emergency: surge in viral fever reported."])
print(f"Raw embedding shape: {sample.shape}")  # (1, 768)"""))

# ========== CELL: Dataset Construction ==========
cells.append(md("## 4. Dataset Construction"))
cells.append(code("""districts = sorted(cases_df['District'].unique())
node_mapping = {d: i for i, d in enumerate(districts)}
print(f"Districts ({len(districts)}): {districts}")

# Build Adjacency Matrix
num_nodes = len(districts)
adj_matrix = np.zeros((num_nodes, num_nodes))
for _, row in conn_df.iterrows():
    src_idx = node_mapping.get(row['Source'])
    tgt_idx = node_mapping.get(row['Target'])
    if src_idx is not None and tgt_idx is not None:
        adj_matrix[src_idx, tgt_idx] = row['Weight']
        adj_matrix[tgt_idx, src_idx] = row['Weight']  # Make symmetric

# Add self-loops (important for GNN)
np.fill_diagonal(adj_matrix, 1.0)

adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
edge_index = adj_tensor.nonzero().t().contiguous().to(device)
print(f"Adjacency Matrix: {adj_tensor.shape}")
print(f"Edge Index: {edge_index.shape} ({edge_index.shape[1]} edges)")

# Visualize
plt.figure(figsize=(6, 5))
sns.heatmap(adj_matrix, annot=True, fmt='.1f', xticklabels=districts, yticklabels=districts, cmap='YlOrRd')
plt.title('District Connectivity (with Self-Loops)')
plt.tight_layout()
plt.savefig('adjacency_matrix.png', dpi=150)
plt.show()"""))

# ========== CELL: Feature Processing ==========
cells.append(md("""### 4.1 Feature Processing (Improved)

**Key improvements:**
- **StandardScaler** normalizes all numeric features to zero mean / unit variance
- **PCA(32)** compresses BioBERT 768-dim embeddings to 32 dims (retains key signal, removes noise)
- Sparse news entries (zeros) no longer dominate the feature space"""))

cells.append(code("""feature_cols = ['dengue', '.MMAX', '.MMIN', '..TMRF', '.RH -0830', '.RH -1730']
dates = sorted(cases_df['Date'].unique())
full_idx = pd.MultiIndex.from_product([dates, districts], names=['Date', 'District'])

relevant_cols = ['Date', 'District'] + feature_cols
cases_filtered = cases_df[relevant_cols].copy()
cases_indexed = cases_filtered.set_index(['Date', 'District'])
cases_filled = cases_indexed.reindex(full_idx, fill_value=0)

news_grouped = news_df.groupby(['Date', 'District'])['Headline'].apply(lambda x: " ".join(x))

num_timesteps = len(dates)
emb_dim_raw = 768

# Step 1: Collect raw features and embeddings separately
raw_case_features = []  # Will be (T*N, num_case_feats)
raw_embeddings = []     # Will be (T*N, 768)
raw_targets = []        # Will be (T, N, 1)

print(f"Processing {num_timesteps} timesteps...")
for t, date in enumerate(dates):
    day_data = cases_filled.loc[date].reindex(districts, fill_value=0)
    feats = day_data[feature_cols].values  # (N, 6)
    raw_case_features.append(feats)
    
    day_embeddings = []
    for district in districts:
        if (date, district) in news_grouped:
            headline = news_grouped.loc[(date, district)]
            emb = bert_encoder.encode([headline]).squeeze(0).numpy()
        else:
            emb = np.zeros(emb_dim_raw)
        day_embeddings.append(emb)
    raw_embeddings.append(np.array(day_embeddings))  # (N, 768)
    raw_targets.append(day_data[['dengue']].values)   # (N, 1)
    
    if (t + 1) % 50 == 0:
        print(f"  Processed {t+1}/{num_timesteps} timesteps")

raw_case_features = np.array(raw_case_features)  # (T, N, 6)
raw_embeddings = np.array(raw_embeddings)          # (T, N, 768)
raw_targets = np.array(raw_targets)                # (T, N, 1)

print(f"\\nRaw case features: {raw_case_features.shape}")
print(f"Raw embeddings: {raw_embeddings.shape}")
print(f"Raw targets: {raw_targets.shape}")"""))

# ========== CELL: Normalization + PCA ==========
cells.append(md("### 4.2 Feature Normalization & PCA Compression"))
cells.append(code("""# Step 2: Normalize case/weather features
T, N, F_case = raw_case_features.shape
case_flat = raw_case_features.reshape(T * N, F_case)

scaler = StandardScaler()
case_scaled = scaler.fit_transform(case_flat)
case_scaled = case_scaled.reshape(T, N, F_case)
print(f"Normalized case features: mean≈{case_scaled.mean():.4f}, std≈{case_scaled.std():.4f}")

# Step 3: PCA compress embeddings (768 → PCA_DIM)
emb_flat = raw_embeddings.reshape(T * N, 768)

# Check how many non-zero embedding rows we have
nonzero_count = (emb_flat.sum(axis=1) != 0).sum()
print(f"Non-zero embedding rows: {nonzero_count}/{T*N} ({100*nonzero_count/(T*N):.1f}%)")

pca = PCA(n_components=PCA_DIM)
emb_pca = pca.fit_transform(emb_flat)
emb_pca = emb_pca.reshape(T, N, PCA_DIM)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Compressed embeddings: {emb_pca.shape}")

# Also normalize PCA embeddings
emb_pca_flat = emb_pca.reshape(T * N, PCA_DIM)
emb_scaler = StandardScaler()
emb_pca_scaled = emb_scaler.fit_transform(emb_pca_flat).reshape(T, N, PCA_DIM)

# Step 4: Combine features
# Input features: normalized case (6) + normalized PCA embeddings (32) = 38
combined_features = np.concatenate([case_scaled, emb_pca_scaled], axis=2)
print(f"\\nFinal feature shape: {combined_features.shape}")
print(f"  Case features: {F_case} dims")
print(f"  PCA embeddings: {PCA_DIM} dims")
print(f"  Total: {combined_features.shape[2]} dims (vs 774 before!)")

# Normalize targets too (for better training, we'll inverse-transform predictions later)
target_flat = raw_targets.reshape(T * N, 1)
target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target_flat).reshape(T, N, 1)

x_tensor = torch.tensor(combined_features, dtype=torch.float32)
y_tensor = torch.tensor(target_scaled, dtype=torch.float32)

print(f"\\nX tensor: {x_tensor.shape}")
print(f"Y tensor: {y_tensor.shape}")"""))

# ========== CELL: Sliding Windows ==========
cells.append(md("### 4.3 Sliding Window Generation & Data Split"))
cells.append(code("""def create_windows(x_tensor, y_tensor, window_size=7, horizon=1):
    X_out, Y_out = [], []
    for i in range(len(x_tensor) - window_size - horizon + 1):
        X_out.append(x_tensor[i : i + window_size])
        Y_out.append(y_tensor[i + window_size : i + window_size + horizon])
    return torch.stack(X_out), torch.stack(Y_out)

x_windows, y_windows = create_windows(x_tensor, y_tensor, WINDOW_SIZE, HORIZON)
print(f"Total windows: {x_windows.shape[0]}")
print(f"X shape: {x_windows.shape}")
print(f"Y shape: {y_windows.shape}")

# Train/Validation/Test Split (70/15/15) - time-based, no shuffle
n_total = len(x_windows)
train_end = int(n_total * 0.70)
val_end = int(n_total * 0.85)

x_train, y_train = x_windows[:train_end].to(device), y_windows[:train_end].to(device)
x_val, y_val = x_windows[train_end:val_end].to(device), y_windows[train_end:val_end].to(device)
x_test, y_test = x_windows[val_end:].to(device), y_windows[val_end:].to(device)

print(f"\\nTrain: {x_train.shape[0]} samples")
print(f"Val:   {x_val.shape[0]} samples")
print(f"Test:  {x_test.shape[0]} samples")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)"""))

# ========== CELL: Model ==========
cells.append(md("""## 5. Model Architecture: GAT + LSTM (Improved)

**Changes from v1:**
- Added **Batch Normalization** after GAT layers
- **2-layer LSTM** with dropout
- **Residual skip connection** from input to LSTM
- Projection layer to align input dims with hidden dims for skip connection"""))

cells.append(code("""class EpiGraphModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim=1, heads=2, dropout=0.4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Input projection (for residual connection)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial Encoder: GAT
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Temporal Encoder: LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        
        # Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        b, t, n, f = x.size()
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
                # Residual connection
                residual = self.input_proj(xt[bi])
                h = h + residual
                batch_spatial.append(h)
            spatial_features_seq.append(torch.stack(batch_spatial))
        
        spatial_seq = torch.stack(spatial_features_seq, dim=1)  # (B, T, N, H)
        spatial_flat = spatial_seq.view(b * n, t, self.hidden_dim)
        
        lstm_out, _ = self.lstm(spatial_flat)  # (B*N, T, H)
        last_out = lstm_out[:, -1, :]          # (B*N, H)
        out = self.fc(last_out)                # (B*N, 1)
        return out.view(b, n, -1)

num_features = x_tensor.shape[2]
model = EpiGraphModel(num_nodes, num_features, HIDDEN_DIM, dropout=DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model Parameters: {total_params:,}")
print(f"Input features: {num_features} (was 774 in v1!)")
print(model)"""))

# ========== CELL: Training ==========
cells.append(md("""## 6. Training with Validation & Early Stopping

**Improvements:**
- **Huber Loss** (robust to outlier case spikes)
- **Weight Decay** (L2 regularization)
- **Gradient Clipping** (prevents exploding gradients)
- **Early Stopping** (stops when val loss stops improving)
- **ReduceLROnPlateau** scheduler"""))

cells.append(code("""optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

print(f"Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
print(f"Loss: HuberLoss | Optimizer: AdamW (lr={LR}, wd={WEIGHT_DECAY})")
print("-" * 60)

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    total_train_loss = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx, edge_index)
        target = by.squeeze(1)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    avg_train = total_train_loss / len(train_loader)
    train_losses.append(avg_train)
    
    # ---- Validation ----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            out = model(bx, edge_index)
            target = by.squeeze(1)
            loss = criterion(out, target)
            total_val_loss += loss.item()
    avg_val = total_val_loss / len(val_loader)
    val_losses.append(avg_val)
    
    scheduler.step(avg_val)
    
    # ---- Early Stopping ----
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        marker = " ★ best"
    else:
        patience_counter += 1
        marker = ""
    
    if (epoch + 1) % 5 == 0 or patience_counter == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}{marker}")
    
    if patience_counter >= PATIENCE:
        print(f"\\n⏹ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\\n✓ Restored best model (val loss: {best_val_loss:.4f})")

torch.save(model.state_dict(), "epigraph_model_v2.pth")
print("Model saved to epigraph_model_v2.pth")"""))

# ========== CELL: Loss Curves ==========
cells.append(md("### 6.1 Training & Validation Loss Curves"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full loss curve
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Huber Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].plot(train_losses, label='Train Loss', linewidth=2)
axes[1].plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Huber Loss (log scale)')
axes[1].set_title('Loss Curves (Log Scale)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves_v2.png', dpi=150)
plt.show()

print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Val Loss:   {val_losses[-1]:.4f}")
print(f"Best Val Loss:    {best_val_loss:.4f}")
print(f"Train/Val ratio:  {train_losses[-1]/val_losses[-1]:.2f} (closer to 1.0 = less overfitting)")"""))

# ========== CELL: Evaluation ==========
cells.append(md("""## 7. Evaluation Metrics

Evaluated on the **held-out test set**. Predictions are inverse-transformed back to original scale for meaningful metrics."""))

cells.append(code("""model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx, edge_index)
        target = by.squeeze(1)
        all_preds.append(out.cpu())
        all_targets.append(target.cpu())

preds_scaled = torch.cat(all_preds).numpy().flatten()
targets_scaled = torch.cat(all_targets).numpy().flatten()

# Inverse-transform to original scale
preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
targets = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

# Clip negative predictions to 0 (cases can't be negative)
preds = np.clip(preds, 0, None)

# ---- Regression Metrics ----
mae = mean_absolute_error(targets, preds)
rmse = np.sqrt(np.mean((targets - preds) ** 2))
r2 = r2_score(targets, preds)

# ---- MAPE ----
nonzero_mask = targets != 0
if nonzero_mask.sum() > 0:
    mape = np.mean(np.abs((targets[nonzero_mask] - preds[nonzero_mask]) / targets[nonzero_mask])) * 100
    accuracy_pct = max(0, 100 - mape)
else:
    mape = float('nan')
    accuracy_pct = float('nan')

# ---- Tolerance-Based Accuracy ----
errors = np.abs(targets - preds)
acc_3 = (errors <= 3).mean() * 100
acc_5 = (errors <= 5).mean() * 100
acc_10 = (errors <= 10).mean() * 100

# ---- Cosine Similarity ----
cos_sim = F.cosine_similarity(
    torch.tensor(preds).unsqueeze(0).float(),
    torch.tensor(targets).unsqueeze(0).float()
).item()

# ---- Outbreak Detection (F1 & Accuracy) ----
threshold = np.median(targets)
pred_outbreak = (preds > threshold).astype(int)
true_outbreak = (targets > threshold).astype(int)
f1 = f1_score(true_outbreak, pred_outbreak, zero_division=0)
outbreak_acc = accuracy_score(true_outbreak, pred_outbreak) * 100

print("=" * 55)
print("         TEST SET EVALUATION METRICS")
print("=" * 55)
print()
print("  📊 ACCURACY METRICS:")
print(f"  ├─ Overall Accuracy (MAPE):     {accuracy_pct:.2f}%")
print(f"  ├─ Accuracy (within ±3 cases):  {acc_3:.1f}%")
print(f"  ├─ Accuracy (within ±5 cases):  {acc_5:.1f}%")
print(f"  └─ Accuracy (within ±10 cases): {acc_10:.1f}%")
print()
print("  📈 REGRESSION METRICS:")
print(f"  ├─ MAE  (Mean Absolute Error):  {mae:.4f}")
print(f"  ├─ RMSE (Root Mean Sq Error):   {rmse:.4f}")
print(f"  ├─ MAPE (Mean Abs % Error):     {mape:.2f}%")
print(f"  └─ R² Score:                    {r2:.4f}")
print()
print("  🔍 CLASSIFICATION METRICS (Outbreak Detection):")
print(f"  ├─ F1-Score:                    {f1:.4f}")
print(f"  ├─ Outbreak Detection Accuracy: {outbreak_acc:.1f}%")
print(f"  └─ Outbreak Threshold:          {threshold:.1f} cases")
print()
print("  🧭 SIMILARITY:")
print(f"  └─ Cosine Similarity:           {cos_sim:.4f}")
print("=" * 55)"""))

# ========== CELL: Prediction Plots ==========
cells.append(md("### 7.1 Predicted vs Actual (Test Set)"))
cells.append(code("""preds_all = torch.cat(all_preds).numpy()  # (samples, nodes, 1)
targets_all = torch.cat(all_targets).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot (original scale)
axes[0].scatter(targets, preds, alpha=0.5, s=20, color='steelblue')
max_val = max(targets.max(), preds.max()) * 1.1
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Cases')
axes[0].set_ylabel('Predicted Cases')
axes[0].set_title('Predicted vs Actual (Test Set)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Time series for first district (inverse-transformed)
d0_preds = target_scaler.inverse_transform(preds_all[:, 0, :]).flatten()
d0_targets = target_scaler.inverse_transform(targets_all[:, 0, :]).flatten()
d0_preds = np.clip(d0_preds, 0, None)

axes[1].plot(d0_targets, label='Actual', linewidth=2, marker='o', markersize=3)
axes[1].plot(d0_preds, label='Predicted', linewidth=2, linestyle='--', marker='s', markersize=3)
axes[1].set_xlabel('Test Sample Index')
axes[1].set_ylabel('Cases')
axes[1].set_title(f'Time Series: {districts[0]} (Test Set)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_comparison_v2.png', dpi=150)
plt.show()"""))

# ========== CELL: Per-District ==========
cells.append(md("### 7.2 Per-District Performance"))
cells.append(code("""district_metrics = []
for i, d in enumerate(districts):
    d_preds = target_scaler.inverse_transform(preds_all[:, i, :]).flatten()
    d_targets = target_scaler.inverse_transform(targets_all[:, i, :]).flatten()
    d_preds = np.clip(d_preds, 0, None)
    
    d_mae = mean_absolute_error(d_targets, d_preds)
    d_rmse = np.sqrt(np.mean((d_targets - d_preds) ** 2))
    d_r2 = r2_score(d_targets, d_preds) if len(np.unique(d_targets)) > 1 else 0
    district_metrics.append({"District": d, "MAE": round(d_mae, 2), "RMSE": round(d_rmse, 2), "R²": round(d_r2, 4)})

metrics_df = pd.DataFrame(district_metrics)
display(metrics_df)

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(districts))
width = 0.35
ax.bar(x_pos - width/2, metrics_df['MAE'], width, label='MAE', color='coral')
ax.bar(x_pos + width/2, metrics_df['RMSE'], width, label='RMSE', color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(districts, rotation=45)
ax.set_ylabel('Error (Cases)')
ax.set_title('Per-District Error Metrics')
ax.legend()
plt.tight_layout()
plt.savefig('district_metrics_v2.png', dpi=150)
plt.show()"""))

# ========== CELL: Risk Dashboard ==========
cells.append(md("## 8. Risk Dashboard (Final Predictions)"))
cells.append(code("""input_seq = x_tensor[-WINDOW_SIZE:].unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    risk_scaled = model(input_seq, edge_index).squeeze().cpu().numpy()

# Inverse transform to get actual case predictions
risk_scores = target_scaler.inverse_transform(risk_scaled.reshape(-1, 1)).flatten()
risk_scores = np.clip(risk_scores, 0, None)

results = pd.DataFrame({
    "District": districts,
    "Predicted_Risk_Score": risk_scores
}).sort_values('Predicted_Risk_Score', ascending=False)

print("\\n=== Predicted Outbreak Risk (Next Day) ===")
display(results)

colors = ['#ff4444' if r > np.median(risk_scores) else '#4CAF50' for r in results['Predicted_Risk_Score']]
plt.figure(figsize=(10, 6))
plt.barh(results['District'], results['Predicted_Risk_Score'], color=colors)
plt.xlabel('Predicted Cases')
plt.title('District-Level Outbreak Risk Assessment')
plt.axvline(x=np.median(risk_scores), color='orange', linestyle='--', label=f'Median: {np.median(risk_scores):.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('risk_dashboard_v2.png', dpi=150)
plt.show()"""))

# ========== CELL: Summary ==========
cells.append(md("""## 9. Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP Encoder** | BioBERT → PCA(32) | Compresses 768-dim medical text embeddings to 32 essential dims |
| **Spatial Encoder** | GATv2 (2-layer, 2-head) + BatchNorm | Captures disease spread patterns between connected districts |
| **Temporal Encoder** | LSTM (2-layer) | Models weekly case count trends and seasonal patterns |
| **Normalization** | StandardScaler | Zero-mean, unit-variance for all features and targets |
| **Loss** | HuberLoss (δ=1.0) | Robust to outlier case count spikes |
| **Regularization** | Dropout(0.4) + WeightDecay + GradClip | Prevents overfitting on small dataset |

### Improvements over v1
| Metric | v1 | v2 (Expected) |
|--------|----|----|
| Feature dims | 774 | 38 |
| Normalization | None | StandardScaler |
| Loss | MSE | Huber |
| Early stopping | No | Yes (patience=15) |
| Target scaling | No | Yes (inverse-transformed) |
| Train/Val gap | ~5x | Significantly reduced |
"""))

# ========== Build Notebook ==========
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.3"
        }
    },
    "cells": cells
}

output_path = os.path.join("notebooks", "EpiGraph_AI.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created at: {output_path}")
print(f"Total cells: {len(cells)}")
