# EpiGraph-AI: Spatiotemporal Graph Neural Network for Epidemic Prediction

## Overview

EpiGraph-AI is a deep learning framework that predicts weekly **dengue fever case counts** across 5 districts in Gujarat, India. It combines:

- **Graph Attention Networks (GATv2)** for spatial modeling — learning how disease spreads between connected districts
- **LSTM** for temporal modeling — capturing week-over-week trends and seasonal patterns
- **BioBERT embeddings** for NLP — encoding health news headlines as features that signal emerging outbreaks
- **Engineered temporal features** — rolling averages, lags, and rate-of-change for stronger predictive signal

The model achieves an **R² of 0.585** and **63% accuracy** (within 50% or 10 cases of the actual count).

---

## Project Structure

```
EpiGraph-AI/
├── data/
│   ├── processed_cases.csv     # Weekly dengue cases + weather per district (860 rows)
│   ├── health_news.csv         # Health news headlines per district (300 rows)
│   └── connectivity.csv        # District connectivity graph (7 edges)
├── src/
│   ├── config.py               # File paths and column name constants
│   ├── preprocessing.py        # BioBERT encoder and sliding window utilities
│   ├── dataset.py              # Data loading, feature engineering, normalization
│   ├── model.py                # GATv2 + LSTM model with skip connections
│   ├── train.py                # Training loop, evaluation, and metrics
│   └── dashboard.py            # Visualization dashboard
├── notebooks/
│   └── EpiGraph_AI.ipynb       # Jupyter notebook version
├── epigraph_model.pth          # Trained model weights
├── requirements.txt            # Python dependencies
├── improvements.md             # Future improvement suggestions
├── run_pipeline.py             # End-to-end pipeline runner
└── README.md                   # Project readme
```

---

## Data

### processed_cases.csv
Weekly records for 5 districts (Ahmedabad, Gandhinagar, Rajkot, Surat, Vadodara) from 2010–2013.

| Column | Description |
|--------|-------------|
| `Date` | Week start date |
| `District` | District name |
| `dengue` | Weekly dengue case count (target variable) |
| `.MMAX` | Maximum temperature (°C) |
| `.MMIN` | Minimum temperature (°C) |
| `..TMRF` | Total rainfall (mm) |
| `.RH -0830` | Morning humidity (%) |
| `.RH -1730` | Evening humidity (%) |

**Key characteristics:** 860 rows, 172 timesteps, extremely right-skewed target (median=5, max=724).

### health_news.csv
300 health-related news headlines with date, district, headline text, and type. These are encoded into 768-dimensional BioBERT embeddings to capture early outbreak signals from media.

### connectivity.csv
7 directed edges defining spatial connectivity between districts with weights (e.g., Ahmedabad→Gandhinagar: 1.0, Ahmedabad→Surat: 0.7). Made bidirectional in preprocessing.

---

## How It Works

### Step 1: Feature Engineering (`dataset.py`)

For each district at each timestep, the model receives:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Base features | 6 | Raw dengue cases + 5 weather variables |
| Engineered features | 7 | Rolling mean (4-week, 8-week), rolling std, lag-1, lag-2, delta, log(cases) |
| BioBERT embeddings | 768 → 16 | News headlines encoded via BioBERT, projected to 16 dims |
| **Total per node** | **29** | After BioBERT projection (13 base/engineered + 16 projected) |

The target variable (dengue cases) is **log-transformed** using `log1p(x)` to handle the extreme right-skew, then **standardized** (zero mean, unit variance) using training set statistics.

### Step 2: Graph Construction (`dataset.py`)

A spatial graph connects the 5 districts:
- Original 7 directed edges are made **bidirectional** (14 edges)
- **Self-loops** are added (5 more edges) so each district can attend to its own features
- Total: **17 edges** with learned attention weights via GATv2

### Step 3: Model Architecture (`model.py`)

```
Input: (Batch, Time=7, Nodes=5, Features=781)
          │
          ├─── DEEP PATH ──────────────────────────────┐
          │    BioBERT Projection (768 → 16)           │
          │         ↓                                   │
          │    GATv2 Layer 1 (29 → 128, 2 heads)       │
          │         ↓ ELU + Dropout                     │
          │    GATv2 Layer 2 (256 → 128, 1 head)       │
          │         ↓ LayerNorm                         │
          │    2-Layer LSTM (128 → 128)                 │
          │         ↓                                   │
          │    Temporal Attention (weighted sum)         │
          │         ↓                                   │
          │    Context Vector (128-dim)                  │
          │                                             │
          ├─── SKIP PATH ──────────────────────────────┐│
          │    Last timestep base features (13-dim)    ││
          │         ↓                                   ││
          │    FC → ReLU → FC (13 → 64 → 32)          ││
          │         ↓                                   ││
          │    Skip Features (32-dim)                   ││
          │                                             ││
          └─── COMBINE ────────────────────────────────┘│
               Concat(Context, Skip) = 160-dim          │
                    ↓                                    │
               FC → ReLU → Dropout → FC (160 → 64 → 1) │
                    ↓                                    │
               Output: (Batch, Nodes=5, 1)               │
```

**Key design decisions:**
- **Skip connections** let raw temporal features (lags, rolling means) bypass the deep path. This is critical because with only 132 training samples, the LSTM cannot learn lag patterns from scratch fast enough.
- **Temporal attention** learns which of the 7 timesteps in the window matters most, instead of just using the last LSTM hidden state.
- **BioBERT projection** (768→16) prevents the sparse NLP features from overwhelming the 13 tabular features.

### Step 4: Training (`train.py`)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Loss | HuberLoss (δ=1.0) | Robust to outlier case spikes |
| Optimizer | Adam (lr=0.001, weight_decay=5e-5) | Standard with L2 regularization |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30) | Periodic LR resets to escape local minima |
| Batch size | 16 | Stable gradients (was 1 originally) |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |
| Epochs | 300 (early stop at ~90) | With patience=50 for early stopping |
| Target transform | log1p → StandardScaler | Handles extreme right-skew |

### Step 5: Evaluation

Predictions are inverse-transformed (`StandardScaler⁻¹ → expm1`) back to original case counts for evaluation:

| Metric | Value | Meaning |
|--------|-------|---------|
| **R²** | **0.585** | Explains 58.5% of variance (>0 = better than mean) |
| **RMSE** | **55.91** | Average error magnitude in cases |
| **MAE** | **26.74** | Median-like error in cases |
| **Accuracy (50%)** | **63.0%** | Predictions within 50% of actual or 10 cases |
| **Accuracy (30%)** | **52.7%** | Predictions within 30% of actual or 5 cases |

---

## How to Run

### Prerequisites
```bash
pip install torch torch-geometric transformers pandas numpy
```

### Train the Model
```bash
cd EpiGraph-AI
python -m src.train
```
This will:
1. Load and preprocess data (~1 min for BioBERT encoding)
2. Train for up to 300 epochs with early stopping
3. Save the best model to `epigraph_model.pth`
4. Print final evaluation metrics

### Run Full Pipeline
```bash
python run_pipeline.py
```
Runs environment check → training → dashboard visualization.

---

## Key Technical Insights

1. **Log-transform is essential for skewed epidemic data.** Dengue cases follow a heavy-tailed distribution. Without log-transform, MSE/Huber loss focuses on rare extreme values while ignoring the common low-case weeks.

2. **Skip connections beat larger models for small datasets.** With only 132 training samples, making the model bigger causes overfitting. Skip connections from raw features to the output let the model learn a strong baseline (from lags/rolling means) plus a residual correction (from the deep GAT+LSTM path).

3. **Feature engineering > architecture complexity.** Simple rolling means and lag features improved R² more than doubling the hidden dimension or adding LSTM layers.

4. **BioBERT needs aggressive compression.** 768 dims of mostly-zero embeddings drown out 13 informative tabular features. Projecting to just 16 dims balances the feature groups.

5. **Bidirectional graphs with self-loops are necessary.** The original directed graph with 7 edges gave each node too few neighbors for GAT attention to be meaningful. Adding reverse edges and self-loops (17 total) dramatically improved spatial message passing.
