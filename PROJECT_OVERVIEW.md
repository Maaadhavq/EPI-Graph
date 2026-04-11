# EpiGraph-AI: Spatiotemporal Graph Neural Network for Epidemic Prediction

## Overview

EpiGraph-AI is a multi-modal deep learning framework designed to forecast weekly **dengue fever case counts** across 5 major districts in Gujarat, India. By merging geographic connectivity, meteorological patterns, and semantic signals from health news, the system identifies outbreak risks before they escalate.

- **Graph Attention Networks (GATv2)** for spatial modeling — learning dynamic influence weights between districts.
- **LSTM** for temporal modeling — capturing 7-week historical trends and seasonal cycles.
- **BioBERT embeddings** — encoding health news headlines into contextual risk features.
- **Explainable AI (XAI)** — a diagnostic layer that breaks down risk into Weather, Temporal, and Spatial components.

The model achieves an **R² of 0.585** and **63% outbreak detection accuracy** (within relative tolerance).

---

## Project Structure

```
EpiGraph-AI/
├── data/
│   ├── processed_cases.csv     # Weekly dengue cases + weather per district
│   ├── health_news.csv         # Local health headlines per district
│   └── connectivity.csv        # Geographic connectivity graph (adjacency weights)
├── notebooks/
│   └── EpiGraph_AI.ipynb       # [CORE] Data processing, training, and metrics
├── website/                     # Frontend dashboard (D3.js, Chart.js)
├── app.py                      # Flask backend & data-driven risk API
├── epigraph_model_v2.pth       # Trained model weights
├── requirements.txt            # Python dependencies
├── create_pdf_report.py        # Utility to generate the technical report
├── run_pipeline.py             # Script to verify environment and logic
└── PROJECT_OVERVIEW.md         # Project documentation (this file)
```

---

## How It Works

### Step 1: Feature Extraction & Compression
For each district, the model processes:
*   **Base Features (6):** Raw dengue cases + 5 weather variables (Temp, Rainfall, Humidity).
*   **Engineered Features (7):** Rolling means, lags (1, 2 weeks), and delta change.
*   **BioBERT Context (32):** News headlines are encoded via **BioBERT** (768d) and then compressed using **PCA** to 32 dimensions to maintain feature balance.

### Step 2: Spatial Graph Construction
A spatial graph represents district connectivity:
*   Nodes are connected based on geographic adjacency weights from `connectivity.csv`.
*   The graph is made **bidirectional** and includes **self-loops** (17 total edges) to optimize GAT message passing.

### Step 3: Model Architecture (`EpiGraphModel`)
Implemented in PyTorch, the model utilizes a deep hybrid pipeline:
1.  **Spatial Layer:** Two `GATv2Conv` layers with multi-head attention to aggregate risk from neighbors.
2.  **Residual Connections:** Skips that allow raw temporal signals to reach the LSTM directly.
3.  **Temporal Layer:** A 2-layer LSTM that processes the 7-week sliding window sequence.
4.  **FC Head:** A multi-layer perceptron to predict the final case count.

### Step 4: Data-Driven Deployment
For production efficiency, the **Flask backend (`app.py`)** uses a high-fidelity logic-based replica of the model:
*   **Risk Scoring:** Normalizes current case/weather data against the district's 90th percentile historical "epidemic threshold."
*   **Spatial Spillover:** Mimics GAT attention by propagating a % of risk from high-risk neighbors based on connectivity weights.
*   **XAI:** Dynamically calculates feature contributions (e.g., how much rainfall contributed to today's risk).

---

## Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | **0.585** | Explains 58.5% of variance in case distribution |
| **Accuracy (50%)** | **63.0%** | Predictions within 50% or 10 cases of actual |
| **MAE** | **26.74** | Average absolute error in weekly case count |
| **RMSE** | **55.91** | Root mean square error (penalizes large misses) |

---

## How to Run

### 1. Prerequisites
```bash
pip install -r requirements.txt
```

### 2. Train and Analyze
Since all modular logic is consolidated for experimentation, open and run:
`notebooks/EpiGraph_AI.ipynb`
This will generate the embeddings, train the model, and save `epigraph_model_v2.pth`.

### 3. Launch Dashboard
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser to interact with the map and XAI reports.

---

## Technical Insights
1.  **Huber Loss beats MSE:** Epidemic data has extreme outliers. Huber Loss prevents the model from "exploding" during massive spikes.
2.  **Skip Connections are Critical:** For small datasets (~800 rows), skip connections act as a strong baseline, preventing the GAT layers from over-complicating the signal.
3.  **Log-Transform Readiness:** While the notebook uses StandardScaler, implementing a `log1p` transform is identified as the next step for better handling of right-skewed data.
