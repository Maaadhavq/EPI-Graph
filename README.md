# EpiGraph-AI

> **Note:** This project is currently undergoing development and is a work in progress.

A deep learning pipeline for predicting **Epidemic Outbreaks** (specifically Dengue fever in this instance) using a combination of clinical/weather data and textual health news.

## What it does

This model approaches outbreak prediction as an interconnected, spatio-temporal problem:
- **Spatial context:** Models how districts are connected using **Graph Attention Networks (GAT)**.
- **Temporal context:** Captures historical trends (weather metrics, past cases) using **LSTMs**.
- **Contextual signals:** Extracts meaningful embeddings from local health news headlines using **BioBERT** and integrates them into the model pipeline.

The architecture (`EpiGraphModel`) handles all of this simultaneously and is optimized with a Huber Loss function (to be robust against sudden huge spikes/outliers in cases).

## How it works
1. **BioBERT Extraction:** Reads news headlines and encodes them into 768-dim embeddings, which are then compressed to 32 dims via PCA for efficiency.
2. **Graph Construction:** Builds an adjacency matrix showing connectivity between different districts.
3. **Sequential combination:** Normalizes and bundles historical weather, cases, and news embeddings into rolling windows (7-day lookback to predict 1 day forward).
4. **Prediction:** Evaluates on unseen data to test Mean Absolute Error, RMSE, and Outbreak Detection Accuracy.

## Data Structure
Expects the following files in a `data/` directory:
```
data/
├── processed_cases.csv   # Target cases and weather variables (.MMAX, .MMIN, etc.)
├── health_news.csv       # News headlines per date/district
└── connectivity.csv      # Adjacency matrix for the districts (Source, Target, Weight)
```

## Dependencies
```bash
pip install -r requirements.txt
```

## How to use
Run `notebooks/EpiGraph_AI.ipynb` top-to-bottom. It will:
- Conduct exploratory data analysis
- Generate and compress embeddings
- Train the model using early stopping
- Display final metrics and export evaluation charts
- Save model weights to `epigraph_model_v2.pth`
