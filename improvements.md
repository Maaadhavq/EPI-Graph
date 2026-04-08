# EpiGraph-AI Improvement Recommendations

This document outlines potential areas for enhancing the performance, scalability, and usability of the EpiGraph-AI framework.

## 1. Model Enhancements
- **Hyperparameter Tuning**: 
  - SYSTEMATICALLY search for optimal `learning_rate` (currently 0.001), `hidden_dim` (64), and `dropout` (0.3).
  - Use libraries like Optuna or Ray Tune.
- **Architecture Upgrades**:
  - **Temporal**: Replace LSTM with a **Transformer Encoder** or **Temporal Convolutional Network (TCN)** for better long-range dependency capture.
  - **Spatial**: Experiment with **GATv2** or **GraphSAGE** for more expressive spatial aggregation.
- **Loss Functions**:
  - Experiment with **Huber Loss** or **Log-Cosh Loss** to be more robust against outliers in case counts.
  - Implement a weighted loss to prioritize predicting *outbreaks* (spikes) over normal low-case days.

## 2. Data Pipeline Improvements
- **Data Augmentation**:
  - The current dataset is small. Augment training data by adding noise to case counts or shuffling news headlines slightly.
- **Advanced NLP**:
  - **Fine-tune BioBERT**: Instead of using frozen embeddings, fine-tune the BioBERT layers on a small labeled medical dataset if available.
  - **Entity Extraction**: Extract specific entities (e.g., "Dengue", "Ahmedabad") to create explicit graph nodes for diseases/locations in addition to the spatial graph.
- **Graph Construction**:
  - Use **Dynamic Graphs**: Allow the adjacency matrix to change over time (e.g., if roads close or travel restrictions apply).

## 3. Engineering & Production
- **Experiment Tracking**:
  - Integrate **Weights & Biases (WandB)** or **MLflow** to track loss curves, hyperparameters, and model versions automatically.
- **API Deployment**:
  - Wrap the inference logic in a **FastAPI** service to allow real-time requests from a frontend.
- **Testing**:
  - Add `pytest` unit tests for:
    - Data loading integrity (null checks, shape checks).
    - Model forward pass (dimension consistency).
    - Loss calculation.

## 4. Visualization & UI
- **Interactive Dashboard**:
  - Upgrade `src/dashboard.py` to use **Streamlit** or **Plotly Dash**.
  - Allow users to select different districts, zoom into time ranges, and view the raw news headlines driving the risk scores.
- **Geospatial Plotting**:
  - Overlay the risk scores on a real map of Gujarat using **Folium** or **Geopandas**.
