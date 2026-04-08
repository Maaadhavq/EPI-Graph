# EpiGraph-AI

**Project Title:** EpiGraph-AI: A Spatiotemporal Multi-Modal Disease Outbreak Forecasting Framework using Bio-Medical NLP.

## Overview
This project integrates semantic risk signals from medical news with temporal case data to predict localized disease outbreaks using Graph Neural Networks and BioBERT.

## Project Structure
- `data/`: Contains the dataset files (`processed_cases.csv`, `health_news.csv`, `connectivity.csv`).
- `src/`: Source code.
  - `config.py`: Configuration and file paths.
  - `dataset.py`: Data loading utilities.
- `notebooks/`: Jupyter notebooks for experimentation.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### 1. Install Python
Ensure Python 3.8+ is installed and available in your system path.

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Verify Setup
Run the verification script to check your environment and data loading:
```bash
python check_env.py
```

## Usage

### Running the Pipeline
To run the entire pipeline (Environment Check -> Training -> Visualization):
```bash
python run_pipeline.py
```

### Individual Steps
1. **Training**:
   ```bash
   python src/train.py
   ```
   This will train the model and save `epigraph_model.pth`.

2. **Dashboard**:
   ```bash
   python src/dashboard.py
   ```
   This will load the trained model and generate `dashboard_risk_plot.png`.

## Project Structure
- `data/`: Datasets.
- `src/`:
  - `dataset.py`: Data loading and preprocessing (BioBERT + Sliding Window).
  - `model.py`: GAT + LSTM Model.
  - `train.py`: Training loop.
  - `preprocessing.py`: Encoder and Windowing classes.
  - `config.py`: configuration.
