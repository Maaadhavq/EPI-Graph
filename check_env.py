import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), "EpiGraph-AI"))

def check_imports():
    print("Checking imports...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch NOT installed.")

    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric NOT installed.")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers NOT installed.")

    try:
        import pandas
        print(f"Pandas version: {pandas.__version__}")
    except ImportError:
        print("Pandas NOT installed.")

def check_data_loading():
    print("\nChecking data loading...")
    try:
        from src.dataset import load_raw_data
        cases, news, conn = load_raw_data()
        print(f"Cases Data loaded. Rows: {len(cases)}")
        print(f"News Data loaded. Rows: {len(news)}")
        print(f"Connectivity Data loaded. Rows: {len(conn)}")
    except Exception as e:
        print(f"Data loading failed: {e}")

if __name__ == "__main__":
    check_imports()
    check_data_loading()
