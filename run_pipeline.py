import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    try:
        # Use simple os.system or subprocess
        # We assume python is available since this script is running
        ret = os.system(f"python {script_name}")
        if ret != 0:
            print(f"Error running {script_name}. Return code: {ret}")
            return False
        return True
    except Exception as e:
        print(f"Failed to execute {script_name}: {e}")
        return False

def main():
    print("Starting EpiGraph-AI Pipeline...")
    
    # 1. Environment Check
    if not run_script("check_env.py"):
        print("Environment check failed. Please check dependencies.")
        # Proceeding anyway? No.
        # return
    
    # 2. Train Model
    # Note: src/train.py is inside src, but we can run it as module or script.
    # Our scripts use sys.path hacks, so likely safe.
    if not run_script("src/train.py"):
        print("Training failed.")
        return
        
    # 3. Dashboard / Visualization
    if not run_script("src/dashboard.py"):
        print("Dashboard failed.")
        return
        
    print("\nPipeline Completed Successfully!")

if __name__ == "__main__":
    main()
