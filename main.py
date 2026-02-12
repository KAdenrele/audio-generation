import pandas as pd
import os
import sys
from src.qwen_runner import run_qwen3_batch
# from lux_runner import run_lux_batch


# Configuration
PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset"

def main():
    # 1. Load Prompts
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: {PROMPT_FILE} not found.")
        print("Please ensure prompts.csv is in the same directory.")
        return

    print(f"Reading prompts from {PROMPT_FILE}...")
    try:
        df = pd.read_csv(PROMPT_FILE)
        # Ensure we have a 'text' column, or use the first column
        if 'text' in df.columns:
            prompts = df['text'].tolist()
        else:
            print("Warning: 'text' column not found. Using the first column.")
            prompts = df.iloc[:, 0].tolist()
            
        # Limit to first 500 for this run
        prompts = prompts[:500]
        print(f"Loaded {len(prompts)} prompts.")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Run Qwen3
    # We wrap this in a try/except so main doesn't crash if Qwen fails
    try:
        run_qwen3_batch(prompts, OUTPUT_DIR)
    except Exception as e:
        print(f"Qwen execution halted: {e}")

    # (Future) Run Lux
    
    # run_lux_batch(prompts, OUTPUT_DIR)

if __name__ == "__main__":
    main()