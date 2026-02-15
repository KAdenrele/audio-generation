import pandas as pd
import os
import sys
from src.qwen_runner import run_qwen3_batch
from src.lux_runner import run_lux_batch
import logging
# from lux_runner import run_lux_batch


# Configuration
PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset"
CHATTERBOX_REF_CLIP = "ref_clip.m4a"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    #Load Prompts
    if not os.path.exists(PROMPT_FILE):
        logging.error(f"Error: {PROMPT_FILE} not found.")
        logging.error("Please ensure prompts.csv is in the same directory.")
        return

    logging.info(f"Reading prompts from {PROMPT_FILE}...")
    try:
        df = pd.read_csv(PROMPT_FILE)
        # Ensure we have a 'text' column, or use the first column
        if 'text' in df.columns:
            prompts = df['text'].tolist()
        else:
            logging.warning("'text' column not found. Using the first column.")
            prompts = df.iloc[:, 0].tolist()
            
        # Limit to first 500 for this run
        prompts = prompts[:500]
        logging.info(f"Loaded {len(prompts)} prompts.")
        
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return

    #Run Qwen3
    # try:
    #     run_qwen3_batch(prompts, OUTPUT_DIR)
    # except Exception as e:
    #     logging.error(f"Qwen execution halted: {e}")

    #Run Lux
    try:
        run_lux_batch(prompts, OUTPUT_DIR)
    except Exception as e:
        logging.error(f"Skipping Lux due to error: {e}")


if __name__ == "__main__":
    main()