import pandas as pd
import torch
import soundfile as sf
import os
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel
from zipvoice.luxtts import LuxTTS 
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_qwen3_batch(prompts):
    logging.info("=== STARTING QWEN3 ===")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    os.makedirs(f"{OUTPUT_DIR}/qwen3", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Qwen3")):
        wavs, sr = model.generate_custom_voice(text=text, speaker="Ryan")
        sf.write(f"{OUTPUT_DIR}/qwen3/sample_{i}.wav", wavs[0], sr)
    del model
    torch.cuda.empty_cache()
    logging.info("=== FINISHED QWEN3 ===")

def run_lux_batch(prompts):
    logging.info("=== STARTING LUX ===")
    model = LuxTTS('YatharthS/LuxTTS', device='cuda') 
    os.makedirs(f"{OUTPUT_DIR}/lux", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Lux")):
        try:
            audio = model.inference(text) 
            sf.write(f"{OUTPUT_DIR}/lux/sample_{i}.wav", audio.cpu().numpy(), 48000)
        except Exception as e:
            logging.error(f"Lux error on prompt {i}: {e}")
    del model
    torch.cuda.empty_cache()
    logging.info("=== FINISHED LUX ===")

if __name__ == "__main__":
    if not os.path.exists(PROMPT_FILE):
        logging.error(f"Error: {PROMPT_FILE} not found.")
        sys.exit(1)
        
    df = pd.read_csv(PROMPT_FILE).head(500)
    prompts = df['text'].tolist()
    logging.info(f"Loaded {len(prompts)} prompts.")

    run_qwen3_batch(prompts)
    run_lux_batch(prompts)
