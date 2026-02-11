import pandas as pd
import torch
import soundfile as sf
import os
import subprocess
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel
from zipvoice.luxtts import LuxTTS 

PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_qwen3_batch(prompts):
    print("\n=== STARTING QWEN3 ===")
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

def run_lux_batch(prompts):
    print("\n=== STARTING LUX ===")
    model = LuxTTS('YatharthS/LuxTTS', device='cuda') 
    os.makedirs(f"{OUTPUT_DIR}/lux", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Lux")):
        try:
            audio = model.inference(text) 
            sf.write(f"{OUTPUT_DIR}/lux/sample_{i}.wav", audio.cpu().numpy(), 48000)
        except Exception as e:
            print(f"Lux error {i}: {e}")
    del model
    torch.cuda.empty_cache()

def run_pocket_batch():
    print("\n=== STARTING POCKET TTS (Subprocess) ===")
    # We call the separate python executable in the 'pocket_env' folder
    subprocess.run(["/app/pocket_env/bin/python", "pocket_worker.py"], check=True)

if __name__ == "__main__":
    if not os.path.exists(PROMPT_FILE):
        print("Error: prompts.csv not found")
        exit(1)
        
    df = pd.read_csv(PROMPT_FILE).head(500)
    prompts = df['text'].tolist()

    run_qwen3_batch(prompts)
    run_lux_batch(prompts)
    run_pocket_batch() # Calls the external worker