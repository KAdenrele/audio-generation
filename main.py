import pandas as pd
import torch
import soundfile as sf
import os
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel
from pocket_tts import TTSModel
from zipvoice.luxtts import LuxTTS 

# Configuration
PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. QWEN3-TTS BATCH GENERATOR
def run_qwen3_batch(prompts):
    print("Loading Qwen3-TTS (1.7B)...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    
    os.makedirs(f"{OUTPUT_DIR}/qwen3", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Qwen3 Progress")):
        try:
            wavs, sr = model.generate_custom_voice(text=text, speaker="Ryan")
            sf.write(f"{OUTPUT_DIR}/qwen3/sample_{i}.wav", wavs[0], sr)
        except Exception as e:
            print(f"Qwen3 error at index {i}: {e}")
    
    # Free VRAM for the next model
    del model
    torch.cuda.empty_cache()

# 2. POCKET TTS BATCH GENERATOR
def run_pocket_batch(prompts):
    print("Loading Pocket TTS...")
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    
    os.makedirs(f"{OUTPUT_DIR}/pocket", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Pocket Progress")):
        try:
            audio_tensor = model.generate_audio(voice_state, text)
            sf.write(f"{OUTPUT_DIR}/pocket/sample_{i}.wav", audio_tensor.numpy(), 24000)
        except Exception as e:
            print(f"Pocket TTS error at index {i}: {e}")

# 3. LUX TTS BATCH GENERATOR (Local Import)
def run_lux_batch(prompts):
    print("Loading Lux TTS...")
    # This downloads weights automatically from HuggingFace on first run
    model = LuxTTS('YatharthS/LuxTTS', device='cuda') 
    
    os.makedirs(f"{OUTPUT_DIR}/lux", exist_ok=True)
    for i, text in enumerate(tqdm(prompts, desc="Lux Progress")):
        try:
            # Lux returns audio as a torch tensor
            audio = model.inference(text) 
            sf.write(f"{OUTPUT_DIR}/lux/sample_{i}.wav", audio.cpu().numpy(), 48000)
        except Exception as e:
            print(f"Lux error at index {i}: {e}")

# MAIN EXECUTION
if __name__ == "__main__":
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: {PROMPT_FILE} not found.")
    else:
        df = pd.read_csv(PROMPT_FILE).head(500)
        prompts = df['text'].tolist()

        # Running them one after another
        run_qwen3_batch(prompts)
        run_pocket_batch(prompts)
        run_lux_batch(prompts)