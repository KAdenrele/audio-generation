import sys
import os
# This forces Python to look in /opt/LuxTTS regardless of where this script lives
lux_docker_path = "/opt/LuxTTS"

if os.path.exists(lux_docker_path) and lux_docker_path not in sys.path:
    # insert(0) puts it at the absolute top of Python's search priority
    sys.path.insert(0, lux_docker_path)

import torch
import soundfile as sf
from tqdm import tqdm
from zipvoice.luxtts import LuxTTS
import logging
def run_lux_batch(prompts, output_dir="generated_dataset"):
    """
    Generates audio for a list of text prompts using Lux TTS.
    """
    target_dir = os.path.join(output_dir, "lux")
    os.makedirs(target_dir, exist_ok=True)
    
    logging.info("=== STARTING LUX TTS ===")   

    try:
        # 2. Load Model
        # Lux downloads its own weights (approx 1.2GB) automatically if missing
        logging.info("Loading LuxTTS model...")
        model = LuxTTS('YatharthS/LuxTTS', device='cuda')
        logging.info("Model loaded successfully.")
        
        success_count = 0
        skip_count = 0
        
        # 3. Generation Loop
        for i, text in enumerate(tqdm(prompts, desc="Lux Generation")):
            filename = os.path.join(target_dir, f"sample_{i}.wav")
            
            # Resume: Skip if file exists
            if os.path.exists(filename):
                skip_count += 1
                continue
            
            try:
                # Lux generates a Torch tensor at 48kHz
                audio = model.inference(str(text))
                
                # Move to CPU and save
                # Note: Lux uses 48000 Hz sample rate
                sf.write(filename, audio.cpu().numpy(), 48000)
                success_count += 1
                
            except Exception as e:
                logging.error(f"Error generating Lux sample {i}: {e}")
        
        logging.info("=== LUX COMPLETE ===")
        logging.info(f"Generated: {success_count}")
        logging.info(f"Skipped: {skip_count}")
        
        # 4. Cleanup (Crucial for multi-model runs)
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logging.critical(f"Lux Failed to Run: {e}")
        raise e

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Running lux_runner.py in standalone mode...")
    test_prompts = ["Lux TTS is high fidelity.", "It uses a lateny diffusion model."]
    run_lux_batch(test_prompts)