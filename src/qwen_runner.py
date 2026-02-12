import torch
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel

def run_qwen3_batch(prompts, output_dir="generated_dataset"):
    """
    Generates audio for a list of text prompts using Qwen3-TTS.
    """
    target_dir = os.path.join(output_dir, "qwen3")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n=== STARTING QWEN3-TTS (1.7B) ===")
    print(f"Output Directory: {target_dir}")
    
    try:
        # Load Model
        # Note: We use bfloat16 for speed and lower VRAM usage on modern GPUs
        print("Loading model weights...")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
            device_map="cuda", 
            dtype=torch.bfloat16
        )
        print("Model loaded successfully.")
        
        # Generation Loop
        success_count = 0
        skip_count = 0
        
        for i, text in enumerate(tqdm(prompts, desc="Qwen3 Generation")):
            filename = os.path.join(target_dir, f"sample_{i}.wav")
            
            # Resume feature: Skip if file exists
            if os.path.exists(filename):
                skip_count += 1
                continue
            
            try:
                # Generate audio (Speaker 'Ryan' is a standard male voice)
                wavs, sr = model.generate_custom_voice(text=str(text), speaker="Ryan")
                
                # Save to disk
                # wavs[0] contains the raw audio data
                sf.write(filename, wavs[0], sr)
                success_count += 1
                
            except Exception as e:
                print(f"\n[!] Error generating sample {i}: {e}")
                # Optional: Save a dummy file or log the error to a file
        
        print(f"\n=== QWEN3 COMPLETE ===")
        print(f"Generated: {success_count}")
        print(f"Skipped (Already Existed): {skip_count}")
        
        # Cleanup VRAM
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\n[CRITICAL] Qwen3 Failed to Load or Run: {e}")
        raise e

if __name__ == "__main__":
    # Test block to run this script standalone
    print("Running qwen_runner.py in standalone mode...")
    test_prompts = ["Hello, this is a test of Qwen3 TTS.", "Generating audio is fun."]
    run_qwen3_batch(test_prompts)