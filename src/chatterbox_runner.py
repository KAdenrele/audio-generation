import torch
import torchaudio
import os
from tqdm import tqdm
import logging
from chatterbox.tts_turbo import ChatterboxTurboTTS

def run_chatterbox_batch(prompts, ref_audio_path, output_dir="generated_dataset"):
    """
    Generates audio for a list of text prompts using Chatterbox Turbo TTS.
    Requires a reference audio clip for voice cloning.
    """
    target_dir = os.path.join(output_dir, "chatterbox")
    os.makedirs(target_dir, exist_ok=True)
    
    logging.info("=== STARTING CHATTERBOX TURBO TTS ===")
    logging.info(f"Output Directory: {target_dir}")
    logging.info(f"Reference Audio: {ref_audio_path}")

    if not os.path.exists(ref_audio_path):
        logging.error(f"Reference audio file not found: {ref_audio_path}")
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

    try:
        # 1. Load Model
        logging.info("Loading Chatterbox Turbo model...")
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        logging.info("Model loaded successfully.")
        
        success_count = 0
        skip_count = 0
        
        # 2. Generation Loop
        for i, text in enumerate(tqdm(prompts, desc="Chatterbox Generation")):
            filename = os.path.join(target_dir, f"sample_{i}.wav")
            
            # Resume: Skip if file exists
            if os.path.exists(filename):
                skip_count += 1
                continue
            
            try:
                # Generate audio using the reference clip
                wav = model.generate(str(text), audio_prompt_path=ref_audio_path)
                
                # Save to disk
                torchaudio.save(filename, wav.cpu(), model.sr)
                success_count += 1
                
            except Exception as e:
                logging.error(f"Error generating Chatterbox sample {i}: {e}")
        
        logging.info("=== CHATTERBOX COMPLETE ===")
        logging.info(f"Generated: {success_count}")
        logging.info(f"Skipped (Already Existed): {skip_count}")
        
        # 3. Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logging.critical(f"Chatterbox Failed to Load or Run: {e}")
        raise e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Running chatterbox_runner.py in standalone mode...")
    
    # --- IMPORTANT ---
    # Chatterbox requires a reference audio clip for voice cloning.
    # Please place a 10-second WAV file named 'ref_clip.wav' in the root directory.
    ref_clip_path = "ref_clip.wav"
    
    if not os.path.exists(ref_clip_path):
        logging.warning(f"Reference clip '{ref_clip_path}' not found. Standalone test cannot run.")
    else:
        test_prompts = ["Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat?"]
        run_chatterbox_batch(test_prompts, ref_audio_path=ref_clip_path)