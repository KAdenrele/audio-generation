import sys
import pandas as pd
import soundfile as sf
import os
from tqdm import tqdm
from pocket_tts import TTSModel

# Load prompts
PROMPT_FILE = "prompts.csv"
OUTPUT_DIR = "generated_dataset/pocket"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    print("Loading Pocket TTS (NumPy 2 Env)...")
    # Load 500 prompts
    df = pd.read_csv(PROMPT_FILE).head(500)
    prompts = df['text'].tolist()

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")

    for i, text in enumerate(tqdm(prompts, desc="Pocket Progress")):
        try:
            # Generate
            audio_tensor = model.generate_audio(voice_state, text)
            # Save
            sf.write(f"{OUTPUT_DIR}/sample_{i}.wav", audio_tensor.numpy(), 24000)
        except Exception as e:
            print(f"Error on sample {i}: {e}")

if __name__ == "__main__":
    run()