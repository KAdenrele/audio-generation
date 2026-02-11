from prompt_file import raw_text
import pandas as pd
import re



def clean_and_split(text):
    # 1. Basic cleaning (remove extra whitespace)
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # 3. Filter: Keep only sentences between 20 and 150 characters
    # (Too short sounds robotic; too long might crash some TTS models)
    cleaned = [s.strip() for s in sentences if 20 < len(s) < 200]
    return cleaned

prompts = clean_and_split(raw_text)

# If you have fewer than 500, we duplicate/shuffle to reach the goal
while len(prompts) < 500:
    prompts.extend(prompts[:500-len(prompts)])

df = pd.DataFrame({'text': prompts})
df.to_csv("prompts.csv", index=False, quoting=1) # quoting=1 handles commas automatically
print(f"Created prompts.csv with {len(df)} samples.")