import os
import pandas as pd
from pathlib import Path

# SET YOUR PATHS
NPY_FOLDER = Path(r"C:\Users\Addhyan\Desktop\Thesis\Thesis2\THESIS WORK new\NewCode\outputs\CAS_ME_2")
OUTPUT_CSV = Path(r"C:\Users\Addhyan\Desktop\Thesis\Thesis2\THESIS WORK new\NewCode\src\stage2_training\casme_square_labels.csv")

# Emotion mapping to match your CASME II training classes
emotion_map = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "happy": "positive",
    "surprise": "surprise"
}

data = []

for file in os.listdir(NPY_FOLDER):
    if file.endswith(".npy") and "micro-expression" in file:
        # Example: s15_anger1_1_micro-expression_268_278.npy
        parts = file.split("_")
        raw_emotion = parts[1] # 'anger1'
        
        # Clean the emotion (remove numbers like 'anger1' -> 'anger')
        clean_emotion = "".join([i for i in raw_emotion if not i.isdigit()])
        
        # Map to our 4-class system
        mapped_emotion = emotion_map.get(clean_emotion, "others")
        
        data.append({"filename": file, "emotion": mapped_emotion})

# Save the CSV
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Generated {len(df)} labels for CAS(ME)^2 Micro-expressions!")
print(df['emotion'].value_counts()) # Shows you the class distribution