import pandas as pd
import os
from pathlib import Path

# ==========================================
# 1. SET YOUR PATHS HERE
# ==========================================
# Update this to where your CASME II Excel file actually is!
EXCEL_PATH = r"C:\Users\Addhyan\Desktop\Thesis\Thesis2\THESIS WORK new\DATASETS\CASME II\CASME2-coding-20140508.xlsx" 

# This is where the new CSV will be saved
OUTPUT_CSV = r"C:\Users\Addhyan\Desktop\Thesis\Thesis2\THESIS WORK new\NewCode\src\stage2_training\casme2_labels_7class.csv"

# ==========================================
# 2. FILE CHECK
# ==========================================
if not os.path.exists(EXCEL_PATH):
    print(f"❌ ERROR: Could not find the Excel file at: {EXCEL_PATH}")
    print("Please find 'CASME2-coding-20140508.xlsx' and copy its full path here.")
    exit()

# ==========================================
# 3. PROCESSING
# ==========================================
print(f"📖 Reading annotations from: {EXCEL_PATH}")
# CASME II Excel files often have the data on the first sheet
df = pd.read_excel(EXCEL_PATH)

# Standard CASME II columns are usually: 'Subject', 'Filename', 'OnsetFrame', 'Estimated Emotion'
# We create the filename to match your .npy files: sub01_EP01_01f_onset.npy
def generate_npy_name(row):
    sub = f"sub{int(row['Subject']):02d}"
    fname = row['Filename']
    onset = int(row['OnsetFrame'])
    return f"{sub}_{fname}_{onset}.npy"

# Create the data list
data = []
for _, row in df.iterrows():
    filename = generate_npy_name(row)
    # Get the raw emotion (disgust, happiness, surprise, etc.)
    emotion = str(row['Estimated Emotion']).lower().strip()
    data.append({"filename": filename, "emotion": emotion})

# Save the 7-class CSV
final_df = pd.DataFrame(data)
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Success! 7-Class CSV saved to: {OUTPUT_CSV}")
print("\nEmotion Distribution for your Thesis:")
print(final_df['emotion'].value_counts())