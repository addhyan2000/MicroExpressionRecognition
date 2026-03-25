import pandas as pd
from pathlib import Path

excel = Path(r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CAS(ME)2/CAS(ME)^2code_final.xlsx")
video_root = Path(r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CAS(ME)2/rawvideo")

df = pd.read_excel(excel, header=None)
print(f"Total Excel Rows: {len(df)}")

found = 0
for i, row in df.iterrows():
    sub = f"s{int(row[0])}"
    vid_name = str(row[1])
    # Check if subject folder exists
    if (video_root / sub).exists():
        # Just check if ANY video exists for this subject
        if any((video_root / sub).glob("*.avi")):
            found += 1

print(f"Excel rows where we have the Subject's folder: {found}")