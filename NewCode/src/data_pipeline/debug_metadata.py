import pandas as pd
from pathlib import Path

# Update these to your actual paths
excel_path = Path(r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CAS(ME)2/CAS(ME)^2code_final.xlsx")
root_path = Path(r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CAS(ME)2/selectedpic")

df = pd.read_excel(excel_path, header=None)
print(f"Total rows in Excel: {len(df)}")

missing_count = 0
found_count = 0

for i, row in df.iterrows():
    try:
        sub_id = int(row[0])
        subject = f"s{sub_id}" 
        video_name = str(row[1])
        
        video_path = root_path / subject / video_name
        
        if video_path.exists():
            found_count += 1
        else:
            missing_count += 1
            if missing_count < 5: # Just show the first few errors
                print(f"❌ Not Found: {video_path}")
    except:
        continue

print(f"\n--- RESULTS ---")
print(f"✅ Paths that exist: {found_count}")
print(f"❌ Paths that failed: {missing_count}")