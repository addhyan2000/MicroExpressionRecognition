import os
from pathlib import Path

# Path to your folder
folder_path = Path(r"C:\Users\Addhyan\Desktop\Thesis\Thesis2\THESIS WORK new\NewCode\outputs\CAS_ME_2")

files = os.listdir(folder_path)
npy_files = [f for f in files if f.endswith('.npy')]
micro_count = [f for f in npy_files if "micro" in f.lower()]
macro_count = [f for f in npy_files if "macro" in f.lower()]

print(f"--- Folder Audit ---")
print(f"Total .npy files: {len(npy_files)}")
print(f"Micro-expressions found: {len(micro_count)}")
print(f"Macro-expressions found: {len(macro_count)}")

if micro_count:
    print(f"\nExample Micro Filename: {micro_count[0]}")
    print(f"Example Macro Filename: {macro_count[0]}")