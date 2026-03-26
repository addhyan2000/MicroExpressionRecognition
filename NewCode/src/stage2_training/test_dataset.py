# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# --- PROJECT ROOT INITIALIZATION ---
# This ensures Python can find the 'src' folder regardless of where you run the script from.
_current_file = Path(__file__).resolve()
# We go up two levels: stage2_training -> src -> NewCode
_project_root = _current_file.parents[2] 
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.stage2_training.dataset import MERDataset
import logging

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

# 2. Define your paths (using Path for Windows/Linux compatibility)
DATA_DIR = _project_root / "outputs" / "CASME_II"
LABEL_CSV = _project_root / "src" / "stage2_training" / "casme2_labels.csv"

def run_test():
    print("\n" + "="*50)
    print(" STAGE 2: DATASET & LABEL VERIFICATION ")
    print("="*50)

    # 3. Initialize Dataset
    if not LABEL_CSV.exists():
        print(f"❌ ERROR: Could not find {LABEL_CSV.name}. Did you run the label script?")
        return

    ds = MERDataset(
        data_dir=DATA_DIR, 
        label_csv=LABEL_CSV, 
        normalize=True,
        preload=False, 
        mmap=True
    )

    # 4. Run the Dry Run
    try:
        ds.dry_run(batch_size=8)
        print("\n✅ SUCCESS: The pipeline is officially production-ready.")
    except Exception as e:
        print(f"\n❌ FAILED: Still hitting an error: {e}")

if __name__ == "__main__":
    run_test()