import threading
import os
import json
import time
from pathlib import Path
from checkpoint import CheckpointManager

def test_final_resilience():
    print("--- Starting Final Checkpoint Resilience Test ---")
    path = Path("final_test_checkpoint.json")
    
    # 1. Test Concurrency
    if path.exists(): path.unlink()
    mgr = CheckpointManager(path)
    
    def worker(tid):
        for i in range(20):
            mgr.mark_processed(f"thread_{tid}_vid_{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    print(f"Count after concurrency: {mgr.processed_count()} (Expected 100)")
    
    # 2. Test Corruption Backup
    print("\nSimulating file corruption...")
    with open(path, "w") as f:
        f.write("NOT A JSON FILE - CORRUPTING ON PURPOSE")
    
    # Re-instantiate - this should trigger the backup
    print("Re-initializing Manager...")
    mgr_new = CheckpointManager(path)
    
    # Check if backup was created
    backups = list(path.parent.glob("*.json.corrupted"))
    if backups:
        print(f"SUCCESS: Found {len(backups)} backup file(s): {backups[0].name}")
    else:
        print("FAILURE: Corrupted file was lost!")

    if mgr_new.processed_count() == 0:
        print("SUCCESS: Fresh state initialized after corruption.")

if __name__ == "__main__":
    test_final_resilience()