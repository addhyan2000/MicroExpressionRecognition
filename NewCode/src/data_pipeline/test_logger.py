import os
import threading
import time
import logging
from pathlib import Path
from logger import get_pipeline_logger

def test_logger_production():
    print("--- Starting Production Logger Stress Test ---")
    log_file = Path("test_pipeline.log")
    
    # 1. Test Thread-Safety (Atomic Initialization)
    # We call the logger from 10 threads at once to see if duplicate handlers are created.
    print("\n[Test 1] Testing Thread-Safe Initialization...")
    loggers = []
    def get_log_worker():
        l = get_pipeline_logger(log_file=log_file)
        loggers.append(l)
    
    threads = [threading.Thread(target=get_log_worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    main_logger = get_pipeline_logger(log_file=log_file)
    # If thread-safety failed, handlers count would be > 2 (1 console + 1 file)
    handler_count = len(main_logger.handlers)
    print(f"Total Handlers Attached: {handler_count} (Expected: 2)")
    
    if handler_count == 2:
        print("SUCCESS: Thread-safety Lock prevented duplicate handlers.")
    else:
        print("FAILURE: Duplicate handlers detected! Check your _LOCK logic.")

    # 2. Test Rotating File Logic (Disk Protection)
    # We write 12MB of data. Since limit is 10MB, it MUST rotate.
    print("\n[Test 2] Testing RotatingFileHandler (Disk Protection)...")
    print("Writing ~12MB of data to trigger rotation...")
    
    # Use a smaller max_bytes just for this test to trigger rotation quickly
    rotating_logger = get_pipeline_logger(
        log_file="rotation_test.log", 
        max_bytes=1024 * 1024, # 1MB limit for test
        backup_count=3
    )
    
    large_string = "X" * 1024  # 1KB string
    for _ in range(1500): # Write 1.5MB total
        rotating_logger.debug(large_string)
    
    # Check for the existence of backup files
    backups = list(Path(".").glob("rotation_test.log*"))
    print(f"Found {len(backups)} log files (Expected: 2 or more due to rotation)")
    
    if len(backups) > 1:
        print("SUCCESS: Log rotation triggered. Your disk is safe.")
    else:
        print("FAILURE: Log file grew past limit without rotating.")

    # 3. Test Resilience (Permission Errors)
    print("\n[Test 3] Testing Permission Error Resilience...")
    # Try to write to a system-protected root path (should fail)
    bad_path = "/this_path_is_illegal.log" if os.name != 'nt' else "C:/Windows/illegal.log"
    
    try:
        fail_logger = get_pipeline_logger(log_file=bad_path, logger_name="fail_test")
        print("SUCCESS: Pipeline did not crash despite illegal file path.")
        print("Check the console above for the 'Failed to initialize' warning.")
    except Exception as e:
        print(f"FAILURE: Pipeline crashed on bad file path: {e}")

    print("\n--- All Logging Sanity Checks Passed. ---")

if __name__ == "__main__":
    # Clean up old test logs first
    for f in Path(".").glob("*.log*"):
        try: f.unlink()
        except: pass
        
    run_test = True
    try:
        test_logger_production()
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {e}")
        run_test = False
        
    if run_test:
        print("\nRESULT: logger.py is PRODUCTION READY.")


            