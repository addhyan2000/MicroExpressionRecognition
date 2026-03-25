import os
import shutil
import tempfile
from pathlib import Path
from config import PipelineConfig

def test_config_logic():
    print("--- Starting Production Config Validation Test ---")
    
    # Setup: Create a real temporary directory to act as input_dir
    tmp_input = Path(tempfile.mkdtemp())
    tmp_output = Path(tempfile.mkdtemp())
    # Remove output dir to test if setup_filesystem creates it
    if tmp_output.exists():
        shutil.rmtree(tmp_output)

    print(f"Using Temp Input: {tmp_input}")
    print(f"Using Temp Output: {tmp_output}")

    # 1. Test Valid Configuration (The Happy Path)
    print("\n[Test 1] Validating standard configuration...")
    try:
        cfg = PipelineConfig(
            dataset_name="CASME_II_TEST",
            input_dir=tmp_input,
            output_dir=tmp_output,
            target_fps=30,
            evm_high_omega=3.0 # Well below 15Hz Nyquist
        )
        print("SUCCESS: Valid config instantiated.")
        
        print("Testing filesystem setup...")
        cfg.setup_filesystem()
        if tmp_output.exists() and tmp_output.is_dir():
            print("SUCCESS: Output directory created correctly.")
    except Exception as e:
        print(f"FAILURE: Valid config raised an error: {e}")

    # 2. Test Input Directory Validation
    print("\n[Test 2] Testing missing input directory...")
    try:
        PipelineConfig(
            dataset_name="FAIL_TEST",
            input_dir=Path("C:/This/Path/Does/Not/Exist/Hopefully"),
            output_dir=tmp_output
        )
        print("FAILURE: Config allowed a non-existent input directory!")
    except FileNotFoundError:
        print("SUCCESS: Caught missing input directory.")

    # 3. Test Frequency Logic (Inversion)
    print("\n[Test 3] Testing Frequency Inversion (low > high)...")
    try:
        PipelineConfig(
            dataset_name="FAIL_TEST",
            input_dir=tmp_input,
            output_dir=tmp_output,
            evm_low_omega=5.0,
            evm_high_omega=2.0
        )
        print("FAILURE: Config allowed low_omega > high_omega!")
    except ValueError as e:
        print(f"SUCCESS: Caught frequency inversion: {e}")

    # 4. Test Scientific Nyquist Limit
    print("\n[Test 4] Testing Nyquist Limit (Scientific Integrity)...")
    # At 30 FPS, the Nyquist limit is 15 Hz. Setting high_omega to 16 Hz should fail.
    try:
        PipelineConfig(
            dataset_name="FAIL_TEST",
            input_dir=tmp_input,
            output_dir=tmp_output,
            target_fps=30,
            evm_high_omega=16.0
        )
        print("FAILURE: Config allowed frequency above Nyquist limit!")
    except ValueError as e:
        print(f"SUCCESS: Caught Nyquist violation: {e}")

    # 5. Test Optical Flow Minimum
    print("\n[Test 5] Testing Optical Flow Frame Minimum...")
    try:
        PipelineConfig(
            dataset_name="FAIL_TEST",
            input_dir=tmp_input,
            output_dir=tmp_output,
            target_frames=1
        )
        print("FAILURE: Config allowed target_frames < 2!")
    except ValueError as e:
        print(f"SUCCESS: Caught frame minimum violation: {e}")

    # 6. Test File/Folder Conflict
    print("\n[Test 6] Testing Output Path Conflict...")
    conflict_file = tmp_output.parent / "conflict_file.txt"
    conflict_file.touch()
    try:
        cfg_conflict = PipelineConfig(
            dataset_name="FAIL_TEST",
            input_dir=tmp_input,
            output_dir=conflict_file
        )
        cfg_conflict.setup_filesystem()
        print("FAILURE: setup_filesystem allowed overwriting a file with a directory!")
    except RuntimeError as e:
        print(f"SUCCESS: Caught path conflict: {e}")

    # Cleanup
    shutil.rmtree(tmp_input)
    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    if conflict_file.exists():
        conflict_file.unlink()

    print("\n--- All Configuration Sanity Checks Passed. ---")

if __name__ == "__main__":
    test_config_logic()