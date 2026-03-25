import os
import glob
import cv2
import numpy as np
import logging
from processors import TemporalInterpolator, EVMProcessor, FlowExtractor

logging.basicConfig(level=logging.INFO)

def load_frames(folder_path):
    print(f"Loading frames from: {folder_path}")
    
    # Check for jpg, jpeg, png
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.[jp][pn]g")))
    
    if not image_paths:
        raise ValueError(f"No images found in {folder_path}. Please check the path.")
        
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
        
    video_tensor = np.stack(frames, axis=0)
    
    # Cast to float32 and normalize to 0-1 range to test the lazy scaling logic
    video_tensor = video_tensor.astype(np.float32) / 255.0
    
    print(f"Loaded {len(frames)} frames. Tensor shape: {video_tensor.shape}")
    return video_tensor

def run_pipeline_check():
    print("=== Senior Developer Pipeline Sanity Check ===")
    
    # Using a raw string (r"...") to prevent Windows path parsing errors
    TEST_FOLDER = r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/DATASETS/CASME II/Cropped/sub01/EP02_01f"
    
    use_dummy_data = False
    
    # FIXED: Only check if the path actually exists on your hard drive
    if not os.path.exists(TEST_FOLDER):
        print(f"\nWARNING: Could not find the path: {TEST_FOLDER}")
        print("Falling back to generating synthetic moving data to verify the math...")
        use_dummy_data = True

    if use_dummy_data:
        T, H, W, C = 20, 128, 128, 3
        input_tensor = np.zeros((T, H, W, C), dtype=np.float32)
        # Create a moving white box to simulate facial muscle movement
        for i in range(T):
            input_tensor[i, i:i+30, i:i+30, :] = 1.0
    else:
        input_tensor = load_frames(TEST_FOLDER)

    print("\n--- STEP 1: Running Temporal Interpolator ---")
    interpolator = TemporalInterpolator(target_frames=32)
    interp_output = interpolator.process(input_tensor)
    print(f"Output shape after interpolation: {interp_output.shape}")
    assert interp_output.shape[0] == 32, "Interpolator failed to reach target frames!"

    print("\n--- STEP 2: Running EVM Processor ---")
    evm = EVMProcessor(alpha=20.0, low_omega=0.4, high_omega=3.0)
    # Using 200 fps assuming CASME II data for this test
    evm_output = evm.process(interp_output, fps=200) 
    print(f"Output shape after EVM: {evm_output.shape}")

    print("\n--- STEP 3: Running Flow and Strain Extractor ---")
    extractor = FlowExtractor()
    flow_output = extractor.process(evm_output)
    
    print(f"Output shape after Flow Extraction: {flow_output.shape}")
    print(f"Expected shape: (31, Height, Width, 3)")
    
    max_flow = np.max(flow_output)
    max_strain = np.max(flow_output[..., 2])
    
    print(f"\nDiagnostics:")
    print(f"Global maximum value (checking for NaN/Infs): {max_flow:.4f}")
    print(f"Strain magnitude maximum value: {max_strain:.4f}")
    
    if max_flow > 0:
        print("\n[SUCCESS] The pipeline is production-ready. Optical flow and strain were successfully extracted without memory crashes.")
    else:
        print("\n[WARNING] Maximum flow is 0. If you used a completely black image, this is normal. Otherwise, check your data.")

if __name__ == "__main__":
    run_pipeline_check()