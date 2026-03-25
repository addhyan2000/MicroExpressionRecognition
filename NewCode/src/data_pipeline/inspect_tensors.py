import numpy as np
import cv2
import random
from pathlib import Path

def inspect_random_tensor(output_dir: str):
    output_path = Path(output_dir)
    # 1. Find all generated .npy files
    tensor_files = list(output_path.glob("*.npy"))
    
    if not tensor_files:
        print(f"No .npy files found in {output_dir}. Did the pipeline run correctly?")
        return

    # 2. Pick a random file
    target_file = random.choice(tensor_files)
    print(f"--- Inspecting: {target_file.name} ---")
    
    # 3. Load the tensor (Shape: T, H, W, 3)
    data = np.load(target_file)
    T, H, W, C = data.shape
    print(f"Tensor Shape: {T} frames | {H}x{W} resolution | {C} channels")

    # 4. Playback Loop
    print("Press 'q' to exit, any other key to see another video.")
    
    # Normalization helper: Maps flow/strain values to 0-255 for display
    def to_vis(img):
        # We use a simple min-max to make the subtle motions visible to the eye
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return (img * 255).astype(np.uint8)

    for i in range(T):
        # Extract individual channels
        flow_x = to_vis(data[i, :, :, 0])
        flow_y = to_vis(data[i, :, :, 1])
        strain = to_vis(data[i, :, :, 2])

        # Apply colormaps for better visual intuition
        # X Flow: Cyan/Blue | Y Flow: Green | Strain: Hot/Red
        vis_x = cv2.applyColorMap(flow_x, cv2.COLORMAP_OCEAN)
        vis_y = cv2.applyColorMap(flow_y, cv2.COLORMAP_SUMMER)
        vis_s = cv2.applyColorMap(strain, cv2.COLORMAP_JET)

        # Concatenate horizontally
        combined = np.hstack((vis_x, vis_y, vis_s))
        
        # Add labels
        cv2.putText(combined, f"FRAME: {i:02d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "FLOW X", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "FLOW Y", (W + 10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "STRAIN", (2*W + 10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("MER Stage 1: Feature Inspection", combined)
        
        # Playback at ~15 FPS
        if cv2.waitKey(60) & 0xFF == ord('q'):
            return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Point this to your actual output directory
    OUTPUT_DIR = r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/NewCode/outputs/CASME_II"
    
    while True:
        inspect_random_tensor(OUTPUT_DIR)
        print("\nClose the video window to pick another random file...")
        key = input("Press 'Enter' for random sample, 'q' to quit: ")
        if key.lower() == 'q':
            break
        