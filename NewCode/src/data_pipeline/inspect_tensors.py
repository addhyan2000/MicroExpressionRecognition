import numpy as np
import cv2
import random
from pathlib import Path

def inspect_random_tensor(output_dir: str):
    output_path = Path(output_dir)
    tensor_files = list(output_path.glob("*.npy"))
    
    if not tensor_files:
        print(f"\n[!] No files found in {output_dir}")
        return False

    target_file = random.choice(tensor_files)
    data = np.load(target_file)
    T, H, W, C = data.shape

    def to_vis(img):
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return (img * 255).astype(np.uint8)

    # We use a window name constant to avoid typos
    WIN_NAME = "MER Feature Inspector"
    cv2.namedWindow(WIN_NAME)

    # Play the clip once
    for i in range(T):
        # ... (same visualization logic as before) ...
        flow_x = to_vis(data[i, :, :, 0])
        flow_y = to_vis(data[i, :, :, 1])
        strain = to_vis(data[i, :, :, 2])
        combined = np.hstack((cv2.applyColorMap(flow_x, cv2.COLORMAP_OCEAN),
                             cv2.applyColorMap(flow_y, cv2.COLORMAP_SUMMER),
                             cv2.applyColorMap(strain, cv2.COLORMAP_JET)))

        cv2.putText(combined, f"FRAME: {i:02d}/31", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WIN_NAME, combined)
        
        # If user presses 'q' during playback, exit
        if cv2.waitKey(60) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False

    # --- THE FIX: ADD ON-SCREEN INSTRUCTIONS ---
    # Create a small overlay so you know what to do at the end
    overlay = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(overlay, "Playback Finished. PRESS ANY KEY for next, 'q' to quit.", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    final_view = np.vstack((combined, overlay))
    
    cv2.imshow(WIN_NAME, final_view)
    
    # Wait for keypress on the window
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q'):
        cv2.destroyAllWindows()
        return False
        
    return True

if __name__ == "__main__":
    # Path Configuration
    PATHS = {
        "1": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/NewCode/outputs/CASME_II",
        "2": r"C:/Users/Addhyan/Desktop/Thesis/Thesis2/THESIS WORK new/NewCode/outputs/CAS_ME_2"
    }

    print("\n" + "="*40)
    print(" MER TENSOR INSPECTION TOOL ")
    print("="*40)
    print("[1] Inspect CASME II Tensors")
    print("[2] Inspect CAS(ME)^2 Tensors")
    print("[q] Quit")
    
    choice = input("\nSelect dataset to inspect: ").strip().lower()
    
    if choice in PATHS:
        selected_dir = PATHS[choice]
        active = True
        while active:
            active = inspect_random_tensor(selected_dir)
    else:
        print("Exiting...")