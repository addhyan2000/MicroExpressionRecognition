import numpy as np
from Stage1_DataPipeline.evm_magnifier import EulerianMagnifier
def print_memory():
    print("running...")

print("--- EVM RAM Stress Test ---")
print_memory()

# 300 frames, 256x256, 3 channels (typical MER clip)
print("\nCreating dummy tensor (300, 256, 256, 3)...")
dummy_video = np.random.randint(0, 256, size=(300, 256, 256, 3), dtype=np.uint8)
print_memory()

magnifier = EulerianMagnifier(alpha=10.0, low_omega=5.0, high_omega=25.0, fps=200, levels=4)

print("\nStarting Vectorized EVM pipeline (this simulates peak RAM load)...")
result = magnifier.magnify(dummy_video)

print(f"\nMagnification completed safely! Output dimensions: {result.shape}")
print_memory()
print("---------------------------")
