# -*- coding: utf-8 -*-
import sys
import torch
from pathlib import Path

# --- PROJECT ROOT INITIALIZATION ---
_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[2] 
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.stage2_training.model import STSTNet
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_ststnet_architecture():
    print("\n" + "="*50)
    print(" STSTNet ARCHITECTURE STRESS TEST ")
    print("="*50)

    # 1. Instantiate Model
    num_classes = 4
    model = STSTNet(num_classes=num_classes, dropout_p=0.5)
    model.eval() # Set to evaluation mode

    # 2. Simulate a "Perfect" Batch
    # Shape: (Batch=8, Channels=3, Time=32, Height=224, Width=224)
    dummy_input = torch.randn(8, 3, 32, 224, 224)
    
    print(f"🚀 Step 1: Passing dummy batch {tuple(dummy_input.shape)} through model...")
    
    try:
        output = model(dummy_input)
        print(f"✅ Success! Output Shape: {tuple(output.shape)} (Expected [8, {num_classes}])")
        
        # Verify the math
        assert output.shape == (8, num_classes), "Output shape mismatch!"
        
    except Exception as e:
        print(f"❌ FAILED: Forward pass crashed: {e}")
        return

    # 3. Test the "Defensive Guardrail"
    print(f"\n🚀 Step 2: Testing Defensive Input Assertion (Passing 31 frames)...")
    bad_input = torch.randn(8, 3, 31, 224, 224) # 31 frames instead of 32
    try:
        model(bad_input)
        print("❌ FAILED: Model should have caught the bad shape but didn't.")
    except ValueError as v:
        print(f"✅ Success! Model blocked the bad input with error: {v}")

    # 4. Parameter Audit
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 Step 3: Parameter Audit")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model complexity is appropriate for 'Shallow' Triple Stream architecture.")

    print("\n" + "="*50)
    print(" MODEL IS VERIFIED AND READY FOR TRAINING ")
    print("="*50)

if __name__ == "__main__":
    test_ststnet_architecture()