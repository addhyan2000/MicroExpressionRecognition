import sys
from pathlib import Path

# --- THE PATH FIX ---
# This finds the 'NewCode' folder so Python can see the 'src' folder
_root = Path(__file__).resolve().parents[2] 
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# --------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.stage2_training.trainer import METrainer
from src.stage2_training.model import STSTNet
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def run_sanity_test():
    logger.info("🚀 Starting Trainer Sanity Test (7-Class Mode)...")
    
    num_classes = 7
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ckpt_dir = Path("test_checkpoints")
    
    if test_ckpt_dir.exists():
        shutil.rmtree(test_ckpt_dir)

    logger.info("📦 Generating dummy tensors...")
    # (3 streams, 32 frames, 224x224)
    dummy_train_x = torch.randn(8, 3, 32, 224, 224)
    dummy_train_y = torch.randint(0, num_classes, (8,))
    dummy_val_x = torch.randn(4, 3, 32, 224, 224)
    dummy_val_y = torch.randint(0, num_classes, (4,))

    train_loader = DataLoader(TensorDataset(dummy_train_x, dummy_train_y), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(dummy_val_x, dummy_val_y), batch_size=batch_size)

    logger.info(f"🧠 Initializing STSTNet with {num_classes} output neurons...")
    model = STSTNet(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = METrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=test_ckpt_dir,
        patience=3
    )

    try:
        logger.info("🏃 Running 2-epoch test loop...")
        trainer.train(num_epochs=2)
        
        if (test_ckpt_dir / "last_checkpoint.pth").exists():
            logger.info("✅ SUCCESS: Checkpoint saved successfully.")
            logger.info("🔥 Your trainer is ready for the thesis run!")
        else:
            logger.error("❌ FAIL: Checkpoint not found.")

    except Exception as e:
        logger.error(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_sanity_test()