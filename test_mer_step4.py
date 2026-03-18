"""Production-Grade Integration Test for MTMNet Step 4 Transfer Learning Suite.

Validates the forward pass, GRL gradient reversal behavior, strict API coupling,
Teacher-Student logic, Dual-Path routing, and absolute numerical stability.
"""

import sys
import os
import copy
import io
import numpy as np
from sklearn.metrics import recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# Add the codebase to Python path so `src.models...` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Main Code')))

import torch
import torch.nn as nn
from src.models.components.spatial_encoder import SpatialEncoder
from src.models.components.temporal_aggregator import SLSTTLSTM
from src.models.components.classification_head import ClassificationHead
from src.models.mtm_net import MTMNet
from src.losses.transfer_loss import MTMTransferLoss

def test_mtm_pipeline():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Testing MTMNet Pipeline on {device} ---")

    # 1. Strict API Validation & Initialization
    num_classes = 5
    embed_dim = 16
    lstm_hidden = 64
    feature_dim = lstm_hidden * 2 # SLSTTLSTM is bidirectional

    encoder = SpatialEncoder(embed_dim=embed_dim).to(device)
    # Aggregator mathematically coupled
    aggregator = SLSTTLSTM(embed_dim=embed_dim, lstm_hidden=lstm_hidden).to(device)
    assert hasattr(aggregator, 'out_features'), "Aggregator missing out_features!"
    
    # Strict out_features coupling
    head = ClassificationHead(input_dim=aggregator.out_features, num_classes=num_classes).to(device)
    
    teacher_net = MTMNet(
        spatial_encoder=encoder,
        temporal_aggregator=aggregator,
        classification_head=head
    ).to(device)
    
    print("\n[1/8] Strict API Validation Passed. MTMNet initialized successfully.")
    
    # 2. Teacher-Student Logic Test
    student_net = MTMNet(
        spatial_encoder=copy.deepcopy(encoder),
        temporal_aggregator=copy.deepcopy(aggregator),
        classification_head=copy.deepcopy(head)
    ).to(device)
    
    # Clone weights
    student_net.clone_from_teacher(teacher_net)
    
    for (k_t, v_t), (k_s, v_s) in zip(teacher_net.state_dict().items(), student_net.state_dict().items()):
        assert torch.equal(v_t, v_s), f"State dict mismatch at {k_t}"
        
    # Freeze as teacher
    teacher_net.freeze_as_teacher()
    for param in teacher_net.parameters():
        assert not param.requires_grad, "Teacher parameter un-frozen!"
    assert not teacher_net.training, "Teacher should be in eval mode!"
    
    print("[2/8] Teacher-Student Logic Test Passed.")
    
    # 3. State Persistence Test
    lambda_val = 0.5
    student_net.update_grl_lambda(lambda_val)
    
    # Mock save/load
    buffer = io.BytesIO()
    torch.save(student_net.state_dict(), buffer)
    buffer.seek(0)
    
    new_student = MTMNet(
        spatial_encoder=copy.deepcopy(encoder),
        temporal_aggregator=copy.deepcopy(aggregator),
        classification_head=copy.deepcopy(head)
    ).to(device)
    new_student.load_state_dict(torch.load(buffer, weights_only=True))
    
    assert 'domain_discriminator.grl.current_lambda' in new_student.state_dict()
    assert new_student.state_dict()['domain_discriminator.grl.current_lambda'].item() == lambda_val, "GRL Lambda not persisted!"
    
    print("[3/8] State Persistence Test Passed.")
    
    # 4. Numerical Stability & Device Check & Dual-Path Routing Validation
    b_micro, b_macro = 4, 6
    frames, c, h, w = 15, 3, 28, 28
    
    micro_videos = torch.randn(b_micro, frames, c, h, w, device=device)
    macro_videos = torch.randn(b_macro, frames, c, h, w, device=device)
    
    # Variable lengths
    micro_lengths = torch.tensor([15, 12, 10, 15], device=device)
    macro_lengths = torch.tensor([15, 14, 11, 15, 13, 10], device=device)
    
    micro_labels = torch.tensor([0, 1, 2, 3], device=device)
    macro_labels = torch.tensor([0, 0, 1, 4, 3, 2], device=device)
    
    student_net.train()
    
    # Unpack features_raw and features_proj directly
    micro_out = student_net(micro_videos, lengths=micro_lengths)
    macro_out = teacher_net(macro_videos, lengths=macro_lengths)
    
    assert 'features_raw' in micro_out and 'features_proj' in micro_out
    
    criterion = MTMTransferLoss(num_classes=num_classes).to(device)
    
    loss_dict = criterion(
        micro_logits=micro_out['class_logits'],
        micro_features_raw=micro_out['features_raw'],
        micro_features_proj=micro_out['features_proj'],
        micro_labels=micro_labels,
        micro_domain_logits=micro_out['domain_logits'],
        macro_logits=macro_out['class_logits'],
        macro_features_raw=macro_out['features_raw'],
        macro_features_proj=macro_out['features_proj'],
        macro_labels=macro_labels,
        macro_domain_logits=macro_out['domain_logits']
    )
    
    total_loss = loss_dict['Total_Loss']
    assert not torch.isnan(total_loss) and not torch.isinf(total_loss), "NaN/Inf in Total Loss!"
    
    # Check that gradients flow appropriately
    total_loss.backward(retain_graph=True)
    
    # Assert zero NaNs in gradients
    for name, param in student_net.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"
            
    student_net.zero_grad()
    print("[4/8] Numerical Stability & Dual-Path Routing Validation Passed.")
    
    # 5. Shared Center Update Verification
    criterion.joint_loss.center_loss.centers.grad = None
    L_macro_train = criterion.joint_loss(macro_out['class_logits'].detach(), macro_out['features_proj'].detach(), macro_labels)
    L_macro_train.backward()
    center_grad = criterion.joint_loss.center_loss.centers.grad
    assert center_grad is not None, "Center Loss grad is None!"
    assert torch.sum(torch.abs(center_grad)).item() > 0.0, "Center Loss did not update!"
    print("[5/9] Shared Center Update Verification Passed.")
    
    # 6. LIR Isolation Verification (CRITICAL)
    student_net.zero_grad()
    teacher_net.zero_grad()
    
    # The "Isolation Proof" (Advanced)
    # Momentarily set requires_grad=True for Teacher's classification_head
    for param in teacher_net.classification_head.parameters():
        param.requires_grad = True
        
    macro_out_teacher = teacher_net(macro_videos, lengths=macro_lengths)
    
    loss_dict_lir = criterion(
        micro_logits=student_net(micro_videos, lengths=micro_lengths)['class_logits'],
        micro_features_raw=student_net(micro_videos, lengths=micro_lengths)['features_raw'],
        micro_features_proj=student_net(micro_videos, lengths=micro_lengths)['features_proj'],
        micro_labels=micro_labels,
        micro_domain_logits=student_net(micro_videos, lengths=micro_lengths)['domain_logits'],
        macro_logits=macro_out_teacher['class_logits'],
        macro_features_raw=macro_out_teacher['features_raw'],
        macro_features_proj=macro_out_teacher['features_proj'],
        macro_labels=macro_labels,
        macro_domain_logits=macro_out_teacher['domain_logits']
    )
    
    # Proof step A: LIR Isolation
    loss_dict_lir['L_LIR'].backward(retain_graph=True)
    for name, param in teacher_net.classification_head.named_parameters():
        if param.grad is not None:
            assert torch.sum(torch.abs(param.grad)).item() == 0.0, f"Teacher Isolation breached! LIR leaked to {name}."
             
    # Proof step B: Classification Flow
    teacher_net.zero_grad()
    loss_dict_lir['L_macro'].backward(retain_graph=True)
    for name, param in teacher_net.classification_head.named_parameters():
        assert param.grad is not None, f"Teacher Classification gradient missing on {name}."
        assert torch.sum(torch.abs(param.grad)).item() > 0.0, f"Teacher Classification gradient is zero on {name}."
        
    # Re-freeze teacher
    for param in teacher_net.classification_head.parameters():
        param.requires_grad = False
        
    student_net.zero_grad()
    teacher_net.zero_grad()
    print("[6/9] LIR Isolation Verification (The Isolation Proof) Passed.")

    # 7. AMP Stability Test
    student_net.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        micro_out_amp = student_net(micro_videos, lengths=micro_lengths)
        macro_out_amp = teacher_net(macro_videos, lengths=macro_lengths)
        
        loss_dict_amp = criterion(
            micro_logits=micro_out_amp['class_logits'],
            micro_features_raw=micro_out_amp['features_raw'],
            micro_features_proj=micro_out_amp['features_proj'],
            micro_labels=micro_labels,
            micro_domain_logits=micro_out_amp['domain_logits'],
            macro_logits=macro_out_amp['class_logits'],
            macro_features_raw=macro_out_amp['features_raw'],
            macro_features_proj=macro_out_amp['features_proj'],
            macro_labels=macro_labels,
            macro_domain_logits=macro_out_amp['domain_logits']
        )
        total_loss_amp = loss_dict_amp['Total_Loss']
        
    scaler.scale(total_loss_amp).backward()
    optimizer_model = torch.optim.AdamW(student_net.parameters(), lr=1e-4)
    scaler.unscale_(optimizer_model)
    
    for name, param in student_net.named_parameters():
        if param.requires_grad and param.grad is not None:
             assert not torch.isnan(param.grad).any(), f"AMP unscale generated NaNs at {name}"
    print("[7/9] AMP Stability Test Passed.")
    
    # 8. Metric Logic Check
    # Simulate an imbalanced batch: 100% accurate on majority class (0, 1), 0% on minority 'Surprise' class (4)
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 1]) # Class 4 is entirely misclassified
    
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    assert per_class_recall[0] == 1.0, "Class 0 recall should be 1.0"
    assert per_class_recall[4] == 0.0, "Class 4 recall should be 0.0"
    assert len(per_class_recall) == 5, "Should have 5 classes"
    
    print("[8/9] Metric Logic Check Passed (Imbalanced Stress-Test).")
    
    # 9. Mathematical Gradient-Sign Test (The 'Senior' Test)
    # GRL Signal Hardening
    student_net.zero_grad()
    
    # Use torch.linspace to ensure a strong, structured non-zero gradient signal
    # proj_features: (b_micro, lstm_hidden*2) = (4, 128)
    structured_proj = torch.linspace(-1.0, 1.0, steps=4*feature_dim, device=device).view(4, feature_dim)
    new_proj = structured_proj.detach().requires_grad_(True)
    
    # Run through domain_discriminator explicitly (Without GRL)
    out_dd = student_net.domain_discriminator.net(new_proj)
    domain_targets_explicit = torch.ones_like(out_dd)
    loss_dd = criterion.bce_loss(out_dd, domain_targets_explicit)
    loss_dd.backward()
    
    # Ensure raw gradient is non-zero
    mu_orig = new_proj.grad.mean().item()
    assert abs(mu_orig) > 1e-8, "Structured signal failed to produce gradient!"
    
    # Run through GRL explicitly
    student_net.zero_grad()
    new_proj_grl = structured_proj.detach().requires_grad_(True)
    grl_out = student_net.domain_discriminator.grl(new_proj_grl)
    out_dd_grl = student_net.domain_discriminator.net(grl_out)
    loss_dd_grl = criterion.bce_loss(out_dd_grl, domain_targets_explicit)
    loss_dd_grl.backward()
    
    mu_rev = new_proj_grl.grad.mean().item()
    expected_reversed_grad = new_proj.grad * (-lambda_val)
    
    # EXPLICIT MEAN SIGN CHECK
    sign_orig = 1 if mu_orig > 0 else -1
    sign_rev = 1 if mu_rev > 0 else -1
    assert sign_orig == -sign_rev, "Mathematical sign reversal failure!"
    
    # Validate against actual GRL reversed grad
    assert torch.allclose(new_proj_grl.grad, expected_reversed_grad, atol=1e-5), "Gradient Reversal Math Failed!"
    
    print("[9/9] Mathematical Gradient-Sign Test (The 'Senior' Test) Passed.")
    
    # 10. Resource Cleanup
    writer = SummaryWriter(log_dir="./test_runs")
    writer.add_scalar("Test/Passed", 1, 1)
    writer.close()
    
    torch.cuda.empty_cache()
    
    print("\n✅ All Executive-Level Hardening MTMNet Tests Passed Successfully! VRAM Cleared.\n")

if __name__ == "__main__":
    test_mtm_pipeline()
