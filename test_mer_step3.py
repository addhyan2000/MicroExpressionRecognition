"""Step 3 Integration Test — Classification Head & Joint Loss.

Validates:
1. ClassificationHead output shape and layer-aware initialization.
2. Batch-size-1 safety (LayerNorm).
3. JointLoss with alpha-weighted Focal Loss (the critical code path).
4. No NaN values in loss or any gradient tensors.
5. Backward pass produces valid gradients on features, centers, AND the MLP.
6. Dedicated Center Loss SGD optimizer can successfully step the centroids.
7. No-alpha regression check with pristine gradient state.

Author  : Addhyan Pant
Project : Master's Thesis — Micro-Expression Recognition
"""

import torch
import sys

sys.path.append(r"C:\Users\Addhyan\Desktop\Thesis\THESIS WORK MAIN\Main Code\src")

try:
    from models.components.classification_head import ClassificationHead
    from losses.joint_loss import JointLoss

    print("=" * 60)
    print("  STEP 3 — Classification Head & Joint Loss Tests")
    print("=" * 60)

    # ── Hyperparameters ──────────────────────────────────────────
    batch_size = 4
    input_dim = 128
    hidden_dim = 64
    num_classes = 5

    # ── 1. ClassificationHead ────────────────────────────────────
    print("\n[1] Testing ClassificationHead...")
    head = ClassificationHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    head.train()

    dummy_features = torch.randn(batch_size, input_dim, requires_grad=True)
    logits = head(dummy_features)

    print(f"    Features shape : {dummy_features.shape}")
    print(f"    Logits shape   : {logits.shape}")
    assert logits.shape == (batch_size, num_classes), \
        "Classification head output shape mismatch"

    # Verify layer-aware initialization
    # Hidden layer [0]: Kaiming, bias = 0
    assert head.classifier[0].bias.abs().sum().item() == 0.0, \
        "Hidden Linear bias should be initialized to zero"
    # Logit layer [4]: Normal(0, 0.01), bias = 0
    assert head.classifier[4].bias.abs().sum().item() == 0.0, \
        "Logit Linear bias should be initialized to zero"
    logit_weight_std = head.classifier[4].weight.std().item()
    assert logit_weight_std < 0.05, \
        f"Logit layer weight std too large ({logit_weight_std:.4f}), expected ~0.01"
    print(f"    Hidden bias    : 0 ✓")
    print(f"    Logit std      : {logit_weight_std:.4f} (target ~0.01) ✓")

    print("    [PASS] ClassificationHead\n")

    # ── 2. Batch-size-1 Safety (LayerNorm) ───────────────────────
    print("[2] Testing batch_size=1 safety (LayerNorm)...")
    head.eval()
    single_sample = torch.randn(1, input_dim)
    logits_single = head(single_sample)
    assert logits_single.shape == (1, num_classes), \
        "Batch-size-1 output shape mismatch"
    print(f"    Output shape   : {logits_single.shape}")
    print("    [PASS] No crash on batch_size=1\n")
    head.train()

    # ── 3. JointLoss with alpha weights ──────────────────────────
    print("[3] Testing JointLoss (with alpha-weighted Focal Loss)...")
    labels = torch.randint(0, num_classes, (batch_size,))

    # Create per-class alpha weights — this exercises the critical
    # code path that previously harboured the fatal p_t bug.
    alpha_weights = torch.rand(num_classes)
    print(f"    Alpha weights  : {alpha_weights}")

    criterion = JointLoss(
        num_classes=num_classes,
        feat_dim=input_dim,
        alpha=alpha_weights,
        gamma=2.0,
        lambda_c=0.1,
    )

    # Create the Center Loss optimizer BEFORE backward so we can
    # zero its gradients properly
    optimizer_centloss = torch.optim.SGD(
        criterion.center_loss.parameters(), lr=0.5
    )

    # Fresh forward pass
    dummy_features = torch.randn(batch_size, input_dim, requires_grad=True)
    logits = head(dummy_features)
    loss = criterion(logits, dummy_features, labels)

    print(f"    Labels         : {labels}")
    print(f"    Joint Loss     : {loss.item():.6f}")

    # ── 4. NaN guard ─────────────────────────────────────────────
    print("\n[4] Checking for NaN values...")
    assert not torch.isnan(loss), "FATAL: Loss is NaN!"
    assert torch.isfinite(loss), "FATAL: Loss is not finite!"
    print("    Loss           : finite ✓")

    # ── 5. Backward pass & gradient validation ───────────────────
    print("\n[5] Running backward pass...")

    # Zero ALL gradients before backward to prevent accumulation
    head.zero_grad()
    optimizer_centloss.zero_grad()
    if dummy_features.grad is not None:
        dummy_features.grad.zero_()

    loss.backward()

    # Verify gradients on input features
    assert dummy_features.grad is not None, \
        "Gradients missing on input features"
    assert not torch.isnan(dummy_features.grad).any(), \
        "FATAL: NaN detected in feature gradients!"
    print(f"    Feature grad   : shape {dummy_features.grad.shape}, no NaN ✓")

    # Verify gradients on Center Loss centroids
    assert criterion.center_loss.centers.grad is not None, \
        "Gradients missing on class centers"
    assert not torch.isnan(criterion.center_loss.centers.grad).any(), \
        "FATAL: NaN detected in center gradients!"
    print(f"    Center grad    : shape {criterion.center_loss.centers.grad.shape}, no NaN ✓")

    # Verify gradients flowed through the ClassificationHead MLP
    assert head.classifier[0].weight.grad is not None, \
        "Gradients missing on ClassificationHead hidden layer!"
    assert head.classifier[4].weight.grad is not None, \
        "Gradients missing on ClassificationHead logit layer!"
    print(f"    MLP hidden grad: shape {head.classifier[0].weight.grad.shape} ✓")
    print(f"    MLP logit grad : shape {head.classifier[4].weight.grad.shape} ✓")

    print("    [PASS] Backward pass\n")

    # ── 6. Center Loss Optimizer Step ────────────────────────────
    print("[6] Testing dedicated Center Loss optimizer step...")
    centers_before = criterion.center_loss.centers.data.clone()

    optimizer_centloss.step()

    centers_after = criterion.center_loss.centers.data
    centers_moved = not torch.equal(centers_before, centers_after)

    assert centers_moved, \
        "FATAL: Center Loss optimizer step did not update centroids!"
    print(f"    Centers moved  : {centers_moved} ✓")
    print("    [PASS] Optimizer step\n")

    # ── 7. JointLoss WITHOUT alpha (regression check) ────────────
    print("[7] Testing JointLoss (without alpha, regression check)...")

    # Clear ALL accumulated gradients for pristine test isolation
    head.zero_grad()

    dummy_features_2 = torch.randn(batch_size, input_dim, requires_grad=True)
    logits_2 = head(dummy_features_2)

    criterion_no_alpha = JointLoss(
        num_classes=num_classes,
        feat_dim=input_dim,
        lambda_c=0.1,
    )
    loss_2 = criterion_no_alpha(logits_2, dummy_features_2, labels)
    loss_2.backward()

    assert not torch.isnan(loss_2), "FATAL: Loss (no alpha) is NaN!"
    assert torch.isfinite(loss_2), "FATAL: Loss (no alpha) is not finite!"
    assert dummy_features_2.grad is not None, \
        "Gradients missing on features (no alpha path)"
    assert not torch.isnan(dummy_features_2.grad).any(), \
        "FATAL: NaN in feature gradients (no alpha path)!"
    print(f"    Loss (no alpha): {loss_2.item():.6f}, no NaN ✓")
    print("    [PASS] No-alpha regression\n")

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("  ALL STEP 3 TESTS PASSED ✓")
    print("=" * 60)

except Exception as e:
    import traceback

    print("\n" + "=" * 60)
    print("  TEST FAILURE")
    print("=" * 60)
    traceback.print_exc()
