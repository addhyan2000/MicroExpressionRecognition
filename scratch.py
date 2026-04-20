import re

with open('Stage3_Training/main_stage3.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Add f1_score import
text = text.replace('from torch.utils.data import DataLoader, Subset\n', 'from torch.utils.data import DataLoader, Subset\nfrom sklearn.metrics import f1_score\n')

# Find the start of the Dataset section
start_marker = '    # ─────────────────────────────────────────────────────────────────\n    # 1. Dataset\n    # ─────────────────────────────────────────────────────────────────'
start_idx = text.find(start_marker)

# Find the end of main
end_marker = 'if __name__ == "__main__":'
end_idx = text.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_logic = '''    # ─────────────────────────────────────────────────────────────────
    # 1. Dataset Initialize
    # ─────────────────────────────────────────────────────────────────
    log.info("\\n[1/7] Initializing Dataset...")

    full_train_dataset = MERDataset(
        csv_path=Path(args.csv_path),
        tensor_dir=Path(args.tensor_dir),
        expression_filter=args.expression_filter,
        transform=True,
        log_dir=log_dir,
    )
    
    full_val_dataset = MERDataset(
        csv_path=Path(args.csv_path),
        tensor_dir=Path(args.tensor_dir),
        expression_filter=args.expression_filter,
        transform=False,
        log_dir=log_dir,
    )

    if len(full_train_dataset) == 0:
        log.error("Dataset is empty! Check CSV path and tensor directory.")
        sys.exit(1)

    log.info("Full dataset: %d samples, %d classes, %d subjects",
             len(full_train_dataset), full_train_dataset.num_classes, full_train_dataset.num_subjects)

    # ─────────────────────────────────────────────────────────────────
    # 2. Leave-One-Subject-Out (LOSO) Cross-Validation
    # ─────────────────────────────────────────────────────────────────
    unique_subjects = full_train_dataset.get_unique_subject_ids()
    log.info("\\n[2/7] Starting Leave-One-Subject-Out Cross-Validation for %d subjects...", len(unique_subjects))

    loso_accuracies = []
    loso_f1_scores = []
    
    base_checkpoint_dir = Path(args.checkpoint_dir)

    for fold_idx, val_subject_id in enumerate(unique_subjects, 1):
        log.info("\\n" + "═" * 70)
        log.info("  FOLD %d/%d — Holding out Subject: %d", fold_idx, len(unique_subjects), val_subject_id)
        log.info("═" * 70)

        # ── Split datasets for this fold ────────────────────────────────
        val_indices = full_val_dataset.get_indices_for_subjects([val_subject_id])
        train_indices = [i for i in range(len(full_train_dataset)) if i not in val_indices]
        
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)

        log.info("Train split: %d samples", len(train_dataset))
        log.info("Val split  : %d samples", len(val_dataset))
        
        # ── Dynamic Class Weights for Trainer ───────────────────────────
        counts = torch.zeros(args.num_emotions, dtype=torch.float32)
        for i in train_indices:
            counts[full_train_dataset.samples[i]["emotion_label"]] += 1.0
        total = counts.sum()
        class_weights = total / (args.num_emotions * counts.clamp(min=1.0))
        log.info("Fold Class weights (inverse freq): %s", [f"{w:.4f}" for w in class_weights.tolist()])
        
        # ── DataLoaders ─────────────────────────────────────────────────
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        # ─────────────────────────────────────────────────────────────────
        # 3. Model
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[3/7] Instantiating AdversarialMERWrapper (with Projection Head)...")
        model = AdversarialMERWrapper(
            num_emotions=args.num_emotions,
            num_subjects=full_train_dataset.num_subjects,
            grl_lambda=args.grl_lambda,
            feature_dim=args.feature_dim,
            proj_dim=args.proj_dim,
            head_dropout=args.head_dropout,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # 4. Loss Functions
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[4/7] Configuring Loss Functions...")
        supcon_criterion = SupConLoss(
            temperature=args.supcon_temperature, memory_size=args.xbm_memory_size, proj_dim=args.proj_dim,
        )
        emotion_criterion = FocalLoss(
            alpha=class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        )
        identity_criterion = nn.CrossEntropyLoss()

        # ─────────────────────────────────────────────────────────────────
        # 5. Optimizer & Scheduler
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[5/7] Configuring Optimizer & Scheduler...")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7,
        )

        # ─────────────────────────────────────────────────────────────────
        # 6. Trainer
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[6/7] Creating AdversarialTrainer...")
        clip_norm = args.gradient_clip_norm if args.gradient_clip_norm > 0 else None
        
        fold_dir = base_checkpoint_dir / f"fold_{val_subject_id}"

        trainer = AdversarialTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            emotion_criterion=emotion_criterion,
            identity_criterion=identity_criterion,
            supcon_criterion=supcon_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            identity_loss_weight=args.identity_loss_weight,
            supcon_loss_weight=args.supcon_weight,
            grl_gamma=args.grl_gamma,
            checkpoint_dir=fold_dir,
            log_dir=log_dir,
            use_amp=args.use_amp,
            gradient_clip_norm=clip_norm,
        )

        # ─────────────────────────────────────────────────────────────────
        # 7. Launch Training
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[7/7] Launching Training...")
        history = trainer.train()
        
        # ─────────────────────────────────────────────────────────────────
        # 8. Fold Evaluation (Load Best Model)
        # ─────────────────────────────────────────────────────────────────
        log.info("\\n[8/8] Evaluating Best Model for Fold...")
        best_ckpt_path = fold_dir / "best_model.pth"
        if best_ckpt_path.exists():
            checkpoint = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info("  Loaded best model from epoch %d.", checkpoint.get("epoch", 0))
        else:
            log.warning("  Could not find best_model.pth. Evaluating with current weights.")
            
        model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for tensors, labels, _ in val_loader:
                tensors = tensors.to(device)
                _, logits, _ = model(tensors)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        if len(all_labels) > 0:
            fold_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            fold_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            fold_acc = 0.0
            fold_f1 = 0.0
            
        loso_accuracies.append(fold_acc)
        loso_f1_scores.append(fold_f1)
        
        log.info("Fold %d Results:", fold_idx)
        log.info("  Validation Accuracy: %.4f", fold_acc)
        log.info("  Validation F1-Score: %.4f", fold_f1)

    # ─────────────────────────────────────────────────────────────────
    # Post-Training Summary
    # ─────────────────────────────────────────────────────────────────
    log.info("\\n" + "═" * 70)
    log.info("  GRAND LOSO SUMMARY")
    log.info("═" * 70)
    
    grand_acc = sum(loso_accuracies) / max(1, len(loso_accuracies))
    mean_f1 = sum(loso_f1_scores) / max(1, len(loso_f1_scores))
    
    log.info("Mean LOSO Accuracy: %.4f", grand_acc)
    log.info("Mean F1-Score     : %.4f", mean_f1)

    log.info("\\n" + "═" * 70)
    log.info("  STAGE 3 COMPLETE")
    log.info("═" * 70)

\n\n'''
    
    text = text[:start_idx] + new_logic + text[end_idx:]
    with open('Stage3_Training/main_stage3.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Done replacing.')
else:
    print('Indices not found.')
