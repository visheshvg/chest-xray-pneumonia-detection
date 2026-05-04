"""Training loop with class weighting, AMP, and two-phase fine-tuning."""
import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config


def _run_one_epoch(model, loader, criterion, optimizer, scaler, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    with torch.set_grad_enabled(train):
        for imgs, labels in pbar:
            imgs   = imgs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                logits = model(imgs).squeeze(-1)
                loss   = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


def _train_n_epochs(model, train_loader, val_loader, criterion, optimizer,
                    epochs, model_name, history):
    scaler    = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2,
                                   patience=3, min_lr=1e-7)
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5
    weights_path = os.path.join(config.WEIGHTS_DIR, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, train=True)
        val_loss,   val_acc   = _run_one_epoch(model, val_loader,   criterion,
                                                optimizer, scaler, train=False)
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(torch.load(weights_path))
    return model, history


def train_model(model, model_name, train_loader, val_loader, pos_weight,
                two_phase=False, unfreeze_fn=None):
    """Single-phase for the custom CNN, two-phase for pretrained models."""
    model = model.to(config.DEVICE)
    pos_weight = pos_weight.to(config.DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history    = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    if two_phase:
        # --- Phase 1: head only ---
        print(f"\n--- {model_name}: PHASE 1 (frozen backbone) ---")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=config.LEARNING_RATE)
        model, history = _train_n_epochs(model, train_loader, val_loader,
                                          criterion, optimizer,
                                          config.EPOCHS_PHASE1, model_name, history)

        # --- Phase 2: unfreeze top layers, lower LR ---
        print(f"\n--- {model_name}: PHASE 2 (fine-tuning) ---")
        model = unfreeze_fn(model, model_name)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=config.FINETUNE_LR)
        model, history = _train_n_epochs(model, train_loader, val_loader,
                                          criterion, optimizer,
                                          config.EPOCHS_PHASE2, model_name, history)
    else:
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        model, history = _train_n_epochs(model, train_loader, val_loader,
                                          criterion, optimizer,
                                          config.EPOCHS_CUSTOM, model_name, history)

    # Save history JSON for the report
    hist_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: list(map(float, v)) for k, v in history.items()}, f, indent=2)

    return model, history
