
import os
import sys
import time
import csv
import random
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset import MultimodalSentimentDataset
from src.models.visual_model import VisualSentimentModel
from src.utils import EnvironmentLoader as env
from src.utils import TrainingHelper as train_helper
from src.utils import EarlyStopper, ProgressTracker

from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        images = batch.get('images', batch.get('image'))
        y = batch.get('labels', batch.get('label'))
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(images)
        p = logits.argmax(dim=1)
        preds.append(p.cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc = (preds == labels).mean()
    num_classes = int(labels.max() + 1) if labels.size > 0 else 1
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return float(acc), float(macro_f1)

def main():

    # Step 1: Environment Set up
    csv_path      = env.get_str("CSV_PATH", "")
    epochs        = env.get_int("EPOCHS", 5)
    batch_size    = env.get_int("BATCH_SIZE", 5)
    lr            = env.get_float("LR", 3e-4)
    weight_decay  = env.get_float("WEIGHT_DECAY", 1e-4)
    model_name    = env.get_str("MODEL_NAME", "resnet50")
    feature_dim   = env.get_int("FEATURE_DIM", 512)
    freeze_backbone     = env.get_boolean("FREEZE_BACKBONE", False)
    unfreeze_backbone   = env.get_boolean("UNFREEZE_BACKBONE", False)
    use_pretrained= env.get_boolean("USE_PRETRAINED", False)
    num_workers   = env.get_int("NUM_WORKERS", 1)
    seed          = env.get_int("SEED", 42)
    outdir        = env.get_str("OUTDIR", "./checkpoints")
    train_split   = env.get_float("TRAIN_SPLIT", 0.7)
    val_split     = env.get_float("VAL_SPLIT", 0.15)
    test_split    = env.get_float("TEST_SPLIT", 0.15)
    amp           = env.get_boolean("AMP", False)
    resume        = env.get_boolean("RESUME", False)
    resume_path   = env.get_str("RESUME_PATH", "")

    if unfreeze_backbone:
        freeze_backbone = False

    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    best_path = outdir_path / "best_visual_model.ckpt"
    last_path = outdir_path / "last.ckpt"
    last_checkpoint_path = Path(resume_path) if resume_path else last_path
    writer, metrics_path = train_helper.metrics_logger(outdir_path)

    # Step 2: Load data from CSV
    df = pd.read_csv(csv_path)
    num_classes = int(df["sentiment_label"].nunique())
    print(f"Image-Only Sentiment training; classes={num_classes}")
    print(df[['dataset','sentiment_label']].groupby('dataset').count().rename(columns={'sentiment_label':'n'}))


    # Step 3: Transform and Split the data
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print(f"[Info] Data Transformations are completed")
    train_ds, val_ds, test_ds = train_helper.stratified_data_split(
    df, train_split, val_split, test_split, train_tfms, eval_tfms, seed=seed
    )

    
    y = train_ds.df['sentiment_label'].to_numpy()
    num_classes = int(y.max() + 1)
    counts = np.bincount(y, minlength=num_classes)
    class_weights = (len(y) / (counts + 1e-12))  # inverse freq
    class_weights = class_weights / class_weights.mean()
    # Label Smoothing
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.05)

    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    print(f"[Info] Training | Validation | Test Data Splits are ready")

    # Step 4: Get the Image-only Sentiment Classifier Model
    model = VisualSentimentModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        use_pretrained=use_pretrained,
        device=device,
    )

    if not freeze_backbone:
        # phase 2 training
        # train_helper.unfreeze_blocks(model, blocks=("layer4",))
        # phase-3 (next)
        train_helper.unfreeze_blocks(model, blocks=("layer3","layer4"))
    train_helper.print_trainable_summary(model)

    # HEAD = classifier + feature_projection
    head_params = list(model.classifier.parameters()) \
           + list(model.feature_extractor.feature_projection.parameters())
    # BACKBONE
    backbone_params = [p for p in model.feature_extractor.backbone.parameters() if p.requires_grad]
    # HEAD to learn faster than BACKBONE
    optimizer = torch.optim.AdamW(
    [
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr * 0.1},
    ],
    weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == "cuda"))

    # Enable resuming training from a checkpoint
    start_epoch = 0
    best_val_f1 = -1.0
    if resume and last_checkpoint_path.exists():
        ckpt = train_helper.safe_load(last_checkpoint_path, device)
        if ckpt is None and resume_path != best_path and best_path.exists():
            print("[Resume] Falling back to best checkpoint...")
            ckpt = train_helper.safe_load(best_path, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_f1 = float(ckpt.get("val_f1", best_val_f1))
            remaining_epochs = max(1, epochs - start_epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            print(f"[Resume] Resumed @ epoch {start_epoch} (best ValF1={best_val_f1:.4f})")
        else:
            print("[Resume] Could not load any checkpoint; starting fresh.")

    # Early stopping
    early = EarlyStopper(patience=6, min_delta=1e-4)

    print(f"[Info] Starting the training")
    # 7) Step 5: Train the model
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        loss_meter = ProgressTracker()
        start = time.time()

        progress_bar = tqdm(train_loader, total=len(train_loader),desc=f"Epoch {epoch}/{epochs}", ncols=100)

        for batch in progress_bar:
            images = batch.get('images', batch.get('image'))
            labels = batch.get('labels', batch.get('label'))
            if isinstance(images, list):
                images = torch.stack(images, dim=0)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to stabilize FT
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to stabilize FT
                optimizer.step()

            loss_meter.update(loss.item(), k=images.size(0))
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}",lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False, ncols=100):
                images = batch.get('images', batch.get('image'))
                labels = batch.get('labels', batch.get('label'))
                if isinstance(images, list):
                    images = torch.stack(images, dim=0)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                val_preds.append(logits.argmax(1).cpu())
                val_labels.append(labels.cpu())

        vp = torch.cat(val_preds).numpy()
        vl = torch.cat(val_labels).numpy()
        val_acc = (vp == vl).mean()
        # macro-F1
        C = int(vl.max() + 1) if vl.size > 0 else 1
        f1s = []
        for c in range(C):
            tp = np.sum((vp == c) & (vl == c))
            fp = np.sum((vp == c) & (vl != c))
            fn = np.sum((vp != c) & (vl == c))
            prec = tp / (tp + fp + 1e-12)
            rec  = tp / (tp + fn + 1e-12)
            f1s.append(2*prec*rec/(prec+rec+1e-12))
        val_f1 = float(np.mean(f1s))

        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}/{epochs} | "
            f"TrainLoss {loss_meter.avg:.4f} | ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        train_helper.log_metrics(writer, metrics_path, epoch, loss_meter, val_acc, val_f1, scheduler, elapsed)

        train_helper.atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "config": {
                "csv_path": csv_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "model_name": model_name,
                "feature_dim": feature_dim,
                "freeze_backbone": freeze_backbone,
                "use_pretrained": use_pretrained,
                "num_workers": num_workers,
                "seed": seed,
                "outdir": str(outdir_path),
                "splits": [train_split, val_split, test_split],
                "amp": amp,
            }
        }, last_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            train_helper.atomic_torch_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": {
                "csv_path": csv_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "model_name": model_name,
                "feature_dim": feature_dim,
                "freeze_backbone": freeze_backbone,
                "use_pretrained": use_pretrained,
                "num_workers": num_workers,
                "seed": seed,
                "outdir": str(outdir_path),
                "splits": [train_split, val_split, test_split],
                "amp": amp,
            }
            }, best_path)
            print(f"Saved new best to: {best_path} (ValF1={val_f1:.4f})")
        
        if early.step(val_f1):
            print(f"[EarlyStop] No improvement for {early.patience} epochs. Stopping at epoch {epoch}.")
            break

        
    # Step 6: Evaluate the model
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded best checkpoint from epoch {ckpt['epoch']} (ValF1={ckpt['val_f1']:.4f})")

    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"[Test] Acc={test_acc:.4f} | Macro-F1={test_f1:.4f}")

    class_names = ["negative","neutral","positive"]  # adjust if different order
    train_helper.evaluate_detailed(model, val_loader, device, class_names, writer, epoch=None)
    train_helper.evaluate_by_dataset(model, val_loader, device)
    writer.close()

    # Step 7: Save the model
    torch.save(model.feature_extractor.backbone.state_dict(), "saved_models/visual_model_backbone.pt")

if __name__ == "__main__":
    main()
