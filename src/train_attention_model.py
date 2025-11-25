#!/usr/bin/env python3

import os
import sys
import time
import csv
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataset import MultimodalSentimentDataset
from src.models.cross_modal_attention import TextFeatureExtractor,  ImageFeatureExtractor, AttentionFusionModel
from src.utils import EnvironmentLoader as env
from src.utils import TrainingHelper as train_helper
from src.utils import EarlyStopper

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ProgressTracker:
    def __init__(self): self.reset()
    def reset(self): self.s=0.0; self.n=0
    def update(self, val, k=1): self.s += float(val)*k; self.n += k
    @property
    def avg(self): return self.s / max(1, self.n)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def detailed_eval_and_cm(model, loader, device, class_names, save_png, split_tag, writer=None):
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    preds, labels = [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        logits, *_ = model(images, input_ids, attention_mask)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()

    print(f"\n[{split_tag}] Detailed classification report")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{split_tag} Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(save_png, dpi=200, bbox_inches="tight")
    if writer:
        writer.add_figure(f"{split_tag}/ConfusionMatrix", fig)
    plt.close(fig)
    print(f"[Artifacts] Saved {split_tag.lower()} confusion matrix to {save_png}")

def save_per_dataset_csv(per_ds_dict, out_path: Path):
    import pandas as pd
    df = pd.DataFrame.from_dict(per_ds_dict, orient="index")
    df.index.name = "dataset"
    df.to_csv(out_path)
    print(f"[Artifacts] Saved per-dataset metrics to {out_path}")


from transformers import AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def mm_collate(batch, max_len=128):
    """
    Collate for multimodal: stacks images, tokenizes text, builds labels.
    Expects each sample to have at least: image tensor, label int, and a text field.
    Text field name can vary; we try several common keys and fall back to ''.
    """
    images, texts, labels, datasets = [], [], [], []

    for sample in batch:
        img = sample.get("images", sample.get("image"))
        images.append(img)  # single tensor [C,H,W]

        lab = sample.get("labels", sample.get("label"))
        labels.append(int(lab) if not torch.is_tensor(lab) else lab.item())

        ds = sample.get("dataset", "unknown")
        if isinstance(ds, list): 
            ds = ds[0] if ds else "unknown"
        datasets.append(ds)

        t = (sample.get("text") or sample.get("caption") or
             sample.get("tweet") or sample.get("raw_text") or "")
        if not isinstance(t, str):
            # handle NaN or non-string
            t = "" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)
        texts.append(t)

    images = torch.stack(images, dim=0)  # [B,C,H,W]
    labels = torch.tensor(labels, dtype=torch.long)

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    batch_out = {
        "images": images,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "dataset": datasets,  # keep for per-dataset reporting if you want
    }
    return batch_out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        images = batch.get('images', batch.get('image'))
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        y = batch.get('labels', batch.get('label'))
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, *_ = model(images, input_ids, attention_mask)
        preds.append(logits.argmax(dim=1).cpu())
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


def create_datasets(df: pd.DataFrame,
                    train_ratio: float,
                    val_ratio: float,
                    test_ratio: float,
                    train_tfms,
                    eval_tfms,
                    seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    train_ds = MultimodalSentimentDataset(train_df, image_transform=train_tfms,
                                          max_text_length=128, filter_missing=False)
    val_ds   = MultimodalSentimentDataset(val_df,   image_transform=eval_tfms,
                                          max_text_length=128, filter_missing=False)
    test_ds  = MultimodalSentimentDataset(test_df,  image_transform=eval_tfms,
                                          max_text_length=128, filter_missing=False)
    return train_ds, val_ds, test_ds


def metrics_logger(base_path: Path):
    metrics_path = base_path / "metrics.csv"
    tb_dir = base_path / "tb_fusion"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_acc", "val_f1", "lr", "epoch_seconds"])
    return writer, metrics_path


def log_metrics(writer, metrics_path, epoch, loss_meter, val_acc, val_f1, scheduler, elapsed):
    writer.add_scalar("Train/Loss", loss_meter.avg, epoch)
    writer.add_scalar("Val/Acc",   val_acc,        epoch)
    writer.add_scalar("Val/MacroF1", val_f1,       epoch)
    writer.add_scalar("LR",        scheduler.get_last_lr()[0], epoch)

    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{loss_meter.avg:.6f}", f"{val_acc:.6f}",
                    f"{val_f1:.6f}", f"{scheduler.get_last_lr()[0]:.8f}",
                    f"{elapsed:.2f}"])

from collections import defaultdict
import numpy as np
import torch

@torch.no_grad()
def evaluate_fusion_by_dataset(model, loader, device, print_table=True):
    model.eval()
    buckets = defaultdict(lambda: {"preds": [], "labels": [], "gate_mean": []})
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        ds = batch.get("dataset", ["unknown"] * images.size(0))

        if input_ids is not None:
            input_ids = input_ids.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        logits, gate, _, _ = model(images, input_ids, attention_mask)
        preds = logits.argmax(1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        # gate: [B, D] → scalar “image trust” per sample
        gate_scalar = gate.mean(dim=1).cpu().numpy()

        for name, p, y, g in zip(ds, preds, labels_np, gate_scalar):
            buckets[name]["preds"].append(p)
            buckets[name]["labels"].append(y)
            buckets[name]["gate_mean"].append(float(g))

    if print_table:
        print("\n[Per-dataset metrics + gate]")
        print("dataset                 N     Acc  MacroF1   Gate(img→txt)")
        print("-"*64)

    out = {}
    for name, d in buckets.items():
        preds = np.array(d["preds"])
        labels = np.array(d["labels"])
        acc = float((preds == labels).mean()) if len(labels) else 0.0
        C = int(labels.max() + 1) if len(labels) else 1
        f1s = []
        for c in range(C):
            tp = ((preds==c)&(labels==c)).sum()
            fp = ((preds==c)&(labels!=c)).sum()
            fn = ((preds!=c)&(labels==c)).sum()
            prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
            f1s.append(2*prec*rec/(prec+rec+1e-12))
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        gate_mean = float(np.mean(d["gate_mean"])) if d["gate_mean"] else 0.0
        out[name] = {"acc": acc, "macro_f1": macro_f1, "gate_mean": gate_mean, "n": len(labels)}
        if print_table:
            print(f"{name:22s} {len(labels):5d}  {acc:6.4f}   {macro_f1:6.4f}   {gate_mean:10.4f}")
    return out


def main():
    csv_path       = env.get_str("CSV_PATH", "data/processed/combined_dataset.csv")
    outdir         = env.get_str("OUTDIR", "./checkpoints")
    epochs         = env.get_int("EPOCHS", 20)
    batch_size     = env.get_int("BATCH_SIZE", 8)
    lr             = env.get_float("LR", 2e-4)
    weight_decay   = env.get_float("WEIGHT_DECAY", 1e-4)
    num_workers    = env.get_int("NUM_WORKERS", 2)
    seed           = env.get_int("SEED", 42)
    amp            = env.get_boolean("AMP", True)

    train_split    = env.get_float("TRAIN_SPLIT", 0.7)
    val_split      = env.get_float("VAL_SPLIT", 0.15)
    test_split     = env.get_float("TEST_SPLIT", 0.15)

    d_model        = env.get_int("MODEL_DIM", 256)
    n_heads        = env.get_int("N_HEADS", 4)

    text_model_path   = env.get_str("TEXT_MODEL_PATH", "saved_models/text_model.pt")
    visual_model_path = env.get_str("VISUAL_MODEL_PATH", "saved_models/visual_model_backbone.pt")

    # Visual model args to construct the base class; weights come from saved model
    visual_args = dict(
        num_classes=3,
        feature_dim=512,
        model_name="resnet50",
        freeze_backbone=False,
        use_pretrained=False,
        device="cpu",
    )

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    best_path = outdir_path / "best_fusion_model.ckpt"
    last_path = outdir_path / "last_fusion_model.ckpt"
    writer, metrics_path = metrics_logger(outdir_path)

    # ========= Data =========
    df = pd.read_csv(csv_path)
    num_classes = int(df["sentiment_label"].nunique())
    print(f"Fusion training; classes={num_classes}")

    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds, val_ds, test_ds = train_helper.stratified_data_split_for_fusion(
        df, train_split, val_split, test_split, train_tfms, eval_tfms, seed=seed
    )

    y = train_ds.df['sentiment_label'].to_numpy()
    num_classes = int(y.max() + 1)
    counts = np.bincount(y, minlength=num_classes)
    class_weights = (len(y) / (counts + 1e-12))
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.02,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=mm_collate, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=mm_collate, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=mm_collate, num_workers=num_workers, pin_memory=pin)


    img_tokens = ImageFeatureExtractor(visual_model_path, visual_args, model_dim=d_model)
    txt_tokens = TextFeatureExtractor(text_model_path, model_dim=d_model)
    model = AttentionFusionModel(img_tokens, txt_tokens,
                                 d_model=d_model, n_heads=n_heads,
                                 num_classes=num_classes, dropout=0.1,
                                 pool='mean', gate_type='vector')
    model.to(device)

    # train only unfrozen params
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[FT] Trainable params: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device == "cuda"))

    best_val_f1 = -1.0
    early = EarlyStopper(patience=6, min_delta=1e-4)

    print(f"[Info] Starting the training")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = ProgressTracker()
        start = time.time()

        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for batch in progress_bar:
            images = batch.get('images', batch.get('image'))
            labels = batch.get('labels', batch.get('label'))
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')

            if isinstance(images, list):
                images = torch.stack(images, dim=0)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits, *_ = model(images, input_ids, attention_mask)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, *_ = model(images, input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            loss_meter.update(loss.item(), k=images.size(0))
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}/{epochs} | TrainLoss {loss_meter.avg:.4f} | ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f} | LR {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

        log_metrics(writer, metrics_path, epoch, loss_meter, val_acc, val_f1, scheduler, elapsed)

        # save last checkpoint
        train_helper.atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }, last_path)

        # save best checkpoint
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
            }, best_path)
            print(f"  ↳ Saved new best to: {best_path} (ValF1={val_f1:.4f})")

        if early.step(val_f1):
            print(f"[EarlyStop] No improvement for {early.patience} epochs. Stopping at epoch {epoch}.")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded best fusion from epoch {ckpt['epoch']} (ValF1={ckpt['val_f1']:.4f})")

    class_names = ["negative", "neutral", "positive"]

    val_cm_path  = outdir_path / "confusion_matrix_val.png"
    detailed_eval_and_cm(model, val_loader, device, class_names, val_cm_path, split_tag="Val", writer=writer)

    val_by_ds = evaluate_fusion_by_dataset(model, val_loader, device, print_table=True)
    save_per_dataset_csv(val_by_ds, outdir_path / "per_dataset_val.csv")

    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"[Test] Acc={test_acc:.4f} | Macro-F1={test_f1:.4f}")

    test_cm_path = outdir_path / "confusion_matrix_test.png"
    detailed_eval_and_cm(model, test_loader, device, class_names, test_cm_path, split_tag="Test", writer=writer)

    test_by_ds = evaluate_fusion_by_dataset(model, test_loader, device, print_table=True)
    save_per_dataset_csv(test_by_ds, outdir_path / "per_dataset_test.csv")

    save_dir = Path("saved_models"); save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / "attention_fusion_model.pt"
    torch.save(model.state_dict(), full_path)

    writer.close()



if __name__ == "__main__":
    main()
