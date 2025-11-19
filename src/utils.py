import os
from dotenv import load_dotenv
load_dotenv()

class ProgressTracker:
    def __init__(self): self.reset()
    def reset(self): self.s=0.0; self.n=0
    def update(self, val, k=1): self.s += float(val)*k; self.n += k
    @property
    def avg(self): return self.s / max(1, self.n)

class EnvironmentLoader:

    @staticmethod
    def get_str(key, default_value):
        return os.environ.get(key, default_value)

    @staticmethod
    def get_int(key, default_value):
        return int(os.environ.get(key, default_value))

    @staticmethod
    def get_float(key, default_value):
        return float(os.environ.get(key, default_value))

    @staticmethod
    def get_boolean(key: str, default: bool):
        val = os.environ.get(key, "")
        if val == "":
            return default
        val = val.strip().lower()
        return val in ("1","true","yes","y","on","t")

import pickle
from pathlib import Path
import torch
import torch.nn as nn

class TrainingHelper:

    @staticmethod
    def atomic_torch_save(obj, path: Path):
        tmp = Path(str(path) + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path) 

    @staticmethod
    def safe_load(path: Path, device):
        try:
            print(f"[Resume] Trying: {path.resolve()}")
            return torch.load(path, map_location=device, weights_only=False)
        except (pickle.UnpicklingError, EOFError, RuntimeError, IsADirectoryError) as e:
            print(f"[Resume] Failed to load '{path}': {e}")
            return None
    
    @staticmethod
    def unfreeze_blocks(model, blocks=("layer4",)):

        def _set_bn_eval(m):
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.track_running_stats = False
                for p in m.parameters():
                    p.requires_grad = False

        bb = model.feature_extractor.backbone
        # freeze all
        for p in bb.parameters(): 
            p.requires_grad = False
        # unfreeze selected blocks
        names = dict(bb.named_children())
        for name, child in names.items():
            if name in blocks:
                for p in child.parameters():
                    p.requires_grad = True
            else:
                child.apply(_set_bn_eval)
            
    @staticmethod
    def metrics_logger(base_path):

        import csv
        from torch.utils.tensorboard import SummaryWriter

        metrics_path = base_path / "metrics.csv"
        tb_dir = base_path / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        # create CSV header first time
        if not metrics_path.exists():
            with open(metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_acc", "val_f1", "lr", "epoch_seconds"])
        return writer, metrics_path

    @staticmethod
    def log_metrics(writer, metrics_path, epoch, loss_meter, val_acc, val_f1, scheduler, elapsed):

        import csv 

        writer.add_scalar("Train/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/Acc",   val_acc,        epoch)
        writer.add_scalar("Val/MacroF1", val_f1,       epoch)
        writer.add_scalar("LR",        scheduler.get_last_lr()[0], epoch)

        # append CSV row
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{loss_meter.avg:.6f}", f"{val_acc:.6f}",
                        f"{val_f1:.6f}", f"{scheduler.get_last_lr()[0]:.8f}",
                        f"{elapsed:.2f}"])

    @staticmethod
    def print_trainable_summary(model, tag="[FT]"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{tag} Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    @staticmethod
    def evaluate_detailed(model, loader, device, class_names=None, writer=None, epoch=None, save_path=None):
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch.get("images", batch.get("image"))
                y = batch.get("labels", batch.get("label"))
                if isinstance(x, list):
                    x = torch.stack(x, dim=0)
                logits = model(x.to(device))
                preds.append(logits.argmax(1).cpu())
                labels.append(y.cpu())
        y_pred = torch.cat(preds).numpy()
        y_true = torch.cat(labels).numpy()

        # Text report 
        print("\n[Detailed Validation Report]")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # Confusion matrix 
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(4,4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(cm.shape[0])
        if class_names: plt.xticks(ticks, class_names, rotation=45, ha="right")
        else: plt.xticks(ticks, ticks)
        if class_names: plt.yticks(ticks, class_names)
        else: plt.yticks(ticks, ticks)
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        if writer and epoch is not None:
            writer.add_figure("Val/ConfusionMatrix", fig, global_step=epoch)

        from pathlib import Path
        save_path = "checkpoints/confusion_matrix.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Artifacts] Saved confusion matrix to {save_path}")

        plt.close(fig)


    @staticmethod
    def evaluate_by_dataset(model, loader, device):
        from collections import defaultdict
        import numpy as np
        import torch

        model.eval()
        buckets = defaultdict(lambda: {'preds': [], 'labels': []})

        with torch.no_grad():
            for batch in loader:
                x = batch.get('images', batch.get('image'))
                y = batch.get('labels', batch.get('label'))
                ds = batch.get('dataset', ['unknown'] * len(y))

                if isinstance(x, list):
                    x = torch.stack(x, dim=0)

                logits = model(x.to(device, non_blocking=True))
                p = logits.argmax(1).cpu().numpy()
                y = y.cpu().numpy()

                for name, pi, yi in zip(ds, p, y):
                    buckets[str(name)]['preds'].append(int(pi))
                    buckets[str(name)]['labels'].append(int(yi))

        print("\n[Per-dataset validation metrics]")
        print(f"{'dataset':<18} {'N':>5} {'Acc':>7} {'MacroF1':>8}")
        print("-" * 42)

        for name in sorted(buckets.keys()):
            preds = np.array(buckets[name]['preds'])
            labels = np.array(buckets[name]['labels'])
            n = labels.size

            if n == 0:
                acc = 0.0
                macro_f1 = 0.0
            else:
                acc = float((preds == labels).mean())
                C = int(labels.max() + 1)
                f1s = []
                for c in range(C):
                    tp = np.sum((preds == c) & (labels == c))
                    fp = np.sum((preds == c) & (labels != c))
                    fn = np.sum((preds != c) & (labels == c))
                    prec = tp / (tp + fp + 1e-12)
                    rec  = tp / (tp + fn + 1e-12)
                    f1s.append(2 * prec * rec / (prec + rec + 1e-12))
                macro_f1 = float(np.mean(f1s))

            print(f"{name:<18} {n:>5} {acc:>7.4f} {macro_f1:>8.4f}")

    @staticmethod
    def stratified_data_split(df, train_ratio, val_ratio, test_ratio,
                                          train_tfms, eval_tfms, seed=42):
        
        import numpy as np
        import pandas as pd
        from src.dataset import MultimodalSentimentDataset

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        rng = np.random.RandomState(seed)

        trains, vals, tests = [], [], []
        for name, g in df.groupby('dataset', dropna=False):
            g = g.sample(frac=1.0, random_state=seed)  # shuffle per dataset
            n = len(g)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            trains.append(g.iloc[:n_train])
            vals.append(g.iloc[n_train:n_train+n_val])
            tests.append(g.iloc[n_train+n_val:])

        train_df = pd.concat(trains).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df   = pd.concat(vals).reset_index(drop=True)
        test_df  = pd.concat(tests).reset_index(drop=True)

        print("[Split] Per-dataset counts:")
        for split_name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            print(split_name, "\n", d["dataset"].value_counts(dropna=False), "\n")

        train_ds = MultimodalSentimentDataset(train_df, image_transform=train_tfms,
                                            max_text_length=128, filter_missing=False)
        val_ds   = MultimodalSentimentDataset(val_df,   image_transform=eval_tfms,
                                            max_text_length=128, filter_missing=False)
        test_ds  = MultimodalSentimentDataset(test_df,  image_transform=eval_tfms,
                                            max_text_length=128, filter_missing=False)
        return train_ds, val_ds, test_ds

    @staticmethod
    def stratified_data_split_for_fusion(df, train_ratio, val_ratio, test_ratio,
                            train_tfms, eval_tfms, seed=42):

        import numpy as np
        import pandas as pd
        from sklearn.model_selection import StratifiedShuffleSplit
        from src.dataset import MultimodalSentimentDataset

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        assert {"dataset", "sentiment_label"}.issubset(df.columns), \
            "df must contain 'dataset' and 'sentiment_label' columns"

        df = df.reset_index(drop=True).copy()
        df["__strat_key__"] = df["dataset"].astype(str) + "§" + df["sentiment_label"].astype(str)

        temp_size = val_ratio + test_ratio
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=seed)
        train_idx, temp_idx = next(sss1.split(df, df["__strat_key__"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df  = df.iloc[temp_idx].reset_index(drop=True)

        test_frac_of_temp = test_ratio / max(1e-12, (val_ratio + test_ratio))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac_of_temp, random_state=seed)
        val_idx, test_idx = next(sss2.split(temp_df, temp_df["__strat_key__"]))
        val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        print("[Split] Per-dataset counts:")
        for split_name, d in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            print(split_name, "\n", d["dataset"].value_counts(dropna=False), "\n")

        def _xtab(name, d):
            ct = pd.crosstab(d["dataset"], d["sentiment_label"])
            print(f"[Split] {name} dataset × label:")
            print(ct, "\n")
        _xtab("TRAIN", train_df)
        _xtab("VAL",   val_df)
        _xtab("TEST",  test_df)

        train_ds = MultimodalSentimentDataset(train_df, image_transform=train_tfms,
                                            max_text_length=128, filter_missing=False)
        val_ds   = MultimodalSentimentDataset(val_df,   image_transform=eval_tfms,
                                            max_text_length=128, filter_missing=False)
        test_ds  = MultimodalSentimentDataset(test_df,  image_transform=eval_tfms,
                                            max_text_length=128, filter_missing=False)
        return train_ds, val_ds, test_ds


class EarlyStopper:
    def __init__(self, patience=6, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_epochs = 0
    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            return False 
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience

