import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_PATH = '/home/ubuntu/multimodal-sentiment-analysis'
CSV_PATH = f'{BASE_PATH}/checkpoints/metrics.csv'
OUTPUT_PATH = f'{BASE_PATH}/checkpoints/training_curves.png'

def main():
    df = pd.read_csv(CSV_PATH)
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_acc"], label="Val Acc", linestyle="--")
    ax2.plot(df["epoch"], df["val_f1"],  label="Val Macro-F1", linestyle="-.")
    ax2.set_ylabel("Score")
    # legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="center right")
    ax1.grid(True, alpha=0.3)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
