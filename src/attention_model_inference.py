#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import torch.nn.functional as F

from src.models.cross_modal_attention import (
    ImageFeatureExtractor, TextFeatureExtractor, AttentionFusionModel
)

D_MODEL = 256
N_HEADS = 4
NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]

BEST_FUSION_MODEL = Path("checkpoints/best_fusion_model.ckpt")
BEST_TEXT_MODEL   = Path("saved_models/text_model.pt")
BEST_VISUAL_MODEL    = Path("saved_models/visual_model_backbone.pt")

VISUAL_MODEL_ARGS = dict(
    num_classes=3,
    feature_dim=512,
    model_name="resnet50",
    freeze_backbone=False,
    use_pretrained=False,
    device="cpu",
)

# Image eval transforms
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")


def load_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    # Build frozen token extractors
    img_tokens = ImageFeatureExtractor(str(BEST_VISUAL_MODEL), VISUAL_MODEL_ARGS, model_dim=D_MODEL)
    txt_tokens = TextFeatureExtractor(str(BEST_TEXT_MODEL), model_dim=D_MODEL)

    model = AttentionFusionModel(
        img_tokens, txt_tokens,
        d_model=D_MODEL, n_heads=N_HEADS, num_classes=NUM_CLASSES,
        dropout=0.1, pool='mean', gate_type='vector'
    ).to(device)

    # Load best attention fusion model checkpoint
    ckpt = torch.load(BEST_FUSION_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, device


@torch.no_grad()
def predict(model, device, image_path: str, text: str):
    # Preprocess image
    img = Image.open(image_path).convert("RGB")
    img_t = IMAGE_TRANSFORMS(img).unsqueeze(0).to(device)

    # Preprocess text
    enc = TOKENIZER([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Forward
    logits, gate, z_img, z_txt = model(img_t, input_ids, attention_mask)
    probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    gate_mean = gate.mean().item()

    # Package result
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "class_probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "image_vs_text_trust": float(gate_mean)
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--text",  required=True, help="Associated text/caption/tweet")
    args = parser.parse_args()

    model, device = load_model()
    out = predict(model, device, args.image, args.text)
    print(json.dumps(out, indent=2))
