import torch
import torch.nn as nn
from src.models.visual_model import VisualSentimentModel


import torch
import torch.nn as nn
from src.models.visual_model import VisualSentimentModel

class ImageFeatureExtractor(nn.Module):
    """
    Adapter to convert CNN features of Image into normalized token sequence
    """
    def __init__(self, model_path, model_args, model_dim=256):
        super().__init__()

        # Build and load the trained image model
        base_model = VisualSentimentModel(**model_args)
        self._load_trained_model(base_model, model_path)
        self._freeze(base_model)
        self.base_model = base_model.eval()

        # Drop avgpool to keep spatial tokens as a feature map [B,C,H,W]
        backbone = self.base_model.feature_extractor.backbone
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # 3) Find the output channel count C
        with torch.no_grad():
            dummy_forward = torch.zeros(1, 3, 224, 224) 
            fmap = self.backbone(dummy_forward) # [1, C, H, W]
            C = int(fmap.shape[1])
        self.proj = nn.Conv2d(C, model_dim, kernel_size=1)
        self.ln = nn.LayerNorm(model_dim)

    @staticmethod
    def _load_trained_model(model, model_path):
        print(f"[ImageFeatureExtractor] Loading state_dict from {model_path}")
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ImageFeatureExtractor] load (missing={len(missing)}, unexpected={len(unexpected)})")

    @staticmethod
    def _freeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        # images is in form [B,3,H,W]
        fmap = self.backbone(images)                 # Convert to [B, C, H, W]
        z = self.proj(fmap)                          # [B, model_dimension, H, W]
        tokens = z.flatten(2).transpose(1, 2)        # [B, L, model_dimension]
        return self.ln(tokens)



import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.models.text_model import TextualSentimentModel

class TextFeatureExtractor(nn.Module):
    def __init__(self, model_path, model_dim=256):
        
        super().__init__()
        base_model = TextualSentimentModel()  # roberta-base + LSTM(hidden=256)
        self._load_trained_model(base_model, model_path)
        self._freeze(base_model)
        base_model.eval()

        self.roberta = base_model.roberta # HF backbone
        self.lstm = base_model.lstm # hidden size = 256

        # Since LSTM outputs have hidden_size=256
        input_dim = self.lstm.hidden_size
        self.proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.ln = nn.LayerNorm(model_dim)

    @staticmethod
    def _load_trained_model(model: nn.Module, model_path: str):
        print(f"[TextFeatureExtractor] Loading weights from {model_path}")
        raw = torch.load(model_path, map_location="cpu")
        # Text model was saved via DataParallel → strip "module." if present
        cleaned = {}
        for k, v in raw.items():
            cleaned[k[7:]] = v if k.startswith("module.") else v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[TextFeatureExtractor] strict=False load "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    @staticmethod
    def _freeze(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ro = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        tokens = ro.last_hidden_state                     # [B, T, 768]
        lstm_out, _ = self.lstm(tokens)              # [B, T, 256]
        return self.ln(self.proj(lstm_out))               # [B, T, d_model]


import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_mask=None):
        # kv_mask: [B, L_kv] with 1=keep, 0=pad → convert to key_padding_mask (True=ignore)
        key_padding_mask = (~kv_mask.bool()) if kv_mask is not None else None
        y, _ = self.attention(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False) # y = change proposed by attention model found after looked at q and kv
        x = self.ln1(q + y) # add the new change to existing information and then normalize
        z = self.ffn(x) # z = change proposed by the feed-forward network
        return self.ln2(x + z) # add z and normalize


class GatedFusionClassifier(nn.Module):
    def __init__(self, d_model=256, num_classes=3, gate_type='vector', dropout=0.2):
        super().__init__()
        self.gate_type = gate_type
        gate_dim = d_model if gate_type == 'vector' else 1
        # Build the gate from both modalities and then concatenate [z_img, z_text] to look at both to decide mix
        self.gated_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gate_dim),
            nn.Sigmoid(),
        )
        # Classifier head
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, z_img, z_txt):
        g = self.gated_fusion(torch.cat([z_img, z_txt], dim=-1))  # learn gate g from image and text
        if g.shape[-1] == 1:
            g = g.expand_as(z_img)
        z = g * z_img + (1.0 - g) * z_txt  # z is the blend that favor the cleaner/stronger modality
        return self.cls(z), g # return class score and gate


class AttentionFusionModel(nn.Module):
    def __init__(self, img_feature_extractor, text_feature_extractor, d_model=256, n_heads=4, num_classes=3,
                 dropout=0.1, pool='mean', gate_type='vector'):
        super().__init__()

        self.img_feature_extractor = img_feature_extractor
        self.text_feature_extractor = text_feature_extractor

        self.text_to_image_attention = CrossModalAttention(d_model, n_heads, dropout)  # text to image
        self.image_to_text_attention = CrossModalAttention(d_model, n_heads, dropout)  # image to text

        self.pool = pool
        self.gated_fusion_classifier = GatedFusionClassifier(d_model=d_model, num_classes=num_classes,
                                            gate_type=gate_type, dropout=dropout)

    @staticmethod
    def _masked_mean(x, mask):
        m = mask.float().unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)

    def _pool(self, x, mask=None, is_text=False):
        # x: [B, L, D]
        if self.pool == 'cls' and is_text:
            return x[:, 0, :]
        if mask is not None:
            return self._masked_mean(x, mask)
        return x.mean(dim=1)

    def forward(self, images, input_ids, attention_mask):

        # Extract the image features [B, L_v, D]
        v = self.img_feature_extractor.forward_tokens(images)  
        # Extract the text features  [B, L_t, D]
        t = self.text_feature_extractor.forward_tokens(input_ids, attention_mask)

        # cross-modal attention
        text_guided_image = self.text_to_image_attention(q=v, kv=t, kv_mask=attention_mask)
        image_guided_text = self.image_to_text_attention(q=t, kv=v, kv_mask=None)

        # compress to single vector for [B, D]
        image_vector = self._pool(text_guided_image)
        text_vector = self._pool(image_guided_text, mask=attention_mask, is_text=True)

        # gated fusion classifier
        logits, gate = self.gated_fusion_classifier(image_vector, text_vector)
        return logits, gate, image_vector, text_vector