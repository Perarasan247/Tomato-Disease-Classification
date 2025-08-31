# backend/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import List, Dict

# Keep class order EXACTLY as training
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]

IMG_SIZE = 224


# Same head as in training
class TomatoDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        try:
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class InferenceModel:
    def __init__(self, weights_path: str, device: str = None):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = TomatoDiseaseClassifier(num_classes=len(CLASS_NAMES))
        ckpt = torch.load(weights_path, map_location=self.device)

        # Accept either plain state_dict or the checkpoint dict used in training
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

        self.transform = get_val_transform()

    @torch.inference_mode()
    def predict_image(self, image: Image.Image, topk: int = 5) -> Dict:
        img_t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(img_t)
        probs = F.softmax(logits, dim=1).squeeze(0)

        # top-k
        topk = min(topk, len(CLASS_NAMES))
        top_p, top_i = torch.topk(probs, k=topk)
        top = [
            {"class_idx": int(i), "class_name": CLASS_NAMES[int(i)], "prob": float(p)}
            for p, i in zip(top_p.tolist(), top_i.tolist())
        ]

        # label + healthy flag
        pred_idx = int(torch.argmax(probs).item())
        pred_name = CLASS_NAMES[pred_idx]
        return {
            "pred_idx": pred_idx,
            "pred_class": pred_name,
            "is_healthy": ("healthy" in pred_name.lower()),
            "topk": top,
            "probs": {
                CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))
            },
        }
