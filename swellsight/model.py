import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SwellSightNet(nn.Module):
    def __init__(self, num_wave_types: int, num_directions: int, dropout: float = 0.35):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.shared = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
        )

        self.head_height = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self.head_wave_type = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_wave_types)
        )

        self.head_direction = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_directions)
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.shared(feats)
        return self.head_height(feats), self.head_wave_type(feats), self.head_direction(feats)
