"""
Multimodal pain-detection model.

Architecture
------------
Rama 1  — VisualEncoder  : EfficientNet-B0 (ImageNet) por frame → 1280-D.
Rama 2  — LandmarkEncoder: two-layer MLP 1434-D → 256-D per frame.
Fusion  — Late fusion at frame level: concat visual + landmark.
Temporal — LSTM (2 layers, 512 hidden) sobre la secuencia de T frames → h_n[-1] → 512-D.
Head    — Dense → BN → GELU → Dropout → Dense → BN → GELU → Dropout → Linear(3)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src import config


# ─── Sub-modules ──────────────────────────────────────────────────────────────

class VisualEncoder(nn.Module):
    """
    Backbone visual: EfficientNet-B0 preentrenado en ImageNet → 1280-D.

    Input : (N, 3, 224, 224)  – frames normalizados
    Output: (N, output_dim)
    """

    def __init__(self, freeze: bool = False):
        super().__init__()
        base = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.pool     = base.avgpool
        self._output_dim = config.VISUAL_DIM  # 1280

        if freeze:
            for p in self.features.parameters():
                p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)       # (N, 1280)


class LandmarkEncoder(nn.Module):
    """
    Two-layer MLP that encodes the flat landmark vector.

    Input : (N, landmark_dim)   e.g. 1434
    Output: (N, output_dim)     e.g. 256
    """

    def __init__(
        self,
        input_dim: int = config.LANDMARK_DIM,
        hidden_dim: int = config.LANDMARK_HIDDEN_DIM,
        output_dim: int = config.LANDMARK_OUT_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Full model ───────────────────────────────────────────────────────────────

class PainDetectionModel(nn.Module):
    """
    End-to-end multimodal pain classifier.

    Parameters
    ----------
    num_frames      : number of frames per video clip (T)
    num_classes     : number of pain levels (3)
    landmark_dim    : flattened landmark feature size (1434 with refine=True)
    landmark_hidden : hidden dim of the landmark MLP
    landmark_out    : output dim of the landmark MLP
    dropout         : dropout rate for the classifier head
    freeze_backbone : if True, freeze EfficientNet weights (Phase-1)
    use_landmarks   : include landmark branch

    Forward inputs
    --------------
    frames    : (B, T, 3, 224, 224)   – normalised RGB clips
    landmarks : (B, T, landmark_dim)  – normalised landmark coords

    Forward output
    --------------
    logits : (B, num_classes)
    """

    def __init__(
        self,
        num_frames: int = config.NUM_FRAMES,
        num_classes: int = config.NUM_CLASSES,
        landmark_dim: int = config.LANDMARK_DIM,
        landmark_hidden: int = config.LANDMARK_HIDDEN_DIM,
        landmark_out: int = config.LANDMARK_OUT_DIM,
        dropout: float = config.DROPOUT,
        lstm_hidden: int = config.LSTM_HIDDEN_DIM,
        lstm_layers: int = config.LSTM_NUM_LAYERS,
        freeze_backbone: bool = False,
        use_landmarks: bool = config.USE_LANDMARKS,
    ):
        super().__init__()
        self.num_frames    = num_frames
        self.num_classes   = num_classes
        self.use_landmarks = use_landmarks

        self.visual_encoder = VisualEncoder(freeze=freeze_backbone)

        if use_landmarks:
            self.landmark_encoder = LandmarkEncoder(
                landmark_dim, landmark_hidden, landmark_out, dropout
            )

        fused_dim = (
            self.visual_encoder.output_dim
            + (landmark_out if use_landmarks else 0)
        )

        self.temporal = nn.LSTM(
            input_size=fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------
    def _fuse_features(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract and concatenate per-frame features from all active branches.

        frames can be either:
          (B, T, 3, H, W)    — raw frames, passed through the visual encoder
          (B, T, visual_dim) — pre-computed features (Phase-1 cache), encoder skipped

        Returns fused : (B, T, fused_dim)
        """
        B, T = frames.shape[:2]

        if frames.ndim == 5:
            vis = self.visual_encoder(frames.view(B * T, *frames.shape[2:]))
        else:
            vis = frames.view(B * T, -1)

        parts = [vis]

        if self.use_landmarks:
            parts.append(self.landmark_encoder(landmarks.view(B * T, -1)))

        return torch.cat(parts, dim=1).view(B, T, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """LSTM last hidden state → classifier. Returns logits (B, num_classes)."""
        fused_seq = self._fuse_features(frames, landmarks)
        _, (h_n, _) = self.temporal(fused_seq)              # h_n: (num_layers, B, hidden)
        return self.classifier(h_n[-1])                     # (B, num_classes)

    # ------------------------------------------------------------------
    def predict_per_frame(
        self,
        frames: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify at each timestep using LSTM outputs (majority-vote inference).
        Returns softmax probabilities (B, T, num_classes).
        """
        B, T = frames.shape[:2]
        fused_seq = self._fuse_features(frames, landmarks)
        outputs, _ = self.temporal(fused_seq)                        # (B, T, hidden)
        logits = self.classifier(outputs.reshape(B * T, -1))         # (B*T, num_classes)
        return logits.view(B, T, self.num_classes).softmax(dim=-1)   # (B, T, num_classes)
