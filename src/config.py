"""
Centralised configuration for the pain-detection system.
All scripts import constants from here so hyperparameters stay in one place.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_NPY_DIR   = DATA_DIR / "processed" / "dataset_npy"
FEATURES_CACHE_PATH = DATA_DIR / "processed" / "features_cache.h5"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# ─── Data ─────────────────────────────────────────────────────────────────────
NUM_FRAMES: int = 16       # equispaced frames sampled from each video
FRAME_SIZE: int = 224       # spatial resolution fed to EfficientNet (px)
NUM_CLASSES: int = 3        # pain levels 0 / 1 / 2

# ─── MediaPipe Face Mesh ───────────────────────────────────────────────────────
# NOTE: with refine_landmarks=True MediaPipe returns 478 landmarks (468 base +
#       10 iris). The spec mentions 468×3=1404 but that value is only correct
#       when refine_landmarks=False. We follow refine_landmarks=True → 478×3=1434.
MP_STATIC_IMAGE_MODE: bool = False
MP_REFINE_LANDMARKS: bool = True
MP_MIN_DETECTION_CONF: float = 0.7
MP_MIN_TRACKING_CONF: float = 0.7
NUM_LANDMARKS: int = 478          # change to 468 if refine_landmarks=False
LANDMARK_DIM: int = NUM_LANDMARKS * 3  # 1434

# ─── Backbone ─────────────────────────────────────────────────────────────────
# EfficientNet-B0 preentrenado en ImageNet.
# Normalización de frames con las estadísticas de ImageNet.
FRAME_MEAN: list = [0.485, 0.456, 0.406]
FRAME_STD:  list = [0.229, 0.224, 0.225]

# ─── Model ────────────────────────────────────────────────────────────────────
VISUAL_DIM: int = 1280        # dimensión de salida de EfficientNet-B0
LANDMARK_HIDDEN_DIM: int = 512
LANDMARK_OUT_DIM: int = 256
USE_LANDMARKS: bool = True    # False → solo rama visual (EfficientNet), sin MLP de landmarks

DROPOUT: float = 0.4

# ─── Loss ordinal ─────────────────────────────────────────────────────────────
ORDINAL_ALPHA: float = 0.5    # peso de la penalización ordinal vs CrossEntropy (0=solo CE, 1=solo ordinal)

# ─── Temporal (LSTM) ──────────────────────────────────────────────────────────
LSTM_HIDDEN_DIM: int = 512
LSTM_NUM_LAYERS: int = 2

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 8
WEIGHT_DECAY: float = 1e-2
# Fase 1 (EfficientNet congelado, --freeze_backbone)
PHASE1_LEARNING_RATE: float = 1e-3
PHASE1_NUM_EPOCHS: int = 20
PHASE1_EARLY_STOPPING_PATIENCE: int = 5
# Fase 2 (fine-tuning completo): learning rate menor para no destruir lo
# aprendido por EfficientNet al descongelarlo
LEARNING_RATE: float = 1e-4
NUM_EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 12

# Data split (must sum to 1.0)
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ─── Augmentation ─────────────────────────────────────────────────────────────
# Geométricas (Fase 2, en GPU, iguales para frames y landmarks)
AUG_ROTATION_DEGREES: float = 10.0  # ± degrees
AUG_FLIP_PROB: float = 0.5          # horizontal flip probability
AUG_TRANSLATE: float = 0.10         # max translation as fraction of image size
AUG_TEMPORAL_JITTER: int   = 2      # máx. desplazamiento aleatorio de índice de frame (±frames)
# Las fotométricas (ruido, brillo, contraste, desenfoque y oclusiones) tienen
# sus probabilidades en GpuAugmentation (train.py).
AUG_MIXUP_ALPHA: float = 0.4        # parámetro Beta para MixUp (0 = desactivado)

# ─── Hardware / Reproducibility ───────────────────────────────────────────────
SEED: int = 42
import platform as _platform
NUM_WORKERS: int = 0 if _platform.system() == "Windows" else 2
PIN_MEMORY: bool = True
