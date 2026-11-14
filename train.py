"""
Training script for the pain-detection model.

Usage
-----
    # Full end-to-end training (Phase 2)
    python train.py

    # Phase 1 — backbone frozen, train classifier head only
    python train.py --freeze_backbone

    # Custom options
    python train.py --npy_dir data/processed/dataset_npy --epochs 50 --batch_size 16

Outputs
-------
    checkpoints/best_model[_phase1].pth  — best validation-loss checkpoint
    checkpoints/latest[_phase1].pth      — most recent epoch checkpoint
    logs/<fecha_hora>_<phase1|phase2>/   — una carpeta por ejecución:
        train.log      — el mismo texto que sale por consola
        config.json    — comando, argumentos, hiperparámetros, entorno y dataset
        metrics.csv    — una fila por época (pérdidas, precisiones, lr, duración)
        summary.json   — resultado final (mejor época, test, duración)
        events.out.tfevents.*  — curvas para TensorBoard
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "12"

import argparse
import csv
import json
import logging
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src import config
from src.dataset import build_dataloaders
from src.model import PainDetectionModel

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ─── GPU geometric augmentation (flip, rotation, translation, temporal jitter) ──

class GpuGeometricAugmentation(nn.Module):
    """
    Aumentaciones geométricas en GPU aplicadas de forma consistente a frames
    y landmarks (Fase 2).

    Entradas:
        frames    : (B, T, 3, H, W) — tensores float normalizados con ImageNet
        landmarks : (B, T, landmark_dim) — coordenadas normalizadas [0, 1]
    Salidas: mismas formas, con aumentaciones aplicadas.
    """

    def forward(
        self, frames: torch.Tensor, landmarks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import kornia.geometry.transform as KGT

        B, T, C, H, W = frames.shape
        N_lm = landmarks.shape[-1] // 3

        # ── Temporal jitter ──────────────────────────────────────────────────
        if config.AUG_TEMPORAL_JITTER > 0:
            jitter = torch.randint(
                -config.AUG_TEMPORAL_JITTER,
                 config.AUG_TEMPORAL_JITTER + 1,
                (B, T), device=frames.device,
            )
            t_idx = (torch.arange(T, device=frames.device).unsqueeze(0) + jitter).clamp(0, T - 1)
            # Indexación vectorizada — sin bucle Python sobre el batch
            # t_idx: (B, T) → expandir a (B, T, C, H, W) para frames
            idx_f = t_idx.view(B, T, 1, 1, 1).expand(B, T, C, H, W)
            frames = torch.gather(frames, 1, idx_f)
            idx_l = t_idx.view(B, T, 1).expand_as(landmarks)
            landmarks = torch.gather(landmarks, 1, idx_l)

        # Parámetros por muestra, iguales en todos sus T frames
        angles = torch.empty(B, device=frames.device).uniform_(
            -config.AUG_ROTATION_DEGREES, config.AUG_ROTATION_DEGREES
        )
        tx = torch.empty(B, device=frames.device).uniform_(
            -config.AUG_TRANSLATE, config.AUG_TRANSLATE
        )
        ty = torch.empty(B, device=frames.device).uniform_(
            -config.AUG_TRANSLATE, config.AUG_TRANSLATE
        )
        do_flip = torch.rand(B, device=frames.device) < config.AUG_FLIP_PROB

        # Expandir a (B*T,) para operar frame a frame
        angles_bt  = angles .repeat_interleave(T)
        tx_bt      = tx     .repeat_interleave(T)
        ty_bt      = ty     .repeat_interleave(T)
        flip_bt    = do_flip.repeat_interleave(T)

        fr  = frames   .view(B * T, C, H, W)
        lm  = landmarks.view(B * T, N_lm, 3)

        # ── Flip horizontal ──────────────────────────────────────────────────
        fi = flip_bt.nonzero(as_tuple=True)[0]
        if fi.numel():
            fr[fi]       = fr[fi].flip(-1)
            lm[fi, :, 0] = 1.0 - lm[fi, :, 0]

        # ── Rotación ─────────────────────────────────────────────────────────
        if config.AUG_ROTATION_DEGREES > 0:
            fr = KGT.rotate(fr, angles_bt, mode="bilinear", padding_mode="zeros")
            rad   = angles_bt * (torch.pi / 180.0)
            cos_a = torch.cos(rad).unsqueeze(1)
            sin_a = torch.sin(rad).unsqueeze(1)
            cx = lm[:, :, 0] - 0.5
            cy = lm[:, :, 1] - 0.5
            lm[:, :, 0] = cos_a * cx - sin_a * cy + 0.5
            lm[:, :, 1] = sin_a * cx + cos_a * cy + 0.5

        # ── Traslación ───────────────────────────────────────────────────────
        if config.AUG_TRANSLATE > 0:
            t_px = torch.stack([tx_bt * W, ty_bt * H], dim=1)
            fr = KGT.translate(fr, t_px, mode="bilinear", padding_mode="zeros")
            lm[:, :, 0] = lm[:, :, 0] + tx_bt.unsqueeze(1)
            lm[:, :, 1] = lm[:, :, 1] + ty_bt.unsqueeze(1)

        return fr.view(B, T, C, H, W), lm.view(B, T, -1)


# ─── GPU augmentation (domain aug moved from CPU DataLoader workers to GPU) ───

class GpuAugmentation(nn.Module):
    """
    Applies domain augmentation on-device after each batch is sent to GPU.

    Moving these transforms from CPU DataLoader workers to GPU eliminates the
    data-loading bottleneck in Phase 2 (CPU workers only read + normalize frames;
    the GPU handles the heavy colour / noise / blur transforms).

    Input / output: (B, T, 3, H, W) — ImageNet-normalised float tensors.
    """

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        # Gaussian noise — independent per pixel and frame (no per-clip coherence needed)
        x = x + 0.03 * torch.randn_like(x)

        for b in range(B):
            s, e = b * T, (b + 1) * T

            # Brightness jitter (50%)
            if torch.rand(()).item() < 0.5:
                delta = torch.empty(1, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(-0.3, 0.3)
                x[s:e] = x[s:e] + delta

            # Contrast jitter (50%)
            if torch.rand(()).item() < 0.5:
                factor = torch.empty(1, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(0.7, 1.3)
                mean   = x[s:e].mean(dim=[2, 3], keepdim=True)
                x[s:e] = (x[s:e] - mean) * factor + mean

            # Gaussian blur (30%)
            if torch.rand(()).item() < 0.3:
                sigma  = float(torch.empty(()).uniform_(0.5, 1.5).item())
                k      = self._gauss_kernel(3, sigma, x.device, x.dtype)
                x[s:e] = F.conv2d(x[s:e], k, padding=1, groups=C)

            # Random erase (50%)
            if torch.rand(()).item() < 0.5:
                h_e  = max(1, int(H * torch.empty(()).uniform_(0.05, 0.30).item()))
                w_e  = max(1, int(W * torch.empty(()).uniform_(0.05, 0.30).item()))
                top  = torch.randint(0, max(1, H - h_e), ()).item()
                left = torch.randint(0, max(1, W - w_e), ()).item()
                x[s:e, :, top:top + h_e, left:left + w_e] = 0.0

            # Oclusión facial — mutuamente excluyente (10 % entrecejo, 20 % boca)
            r = torch.rand(()).item()
            if r < 0.10:
                x[s:e, :, :int(H * 0.35), int(W * 0.35):int(W * 0.65)] = 0.0   # entrecejo
            elif r < 0.30:
                x[s:e, :, int(H * 0.40):, int(W * 0.15):int(W * 0.85)] = 0.0   # boca

        return x.view(B, T, C, H, W)

    @staticmethod
    def _gauss_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
        coords = torch.arange(size, device=device, dtype=dtype) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        k = g.unsqueeze(0) * g.unsqueeze(1)
        k = k / k.sum()
        return k.view(1, 1, size, size).expand(3, 1, size, size).contiguous()


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Ordinal loss ─────────────────────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    """
    Combina CrossEntropyLoss con una penalización ordinal.

    La penalización es el valor esperado de |clase_predicha - clase_real|
    bajo la distribución softmax del modelo. Así, confundir clase 0 con
    clase 2 se penaliza el doble que confundir clases adyacentes.

    loss = (1 - alpha) * CE  +  alpha * E[|pred - true|]

    alpha=0 → solo CrossEntropy (sin conciencia ordinal)
    alpha=1 → solo penalización ordinal
    alpha=0.5 → equilibrio (valor por defecto)
    """

    def __init__(self, num_classes: int, weight=None, alpha: float = config.ORDINAL_ALPHA):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=weight)
        self.alpha = alpha
        idx = torch.arange(num_classes).float()
        # dist[i, j] = |i - j|: coste ordinal de predecir clase j cuando la real es i
        self.register_buffer("dist", (idx.unsqueeze(0) - idx.unsqueeze(1)).abs())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce      = self.ce(logits, targets)
        probs   = logits.softmax(dim=1)                        # (B, C)
        ordinal = (probs * self.dist[targets]).sum(dim=1).mean()
        return (1.0 - self.alpha) * ce + self.alpha * ordinal


# ─── Class weights ────────────────────────────────────────────────────────────

def compute_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights from the training split labels."""
    counts  = np.bincount(train_labels, minlength=config.NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (config.NUM_CLASSES * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


# ─── One epoch ───────────────────────────────────────────────────────────────

def _mixup_batch(frames, landmarks, labels, alpha):
    """
    Mezcla aleatoriamente dos muestras del batch con un coeficiente λ ~ Beta(alpha, alpha).
    Devuelve los tensores mezclados y las etiquetas originales de ambas muestras.
    La loss se computa como: λ·loss(logits, labels_a) + (1-λ)·loss(logits, labels_b),
    que es matemáticamente equivalente a usar etiquetas suaves.
    """
    lam  = np.random.beta(alpha, alpha)
    perm = torch.randperm(frames.size(0), device=frames.device)
    frames_mix    = lam * frames    + (1 - lam) * frames[perm]
    landmarks_mix = lam * landmarks + (1 - lam) * landmarks[perm]
    return frames_mix, landmarks_mix, labels, labels[perm], lam


def train_epoch(model, loader, optimiser, criterion, device, writer, epoch, scaler, gpu_aug=None, gpu_geom_aug=None):
    model.train()
    total_loss  = 0.0
    correct     = 0
    n_samples   = 0
    use_mixup   = config.AUG_MIXUP_ALPHA > 0
    amp_enabled = device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for frames, landmarks, labels in pbar:
        frames    = frames.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        # Aumentaciones geométricas en GPU (flip, rotación, traslación, jitter)
        if gpu_geom_aug is not None and frames.ndim == 5:
            frames, landmarks = gpu_geom_aug(frames, landmarks)

        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            # GPU domain augmentation (only for raw frames, not cached features)
            if gpu_aug is not None and frames.ndim == 5:
                frames = gpu_aug(frames)
            # MixUp after all other augmentations so each sample is degraded independently
            if use_mixup:
                frames, landmarks, labels_a, labels_b, lam = _mixup_batch(
                    frames, landmarks, labels, config.AUG_MIXUP_ALPHA
                )
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            logits = model(frames, landmarks)
            loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * labels_a.size(0)
        correct    += (logits.argmax(1) == labels_a).sum().item()
        n_samples  += labels_a.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / n_samples
    acc      = correct    / n_samples
    writer.add_scalar("train/loss",     avg_loss, epoch)
    writer.add_scalar("train/accuracy", acc,      epoch)
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, writer, epoch, split: str):
    model.eval()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    amp_enabled = device.type == "cuda"
    for frames, landmarks, labels in tqdm(loader, desc=f"Epoch {epoch:03d} [{split:4s}]", leave=False):
        frames    = frames.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(frames, landmarks)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n_samples  += labels.size(0)

    avg_loss = total_loss / n_samples
    acc      = correct    / n_samples
    writer.add_scalar(f"{split}/loss",     avg_loss, epoch)
    writer.add_scalar(f"{split}/accuracy", acc,      epoch)
    return avg_loss, acc


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _rng_state() -> dict:
    """
    Estado de todos los generadores aleatorios (para reanudar exactamente).
    Solo con tipos básicos y tensores, para que el checkpoint se pueda cargar
    con torch.load(weights_only=True), el modo por defecto de PyTorch.
    """
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy":  (name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached)),
        "torch":  torch.get_rng_state(),
        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    # torch.load(map_location="cuda") mueve también estos tensores a la GPU;
    # los generadores necesitan su estado en CPU.
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: Path, model, optimiser, scheduler, scaler, epoch, val_loss, val_acc,
                    run_name: str, progress: dict):
    """
    progress: estado del bucle necesario para reanudar sin diferencias
              (best_val_loss, best_val_acc, best_epoch, patience_count).
    """
    torch.save(
        {
            "run":                 run_name,   # carpeta de logs/ que generó este checkpoint
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":   scaler.state_dict(),
            "rng_state":           _rng_state(),
            "progress":            progress,
            "val_loss":            val_loss,
            "val_acc":             val_acc,
            "config": {
                "num_frames":      config.NUM_FRAMES,
                "landmark_dim":    config.LANDMARK_DIM,
                "num_classes":     config.NUM_CLASSES,
                "use_landmarks":   config.USE_LANDMARKS,
                "landmark_hidden": config.LANDMARK_HIDDEN_DIM,
                "landmark_out":    config.LANDMARK_OUT_DIM,
                "lstm_hidden":     config.LSTM_HIDDEN_DIM,
                "lstm_layers":     config.LSTM_NUM_LAYERS,
                "dropout":         config.DROPOUT,
            },
        },
        path,
    )


def load_checkpoint(path: Path, model, optimiser=None, scheduler=None, device="cpu"):
    """Carga los pesos (y, si se pasan, optimizador y scheduler). Devuelve el checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimiser is not None and "optimizer_state_dict" in ckpt:
        optimiser.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ─── Run logging ──────────────────────────────────────────────────────────────

def _config_snapshot() -> dict:
    """Hiperparámetros de src/config.py (constantes en mayúsculas) en formato JSON."""
    snap = {}
    for key in dir(config):
        if not key.isupper():
            continue
        value = getattr(config, key)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (bool, int, float, str, list, tuple)):
            snap[key] = list(value) if isinstance(value, tuple) else value
    return snap


class RunLogger:
    """
    Guarda todo lo de una ejecución en su propia carpeta logs/<fecha_hora>_<fase>/:

      train.log      — el mismo texto que sale por consola
      config.json    — comando, argumentos, hiperparámetros, entorno y dataset
      metrics.csv    — una fila por época, escrita al terminar cada una
      summary.json   — resultado final
      events.out.tfevents.*  — TensorBoard
    """

    def __init__(self, log_root: str, phase: str):
        self.name = f"{datetime.now():%Y%m%d_%H%M%S}_{phase}"
        self.dir  = Path(log_root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

        self._file_handler = logging.FileHandler(self.dir / "train.log", encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(self._file_handler)

        self.tb = SummaryWriter(log_dir=str(self.dir))

        self._csv_file = open(self.dir / "metrics.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                            "lr", "epoch_seconds"])

    def save_json(self, filename: str, data: dict) -> None:
        (self.dir / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, seconds) -> None:
        self._csv.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                            f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr:.3e}", f"{seconds:.1f}"])
        self._csv_file.flush()
        self.tb.add_scalar("lr", lr, epoch)

    def elapsed_minutes(self) -> float:
        return (time.time() - self.start) / 60

    def close(self) -> None:
        self.tb.close()
        self._csv_file.close()
        logging.getLogger().removeHandler(self._file_handler)
        self._file_handler.close()


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    phase = "phase1" if args.freeze_backbone else "phase2"
    run = RunLogger(args.log_dir, phase)
    try:
        _train(args, run, phase)
    except KeyboardInterrupt:
        log.warning("Entrenamiento interrumpido por el usuario (Ctrl+C)")
        raise
    except SystemExit:
        raise
    except Exception:
        log.exception("Entrenamiento interrumpido por un error")   # traza completa en train.log
        raise
    finally:
        run.close()


def _train(args, run: RunLogger, phase: str):
    set_seed(config.SEED)
    log.info("Run: %s  →  %s", run.name, run.dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  (VRAM ≈ %.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        log.warning("No se detecta GPU: PyTorch %s no tiene soporte CUDA o no hay GPU NVIDIA. "
                    "El entrenamiento irá en la CPU y será muchísimo más lento. "
                    "Instala PyTorch con CUDA (ver requirements.txt).", torch.__version__)
    log.info("AMP (mixed precision): %s", "enabled" if device.type == "cuda" else "disabled (no CUDA)")

    # GPU domain augmentation — only for Phase 2 (Phase 1 uses cached features)
    use_cache    = args.freeze_backbone and args.cache_path is not None and Path(args.cache_path).exists()
    effective_cache = args.cache_path if use_cache else None
    gpu_aug      = None if use_cache else GpuAugmentation().to(device)
    gpu_geom_aug = None if use_cache else GpuGeometricAugmentation().to(device)
    log.info("Data pipeline: %s", "cached features (Phase-1)" if use_cache else "raw frames + GPU aug")
    if use_cache:
        log.info("Fase 1 — caché: %s", effective_cache)
    log.info("GPU geometric augmentation: %s", "enabled" if gpu_geom_aug is not None else "disabled")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    log.info("Dataset: npy_dir=%s", args.npy_dir)
    train_loader, val_loader, test_loader, train_labels = \
        build_dataloaders(npy_dir=args.npy_dir, batch_size=args.batch_size,
                          cache_path=effective_cache)

    split_sizes = {"train": len(train_loader.dataset),
                   "val":   len(val_loader.dataset),
                   "test":  len(test_loader.dataset)}
    log.info("Split sizes — train: %d  val: %d  test: %d",
             split_sizes["train"], split_sizes["val"], split_sizes["test"])

    # ── Model ────────────────────────────────────────────────────────────────
    model = PainDetectionModel(
        num_frames=config.NUM_FRAMES,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=args.freeze_backbone,
        use_landmarks=config.USE_LANDMARKS,
    ).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters — total: %s  trainable: %s",
             f"{n_total:,}", f"{n_trainable:,}")

    # ── Loss with class weights ───────────────────────────────────────────────
    class_weights = compute_class_weights(train_labels).to(device)
    log.info("Class weights: %s", class_weights.tolist())
    criterion = OrdinalLoss(num_classes=config.NUM_CLASSES, weight=class_weights, alpha=config.ORDINAL_ALPHA).to(device)
    log.info("Loss: OrdinalLoss(alpha=%.2f)", config.ORDINAL_ALPHA)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimiser = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Optionally resume or transfer weights from a checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch, best_val_acc = None, None
    patience_count = 0
    resume_rng = None
    if args.transfer_weights:
        if not Path(args.transfer_weights).exists():
            log.error("--transfer_weights: fichero no encontrado: %s", args.transfer_weights)
            sys.exit(1)
        # Solo carga pesos del modelo; optimizer y scheduler arrancan desde cero.
        # Usar para Fase 2: transfiere lo aprendido en Fase 1 sin arrastrar su LR schedule.
        load_checkpoint(Path(args.transfer_weights), model, device=device)
        log.info("Pesos transferidos desde %s (optimizer/scheduler reiniciados)", args.transfer_weights)
    elif args.resume:
        if not Path(args.resume).exists():
            log.error("--resume: fichero no encontrado: %s", args.resume)
            sys.exit(1)
        ckpt = load_checkpoint(Path(args.resume), model, optimiser, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        progress = ckpt.get("progress")
        if progress is not None:
            # Continúa exactamente donde se quedó: mejor modelo, paciencia,
            # escala de la precisión mixta y generadores aleatorios.
            best_val_loss  = progress["best_val_loss"]
            best_val_acc   = progress["best_val_acc"]
            best_epoch     = progress["best_epoch"]
            patience_count = progress["patience_count"]
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            resume_rng = ckpt.get("rng_state")
            if progress.get("max_epochs") not in (None, args.epochs):
                log.warning("--epochs=%d distinto del de la ejecución original (%d): el learning rate "
                            "seguirá la curva original pero el bucle terminará en la época %d.",
                            args.epochs, progress["max_epochs"], args.epochs)
            log.info("Resumed from %s (run %s) at epoch %d — best epoch %s (val_loss=%.4f), "
                     "patience %d/%d",
                     args.resume, ckpt.get("run", "?"), start_epoch - 1, best_epoch,
                     best_val_loss, patience_count, args.patience)
        else:
            # Checkpoint antiguo, sin estado del bucle: se toma su val_loss como referencia
            best_val_loss = ckpt.get("val_loss", float("inf"))
            log.warning("Checkpoint sin estado de entrenamiento (versión anterior): se reanuda con "
                        "la paciencia a 0 y su val_loss como mejor valor.")
            log.info("Resumed from %s at epoch %d", args.resume, start_epoch - 1)

    # ── Checkpoints ──────────────────────────────────────────────────────────
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    phase_suffix = "_phase1" if args.freeze_backbone else ""
    best_ckpt   = Path(args.checkpoint_dir) / f"best_model{phase_suffix}.pth"
    latest_ckpt = Path(args.checkpoint_dir) / f"latest{phase_suffix}.pth"

    # ── Configuración de la ejecución ────────────────────────────────────────
    run.save_json("config.json", {
        "run":          run.name,
        "phase":        phase,
        "started":      datetime.now().isoformat(timespec="seconds"),
        "command":      " ".join([Path(sys.executable).name] + sys.argv),
        "args":         vars(args),
        "hyperparameters": _config_snapshot(),
        "environment": {
            "device":  str(device),
            "gpu":     torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python":  platform.python_version(),
            "torch":   torch.__version__,
            "os":      platform.platform(),
        },
        "dataset": {
            "source":        effective_cache or args.npy_dir,
            "split_sizes":   split_sizes,
            "class_weights": class_weights.tolist(),
        },
        "parameters": {"total": n_total, "trainable": n_trainable},
        "checkpoints": {"best": str(best_ckpt), "latest": str(latest_ckpt)},
    })

    # ── Training loop ────────────────────────────────────────────────────────
    writer = run.tb
    last_epoch, early_stopped = start_epoch - 1, False

    log.info("Starting training for up to %d epochs (patience=%d) …",
             args.epochs, args.patience)

    # Al reanudar, los generadores aleatorios vuelven al estado del final de la
    # última época guardada (orden de los datos, aumentaciones, dropout, MixUp).
    if resume_rng is not None:
        _set_rng_state(resume_rng)

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        lr = optimiser.param_groups[0]["lr"]
        train_loss, train_acc = train_epoch(
            model, train_loader, optimiser, criterion, device, writer, epoch, scaler,
            gpu_aug, gpu_geom_aug,
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device, writer, epoch, "val"
        )

        scheduler.step()
        last_epoch = epoch
        run.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr, time.time() - t_epoch)

        gpu_mem = (f"  GPU peak {torch.cuda.max_memory_allocated(device)/1e9:.1f}/"
                   f"{torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB"
                   ) if device.type == "cuda" else ""
        log.info("Epoch %03d/%03d | train loss %.4f  acc %.4f | val loss %.4f  acc %.4f%s",
                 epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, gpu_mem)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss  = val_loss
            best_epoch, best_val_acc = epoch, val_acc
            patience_count = 0
        else:
            patience_count += 1

        # El estado del bucle se guarda ya actualizado con esta época, para que
        # --resume continúe exactamente desde aquí.
        progress = {
            "best_val_loss":  best_val_loss,
            "best_val_acc":   best_val_acc,
            "best_epoch":     best_epoch,
            "patience_count": patience_count,
            "max_epochs":     args.epochs,
        }
        ckpt_args = (model, optimiser, scheduler, scaler, epoch, val_loss, val_acc, run.name, progress)

        # Primero el mejor y después el último: si el proceso muere entre los dos,
        # latest.pth nunca apunta a un mejor modelo que no se llegó a guardar.
        if improved:
            save_checkpoint(best_ckpt, *ckpt_args)
        save_checkpoint(latest_ckpt, *ckpt_args)

        if improved:
            log.info("  ✓ New best model saved  (val_loss=%.4f)", val_loss)
        else:
            log.info("  No improvement for %d / %d epochs.", patience_count, args.patience)
            if patience_count >= args.patience:
                log.info("Early stopping triggered.")
                early_stopped = True
                break

    log.info("Training complete.  Best val loss: %.4f", best_val_loss)

    # ── Quick test-set evaluation with best model ─────────────────────────────
    test_loss = test_acc = None
    if best_ckpt.exists() and len(test_loader.dataset) > 0:
        load_checkpoint(best_ckpt, model, device=device)
        test_loss, test_acc = evaluate_epoch(
            model, test_loader, criterion, device, writer, best_epoch or last_epoch, "test"
        )
        log.info("Test set — loss %.4f  accuracy %.4f", test_loss, test_acc)

    run.save_json("summary.json", {
        "run":            run.name,
        "phase":          phase,
        "finished":       datetime.now().isoformat(timespec="seconds"),
        "duration_min":   round(run.elapsed_minutes(), 1),
        "epochs_run":     last_epoch - start_epoch + 1,
        "last_epoch":     last_epoch,
        "early_stopped":  early_stopped,
        "best_epoch":     best_epoch,
        "best_val_loss":  best_val_loss,
        "best_val_acc":   best_val_acc,
        "test_loss":      test_loss,
        "test_acc":       test_acc,
        "best_checkpoint": str(best_ckpt),
    })
    log.info("Logs de la ejecución → %s", run.dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pain-detection model")
    parser.add_argument("--checkpoint_dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--log_dir",        default=str(config.LOG_DIR))
    parser.add_argument("--batch_size",     type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",             type=float, default=None,
                        help="Learning rate inicial (por defecto: 1e-3 en Fase 1, 1e-4 en Fase 2)")
    parser.add_argument("--weight_decay",   type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--epochs",         type=int,   default=None,
                        help="Épocas máximas (por defecto: 20 en Fase 1, 50 en Fase 2)")
    parser.add_argument("--patience",       type=int,   default=None,
                        help="Paciencia del early stopping (por defecto: 5 en Fase 1, 12 en Fase 2)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone weights (Phase-1 training)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from (mantiene optimizer/scheduler)")
    parser.add_argument("--transfer_weights", default=None,
                        help="Path to checkpoint del que transferir solo pesos (Fase 1 → Fase 2)")
    parser.add_argument("--cache_path", default=str(config.FEATURES_CACHE_PATH),
                        help="Path to features_cache.h5 (activa Phase-1 rápida con features pre-computadas)")
    parser.add_argument("--npy_dir", default=str(config.PROCESSED_NPY_DIR),
                        help="Carpeta con frames.npy/landmarks.npy/labels.npy (memmap, por defecto)")
    args = parser.parse_args()

    # Valores por defecto según la fase
    if args.lr is None:
        args.lr = config.PHASE1_LEARNING_RATE if args.freeze_backbone else config.LEARNING_RATE
    if args.epochs is None:
        args.epochs = config.PHASE1_NUM_EPOCHS if args.freeze_backbone else config.NUM_EPOCHS
    if args.patience is None:
        args.patience = (config.PHASE1_EARLY_STOPPING_PATIENCE if args.freeze_backbone
                         else config.EARLY_STOPPING_PATIENCE)

    train(args)
