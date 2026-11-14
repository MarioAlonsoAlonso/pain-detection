"""
PainDataset — PyTorch Dataset backed by the preprocessed NPY files.

Each sample is a video clip represented as:
  frames    : (T, 3, 224, 224) — ImageNet-normalised tensor
  landmarks : (T, landmark_dim) — normalised landmark coordinates
  label     : int (0 / 1 / 2)

La aumentación de la Fase 2 se aplica en la GPU (train.py). En la Fase 1, con
las características cacheadas, CachedFeaturesDataset aplica el jitter temporal
y el volteo de los landmarks.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
import cv2

from src import config

_to_tensor = transforms.ToTensor()
_normalise = transforms.Normalize(config.FRAME_MEAN, config.FRAME_STD)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PainDataset(Dataset):
    """
    Clips preprocesados por preprocess.py (formato NPY).

    Solo lee y normaliza: la aumentación de datos de la Fase 2 se aplica en la
    GPU, sobre el lote completo, en train.py (GpuGeometricAugmentation y
    GpuAugmentation).

    Los tres splits comparten los mismos arrays (memmap) — sin copias en RAM.

    Parameters
    ----------
    frames    : (N, T, H, W, 3) uint8 BGR — frames.npy
    landmarks : (N, T, landmark_dim) float32 — landmarks.npy
    labels    : (N,) int — labels.npy
    indices   : list of video indices for this split
    """

    def __init__(
        self,
        frames: np.ndarray,
        landmarks: np.ndarray,
        labels: np.ndarray,
        indices: list,
    ):
        super().__init__()
        self.indices = indices
        self._frames_all    = frames
        self._landmarks_all = landmarks
        self.labels         = labels
        self.num_frames     = int(frames.shape[1])
        self.landmark_dim   = int(landmarks.shape[-1])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        i = self.indices[idx]

        frames_raw = self._frames_all   [i]   # (T, H, W, 3) uint8 BGR
        lm_raw     = self._landmarks_all[i]   # (T, landmark_dim) float32
        label = int(self.labels[i])

        out_frames = []
        for t in range(self.num_frames):
            # BGR → RGB → Tensor → normalización ImageNet
            frame_rgb = cv2.cvtColor(frames_raw[t], cv2.COLOR_BGR2RGB)
            out_frames.append(_normalise(_to_tensor(frame_rgb)))   # (3, H, W)

        frames_t = torch.stack(out_frames)                          # (T, 3, H, W)
        lm_t     = torch.tensor(np.asarray(lm_raw), dtype=torch.float32)   # (T, landmark_dim)

        return frames_t, lm_t, torch.tensor(label, dtype=torch.long)


# ─── Cached-features dataset (Phase-1 fast training) ────────────────────────

class CachedFeaturesDataset(Dataset):
    """
    Loads pre-computed EfficientNet features instead of raw frames.

    Used for Phase-1 training (backbone frozen): eliminates EfficientNet
    forward passes and CPU image transforms, reducing step time from ~10 s
    to ~50 ms.

    Each sample:
      visual_features : (T, visual_dim)  — pre-computed by cache_features.py
      landmarks       : (T, landmark_dim)
      label           : int
    """

    def __init__(self, cache_path: str, indices: list, augment: bool = False):
        super().__init__()
        self.indices = indices
        self.augment = augment

        # Carga todo en RAM de una vez (~365 MB) — elimina la latencia de HDF5
        # en cada __getitem__ y hace el acceso completamente instantáneo.
        with h5py.File(cache_path, "r") as f:
            self._features  = f["features"] [:]   # (N, T, visual_dim)  float32
            self._landmarks = f["landmarks"][:]   # (N, T, landmark_dim) float32
            self.labels     = f["labels"]   [:]
            self.num_frames = int(self._features.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i         = self.indices[idx]
        features  = self._features [i].copy()
        landmarks = self._landmarks[i].copy()
        label = int(self.labels[i])

        if self.augment:
            # Temporal jitter: desplaza índices de frame ±AUG_TEMPORAL_JITTER
            jitter = np.random.randint(
                -config.AUG_TEMPORAL_JITTER,
                 config.AUG_TEMPORAL_JITTER + 1,
                size=self.num_frames,
            )
            t_idx     = np.clip(np.arange(self.num_frames) + jitter, 0, self.num_frames - 1)
            features  = features[t_idx]
            landmarks = landmarks[t_idx]

            # Horizontal flip en coordenadas de landmarks (x → 1-x)
            if np.random.random() < config.AUG_FLIP_PROB:
                lm_2d          = landmarks.reshape(self.num_frames, -1, 3).copy()
                lm_2d[:, :, 0] = 1.0 - lm_2d[:, :, 0]
                landmarks      = lm_2d.reshape(self.num_frames, -1)

        return (
            torch.tensor(features,  dtype=torch.float32),
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(label,     dtype=torch.long),
        )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    npy_dir: str | None = None,
    batch_size: int  = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
    use_weighted_sampler: bool = True,
    seed: int = config.SEED,
    cache_path: str | None = None,
):
    """
    Build stratified train / val / test DataLoaders.

    npy_dir    : carpeta generada por preprocess.py (frames/landmarks/labels .npy).
    cache_path : features_cache.h5 generado por cache_features.py. Si se indica,
                 los loaders devuelven features pre-computadas (Fase 1) en lugar
                 de frames.

    Returns
    -------
    train_loader, val_loader, test_loader, train_labels
      train_labels: np.ndarray con las etiquetas del split de entrenamiento.
    """
    from sklearn.model_selection import train_test_split

    # Solo usar npy_dir si los archivos existen realmente
    npy = Path(npy_dir) if npy_dir is not None else None
    if npy is not None and not (npy / "labels.npy").exists():
        npy = None

    if npy is not None:
        labels = np.load(npy / "labels.npy")
    elif cache_path is not None:
        # Fase 1: features_cache.h5 contiene las etiquetas — no hace falta npy_dir
        with h5py.File(cache_path, "r") as f:
            labels = f["labels"][:]
    else:
        raise ValueError(
            "No se encontró el dataset preprocesado: ejecuta preprocess.py "
            "(o indica npy_dir / cache_path)."
        )

    indices = list(range(len(labels)))

    # Stratified split
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - config.TRAIN_RATIO,
        stratify=labels[indices],
        random_state=seed,
    )
    if config.TEST_RATIO == 0:
        val_idx, test_idx = temp_idx, []
    else:
        val_frac = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_frac,
            stratify=labels[temp_idx],
            random_state=seed,
        )

    if cache_path is not None:
        train_ds = CachedFeaturesDataset(cache_path, train_idx, augment=True)
        val_ds   = CachedFeaturesDataset(cache_path, val_idx,   augment=False)
        test_ds  = CachedFeaturesDataset(cache_path, test_idx,  augment=False)
    else:
        # Memmap: mapeo instantáneo del archivo, sin cargar nada en RAM.
        # Cada __getitem__ lee solo la muestra solicitada; el SO cachea lo caliente.
        frames_mm    = np.load(npy / "frames.npy",    mmap_mode="r")
        landmarks_mm = np.load(npy / "landmarks.npy", mmap_mode="r")

        train_ds = PainDataset(frames_mm, landmarks_mm, labels, train_idx)
        val_ds   = PainDataset(frames_mm, landmarks_mm, labels, val_idx)
        test_ds  = PainDataset(frames_mm, landmarks_mm, labels, test_idx)

    train_labels = labels[np.array(train_idx)]   # etiquetas del split de entrenamiento

    # ── Weighted sampler ─────────────────────────────────────────────────────
    sampler = None
    if use_weighted_sampler:
        class_counts  = np.bincount(train_labels, minlength=config.NUM_CLASSES).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_w      = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_w, num_samples=len(sample_w), replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, train_labels
