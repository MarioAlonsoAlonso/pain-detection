"""
PainDataset — PyTorch Dataset backed by the preprocessed NPY files.

Each sample is a video clip represented as:
  frames    : (T, 3, 224, 224) — ImageNet-normalised tensor
  landmarks : (T, landmark_dim) — normalised landmark coordinates
  label     : int (0 / 1 / 2)

On-the-fly augmentation is applied only to the training split and uses the
same geometric transform for frames and landmarks to keep them consistent.
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


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def _augment(
    frame_bgr: np.ndarray,
    lm: np.ndarray,
    rotation_deg: float,
    translate: float,
    brightness_delta: float,
    do_flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply consistent geometric augmentation to a frame and its landmarks.

    Parameters
    ----------
    frame_bgr     : (H, W, 3) uint8 BGR
    lm            : (N, 3) float32  — x,y already in [0,1]; z is relative depth
    rotation_deg  : rotation angle in degrees (positive = counter-clockwise)
    translate     : translation magnitude as a fraction of image size (applied
                    equally in x and y direction with the supplied sign)
    brightness_delta : additive brightness change in [0,1] range
    do_flip       : horizontal flip flag

    Returns
    -------
    augmented frame (H, W, 3) uint8, flattened landmarks (N*3,) float32
    """
    H, W = frame_bgr.shape[:2]
    lm = lm.copy()  # avoid mutating caller's array

    # Rotation — applied to image and to the (x,y) pixel coordinates of landmarks
    if rotation_deg != 0.0:
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), rotation_deg, 1.0)
        frame_bgr = cv2.warpAffine(frame_bgr, M, (W, H))

        xy_px   = lm[:, :2] * np.array([W, H], dtype=np.float32)
        xy_px_h = np.hstack([xy_px, np.ones((len(xy_px), 1), dtype=np.float32)])
        xy_rot  = (M @ xy_px_h.T).T
        lm[:, :2] = xy_rot / np.array([W, H], dtype=np.float32)

    # Translation (normalised units)
    if translate != 0.0:
        tx_px = int(translate * W)
        ty_px = int(translate * H)
        M_t = np.float32([[1, 0, tx_px], [0, 1, ty_px]])
        frame_bgr  = cv2.warpAffine(frame_bgr, M_t, (W, H))
        lm[:, 0] += translate
        lm[:, 1] += translate

    # Horizontal flip
    if do_flip:
        frame_bgr = cv2.flip(frame_bgr, 1)
        lm[:, 0]  = 1.0 - lm[:, 0]

    # Brightness — photometric only (does not affect landmarks)
    if brightness_delta != 0.0:
        frame_f   = frame_bgr.astype(np.float32) / 255.0
        frame_f   = np.clip(frame_f + brightness_delta, 0.0, 1.0)
        frame_bgr = (frame_f * 255).astype(np.uint8)

    return frame_bgr, lm.flatten()


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PainDataset(Dataset):
    """
    Clips preprocesados por preprocess.py (formato NPY).

    Los tres splits comparten los mismos arrays (memmap) — sin copias en RAM.

    Parameters
    ----------
    frames      : (N, T, H, W, 3) uint8 BGR — frames.npy
    landmarks   : (N, T, landmark_dim) float32 — landmarks.npy
    labels      : (N,) int — labels.npy
    indices     : list of video indices for this split
    augment     : whether to apply data augmentation
    gpu_augment : si True, las aumentaciones geométricas se dejan para la GPU
    """

    def __init__(
        self,
        frames: np.ndarray,
        landmarks: np.ndarray,
        labels: np.ndarray,
        indices: list,
        augment: bool = False,
        gpu_augment: bool = False,
    ):
        super().__init__()
        self.indices = indices
        self.augment = augment
        # gpu_augment=True: los workers solo leen y normalizan; las transformaciones
        # geométricas (flip, rotación, traslación, jitter temporal) las aplica
        # GpuGeometricAugmentation en train_epoch, eliminando el cuello de botella CPU.
        self.gpu_augment = gpu_augment

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

        if self.augment and not self.gpu_augment:
            rot   = np.random.uniform(-config.AUG_ROTATION_DEGREES,
                                       config.AUG_ROTATION_DEGREES)
            trans = np.random.uniform(-config.AUG_TRANSLATE,
                                       config.AUG_TRANSLATE)
            brit  = np.random.uniform(-config.AUG_BRIGHTNESS,
                                       config.AUG_BRIGHTNESS)
            flip  = np.random.random() < config.AUG_FLIP_PROB

            jitter  = np.random.randint(
                -config.AUG_TEMPORAL_JITTER,
                 config.AUG_TEMPORAL_JITTER + 1,
                size=self.num_frames,
            )
            t_indices = np.clip(np.arange(self.num_frames) + jitter, 0, self.num_frames - 1)
            frames_raw = frames_raw[t_indices]
            lm_raw     = lm_raw[t_indices]
        else:
            rot = trans = brit = 0.0
            flip = False

        out_frames = []
        out_lm     = []

        for t in range(self.num_frames):
            frame = frames_raw[t]               # (H, W, 3) uint8 BGR
            lm_2d = lm_raw[t].reshape(-1, 3)   # (N_landmarks, 3)

            if self.augment:
                frame, lm_flat = _augment(frame, lm_2d, rot, trans, brit, flip)
            else:
                lm_flat = lm_2d.flatten()

            # BGR → RGB → Tensor → normalización
            # (domain aug: color jitter, blur, noise, erase se aplican en GPU en train.py)
            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = _normalise(_to_tensor(frame_rgb))          # (3, H, W)
            lm_tensor    = torch.tensor(lm_flat, dtype=torch.float32)

            out_frames.append(frame_tensor)
            out_lm.append(lm_tensor)

        frames_t = torch.stack(out_frames)  # (T, 3, H, W)
        lm_t     = torch.stack(out_lm)     # (T, landmark_dim)

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
    gpu_augment: bool = False,
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

        train_ds = PainDataset(frames_mm, landmarks_mm, labels, train_idx,
                               augment=True, gpu_augment=gpu_augment)
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
