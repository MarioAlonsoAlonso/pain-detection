"""
Extract and cache EfficientNet visual features for Phase-1 training.

Reads frames directly from the NPY dataset (no DataLoader) for maximum I/O
throughput, then processes them in GPU batches. The result is stored in a
single HDF5 file (features_cache.h5).

Usage
-----
    python cache_features.py
    python cache_features.py --npy_dir data/processed/dataset_npy --output data/processed/features_cache.h5
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src import config
from src.model import VisualEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ImageNet normalization constants (same as _normalise in dataset.py)
_MEAN = torch.tensor(config.FRAME_MEAN).view(1, 3, 1, 1)
_STD  = torch.tensor(config.FRAME_STD ).view(1, 3, 1, 1)


def frames_to_tensor(frames_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert (N, H, W, 3) uint8 BGR numpy array to (N, 3, H, W) normalised GPU tensor.
    Does BGR→RGB, /255, and ImageNet normalisation inline — no DataLoader overhead.
    """
    # BGR → RGB and uint8 → float32 [0, 1]
    rgb = frames_bgr[:, :, :, ::-1].copy()          # (N, H, W, 3) RGB
    t   = torch.from_numpy(rgb).permute(0, 3, 1, 2)  # (N, 3, H, W)
    t   = t.float().div_(255.0).to(device, non_blocking=True)
    mean = _MEAN.to(device, dtype=t.dtype)
    std  = _STD .to(device, dtype=t.dtype)
    return t.sub_(mean).div_(std)                    # (N, 3, H, W) normalised


def cache_features(args):
    import json as _json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    encoder = VisualEncoder(freeze=True).to(device).eval()
    log.info("VisualEncoder — %s params", f"{sum(p.numel() for p in encoder.parameters()):,}")

    npy = Path(args.npy_dir)
    if not (npy / "frames.npy").exists():
        log.error("No se encontró el dataset preprocesado en %s — ejecuta antes preprocess.py", npy)
        sys.exit(1)
    meta = _json.loads((npy / "metadata.json").read_text())
    n_total      = meta["num_samples"]
    landmark_dim = meta["landmark_dim"]
    num_frames   = meta["num_frames"]
    labels_all   = np.load(npy / "labels.npy")
    frames_src   = np.load(npy / "frames.npy",    mmap_mode="r")
    lm_src       = np.load(npy / "landmarks.npy", mmap_mode="r")
    log.info("Source: %d samples, %d frames, landmark_dim=%d", n_total, num_frames, landmark_dim)

    visual_dim = config.VISUAL_DIM
    features   = np.empty((n_total, num_frames, visual_dim),   dtype=np.float32)
    landmarks  = np.empty((n_total, num_frames, landmark_dim), dtype=np.float32)

    log.info("Extracting features (batch_size=%d) …", args.batch_size)
    for start in tqdm(range(0, n_total, args.batch_size), desc="cache_features"):
        end = min(start + args.batch_size, n_total)
        B   = end - start

        batch_frames    = np.array(frames_src[start:end])    # (B, T, H, W, 3) uint8
        batch_landmarks = np.array(lm_src    [start:end])    # (B, T, landmark_dim)

        flat_frames = batch_frames.reshape(B * num_frames, *batch_frames.shape[2:])
        t_frames    = frames_to_tensor(flat_frames, device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                feats = encoder(t_frames)

        features [start:end] = feats.view(B, num_frames, -1).float().cpu().numpy()
        landmarks[start:end] = batch_landmarks.astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Saving to %s …", out_path)
    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("features",  data=features,   compression="gzip", chunks=True)
        dst.create_dataset("landmarks", data=landmarks,  compression="gzip", chunks=True)
        dst.create_dataset("labels",    data=labels_all)
        dst.attrs["backbone"]   = "efficientnet"
        dst.attrs["visual_dim"] = visual_dim
        dst.attrs["num_frames"] = num_frames
        dst.attrs["source"]     = str(npy)

    size_mb = out_path.stat().st_size / 1e6
    log.info("Done — %d samples, %.0f MB → %s", n_total, size_mb, out_path)
    log.info("  features  : %s", features.shape)
    log.info("  landmarks : %s", landmarks.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache EfficientNet visual features")
    parser.add_argument("--npy_dir",    default=str(config.PROCESSED_NPY_DIR),
                        help="Carpeta con frames.npy/landmarks.npy/labels.npy")
    parser.add_argument("--output",     default=str(config.FEATURES_CACHE_PATH),
                        help="Ruta del HDF5 de salida")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    log.info("npy_dir: %s  →  output: %s", args.npy_dir, args.output)
    cache_features(args)
