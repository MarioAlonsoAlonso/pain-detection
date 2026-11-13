"""
Inclinación de la cámara respecto a la cara (pitch) en el dataset.

Pitch = z[barbilla] - z[frente]   (landmarks 152 y 10 de MediaPipe)
  Un z menor significa más cerca de la cámara, así que un pitch negativo
  indica que la cámara está más cerca de la barbilla que de la frente.

Calcula la inclinación media de cada vídeo y dibuja su histograma con la
mediana (Ilustración 4 de la memoria). La mediana es el valor de referencia
DATASET_PITCH_MEDIAN que usa inference.py para la barra de inclinación.

Uso:
    python scripts/visualize_pitch.py
    python scripts/visualize_pitch.py --npy_dir data/processed/dataset_npy
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

FOREHEAD_IDX = 10    # frente superior centro
CHIN_IDX     = 152   # barbilla inferior centro
NUM_LM       = config.NUM_LANDMARKS   # 478


def compute_pitch(landmarks: np.ndarray) -> np.ndarray:
    """landmarks (N, T, D) → (N,) inclinación media de cada vídeo, con signo."""
    lm_3d = landmarks.reshape(landmarks.shape[0], landmarks.shape[1], NUM_LM, 3)
    pitch = lm_3d[:, :, CHIN_IDX, 2] - lm_3d[:, :, FOREHEAD_IDX, 2]   # (N, T)
    return pitch.mean(axis=1)


def plot_histogram(pitch: np.ndarray) -> plt.Figure:
    median = float(np.median(pitch))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pitch, bins=30, alpha=0.70, color="#4a90d9", edgecolor="white", linewidth=0.4,
            label=f"Todas las muestras  (n={len(pitch)})")
    ax.axvline(median, color="black", linestyle="--", linewidth=1.3, alpha=0.8,
               label=f"Mediana  ({median:.3f})")
    ax.set_xlabel("Pitch  (z_barbilla − z_frente)", fontsize=11)
    ax.set_ylabel("Nº muestras", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def main(npy_dir: Path, out: Path) -> None:
    lm_path = npy_dir / "landmarks.npy"
    if not lm_path.exists():
        sys.exit(f"No se encontró {lm_path} — ejecuta antes preprocess.py")

    landmarks = np.load(lm_path, mmap_mode="r")
    print(f"Dataset: {landmarks.shape[0]} muestras × {landmarks.shape[1]} frames")
    pitch = compute_pitch(np.asarray(landmarks))

    print(f"\nPitch medio por vídeo (z_barbilla − z_frente):")
    print(f"  Mediana: {np.median(pitch):+.4f}   ← DATASET_PITCH_MEDIAN de inference.py")
    print(f"  Media:   {pitch.mean():+.4f}")
    print(f"  Std:      {pitch.std():.4f}")
    print(f"  Rango:   [{pitch.min():+.4f}, {pitch.max():+.4f}]")

    out.parent.mkdir(parents=True, exist_ok=True)
    plot_histogram(pitch).savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nHistograma → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histograma de la inclinación (pitch) del dataset")
    parser.add_argument("--npy_dir", default=str(config.PROCESSED_NPY_DIR))
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "pitch_histogram.png"))
    args = parser.parse_args()
    main(Path(args.npy_dir), Path(args.out))
