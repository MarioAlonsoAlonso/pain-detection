"""
Histogramas de los valores z de la barbilla (lm 152) y la frente (lm 10).

Uso:
    python scripts/visualize_z_landmarks.py
    python scripts/visualize_z_landmarks.py --npy_dir data/processed/dataset_npy
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

FOREHEAD_IDX = 10
CHIN_IDX     = 152
NUM_LM       = config.NUM_LANDMARKS  # 478


def main(npy_dir: Path) -> None:
    print("Cargando landmarks…")
    landmarks = np.load(npy_dir / "landmarks.npy")   # (N, T, 1434)
    N, T, _   = landmarks.shape
    print(f"Dataset: {N} muestras × {T} frames")

    lm_3d = landmarks.reshape(N, T, NUM_LM, 3)       # (N, T, 478, 3)

    # z por muestra (media sobre T frames)
    z_forehead = lm_3d[:, :, FOREHEAD_IDX, 2].mean(axis=1)   # (N,)
    z_chin     = lm_3d[:, :, CHIN_IDX,     2].mean(axis=1)   # (N,)

    # Estadísticas
    for name, z in [("Frente (lm 10)", z_forehead), ("Barbilla (lm 152)", z_chin)]:
        print(f"\n{name}")
        print(f"  Media:   {z.mean():+.4f}")
        print(f"  Mediana: {np.median(z):+.4f}")
        print(f"  Std:     {z.std():.4f}")
        print(f"  Rango:   [{z.min():+.4f}, {z.max():+.4f}]")

    # Figura
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribución de valores z (profundidad MediaPipe) por landmark", fontsize=12)

    for ax, z, title, color in [
        (axes[0], z_forehead, f"Frente — landmark {FOREHEAD_IDX}", "#4fc3f7"),
        (axes[1], z_chin,     f"Barbilla — landmark {CHIN_IDX}",   "#e57373"),
    ]:
        ax.hist(z, bins=40, color=color, alpha=0.80, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(z), color="black", linestyle="--", linewidth=1.4,
                   label=f"Mediana: {np.median(z):+.4f}")
        ax.axvline(z.mean(), color="gray", linestyle=":", linewidth=1.2,
                   label=f"Media: {z.mean():+.4f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("z (profundidad relativa MediaPipe)", fontsize=10)
        ax.set_ylabel("Nº muestras", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out = Path(config.RESULTS_DIR) / "z_landmarks_histogram.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGuardado en {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", default=str(config.PROCESSED_NPY_DIR))
    args = parser.parse_args()
    main(Path(args.npy_dir))
