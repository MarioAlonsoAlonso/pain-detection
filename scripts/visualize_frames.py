"""
Cuadrícula de clips aleatorios del dataset preprocesado (lo que entra al modelo)
y la guarda en results/sample_frames.png.

Uso:
    python scripts/visualize_frames.py
    python scripts/visualize_frames.py --n_samples 5 --seed 4425
    python scripts/visualize_frames.py --out results/mis_frames.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

CLASS_NAMES  = {0: "Sin dolor", 1: "Dolor leve", 2: "Dolor intenso"}
CLASS_COLORS = {0: "#2ecc71",   1: "#f39c12",    2: "#e74c3c"}


def show_samples(
    frames: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    cols: int,
    seed: int,
) -> plt.Figure:
    """
    Dibuja n_samples clips aleatorios. Por cada clip muestra todos los frames
    en una fila, con el título y borde coloreados según la clase.

    frames : (N, T, H, W, 3) uint8 BGR
    labels : (N,) int
    """
    N, T = frames.shape[:2]
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(N, size=min(n_samples, N), replace=False)

    fig, axes = plt.subplots(
        n_samples, T,
        figsize=(cols * T / 8, 2.4 * n_samples),
        squeeze=False,
    )
    fig.suptitle(
        f"{n_samples} muestras aleatorias  ·  {T} frames por clip",
        fontsize=11, y=1.01,
    )

    for row, sample_idx in enumerate(idxs):
        cls   = int(labels[sample_idx])
        color = CLASS_COLORS[cls]
        label = CLASS_NAMES[cls]

        for t in range(T):
            ax  = axes[row, t]
            img = cv2.cvtColor(frames[sample_idx, t], cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis("off")

            # Título solo en el primer frame de cada fila
            if t == 0:
                ax.set_title(
                    f"#{sample_idx}  {label}",
                    fontsize=7.5, color=color, loc="left",
                )

            # Borde de color según clase
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
                spine.set_visible(True)

    plt.tight_layout()
    return fig


def main(npy_dir: Path, n_samples: int, cols: int, seed: int | None, out: Path) -> None:
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 10_000)
    print(f"Semilla: {seed}   (repite esta selección con --seed {seed})")
    print("Cargando datos…")
    frames = np.load(npy_dir / "frames.npy", mmap_mode="r")  # (N, T, H, W, 3)
    labels = np.load(npy_dir / "labels.npy")                  # (N,)

    N, T, H, W = frames.shape[:4]
    print(f"Dataset: {N} muestras × {T} frames  ({H}×{W} px)")
    for cls in range(config.NUM_CLASSES):
        n = int((labels == cls).sum())
        print(f"  Clase {cls} {CLASS_NAMES[cls]}: {n} muestras")

    fig = show_samples(frames, labels, n_samples=n_samples, cols=cols, seed=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza frames preprocesados del dataset")
    parser.add_argument("--npy_dir",   default=str(config.PROCESSED_NPY_DIR))
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Número de clips a mostrar (default: 6)")
    parser.add_argument("--cols",      type=int, default=8,
                        help="Ancho de cada miniatura en unidades relativas (default: 8)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Semilla para repetir una selección concreta "
                             "(por defecto, una distinta en cada ejecución)")
    parser.add_argument("--out",       default=str(config.RESULTS_DIR / "sample_frames.png"),
                        help="Ruta de la imagen (default: results/sample_frames.png)")
    args = parser.parse_args()

    main(Path(args.npy_dir), args.n_samples, args.cols, args.seed, Path(args.out))
