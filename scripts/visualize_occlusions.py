"""
Visualiza los tipos de oclusión facial aplicados a muestras del dataset.

Uso
---
    python scripts/visualize_occlusions.py                        # 10 muestras aleatorias
    python scripts/visualize_occlusions.py --frame_t 8            # frame del clip (0-15)
    python scripts/visualize_occlusions.py --n_samples 6          # número de muestras
    python scripts/visualize_occlusions.py --seed 7               # semilla aleatoria
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

# Formato: (top_ratio, bot_ratio, left_ratio, right_ratio, probabilidad)
REGIONS = {
    "Entrecejo (10%)": (0.00, 0.35, 0.35, 0.65, 0.10),
    "Boca (20%)":      (0.40, 1.00, 0.15, 0.85, 0.20),
}
REGION_COLORS = {
    "Entrecejo (10%)": "#4fc3f7",
    "Boca (20%)":      "#e57373",
}
CLASS_NAMES = {0: "Sin dolor", 1: "Dolor leve", 2: "Dolor intenso"}


def load_frame(frames_mmap: np.ndarray, sample_idx: int, frame_t: int) -> np.ndarray:
    return cv2.cvtColor(frames_mmap[sample_idx, frame_t], cv2.COLOR_BGR2RGB)


def apply_region(image: np.ndarray, top: float, bot: float,
                 left: float, right: float) -> np.ndarray:
    H, W = image.shape[:2]
    out  = image.copy()
    out[int(H * top):int(H * bot), int(W * left):int(W * right)] = 0
    return out


def draw_region_outline(ax, image: np.ndarray, top: float, bot: float,
                        left: float, right: float, color: str):
    H, W = image.shape[:2]
    y0 = int(H * top);  y1 = int(H * bot)
    x0 = int(W * left); x1 = int(W * right)
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor=color, facecolor="none",
    ))


def run(args):
    npy        = Path(config.PROCESSED_NPY_DIR)
    frames_mm  = np.load(npy / "frames.npy", mmap_mode="r")
    labels_arr = np.load(npy / "labels.npy")
    N          = len(labels_arr)

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(N, size=min(args.n_samples, N), replace=False)

    region_names = list(REGIONS.keys())
    # Columnas: original | entrecejo | boca
    col_titles = ["Original"] + region_names
    n_cols     = len(col_titles)
    n_rows     = len(indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{n_rows} muestras  ·  frame t={args.frame_t}  |  "
        "Entrecejo: 10%   Boca: 20%   (mutuamente excluyentes)",
        color="white", fontsize=11, y=1.005,
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sample_idx in enumerate(indices):
        image = load_frame(frames_mm, int(sample_idx), args.frame_t)
        clase = int(labels_arr[sample_idx])
        H, W  = image.shape[:2]

        # ── Col 0: original con contornos ────────────────────────────────────
        ax = axes[row, 0]
        ax.imshow(image)
        ax.set_title(
            f"#{sample_idx} · {CLASS_NAMES[clase]}",
            color="white", fontsize=8,
        )
        ax.axis("off")
        for name, (top, bot, left, right, _) in REGIONS.items():
            draw_region_outline(ax, image, top, bot, left, right, REGION_COLORS[name])

        # ── Cols 1-N: regiones ocluidas ───────────────────────────────────────
        for col, name in enumerate(region_names, start=1):
            top, bot, left, right, _ = REGIONS[name]
            ax = axes[row, col]
            occluded = apply_region(image, top, bot, left, right)
            ax.imshow(occluded)
            ax.axis("off")

            y0 = int(H * top);  y1 = int(H * bot)
            x0 = int(W * left); x1 = int(W * right)
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor=REGION_COLORS[name],
                facecolor=REGION_COLORS[name], alpha=0.15,
            ))
            ax.text((x0 + x1) // 2, (y0 + y1) // 2, "OCLUIDO",
                    color="white", fontsize=7, ha="center", va="center",
                    fontweight="bold")

            # Título de columna solo en la primera fila
            if row == 0:
                ax.set_title(name, color=REGION_COLORS[name], fontsize=9,
                             fontweight="bold")

    plt.tight_layout()
    out_path = Path(config.RESULTS_DIR) / f"occlusions_grid_t{args.frame_t}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Guardado → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizar tipos de oclusión facial")
    parser.add_argument("--frame_t",    type=int, default=8,
                        help="Frame del clip a mostrar (0-15)")
    parser.add_argument("--n_samples",  type=int, default=10,
                        help="Número de muestras aleatorias (default: 10)")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Semilla aleatoria")
    args = parser.parse_args()
    run(args)
