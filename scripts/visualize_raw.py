"""
visualize_raw.py
----------------
Muestra fotogramas sin procesar de un vídeo aleatorio de cada clase.
No se aplica ningún preprocesado: los frames se extraen tal cual del vídeo.

Uso
---
    python scripts/visualize_raw.py
    python scripts/visualize_raw.py --frames 8 --out results/raw_preview.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

RAW_DIR     = config.RAW_DATA_DIR
RESULTS_DIR = config.RESULTS_DIR
CLASSES     = [0, 1, 2]
CLASS_NAMES = {0: "Clase 0 — sin dolor", 1: "Clase 1 — moderado", 2: "Clase 2 — intenso"}
VIDEO_EXTS  = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def pick_random_video(cls: int) -> Path:
    cls_dir = RAW_DIR / str(cls)
    videos  = [p for ext in VIDEO_EXTS for p in cls_dir.glob(f"*{ext}")]
    if not videos:
        sys.exit(f"No se encontraron vídeos en {cls_dir}")
    return videos[np.random.randint(len(videos))]


def extract_raw_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = set(np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist())
    frames  = []
    frame_num = 0

    while frame_num <= max(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_num += 1

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=config.NUM_FRAMES,
                        help=f"Fotogramas a mostrar por vídeo (default: {config.NUM_FRAMES})")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "raw_preview.png"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    n_frames = args.frames
    fig, axes = plt.subplots(
        len(CLASSES), n_frames,
        figsize=(n_frames * 2.0, len(CLASSES) * 2.0),
    )
    fig.suptitle("Fotogramas sin procesar — vídeo aleatorio por clase", fontsize=12)

    for row, cls in enumerate(CLASSES):
        video  = pick_random_video(cls)
        print(f"Clase {cls}: {video.name}")
        frames = extract_raw_frames(video, n_frames)

        for col in range(n_frames):
            ax = axes[row][col]
            if col < len(frames):
                ax.imshow(frames[col])
            else:
                ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
            ax.axis("off")
            if col == 0:
                ax.set_title(CLASS_NAMES[cls], fontsize=8, loc="left", pad=3)

    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nImagen guardada en: {out_path}")


if __name__ == "__main__":
    main()
