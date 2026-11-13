"""
visualize_landmarks.py
----------------------
Muestra los fotogramas muestreados de un vídeo aleatorio por clase con las
landmarks de FaceMesh superpuestas.

El preprocesado es idéntico al de preprocess.py (muestreo equiespaciado,
detección de cara, recorte por bounding box de landmarks, resize).

Genera una imagen con 3 pares de filas (una por clase):
  • fila superior de cada par : face crop limpio
  • fila inferior de cada par : mismo crop con la malla de FaceMesh

Uso
---
    python scripts/visualize_landmarks.py
    python scripts/visualize_landmarks.py --frames 8 --out results/landmarks_preview.png
    python scripts/visualize_landmarks.py --class 1 --video data/raw/1/mi_video.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
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

mp_face_mesh_mod = mp.solutions.face_mesh


def pick_random_video(cls: int) -> Path:
    cls_dir = RAW_DIR / str(cls)
    videos = [p for ext in VIDEO_EXTS for p in cls_dir.glob(f"*{ext}")]
    if not videos:
        sys.exit(f"No se encontraron vídeos en {cls_dir}")
    return videos[np.random.randint(len(videos))]


def extract_frames_with_landmarks(
    video_path: Path,
    num_frames: int,
    frame_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Devuelve lista de (crop_rgb, pts_xy) donde:
      crop_rgb : (frame_size, frame_size, 3) uint8 RGB — face crop limpio
      pts_xy   : (N_landmarks, 2) float32 — coordenadas en píxeles del crop
                 (misma normalización que preprocess.py: relativas al crop resizado)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        cap.release()
        return []

    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    indices_sorted = sorted(indices)

    results_out: list[tuple[np.ndarray, np.ndarray]] = []
    last_crop: np.ndarray | None = None
    last_pts:  np.ndarray | None = None

    with mp_face_mesh_mod.FaceMesh(
        static_image_mode=False,
        refine_landmarks=config.MP_REFINE_LANDMARKS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
        max_num_faces=1,
    ) as face_mesh:
        frame_num = 0
        while frame_num <= indices_sorted[-1]:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if frame_num in indices:
                if result.multi_face_landmarks:
                    face_lm = result.multi_face_landmarks[0]

                    # Bounding box idéntico al de preprocess.py
                    x_px = np.array([lm.x for lm in face_lm.landmark]) * w
                    y_px = np.array([lm.y for lm in face_lm.landmark]) * h
                    x0 = max(0, int(x_px.min()))
                    y0 = max(0, int(y_px.min()))
                    x1 = min(w, int(x_px.max()))
                    y1 = min(h, int(y_px.max()))
                    crop_w = max(x1 - x0, 1)
                    crop_h = max(y1 - y0, 1)

                    crop = rgb[y0:y1, x0:x1]
                    if crop.size == 0:
                        crop = rgb
                        x0 = y0 = 0
                        crop_w, crop_h = w, h
                    crop = cv2.resize(crop, (frame_size, frame_size))

                    # Coordenadas normalizadas al crop, luego a píxeles del crop resizado.
                    # Igual que preprocess.py pero sin aplanar, para poder hacer scatter.
                    xs = (x_px - x0) / crop_w * frame_size
                    ys = (y_px - y0) / crop_h * frame_size
                    pts = np.stack([xs, ys], axis=1).astype(np.float32)

                    last_crop = crop
                    last_pts  = pts
                    results_out.append((crop, pts))

                elif last_crop is not None:
                    results_out.append((last_crop, last_pts))

            frame_num += 1

    cap.release()
    return results_out


def main():
    parser = argparse.ArgumentParser(
        description="Visualiza fotogramas preprocesados con landmarks de FaceMesh"
    )
    parser.add_argument("--frames", type=int, default=config.NUM_FRAMES,
                        help=f"Fotogramas a mostrar (default: {config.NUM_FRAMES})")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "landmarks_preview.png"),
                        help="Ruta de la imagen de salida")
    parser.add_argument("--video", type=str, default=None,
                        help="Usar este vídeo concreto en vez de uno aleatorio")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames   = args.frames
    frame_size = config.FRAME_SIZE
    blank      = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    # 2 filas por clase (crop + landmarks)
    n_rows = len(CLASSES) * 2
    fig, axes = plt.subplots(
        n_rows, n_frames,
        figsize=(n_frames * 1.8, n_rows * 2.0),
    )
    fig.suptitle("Fotogramas muestreados + landmarks FaceMesh — vídeo aleatorio por clase",
                 fontsize=12, y=1.01)

    for cls_idx, cls in enumerate(CLASSES):
        video = Path(args.video) if args.video else pick_random_video(cls)
        print(f"Clase {cls}: {video.name}")

        frames_lm = extract_frames_with_landmarks(video, n_frames, frame_size)

        row_clean = cls_idx * 2
        row_lm    = cls_idx * 2 + 1

        for col in range(n_frames):
            ax_clean = axes[row_clean][col]
            ax_lm    = axes[row_lm][col]

            if col < len(frames_lm):
                crop, pts = frames_lm[col]
                ax_clean.imshow(crop)
                ax_lm.imshow(crop)
                ax_lm.scatter(pts[:, 0], pts[:, 1], s=6, c="lime",
                              linewidths=0, rasterized=True)
                ax_lm.set_xlim(0, frame_size)
                ax_lm.set_ylim(frame_size, 0)
            else:
                ax_clean.imshow(blank)
                ax_lm.imshow(blank)

            ax_clean.axis("off")
            ax_lm.axis("off")

            if col == 0:
                ax_clean.set_ylabel(
                    f"{CLASS_NAMES[cls]}\ncrop",
                    fontsize=7, rotation=0, labelpad=85, va="center",
                )
                ax_lm.set_ylabel(
                    "landmarks",
                    fontsize=7, rotation=0, labelpad=85, va="center",
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nImagen guardada en: {out_path}")


if __name__ == "__main__":
    main()
