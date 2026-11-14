"""
Calcula el ratio medio de tamaño facial (face_width / frame_width) en los vídeos raw.

Este valor sirve como referencia de distancia cámara-cara en el dataset de
entrenamiento. Úsalo para configurar DATASET_FACE_RATIO en inference.py.

Uso:
    python scripts/compute_face_size.py
    python scripts/compute_face_size.py --raw_dir data/raw --frames_per_video 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def find_videos(raw_dir: Path) -> list[tuple[Path, int]]:
    entries = []
    for label in range(config.NUM_CLASSES):
        class_dir = raw_dir / str(label)
        if not class_dir.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            for p in sorted(class_dir.glob(f"*{ext}")):
                entries.append((p, label))
    return entries


def sample_ratios(video_path: Path, face_mesh, frames_per_video: int) -> list[float]:
    """Devuelve una lista de ratios face_width/frame_width para frames muestreados."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < frames_per_video:
        cap.release()
        return []

    targets = set(np.linspace(0, total - 1, frames_per_video, dtype=int).tolist())
    ratios  = []
    frame_num = 0

    while frame_num <= max(targets):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if frame_num in targets:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm    = results.multi_face_landmarks[0]
                x_px  = np.array([p.x for p in lm.landmark]) * w
                face_w = x_px.max() - x_px.min()
                ratios.append(float(face_w / w))

        frame_num += 1

    cap.release()
    return ratios


def main(raw_dir: Path, frames_per_video: int, max_videos: int) -> None:
    entries = find_videos(raw_dir)
    if not entries:
        print(f"No se encontraron vídeos en {raw_dir}")
        return

    if max_videos and len(entries) > max_videos:
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(entries), size=max_videos, replace=False)
        entries = [entries[i] for i in sorted(indices)]

    print(f"Procesando {len(entries)} vídeos ({frames_per_video} frames cada uno)…\n")

    all_ratios:   list[float] = []
    class_ratios: dict[int, list[float]] = {c: [] for c in range(config.NUM_CLASSES)}

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=config.MP_REFINE_LANDMARKS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
        max_num_faces=1,
    ) as face_mesh:
        for video_path, label in tqdm(entries, desc="Vídeos", unit="vid"):
            r = sample_ratios(video_path, face_mesh, frames_per_video)
            all_ratios.extend(r)
            class_ratios[label].extend(r)

    if not all_ratios:
        print("No se detectaron caras en ningún vídeo.")
        return

    arr = np.array(all_ratios)
    print("── Estadísticas globales (face_size / frame_size) ──────────────────")
    print(f"  Media:   {arr.mean():.4f}")
    print(f"  Mediana: {np.median(arr):.4f}")
    print(f"  Std:     {arr.std():.4f}")
    print(f"  Rango:   [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  N muestras (frames×ejes): {len(arr)}")

    for cls in range(config.NUM_CLASSES):
        r = np.array(class_ratios[cls])
        if len(r) == 0:
            continue
        names = {0: "Sin dolor", 1: "Dolor leve", 2: "Dolor intenso"}
        print(f"\n  Clase {cls} {names[cls]}: media={r.mean():.4f}  "
              f"mediana={np.median(r):.4f}  std={r.std():.4f}")

    mean_val = arr.mean()
    print("\n══ Valor recomendado para inference.py ══════════════════════════════")
    print(f"  DATASET_FACE_RATIO = {mean_val:.4f}")
    print("  (pon este valor en la constante DATASET_FACE_RATIO de inference.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula el ratio de tamaño facial medio en los vídeos raw"
    )
    parser.add_argument("--raw_dir",          default=str(config.RAW_DATA_DIR))
    parser.add_argument("--frames_per_video", type=int, default=5,
                        help="Frames a muestrear por vídeo (default: 5)")
    parser.add_argument("--max_videos", type=int, default=100,
                        help="Máximo de vídeos a procesar (default: 100)")
    args = parser.parse_args()
    main(Path(args.raw_dir), args.frames_per_video, args.max_videos)
