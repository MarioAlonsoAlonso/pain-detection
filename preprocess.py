"""
Preprocessing pipeline: raw videos → NPY dataset.

Usage
-----
    python preprocess.py
    python preprocess.py --raw_dir data/raw --output_dir data/processed/dataset_npy --num_frames 16

Expected raw data layout
------------------------
    data/raw/
        0/   ← videos labelled class 0  (sin dolor)
        1/   ← videos labelled class 1  (dolor leve)
        2/   ← videos labelled class 2  (dolor intenso)

Output (data/processed/dataset_npy/)
------------------------------------
    frames.npy     : (N, T, H, W, 3)  uint8  — BGR face crops
    landmarks.npy  : (N, T, D)        float32 — normalised landmark coords
    labels.npy     : (N,)             int32
    metadata.json  : num_samples, num_frames, frame_size, landmark_dim
"""

import argparse
import logging
import sys
from pathlib import Path

import json

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


# ─── Video scanning ───────────────────────────────────────────────────────────

def find_videos(raw_dir: Path) -> list[tuple[Path, int]]:
    """Return (video_path, label) for every video in raw_dir/{0,1,2}/."""
    entries: list[tuple[Path, int]] = []
    for label in range(config.NUM_CLASSES):
        class_dir = raw_dir / str(label)
        if not class_dir.exists():
            log.warning("Class directory not found: %s", class_dir)
            continue
        for ext in VIDEO_EXTENSIONS:
            for p in sorted(class_dir.glob(f"*{ext}")):
                entries.append((p, label))
    return entries


# ─── Per-video processing ────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    face_mesh,
    num_frames: int,
    frame_size: int,
    landmark_dim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract *num_frames* equispaced face crops + landmark vectors from a video.

    Parameters
    ----------
    video_path   : path to the video file
    face_mesh    : an already-initialised (open) FaceMesh context
    num_frames   : how many frames to sample
    frame_size   : resize crops to (frame_size × frame_size)
    landmark_dim : expected flattened landmark vector length

    Returns
    -------
    (frames, landmarks) or None on failure.
      frames    : (num_frames, frame_size, frame_size, 3) uint8 BGR
      landmarks : (num_frames, landmark_dim) float32
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open: %s", video_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        log.warning(
            "Too few frames (%d < %d) in: %s — skipped", total, num_frames, video_path
        )
        cap.release()
        return None

    # Equispaced target indices (0-based)
    target_set = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    target_sorted = sorted(target_set)

    out_frames: list[np.ndarray] = []
    out_lm:     list[np.ndarray] = []

    last_crop = None
    last_lm   = None
    frame_num = 0

    while frame_num <= target_sorted[-1]:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Always pass to FaceMesh (maintains tracking state even on non-target frames)
        results = face_mesh.process(rgb)

        if frame_num in target_set:
            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0]

                # Face crop from landmark bounding box
                x_px = np.array([lm.x for lm in face_lm.landmark]) * w
                y_px = np.array([lm.y for lm in face_lm.landmark]) * h
                x0   = max(0, int(x_px.min()))
                y0   = max(0, int(y_px.min()))
                x1   = min(w, int(x_px.max()))
                y1   = min(h, int(y_px.max()))

                crop_w = x1 - x0
                crop_h = y1 - y0

                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = frame
                    x0 = y0 = 0
                    crop_w, crop_h = w, h
                crop = cv2.resize(crop, (frame_size, frame_size))

                # Renormalizar landmarks al sistema de coordenadas del crop:
                # lm.x/y están en [0,1] relativo al frame completo; los
                # convertimos a [0,1] relativo al crop para que sean
                # coherentes con lo que ve EfficientNet (la cara recortada).
                lm_raw = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_lm.landmark],
                    dtype=np.float32,
                )
                lm_raw[:, 0] = (lm_raw[:, 0] * w - x0) / max(crop_w, 1)
                lm_raw[:, 1] = (lm_raw[:, 1] * h - y0) / max(crop_h, 1)
                # z ya es relativo (profundidad normalizada por MediaPipe), no cambia
                lm_arr = lm_raw.flatten()

                # Clip to expected size (handles refine_landmarks quirks)
                if lm_arr.size > landmark_dim:
                    lm_arr = lm_arr[:landmark_dim]
                elif lm_arr.size < landmark_dim:
                    lm_arr = np.pad(lm_arr, (0, landmark_dim - lm_arr.size))

                last_crop = crop
                last_lm   = lm_arr
                out_frames.append(crop)
                out_lm.append(lm_arr)
            else:
                # No face detected — reuse last valid detection
                if last_crop is not None:
                    log.debug("Frame %d: no face — reusing previous. (%s)", frame_num, video_path.name)
                    out_frames.append(last_crop)
                    out_lm.append(last_lm)
                else:
                    log.warning("No face in first target frame of %s — skipped", video_path.name)
                    cap.release()
                    return None

        frame_num += 1

    cap.release()

    if len(out_frames) != num_frames:
        log.warning(
            "Expected %d frames, got %d in %s — skipped", num_frames, len(out_frames), video_path.name
        )
        return None

    return np.stack(out_frames), np.stack(out_lm)


# ─── Main preprocessing pipeline ─────────────────────────────────────────────

def preprocess_dataset(
    raw_dir:          str  = str(config.RAW_DATA_DIR),
    output_dir:       str  = str(config.PROCESSED_NPY_DIR),
    num_frames:       int  = config.NUM_FRAMES,
    frame_size:       int  = config.FRAME_SIZE,
):
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    entries = find_videos(Path(raw_dir))   # (video_path, label)

    if not entries:
        log.error("No videos found (raw_dir=%s)", raw_dir)
        return

    log.info("Found %d videos.  Processing…", len(entries))
    landmark_dim = config.LANDMARK_DIM

    # Pre-allocate a disk-backed memmap so frames are never held in RAM.
    # We over-allocate for the worst case (all videos valid) and trim later.
    N_max = len(entries)
    frames_tmp_path = output_dir_p / "_frames_tmp.bin"
    frames_mm = np.memmap(
        frames_tmp_path,
        dtype=np.uint8,
        mode="w+",
        shape=(N_max, num_frames, frame_size, frame_size, 3),
    )
    frame_count = 0
    all_landmarks: list[np.ndarray] = []
    all_labels:    list[int]        = []

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=config.MP_STATIC_IMAGE_MODE,
        refine_landmarks=config.MP_REFINE_LANDMARKS,
        min_detection_confidence=config.MP_MIN_DETECTION_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
        max_num_faces=1,
    ) as face_mesh:
        for video_path, label in tqdm(entries, desc="Preprocessing", unit="video"):
            result = process_video(
                video_path, face_mesh, num_frames, frame_size, landmark_dim
            )
            if result is None:
                continue

            frames, landmarks = result

            frames_mm[frame_count] = frames
            frame_count += 1
            all_landmarks.append(landmarks)
            all_labels.append(label)

    if frame_count == 0:
        log.error("No videos were successfully processed. Aborting.")
        frames_tmp_path.unlink(missing_ok=True)
        return

    n = frame_count
    log.info("Successfully processed %d / %d videos. Saving…", n, len(entries))

    frames_arr    = frames_mm[:n]              # memmap view — no RAM copy
    landmarks_arr = np.stack(all_landmarks)    # (N, T, D)
    labels_arr    = np.array(all_labels,   dtype=np.int32)

    # ── Save NPY (fast memmap access during training) ──────────────────────────
    np.save(output_dir_p / "frames.npy",    frames_arr)
    np.save(output_dir_p / "landmarks.npy", landmarks_arr)
    np.save(output_dir_p / "labels.npy",    labels_arr)
    (output_dir_p / "metadata.json").write_text(
        json.dumps({
            "num_samples":  n,
            "num_frames":   num_frames,
            "frame_size":   frame_size,
            "landmark_dim": landmark_dim,
        }, indent=2)
    )
    log.info("Dataset saved → %s", output_dir_p)

    del frames_arr, frames_mm
    frames_tmp_path.unlink(missing_ok=True)

    # Class distribution report
    for cls in range(config.NUM_CLASSES):
        cnt = int((labels_arr == cls).sum())
        log.info("  Class %d : %4d samples  (%.1f %%)", cls, cnt, cnt / n * 100)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pain-detection video dataset")
    parser.add_argument("--raw_dir",    default=str(config.RAW_DATA_DIR),
                        help="Directory with class sub-folders 0/1/2")
    parser.add_argument("--output_dir", default=str(config.PROCESSED_NPY_DIR),
                        help="Output directory for .npy files")
    parser.add_argument("--num_frames", type=int, default=config.NUM_FRAMES,
                        help="Frames to sample per video")
    parser.add_argument("--frame_size", type=int, default=config.FRAME_SIZE,
                        help="Spatial resolution of face crops")
    args = parser.parse_args()

    preprocess_dataset(args.raw_dir, args.output_dir, args.num_frames, args.frame_size)
