"""
Real-time pain-detection inference using a webcam.

Usage
-----
    # Default webcam (index 0)
    python inference.py --checkpoint checkpoints/best_model.pth

    # Specific webcam
    python inference.py --source 1

    # Use majority voting instead of mean-pooled softmax
    python inference.py --majority_vote

Controls
--------
    q   — quit
    r   — reset sliding-window buffer
"""

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from src import config
from src.model import PainDetectionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_to_tensor = transforms.ToTensor()
_normalise = transforms.Normalize(config.FRAME_MEAN, config.FRAME_STD)

# Pain-level display colours (BGR)
PAIN_COLOUR = {
    0: (0,  200,  0),    # green  — no pain
    1: (0,  200, 255),   # yellow — mild pain
    2: (0,    0, 220),   # red    — severe pain
}
PAIN_LABEL = {
    0: "Sin dolor  (0)",
    1: "Dolor leve (1)",
    2: "Dolor intenso (2)",
}

# Pitch mediano del dataset de entrenamiento (z_barbilla - z_frente).
# La barra de pitch en pantalla muestra la desviación respecto a este valor.
DATASET_PITCH_MEDIAN = -0.085

# Ratio medio de tamaño facial en el dataset de entrenamiento (face_width / frame_width).
# Calculado con scripts/compute_face_size.py sobre los vídeos raw (1280x720).
# Mayor ratio = cara más grande = cámara más cerca.
# Solo afecta a la barra "Dist" de la interfaz, no a la predicción.
DATASET_FACE_RATIO = 0.266


# ─── Real-time inference engine ───────────────────────────────────────────────

class InferenceEngine:
    """
    Maintains a sliding-window buffer of feature tensors extracted from live
    camera frames.  When the buffer is full a pain-level prediction is updated
    with every new frame.

    Parameters
    ----------
    model         : trained PainDetectionModel in eval mode
    device        : torch device
    buffer_size   : number of frames in the sliding window (= config.NUM_FRAMES)
    video_duration: duración en segundos de los clips de entrenamiento (default 4s)
    majority_vote : if True, aggregate per-frame predictions via majority vote;
                    if False (default), average softmax probabilities
    """

    def __init__(
        self,
        model: PainDetectionModel,
        device: torch.device,
        buffer_size: int = config.NUM_FRAMES,
        video_duration: float = 4.0,
        majority_vote: bool = False,
        smooth_alpha: float = 0.35,
    ):
        self.model              = model.eval()
        self.device             = device
        self.buffer_size        = buffer_size
        self.majority_vote      = majority_vote
        self.smooth_alpha       = smooth_alpha   # peso del frame nuevo en la EMA (1.0 = sin suavizado)

        # Intervalo entre frames para replicar el muestreo del entrenamiento.
        # Entrenamiento: 16 frames de 4s → 1 frame cada 250ms.
        self._sample_interval = video_duration / buffer_size   # segundos
        self._last_sample_t   = 0.0                            # timestamp último frame añadido

        # Circular buffers — store pre-processed tensors (cheap to keep in RAM)
        self.frame_buf = collections.deque(maxlen=buffer_size)
        self.lm_buf    = collections.deque(maxlen=buffer_size)

        # Latest stable prediction
        self.pred       = None          # int (0/1/2)
        self.conf       = None          # float
        self.probs      = None          # list[float] — probabilidades suavizadas (mostradas en pantalla)
        self.face_bbox  = None          # (x0,y0,x1,y1) in pixel coords
        self.face_pitch      = 0.0   # z_barbilla - z_frente: >0 cámara arriba, <0 cámara abajo
        self.face_size_ratio = 0.0   # face_width / frame_width: proxy de distancia cámara-cara
        self._ema_probs = None          # np.ndarray — acumulador EMA interno

        # MediaPipe Face Mesh (kept open for the lifetime of the engine)
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=config.MP_REFINE_LANDMARKS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONF,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
            max_num_faces=1,
        )

    # ------------------------------------------------------------------
    def close(self):
        self._face_mesh.close()

    # ------------------------------------------------------------------
    def reset_buffer(self):
        self.frame_buf.clear()
        self.lm_buf.clear()
        self.pred = self.conf = self.probs = self.face_bbox = None
        self._ema_probs = None

    # ------------------------------------------------------------------
    def _preprocess(self, frame_bgr: np.ndarray):
        """
        Detect face, extract MediaPipe landmarks, build model-ready tensors.

        Returns (face_tensor, lm_tensor, bbox, crop_bgr, pitch, face_ratio),
        or a tuple of six None if no face is detected.
          face_tensor : (3, H, W) normalised
          lm_tensor   : (landmark_dim,) float32
          bbox        : (x0, y0, x1, y1) in pixel coordinates
          crop_bgr    : (H, W, 3) uint8 face crop
          pitch       : z_barbilla - z_frente
          face_ratio  : face_width / frame_width
        """
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None, None, None, None, None, None

        face_lm = res.multi_face_landmarks[0]

        # ── Face bounding box from landmarks ──────────────────────────────────
        x_px = np.array([lm.x for lm in face_lm.landmark]) * w
        y_px = np.array([lm.y for lm in face_lm.landmark]) * h
        x0   = max(0, int(x_px.min()))
        y0   = max(0, int(y_px.min()))
        x1   = min(w, int(x_px.max()))
        y1   = min(h, int(y_px.max()))
        bbox = (x0, y0, x1, y1)

        crop_w = x1 - x0
        crop_h = y1 - y0

        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame_bgr
            x0 = y0 = 0
            crop_w, crop_h = w, h
            return None, None, None, None, None, None
        crop = cv2.resize(crop, (config.FRAME_SIZE, config.FRAME_SIZE))

        # ── Landmark vector renormalizado al crop ─────────────────────────────
        # Coherente con preprocess.py: coordenadas relativas al crop, no al frame.
        expected = config.LANDMARK_DIM
        lm_raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_lm.landmark],
            dtype=np.float32,
        )
        lm_raw[:, 0] = (lm_raw[:, 0] * w - x0) / max(crop_w, 1)
        lm_raw[:, 1] = (lm_raw[:, 1] * h - y0) / max(crop_h, 1)
        lm_arr = lm_raw.flatten()
        if lm_arr.size > expected:
            lm_arr = lm_arr[:expected]
        elif lm_arr.size < expected:
            lm_arr = np.pad(lm_arr, (0, expected - lm_arr.size))

        # ── Pitch del plano facial (inclinación cámara-cara) ─────────────────────
        # Landmark 10 = frente superior centro; 152 = barbilla inferior centro.
        # z en MediaPipe es profundidad relativa normalizada al ancho de la cara:
        # valores menores = más cerca de la cámara.
        # pitch > 0 → barbilla más lejos que frente → cámara por encima del paciente.
        # pitch < 0 → frente más lejos que barbilla → cámara por debajo del paciente.
        pitch      = float(lm_raw[152, 2] - lm_raw[10, 2])
        face_ratio = float((x1 - x0) / max(w, 1))   # face_width / frame_width

        # BGR → RGB → tensor → ImageNet normalise
        face_tensor = _normalise(_to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        return face_tensor, torch.tensor(lm_arr, dtype=torch.float32), bbox, crop, pitch, face_ratio

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer(self) -> tuple[int, float]:
        """Run the model on the current buffer contents."""
        frames    = torch.stack(list(self.frame_buf)).unsqueeze(0).to(self.device)   # (1,T,3,H,W)
        landmarks = torch.stack(list(self.lm_buf)).unsqueeze(0).to(self.device)      # (1,T,D)

        if self.majority_vote:
            per_frame_probs = self.model.predict_per_frame(frames, landmarks)[0]       # (T, 3)
            probs = per_frame_probs.mean(dim=0)                                        # (3,)
        else:
            logits = self.model(frames, landmarks)
            probs  = F.softmax(logits, dim=1)[0]                                       # (3,)

        raw = np.array([float(probs[c].item()) for c in range(probs.shape[0])], dtype=np.float32)

        # EMA: suaviza los saltos entre inferencias consecutivas.
        # Primera inferencia: inicializa directamente con los valores brutos.
        if self._ema_probs is None:
            self._ema_probs = raw.copy()
        else:
            self._ema_probs = self.smooth_alpha * raw + (1.0 - self.smooth_alpha) * self._ema_probs

        self.probs = self._ema_probs.tolist()

        pred = int(np.argmax(self._ema_probs))
        conf = self.probs[pred]
        return pred, conf

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> tuple[int | None, float | None]:
        """
        Feed one camera frame into the engine.

        Solo añade el frame al buffer si ha pasado suficiente tiempo desde el
        último frame añadido, replicando el muestreo equiespaciado del entrenamiento
        (1 frame cada video_duration/buffer_size segundos).

        Returns (pred, conf) or (None, None) if the buffer is not full yet
        or if no face was detected.
        """
        now = time.perf_counter()

        # Ejecutar MediaPipe en todos los frames para mantener el estado de
        # tracking, igual que en preprocess.py (que procesa cada frame del vídeo
        # aunque no sea frame objetivo).
        face_tensor, lm_tensor, bbox, crop_bgr, pitch, face_ratio = self._preprocess(frame_bgr)
        if bbox is not None:
            self.face_bbox  = bbox
        if pitch is not None:
            self.face_pitch = pitch
        if face_ratio is not None:
            self.face_size_ratio = face_ratio

        if now - self._last_sample_t < self._sample_interval:
            return self.pred, self.conf

        if face_tensor is None:
            return None, None

        self._last_sample_t = now
        self.frame_buf.append(face_tensor)
        self.lm_buf.append(lm_tensor)

        if len(self.frame_buf) == self.buffer_size:
            self.pred, self.conf = self._infer()

        return self.pred, self.conf


# ─── On-screen overlay ────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    engine: InferenceEngine,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    """Render the prediction overlay onto *frame* (in-place)."""
    h, w = frame.shape[:2]
    pred  = engine.pred
    conf  = engine.conf
    bbox  = engine.face_bbox
    buf_n = len(engine.frame_buf)
    buf_t = engine.buffer_size
    num_classes = config.NUM_CLASSES

    # ── Face bounding box ─────────────────────────────────────────────────────
    colour = PAIN_COLOUR.get(pred, (180, 180, 180)) if pred is not None else (180, 180, 180)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)

    # ── Semi-transparent info panel (top-left) ────────────────────────────────
    panel_w, panel_h = 300, 278
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y_text = 35
    if pred is not None:
        label_txt = PAIN_LABEL.get(pred, "—")
        cv2.putText(frame, label_txt, (18, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, colour, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Calentando buffer…", (18, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Barras de confianza por clase ─────────────────────────────────────────
    bar_x   = 18
    bar_w   = panel_w - 20
    bar_h   = 14
    bar_gap = 22          # espacio entre barras
    labels_short = ["Sin dolor  ", "Moderado   ", "Intenso    "]
    bar_colours  = [(0, 200, 0), (0, 200, 255), (0, 0, 220)]

    y_text += 18
    probs_display = engine.probs if engine.probs is not None else [0.0] * num_classes

    for cls in range(num_classes):
        p     = probs_display[cls]
        is_active = (pred == cls) and (pred is not None)
        bcol  = bar_colours[cls]
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, y_text), (bar_x + bar_w, y_text + bar_h),
                      (50, 50, 50), -1)
        # Relleno proporcional a la probabilidad
        fill = int(bar_w * p)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, y_text), (bar_x + fill, y_text + bar_h),
                          bcol, -1)
        # Borde resaltado si es la clase activa
        border = 2 if is_active else 1
        border_col = (255, 255, 255) if is_active else (100, 100, 100)
        cv2.rectangle(frame, (bar_x, y_text), (bar_x + bar_w, y_text + bar_h),
                      border_col, border)
        # Etiqueta y porcentaje encima
        txt = f"{labels_short[cls]}{p * 100:4.1f}%"
        cv2.putText(frame, txt, (bar_x + 4, y_text + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
        y_text += bar_gap

    # ── Buffer y FPS ──────────────────────────────────────────────────────────
    y_text += 2
    cv2.putText(frame, f"Buffer: {buf_n}/{buf_t}", (18, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    buf_bar_x, buf_bar_y = 18, y_text + 6
    buf_bar_w = panel_w - 20
    cv2.rectangle(frame, (buf_bar_x, buf_bar_y),
                  (buf_bar_x + buf_bar_w, buf_bar_y + 8), (70, 70, 70), -1)
    fill = int(buf_bar_w * buf_n / buf_t)
    cv2.rectangle(frame, (buf_bar_x, buf_bar_y),
                  (buf_bar_x + fill, buf_bar_y + 8), (0, 200, 100), -1)

    y_text += 28
    cv2.putText(frame, f"FPS: {fps:.1f}   Latencia: {latency_ms:.1f} ms", (18, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Indicador de inclinación del plano facial (pitch) ─────────────────────
    # La barra muestra la desviación respecto al pitch mediano del dataset (-0.085).
    # Centro de la barra = ángulo de grabación del entrenamiento.
    # delta > 0 → cámara más arriba que en el entrenamiento.
    # delta < 0 → cámara más abajo que en el entrenamiento.
    y_text += 18
    pitch       = engine.face_pitch
    pitch_delta = pitch - DATASET_PITCH_MEDIAN          # desviación respecto al dataset
    pitch_max   = 0.10                                  # rango ± para la barra
    pitch_norm  = max(-1.0, min(1.0, pitch_delta / pitch_max))

    if abs(pitch_delta) < 0.02:
        pitch_label = f"Pitch: OK  ({pitch:+.3f})"
        pitch_col   = (0, 220, 0)
    elif pitch_delta > 0:
        pitch_label = f"Pitch: cam. arriba  ({pitch:+.3f})"
        pitch_col   = (0, 200, 255)
    else:
        pitch_label = f"Pitch: cam. abajo  ({pitch:+.3f})"
        pitch_col   = (0, 180, 255)

    cv2.putText(frame, pitch_label, (18, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, pitch_col, 1, cv2.LINE_AA)

    # Barra centrada en el pitch del dataset: izq = por debajo, der = por encima
    pb_x, pb_y = 18, y_text + 4
    pb_w = panel_w - 20
    pb_h = 7
    mid  = pb_x + pb_w // 2
    cv2.rectangle(frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (60, 60, 60), -1)
    cv2.line(frame, (mid, pb_y), (mid, pb_y + pb_h), (120, 120, 120), 1)
    fill_px = int(abs(pitch_norm) * (pb_w // 2))
    if pitch_norm >= 0:
        cv2.rectangle(frame, (mid, pb_y), (mid + fill_px, pb_y + pb_h), pitch_col, -1)
    else:
        cv2.rectangle(frame, (mid - fill_px, pb_y), (mid, pb_y + pb_h), pitch_col, -1)
    cv2.rectangle(frame, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (100, 100, 100), 1)

    # ── Indicador de distancia cámara-cara (ratio tamaño facial) ─────────────
    # face_size_ratio = face_width / frame_width. Referencia = media del dataset.
    # Barra centrada: izq = más lejos que en entrenamiento, der = más cerca.
    y_text += 18
    fsr       = engine.face_size_ratio
    fsr_delta = fsr - DATASET_FACE_RATIO        # >0 más cerca, <0 más lejos
    fsr_max   = 0.15                            # rango ± para la barra
    fsr_norm  = max(-1.0, min(1.0, fsr_delta / fsr_max))

    if abs(fsr_delta) < 0.03:
        fsr_label = f"Dist: OK  ({fsr:.3f})"
        fsr_col   = (0, 220, 0)
    elif fsr_delta > 0:
        fsr_label = f"Dist: muy cerca  ({fsr:.3f})"
        fsr_col   = (0, 200, 255)
    else:
        fsr_label = f"Dist: muy lejos  ({fsr:.3f})"
        fsr_col   = (0, 180, 255)

    cv2.putText(frame, fsr_label, (18, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, fsr_col, 1, cv2.LINE_AA)

    fb_x, fb_y = 18, y_text + 4
    fb_w = panel_w - 20
    fb_h = 7
    mid  = fb_x + fb_w // 2
    cv2.rectangle(frame, (fb_x, fb_y), (fb_x + fb_w, fb_y + fb_h), (60, 60, 60), -1)
    cv2.line(frame, (mid, fb_y), (mid, fb_y + fb_h), (120, 120, 120), 1)
    fill_px = int(abs(fsr_norm) * (fb_w // 2))
    if fsr_norm >= 0:
        cv2.rectangle(frame, (mid, fb_y), (mid + fill_px, fb_y + fb_h), fsr_col, -1)
    else:
        cv2.rectangle(frame, (mid - fill_px, fb_y), (mid, fb_y + fb_h), fsr_col, -1)
    cv2.rectangle(frame, (fb_x, fb_y), (fb_x + fb_w, fb_y + fb_h), (100, 100, 100), 1)

    # ── Help text (bottom) ────────────────────────────────────────────────────
    cv2.putText(frame, "q = salir   r = resetear buffer", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ─── Main loop ───────────────────────────────────────────────────────────────

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt     = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    sd       = ckpt["model_state_dict"]

    # Inferir arquitectura del state dict para compatibilidad con checkpoints antiguos
    use_landmarks   = "landmark_encoder.net.0.weight" in sd
    landmark_hidden = ckpt_cfg.get(
        "landmark_hidden",
        int(sd["landmark_encoder.net.0.weight"].shape[0]) if use_landmarks else config.LANDMARK_HIDDEN_DIM,
    )
    landmark_out = ckpt_cfg.get(
        "landmark_out",
        int(sd["landmark_encoder.net.4.weight"].shape[0]) if use_landmarks else config.LANDMARK_OUT_DIM,
    )
    lstm_hidden = ckpt_cfg.get("lstm_hidden", config.LSTM_HIDDEN_DIM)
    lstm_layers = ckpt_cfg.get("lstm_layers", config.LSTM_NUM_LAYERS)

    model = PainDetectionModel(
        num_frames      = ckpt_cfg.get("num_frames",   config.NUM_FRAMES),
        num_classes     = config.NUM_CLASSES,
        landmark_dim    = ckpt_cfg.get("landmark_dim", config.LANDMARK_DIM),
        landmark_hidden = landmark_hidden,
        landmark_out    = landmark_out,
        lstm_hidden     = lstm_hidden,
        lstm_layers     = lstm_layers,
        use_landmarks   = use_landmarks,
    ).to(device)
    model.load_state_dict(sd)
    log.info("Model loaded from epoch %s | landmarks=%s",
             ckpt.get("epoch", "?"), use_landmarks)

    buf_size = ckpt_cfg.get("num_frames", config.NUM_FRAMES)
    engine   = InferenceEngine(
        model=model,
        device=device,
        buffer_size=buf_size,
        majority_vote=args.majority_vote,
        smooth_alpha=args.smooth,
    )

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Cannot open camera with index %d", args.source)
        engine.close()
        sys.exit(1)

    MAIN_WIN = "Pain Detection — Real-Time"

    log.info("Inference running.  Press 'q' to quit, 'r' to reset buffer.")

    fps_ema   = 30.0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame capture failed — exiting.")
            break

        t0 = time.perf_counter()
        engine.process_frame(frame)
        latency_ms = (time.perf_counter() - t0) * 1000

        # EMA de FPS
        now     = time.perf_counter()
        instant = 1.0 / max(now - prev_time, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * instant
        prev_time = now

        draw_overlay(frame, engine, fps_ema, latency_ms)
        cv2.imshow(MAIN_WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            engine.reset_buffer()
            log.info("Buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    engine.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time pain detection")
    parser.add_argument(
        "--checkpoint",
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Webcam index (0, 1 …)",
    )
    parser.add_argument(
        "--majority_vote",
        action="store_true",
        help="Aggregate per-frame predictions via majority vote (default: mean pooling)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.35,
        help="Factor EMA para suavizar probabilidades entre frames (0–1). "
             "1.0 = sin suavizado, 0.1 = muy suave (default: 0.35)",
    )
    args = parser.parse_args()
    run_inference(args)
