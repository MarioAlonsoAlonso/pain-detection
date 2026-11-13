"""
Comprehensive evaluation of a trained pain-detection model.

Usage
-----
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth --output_dir results/

Outputs (per split: val / test)
-------
    results/{split}_report.txt          — classification report (per-class + weighted)
    results/{split}_confusion_matrix.png
    results/{split}_roc_curves.png
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless rendering — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src import config
from src.dataset import build_dataloaders
from src.model import PainDetectionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

CLASS_NAMES = ["No pain (0)", "Mild pain (1)", "Severe pain (2)"]


# ─── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run the model on *loader* and collect predictions / probabilities / labels.

    Returns
    -------
    probs  : (N, num_classes) float32 — softmax probabilities
    preds  : (N,)             int     — argmax predictions
    labels : (N,)             int     — ground-truth labels
    """
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []

    for frames, landmarks, labels in tqdm(loader, desc="Inference", leave=False):
        frames    = frames.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)

        logits = model(frames, landmarks)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: Path, num_classes: int = 3) -> None:
    labels_list = list(range(num_classes))
    cm   = confusion_matrix(y_true, y_pred, labels=labels_list)
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in labels_list])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("  Confusion matrix → %s", save_path)


def plot_roc_curves(y_true, y_probs, save_path: Path) -> None:
    num_classes = y_probs.shape[1]
    colours = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(8, 6))

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
        auc         = roc_auc_score(y_bin[:, c], y_probs[:, c])
        ax.plot(fpr, tpr, color=colours[c], linewidth=2,
                label=f"Class {c}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("  ROC curves → %s", save_path)


# ─── Evaluation per split ─────────────────────────────────────────────────────

def evaluate_split(
    model, loader, device: torch.device, split: str, out_dir: Path, num_classes: int = 3
) -> None:
    log.info("\n%s  Evaluating on %-4s set  %s", "=" * 20, split.upper(), "=" * 20)

    probs, preds, labels = collect_predictions(model, loader, device)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        labels, preds,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    log.info("\n%s", report)
    report_path = out_dir / f"{split}_report.txt"
    report_path.write_text(report)
    log.info("  Classification report → %s", report_path)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    unique_cls = np.unique(labels)
    if len(unique_cls) > 1:
        classes_list = list(range(num_classes))
        y_bin        = label_binarize(labels, classes=classes_list)
        macro_auc    = roc_auc_score(y_bin, probs, average="macro",    multi_class="ovr")
        weighted_auc = roc_auc_score(y_bin, probs, average="weighted", multi_class="ovr")
        log.info("  Macro AUC    : %.4f", macro_auc)
        log.info("  Weighted AUC : %.4f", weighted_auc)

        # Append AUC to the report file
        with open(report_path, "a") as f:
            f.write(f"\nMacro AUC    : {macro_auc:.4f}\n")
            f.write(f"Weighted AUC : {weighted_auc:.4f}\n")
    else:
        log.warning("  Only one class present in %s set — skipping AUC.", split)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, out_dir / f"{split}_confusion_matrix.png", num_classes=num_classes)
    if len(unique_cls) > 1:
        plot_roc_curves(labels, probs, out_dir / f"{split}_roc_curves.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt       = torch.load(ckpt_path, map_location=device)
    ckpt_cfg   = ckpt.get("config", {})
    sd         = ckpt["model_state_dict"]
    trained_at = ckpt.get("epoch", "?")
    log.info("Loaded checkpoint from epoch %s (run: logs/%s)", trained_at, ckpt.get("run", "?"))

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

    num_classes = config.NUM_CLASSES

    model = PainDetectionModel(
        num_frames      = ckpt_cfg.get("num_frames",   config.NUM_FRAMES),
        num_classes     = num_classes,
        landmark_dim    = ckpt_cfg.get("landmark_dim", config.LANDMARK_DIM),
        landmark_hidden = landmark_hidden,
        landmark_out    = landmark_out,
        lstm_hidden     = lstm_hidden,
        lstm_layers     = lstm_layers,
        use_landmarks   = use_landmarks,
    ).to(device)
    model.load_state_dict(sd)
    log.info("landmarks=%s | lstm_hidden=%d", use_landmarks, lstm_hidden)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    _, val_loader, test_loader, _ = build_dataloaders(
        npy_dir=args.npy_dir, batch_size=args.batch_size,
    )

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    for split, loader in [("val", val_loader), ("test", test_loader)]:
        if len(loader.dataset) == 0:
            log.info("Split '%s' vacío — omitido.", split)
            continue
        evaluate_split(model, loader, device, split, out_dir, num_classes)

    log.info("\nAll results saved to %s", out_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pain-detection model")
    parser.add_argument(
        "--checkpoint",
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to the trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(config.RESULTS_DIR),
        help="Directory where plots and reports are saved",
    )
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--npy_dir", default=str(config.PROCESSED_NPY_DIR),
                        help="Carpeta NPY del dataset")
    args = parser.parse_args()

    evaluate(args)
