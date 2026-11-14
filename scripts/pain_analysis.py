"""
pain_analysis.py
----------------
Analiza el nivel de dolor usando los CSVs generados por OpenFace 2.0.

Estructura esperada:
    data/openface/0/*.csv   (vídeos sin dolor)
    data/openface/1/*.csv   (dolor moderado)
    data/openface/2/*.csv   (dolor intenso)

Para cada CSV calcula el PSPI frame a frame y reporta el percentil 90:
    PSPI(frame) = AU04_r + max(AU06_r, AU07_r) + max(AU09_r, AU10_r) + AU45_r  → [0, ~16]
    pspi_p90    = percentil 90 de todos los frames válidos del vídeo

El percentil 90 captura los momentos de mayor expresión de dolor sin que
los frames neutros (mayoría) enmascaren la señal.

Solo se usan frames con success=1 y confidence > 0.5.

Uso
---
    python scripts/pain_analysis.py
    python scripts/pain_analysis.py --openface-dir data/openface
    python scripts/pain_analysis.py --confidence 0.7
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR       = Path(__file__).parent.parent
OPENFACE_DIR   = BASE_DIR / "data" / "openface"
RESULTS_DIR    = BASE_DIR / "results"
CLASSES        = [0, 1, 2]

CLASS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"}
CLASS_LABELS = {
    0: "Clase 0",
    1: "Clase 1",
    2: "Clase 2",
}

AU_COLS = {
    "AU04": "AU04_r",
    "AU06": "AU06_r",
    "AU07": "AU07_r",
    "AU09": "AU09_r",
    "AU10": "AU10_r",
    "AU45": "AU45_r",
}

CONFIDENCE_THRESHOLD = 0.5


# ── Lectura de CSV OpenFace ───────────────────────────────────────────────────

def read_openface_csv(csv_path: Path) -> dict | None:
    """
    Lee un CSV de OpenFace, calcula el PSPI por frame y devuelve
    el percentil 90 junto con las medias de cada AU.
    Devuelve None si no hay frames válidos.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {csv_path.name}: {e}")
        return None

    df.columns = df.columns.str.strip()

    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] > CONFIDENCE_THRESHOLD]

    if len(df) == 0:
        return None

    # Verificar columnas
    for col in AU_COLS.values():
        if col not in df.columns:
            print(f"  [AVISO] Columna {col} no encontrada en {csv_path.name}")

    def get_col(col):
        return df[col].clip(lower=0) if col in df.columns else pd.Series([0.0] * len(df))

    # PSPI por frame
    pspi_frames = (
        get_col("AU04_r")
        + get_col("AU06_r").combine(get_col("AU07_r"), np.maximum)
        + get_col("AU09_r").combine(get_col("AU10_r"), np.maximum)
        + get_col("AU45_r")
    )

    au_means = {key: float(get_col(col).mean()) for key, col in AU_COLS.items()}

    return {
        "pspi_p90":  float(np.percentile(pspi_frames, 90)),
        "pspi_mean": float(pspi_frames.mean()),
        "pspi_max":  float(pspi_frames.max()),
        "au_means":  au_means,
        "detected":  len(df),
    }


# ── Salidas ───────────────────────────────────────────────────────────────────

def print_summary(rows: list) -> None:
    valid = [r for r in rows if r["pspi_mean"] != ""]
    print(f"\n{'═'*65}")
    print(f"  Total vídeos procesados : {len(rows)}")
    print(f"  Con cara detectada      : {len(valid)}")
    print(f"  Sin cara detectada      : {len(rows) - len(valid)}")

    if not valid:
        return

    print(f"\n  {'Clase':>6} | {'N':>5} | {'PSPI medio':>11} | {'PSPI máx':>9}")
    print(f"  {'─'*42}")
    for cls in CLASSES:
        sub = [r for r in valid if r["assigned_class"] == cls]
        if not sub:
            continue
        sm = np.mean([float(r["pspi_mean"]) for r in sub])
        sx = np.max( [float(r["pspi_mean"]) for r in sub])
        print(f"  {cls:>6} | {len(sub):>5} | {sm:>11.3f} | {sx:>9.3f}")

    print(f"  {'─'*42}")


def save_histogram(rows: list, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r["pspi_mean"] != ""]
    if not valid:
        print("Sin datos para el histograma.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PSPI medio por vídeo", fontsize=13)

    # Panel izquierdo: histograma
    ax = axes[0]
    all_scores = [float(r["pspi_mean"]) for r in valid]
    bins = np.linspace(0, max(max(all_scores) * 1.05, 4), 35)

    for cls in CLASSES:
        vals = [float(r["pspi_mean"]) for r in valid if r["assigned_class"] == cls]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.55,
                    color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.4,
                    label=CLASS_LABELS[cls])

    ax.set_xlabel("PSPI medio por vídeo", fontsize=11)
    ax.set_ylabel("Número de vídeos", fontsize=11)
    ax.set_title("Distribución PSPI medio por clase asignada")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(axis="y", alpha=0.3)

    # Panel derecho: scatter
    ax2 = axes[1]
    rng = np.random.default_rng(42)
    for cls in CLASSES:
        sub = [r for r in valid if r["assigned_class"] == cls]
        if not sub:
            continue
        jitter = rng.uniform(-0.15, 0.15, len(sub))
        x = [cls + j for j in jitter]
        y = [float(r["pspi_mean"]) for r in sub]
        ax2.scatter(x, y, color=CLASS_COLORS[cls], alpha=0.6, s=30,
                    label=CLASS_LABELS[cls])

    ax2.set_xticks(CLASSES)
    ax2.set_xticklabels([f"Clase {c}" for c in CLASSES])
    ax2.set_ylabel("PSPI medio (escala 0–16)", fontsize=11)
    ax2.set_title("PSPI medio vs clase asignada")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "pain_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Gráfico guardado en: {out_path}")


def save_au_distributions(rows: list, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r["pspi_mean"] != ""]
    if not valid:
        return

    aus = list(AU_COLS.keys())   # AU04, AU06, AU07, AU09, AU45
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribución de cada AU por clase — OpenFace 2.0", fontsize=13)
    axes_flat = axes.flatten()

    for i, au in enumerate(aus):
        ax = axes_flat[i]
        col = f"au_{au.lower()}_mean"
        all_vals = [float(r[col]) for r in valid if r[col] != ""]
        if not all_vals:
            continue
        bins = np.linspace(0, max(max(all_vals) * 1.05, 0.5), 30)

        for cls in CLASSES:
            vals = [float(r[col]) for r in valid
                    if r["assigned_class"] == cls and r[col] != ""]
            if vals:
                ax.hist(vals, bins=bins, alpha=0.55,
                        color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.4,
                        label=f"Clase {cls}")

        ax.set_title(au, fontsize=11, fontweight="bold")
        ax.set_xlabel("Intensidad media (0–5)", fontsize=9)
        ax.set_ylabel("Nº vídeos", fontsize=9)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Ocultar subplots sobrantes si hay menos AUs que celdas
    for ax in axes_flat[len(aus):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "au_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico de AUs guardado en: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global CONFIDENCE_THRESHOLD
    parser = argparse.ArgumentParser(
        description="Análisis PSPI P90 usando CSVs de OpenFace 2.0"
    )
    parser.add_argument("--openface-dir", type=str,
                        default=str(OPENFACE_DIR),
                        help=f"Directorio con subcarpetas 0/1/2 (default: {OPENFACE_DIR})")
    parser.add_argument("--confidence",   type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Umbral de confianza OpenFace (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--out",          type=str,   default=str(RESULTS_DIR / "pain_analysis.csv"))
    parser.add_argument("--no-plots",     action="store_true")

    args = parser.parse_args()
    CONFIDENCE_THRESHOLD = args.confidence

    openface_dir = Path(args.openface_dir)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out)

    csvs: list[tuple[Path, int]] = []
    for cls in CLASSES:
        cls_dir = openface_dir / str(cls)
        if not cls_dir.exists():
            print(f"[AVISO] No existe: {cls_dir}")
            continue
        for p in sorted(cls_dir.glob("*.csv")):
            csvs.append((p, cls))

    if not csvs:
        sys.exit(
            f"No se encontraron CSVs en {openface_dir}\n"
            f"Estructura esperada: {openface_dir}/0/*.csv, /1/*.csv, /2/*.csv"
        )

    print(f"CSVs a procesar   : {len(csvs)}")
    print(f"Directorio        : {openface_dir}")
    print(f"Confianza mínima  : {CONFIDENCE_THRESHOLD}")
    print("PSPI fórmula      : AU04_r + max(AU06_r, AU07_r) + max(AU09_r, AU10_r) + AU45_r")
    print("Métrica           : media de los frames válidos por vídeo")
    print(f"CSV de salida     : {out_path}\n")

    au_cols    = [f"au_{k.lower()}_mean" for k in AU_COLS]
    fieldnames = ["video", "assigned_class",
                  "pspi_p90", "pspi_mean", "pspi_max",
                  "detected_frames"] + au_cols

    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for csv_path, assigned_cls in tqdm(csvs, desc="Calculando PSPI", unit="vídeo"):
            result = read_openface_csv(csv_path)

            if result is not None:
                au_means = result["au_means"]
                row = {
                    "video":           csv_path.stem,
                    "assigned_class":  assigned_cls,
                    "pspi_p90":        round(result["pspi_p90"],  4),
                    "pspi_mean":       round(result["pspi_mean"], 4),
                    "pspi_max":        round(result["pspi_max"],  4),
                    "detected_frames": result["detected"],
                    **{f"au_{k.lower()}_mean": round(au_means[k], 4) for k in AU_COLS},
                }
            else:
                row = {
                    "video":           csv_path.stem,
                    "assigned_class":  assigned_cls,
                    "pspi_p90":        "",
                    "pspi_mean":       "",
                    "pspi_max":        "",
                    "detected_frames": 0,
                    **{f"au_{k.lower()}_mean": "" for k in AU_COLS},
                }

            writer.writerow(row)
            f.flush()
            rows.append(row)

    print_summary(rows)

    if not args.no_plots:
        save_histogram(rows, RESULTS_DIR)
        save_au_distributions(rows, RESULTS_DIR)

    print(f"\n  CSV guardado en: {out_path}")


if __name__ == "__main__":
    main()
