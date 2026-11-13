"""
corrupt_labels.py — Experimento de etiquetas corrompidas.

Intercambia aleatoriamente N etiquetas de clase 0 → clase 2 y N de clase 2 → clase 0,
simulando un dataset con errores masivos de anotación.

Guarda una copia de seguridad en labels_original.npy antes de modificar nada.

Uso
---
    # Aplicar corrupción (por defecto 400 muestras por clase)
    python scripts/corrupt_labels.py

    # Número personalizado
    python scripts/corrupt_labels.py --n 200

    # Restaurar labels originales
    python scripts/corrupt_labels.py --restore

    # Solo mostrar estadísticas sin modificar nada
    python scripts/corrupt_labels.py --dry_run
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

NPY_DIR      = Path(config.PROCESSED_NPY_DIR)
LABELS_PATH  = NPY_DIR / "labels.npy"
BACKUP_PATH  = NPY_DIR / "labels_original.npy"
SEED         = config.SEED


def _print_dist(labels: np.ndarray, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    total = len(labels)
    for c in range(config.NUM_CLASSES):
        n = int((labels == c).sum())
        print(f"  Clase {c}: {n:4d}  ({n / total * 100:.1f} %)")
    print(f"  Total : {total}")


def corrupt(n: int, dry_run: bool) -> None:
    if not LABELS_PATH.exists():
        print(f"ERROR: no se encontró {LABELS_PATH}")
        sys.exit(1)

    labels = np.load(LABELS_PATH)
    _print_dist(labels, "Distribución ANTES de corromper:")

    idx_0 = np.where(labels == 0)[0]
    idx_2 = np.where(labels == 2)[0]

    if len(idx_0) < n:
        print(f"ERROR: solo hay {len(idx_0)} muestras de clase 0, no se pueden corromper {n}.")
        sys.exit(1)
    if len(idx_2) < n:
        print(f"ERROR: solo hay {len(idx_2)} muestras de clase 2, no se pueden corromper {n}.")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    chosen_0 = rng.choice(idx_0, size=n, replace=False)  # 0 → 2
    chosen_2 = rng.choice(idx_2, size=n, replace=False)  # 2 → 0

    corrupted = labels.copy()
    corrupted[chosen_0] = 2
    corrupted[chosen_2] = 0

    _print_dist(corrupted, f"Distribución DESPUÉS de corromper ({n} muestras por clase):")

    swapped = (corrupted != labels).sum()
    print(f"\n  Muestras modificadas: {swapped} / {len(labels)} ({swapped / len(labels) * 100:.1f} %)")
    print(f"  Semilla usada: {SEED}  (reproducible)")

    if dry_run:
        print("\n[dry_run] No se escribió nada en disco.")
        return

    # Backup solo si no existe ya (no sobreescribir el original con una versión ya corrupta)
    if not BACKUP_PATH.exists():
        np.save(BACKUP_PATH, labels)
        print(f"\nBackup guardado en: {BACKUP_PATH}")
    else:
        print(f"\nBackup ya existente en {BACKUP_PATH} — no se sobreescribe.")

    np.save(LABELS_PATH, corrupted)
    print(f"Labels corrompidas guardadas en: {LABELS_PATH}")


def restore() -> None:
    if not BACKUP_PATH.exists():
        print(f"ERROR: no hay backup en {BACKUP_PATH}. No se puede restaurar.")
        sys.exit(1)

    original = np.load(BACKUP_PATH)
    np.save(LABELS_PATH, original)
    print(f"Labels restauradas desde {BACKUP_PATH}")
    _print_dist(original, "Distribución restaurada:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrompe o restaura las labels del dataset NPY")
    parser.add_argument("--n",       type=int, default=400,
                        help="Número de muestras a intercambiar por clase (default: 400)")
    parser.add_argument("--restore", action="store_true",
                        help="Restaurar las labels originales desde el backup")
    parser.add_argument("--dry_run", action="store_true",
                        help="Mostrar estadísticas sin modificar nada en disco")
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        corrupt(args.n, args.dry_run)
