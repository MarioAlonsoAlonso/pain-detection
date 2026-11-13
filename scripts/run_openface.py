"""
run_openface.py
---------------
Ejecuta OpenFace 2.x (FeatureExtraction) sobre los vídeos de data/raw/{0,1,2}/
y guarda un CSV por vídeo en data/openface/{0,1,2}/, que es lo que lee
scripts/pain_analysis.py.

Solo se extraen las Action Units (-aus). Los vídeos que ya tienen CSV se
saltan, así que se puede interrumpir y relanzar sin perder lo hecho.

Uso
---
    # Todos los vídeos
    python scripts/run_openface.py --openface "C:/ruta/OpenFace_2.2.0_win_x64"

    # Muestra aleatoria de 300 vídeos por clase (como en la memoria del TFG)
    python scripts/run_openface.py --openface "C:/ruta/OpenFace_2.2.0_win_x64" --per_class 300

    # Ver qué se procesaría sin ejecutar OpenFace
    python scripts/run_openface.py --openface "..." --per_class 300 --dry_run
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
OPENFACE_DIR = config.DATA_DIR / "openface"


def find_exe(openface: str) -> Path:
    """Acepta la carpeta de OpenFace o la ruta directa al ejecutable."""
    p = Path(openface)
    if p.is_file():
        return p
    for name in ("FeatureExtraction.exe", "FeatureExtraction"):
        if (p / name).is_file():
            return p / name
    sys.exit(f"No se encontró FeatureExtraction en {p}")


def select_videos(raw_dir: Path, per_class: int | None, seed: int) -> dict[int, list[Path]]:
    rng = random.Random(seed)
    selected: dict[int, list[Path]] = {}
    for cls in range(config.NUM_CLASSES):
        vids = sorted(p for p in (raw_dir / str(cls)).glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
        if per_class is not None and per_class < len(vids):
            vids = sorted(rng.sample(vids, per_class))
        selected[cls] = vids
    return selected


def run_batch(exe: Path, videos: list[Path], out_dir: Path) -> str:
    """Una llamada a FeatureExtraction con varios -f (los modelos se cargan una sola vez)."""
    cmd = [str(exe)]
    for v in videos:
        cmd += ["-f", str(v)]
    cmd += ["-out_dir", str(out_dir), "-aus"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent))
    return "" if res.returncode == 0 else (res.stderr or res.stdout)[-300:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae AUs con OpenFace para pain_analysis.py")
    parser.add_argument("--openface", required=True,
                        help="Carpeta de OpenFace (con FeatureExtraction.exe) o ruta al ejecutable")
    parser.add_argument("--raw_dir", default=str(config.RAW_DATA_DIR))
    parser.add_argument("--output",  default=str(OPENFACE_DIR))
    parser.add_argument("--per_class", type=int, default=None,
                        help="Vídeos aleatorios por clase (por defecto: todos)")
    parser.add_argument("--seed",    type=int, default=config.SEED)
    parser.add_argument("--batch",   type=int, default=20,
                        help="Vídeos por llamada a FeatureExtraction (default: 20)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    raw_dir, out_root = Path(args.raw_dir), Path(args.output)
    selected = select_videos(raw_dir, args.per_class, args.seed)

    pending: list[tuple[int, Path]] = []
    for cls, vids in selected.items():
        done = sum((out_root / str(cls) / f"{v.stem}.csv").exists() for v in vids)
        todo = [v for v in vids if not (out_root / str(cls) / f"{v.stem}.csv").exists()]
        print(f"Clase {cls}: {len(vids)} vídeos seleccionados, {done} ya procesados, {len(todo)} pendientes")
        pending += [(cls, v) for v in todo]

    if args.dry_run or not pending:
        print("Nada que hacer." if not pending else "--dry_run: no se ejecuta OpenFace.")
        return

    exe = find_exe(args.openface)
    errors = 0
    with tqdm(total=len(pending), desc="OpenFace", unit="vídeo") as bar:
        for cls in range(config.NUM_CLASSES):
            vids = [v for c, v in pending if c == cls]
            out_dir = out_root / str(cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(vids), args.batch):
                chunk = vids[i:i + args.batch]
                err = run_batch(exe, chunk, out_dir)
                missing = [v.name for v in chunk if not (out_dir / f"{v.stem}.csv").exists()]
                if err or missing:
                    errors += len(missing)
                    tqdm.write(f"[clase {cls}] sin CSV: {missing[:5]}{' …' if len(missing) > 5 else ''} {err.strip()[:200]}")
                bar.update(len(chunk))

    # OpenFace deja también un *_of_details.txt por vídeo; no hace falta
    for p in out_root.glob("*/*_of_details.txt"):
        p.unlink()

    print(f"\nHecho. CSV en {out_root}  ({errors} vídeos sin CSV)")
    print("Siguiente paso: python scripts/pain_analysis.py")


if __name__ == "__main__":
    main()
