"""Build the x2 LR/HR dataset (train/val/test) from a folder of source images.

Kaggle:
    python scripts/prepare_data.py --src /kaggle/input/anime-faces-waifu2x --out /kaggle/working/data
Pilot (subset; marked as such in dataset_info.json):
    python scripts/prepare_data.py --src ... --out ... --limit 2000
Synthetic images for a dry run without the real dataset:
    python scripts/prepare_data.py --synthetic 200 --out ./tmp/data
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.data import make_synthetic_source, prepare_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder with source images (searched recursively)")
    ap.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    ap.add_argument("--hr-size", type=int, default=128)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42, help="split seed (keep 42 for the paper)")
    ap.add_argument("--limit", type=int, default=None, help="use only N source images (pilot)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic images instead of --src")
    args = ap.parse_args()

    src = args.src
    if args.synthetic:
        src = make_synthetic_source(Path(tempfile.mkdtemp(prefix="edgesr_synth_")), n=args.synthetic)
    if not src:
        ap.error("--src is required unless --synthetic is given")
    prepare_dataset(src, args.out, hr_size=args.hr_size, scale=args.scale, ratios=tuple(args.ratios),
                    seed=args.seed, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
