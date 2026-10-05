"""Score a finished run, or bicubic interpolation, on the val or test split.

    python scripts/evaluate.py --data DATA --run RUNS/light_x2_l1_s1            # test split
    python scripts/evaluate.py --data DATA --bicubic --out RUNS/bicubic
    python scripts/evaluate.py --data DATA --run ... --split val --lpips

Writes {split}_per_image.csv, {split}_summary.json and samples_{split}/ (first N images).
The model is evaluated in fp32 regardless of the AMP setting used for training.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.config import TrainConfig  # noqa: E402
from edgesr.engine import evaluate_bicubic, evaluate_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run folder containing model_final.pt")
    g.add_argument("--bicubic", action="store_true")
    ap.add_argument("--out", help="output folder for --bicubic")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.json"),
                    help="metric settings for --bicubic are read from here")
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs `pip install lpips`)")
    ap.add_argument("--save-n", type=int, default=16, help="save the first N SR images")
    args = ap.parse_args()

    if args.bicubic:
        if not args.out:
            ap.error("--out is required with --bicubic")
        metrics = TrainConfig.load(args.config).metrics
        s = evaluate_bicubic(args.data, args.out, args.split, metrics, lpips=args.lpips, save_n=args.save_n)
    else:
        s = evaluate_run(args.run, args.data, args.split, lpips=args.lpips, save_n=args.save_n)
    print(json.dumps({k: v for k, v in s.items() if k != "metric_config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
