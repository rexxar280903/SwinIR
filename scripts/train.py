"""Train a single run (resumes automatically if the run folder has last.pt).

    python scripts/train.py --data /kaggle/working/data --runs /kaggle/working/runs \
        --set edge_mode=adaptive edge_weight=0.1 seed=1

Every setting not given with --set comes from --config (default configs/base.json).
Add --test to score the finished model on the test split right away.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.config import TrainConfig, apply_overrides  # noqa: E402
from edgesr.engine import evaluate_run, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="parent folder; the run gets a sub-folder")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test)")
    ap.add_argument("--test", action="store_true", help="evaluate on the test split when done")
    args = ap.parse_args()

    cfg = apply_overrides(TrainConfig.load(args.config), args.set)
    run_dir = Path(args.runs) / cfg.run_name
    result = train(cfg, args.data, run_dir, max_steps=args.max_steps)
    if args.test and result.get("status") == "done":
        s = evaluate_run(run_dir, args.data, "test")
        print(f"[test] PSNR-Y {s['psnr_y']:.3f}  SSIM-Y {s['ssim_y']:.4f}  Edge-PSNR-Y {s['edge_psnr_y']:.3f}")


if __name__ == "__main__":
    main()
