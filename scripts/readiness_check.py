"""Readiness gate before the full ablation (see docs/EXPERIMENTS.md, section 8).

Kaggle (GPU, real data):
    python scripts/readiness_check.py --data /kaggle/working/data --out /kaggle/working/readiness \
        --gpu-hours 25 --require-gpu
Local CPU check of the code paths on synthetic images:
    python scripts/readiness_check.py --synthetic 64 --out ./tmp/readiness --quick

Writes readiness_report.md / .json and protocol_lock.json to --out.
Exit code 0 = READY or READY WITH WARNINGS, 1 = NOT READY.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.config import TrainConfig, apply_overrides  # noqa: E402
from edgesr.data import make_synthetic_source, prepare_dataset  # noqa: E402
from edgesr.readiness import run_gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset folder from prepare_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--gpu-hours", type=float, default=25.0, help="GPU hours you can spend on the ablation")
    ap.add_argument("--compare", nargs="*", default=["medium"], help="other model presets to time")
    ap.add_argument("--require-gpu", action="store_true", help="fail when no GPU is visible")
    ap.add_argument("--quick", action="store_true", help="smaller samples, no GPU budget measurement")
    ap.add_argument("--synthetic", type=int, default=0, help="build N synthetic images and check on those")
    args = ap.parse_args()

    base = apply_overrides(TrainConfig.load(args.config), args.set)
    data = args.data
    if args.synthetic:
        data = Path(args.out) / "synthetic_data"
        shutil.rmtree(data, ignore_errors=True)
        src = make_synthetic_source(Path(tempfile.mkdtemp(prefix="edgesr_synth_")), n=args.synthetic)
        prepare_dataset(src, data, workers=0)
    if not data:
        ap.error("--data or --synthetic is required")

    rep = run_gate(data, args.out, base, gpu_hours=args.gpu_hours, require_gpu=args.require_gpu,
                   quick=args.quick, compare_presets=args.compare)
    print(f"\n==> {rep.verdict()}  (report: {Path(args.out) / 'readiness_report.md'})")
    return 1 if rep.verdict() == "NOT READY" else 0


if __name__ == "__main__":
    sys.exit(main())
