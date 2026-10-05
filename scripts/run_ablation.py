"""Run the ablation plan (docs/EXPERIMENTS.md). Safe to re-run: finished runs are skipped
and interrupted runs resume from their last checkpoint.

    # show the plan and what is already done
    python scripts/run_ablation.py --data D --runs R --stage all --list
    # stage 1: L1 + static/adaptive weight sweep (seed 0), then bicubic
    python scripts/run_ablation.py --data D --runs R --stage sweep
    # only if a selected weight sits at the end of the grid: 2 extra weights, once
    python scripts/run_ablation.py --data D --runs R --stage extend
    # stage 2: pick weights from the sweep's *validation* logs, then seeds 1-3
    python scripts/run_ablation.py --data D --runs R --stage confirm
    # everything in one go
    python scripts/run_ablation.py --data D --runs R --stage all

Splitting the work over Kaggle sessions:
    --shard 1/2 and --shard 2/2   run every other run of the stage
    --import /kaggle/input/<previous-output>/runs   copy finished runs from an earlier session
    --hours 11                    do not start a run that would not finish within 11 h
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.ablation import (SWEEP_WEIGHTS, confirm_plan, extension_plan, select_weights,  # noqa: E402
                             sweep_plan)
from edgesr.config import TrainConfig, apply_overrides  # noqa: E402
from edgesr.engine import CKPT_NAME, DONE_NAME, evaluate_bicubic, evaluate_run, train  # noqa: E402
from edgesr.utils import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def status(run_dir: Path) -> str:
    if (run_dir / DONE_NAME).exists():
        return "evaluated" if (run_dir / "test_summary.json").exists() else "trained"
    if (run_dir / CKPT_NAME).exists():
        return "partial"
    return "todo"


def finished(run_dir: Path) -> bool:
    return status(run_dir) in ("trained", "evaluated")


def import_runs(sources, runs_root: Path):
    for src in sources or []:
        for done in sorted(Path(src).glob(f"*/{DONE_NAME}")) + sorted(Path(src).glob("*/bicubic_done.json")):
            dst = runs_root / done.parent.name
            if dst.exists():
                continue
            shutil.copytree(done.parent, dst, ignore=shutil.ignore_patterns(CKPT_NAME, "*.tmp"))
            print(f"[import] {done.parent.name}")


def shard_filter(runs, shard):
    if not shard:
        return runs
    k, n = (int(x) for x in shard.split("/"))
    if not 1 <= k <= n:
        raise SystemExit(f"--shard must look like k/n with 1 <= k <= n, got {shard}")
    return [r for i, r in enumerate(runs) if i % n == k - 1]


def run_all(cfgs, args, runs_root: Path, t_session: float):
    durations = []
    for cfg in cfgs:
        run_dir = runs_root / cfg.run_name
        st = status(run_dir)
        if st == "evaluated":
            print(f"[skip] {cfg.run_name} (done)")
            continue
        if st != "trained":
            if args.hours and durations:
                left = args.hours * 3600 - (time.time() - t_session)
                if left < 1.1 * max(durations):
                    print(f"[stop] {cfg.run_name} would not finish within --hours {args.hours}; "
                          "re-run this command in a new session")
                    return False
            t0 = time.time()
            result = train(cfg, args.data, run_dir)
            durations.append(time.time() - t0)
            if result.get("status") != "done":
                return False
        evaluate_run(run_dir, args.data, "test", lpips=args.lpips)
        if not args.keep_ckpt and (run_dir / CKPT_NAME).exists():
            (run_dir / CKPT_NAME).unlink()  # model_final.pt is kept; last.pt only serves resuming
        print(f"[done] {cfg.run_name}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--stage", required=True, choices=["sweep", "bicubic", "extend", "confirm", "all"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override base settings for every run (changes the protocol hash!)")
    ap.add_argument("--static-weight", type=float, help="skip selection and use this weight")
    ap.add_argument("--adaptive-weight", type=float, help="skip selection and use this weight")
    ap.add_argument("--shard", help="k/n: run only every n-th run starting at k")
    ap.add_argument("--import", dest="imports", nargs="*", help="folders with finished runs to copy in")
    ap.add_argument("--hours", type=float, help="session time budget in hours")
    ap.add_argument("--keep-ckpt", action="store_true", help="keep last.pt after a run finishes")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the plan with status and exit")
    args = ap.parse_args()

    t_session = time.time()
    runs_root = Path(args.runs)
    runs_root.mkdir(parents=True, exist_ok=True)
    import_runs(args.imports, runs_root)
    base = apply_overrides(TrainConfig.load(args.config), args.set)
    write_json(runs_root / "base_config.json", base.to_dict())

    stages = ["sweep", "bicubic", "extend", "confirm"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "bicubic":
            out = runs_root / "bicubic"
            if args.list:
                print(f"  bicubic: {'done' if (out / 'bicubic_done.json').exists() else 'todo'}")
            elif not (out / "bicubic_done.json").exists():
                s = evaluate_bicubic(args.data, out, "test", base.metrics, scale=base.scale, lpips=args.lpips)
                print(f"[bicubic] PSNR-Y {s['psnr_y']:.3f}")
            continue

        if stage == "sweep":
            cfgs = sweep_plan(base)
        elif args.static_weight is not None and args.adaptive_weight is not None:
            if stage == "extend":
                continue
            cfgs = confirm_plan(base, args.static_weight, args.adaptive_weight)
        else:
            missing = [c.run_name for c in sweep_plan(base) if not finished(runs_root / c.run_name)]
            if missing:
                print(f"[{stage}] waiting for {len(missing)} sweep runs: {', '.join(missing)}")
                if args.list:
                    continue
                return 1
            # The boundary rule is applied to the base grid only, so a half-finished
            # extension cannot change which extension is needed.
            ext = extension_plan(base, select_weights([runs_root], allowed_weights=SWEEP_WEIGHTS))
            if stage == "extend":
                cfgs = ext
                if not ext:
                    print("[extend] no extension needed: selected weights are inside the grid")
            else:
                missing = [c.run_name for c in ext if not finished(runs_root / c.run_name)]
                if missing:
                    print(f"[confirm] run --stage extend first; missing: {', '.join(missing)}")
                    if args.list:
                        continue
                    return 1
                sel = select_weights([runs_root])
                if "static" not in sel["chosen"] or "adaptive" not in sel["chosen"]:
                    print(f"[confirm] sweep incomplete: {sel['notes']}")
                    if args.list:
                        continue
                    return 1
                write_json(runs_root / "selection.json", sel)
                sw, aw = sel["chosen"]["static"], sel["chosen"]["adaptive"]
                print(f"[confirm] selected weights (validation only): static={sw:g}, adaptive={aw:g}")
                cfgs = confirm_plan(base, sw, aw)
        cfgs = shard_filter(cfgs, args.shard)

        if args.list:
            print(f"stage {stage}: {len(cfgs)} runs")
            for c in cfgs:
                print(f"  {c.run_name:45s} {status(runs_root / c.run_name)}")
            continue
        if not run_all(cfgs, args, runs_root, t_session):
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
