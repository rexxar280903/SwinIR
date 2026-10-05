"""The ablation plan and the pre-registered rule for choosing the edge-loss weight.

Stage "sweep"   (seed 0): L1, static x 5 weights, adaptive x 5 weights      = 11 runs
Stage "extend"  (seed 0, only if needed): when the selected weight of a mode is the
                largest (smallest) grid value, that mode gets 2 more weights beyond
                that end, once                                                0-4 runs
Stage "confirm" (seeds 1, 2, 3): L1, static(w*_static), adaptive(w*_adaptive) =  9 runs
Bicubic interpolation is evaluated without training.

The confirmation seeds differ from the sweep seed, so the weight that wins the sweep
cannot profit from a lucky seed in the confirmatory comparison. The weights are
chosen from the *validation* logs only (select_weights); the test split is not read.
"""
import dataclasses
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .utils import read_csv, read_json

SWEEP_WEIGHTS = (0.05, 0.1, 0.2, 0.5, 1.0)
EXTEND_UP = (2.0, 5.0)            # added once if the largest grid weight is selected
EXTEND_DOWN = (0.02, 0.01)        # added once if the smallest grid weight is selected
SWEEP_SEEDS = (0,)
CONFIRM_SEEDS = (1, 2, 3)
EDGE_MODES_TESTED = ("static", "adaptive")

# Selection rule (docs/EXPERIMENTS.md, section "Weight selection"):
SELECT_METRIC = "edge_psnr_y"     # maximise validation Edge-PSNR-Y ...
GUARD_METRIC = "psnr_y"           # ... among weights whose validation PSNR-Y is
PSNR_TOLERANCE_DB = 0.10          # at most this much below the L1 run of the same seed
LAST_K_VALIDATIONS = 3            # scores = mean of the last K validation points
TIE_DB = 0.01                     # differences below this count as ties -> smaller weight


def with_condition(base: TrainConfig, edge_mode: str, edge_weight: float, seed: int) -> TrainConfig:
    return dataclasses.replace(base, edge_mode=edge_mode, seed=seed,
                               edge_weight=0.0 if edge_mode == "none" else float(edge_weight))


def sweep_plan(base: TrainConfig, weights=SWEEP_WEIGHTS, seeds=SWEEP_SEEDS) -> list:
    runs = []
    for seed in seeds:
        runs.append(with_condition(base, "none", 0.0, seed))
        for mode in EDGE_MODES_TESTED:
            runs += [with_condition(base, mode, w, seed) for w in weights]
    return runs


def confirm_plan(base: TrainConfig, static_weight: float, adaptive_weight: float,
                 seeds=CONFIRM_SEEDS) -> list:
    runs = []
    for seed in seeds:
        runs.append(with_condition(base, "none", 0.0, seed))
        runs.append(with_condition(base, "static", static_weight, seed))
        runs.append(with_condition(base, "adaptive", adaptive_weight, seed))
    return runs


def extension_plan(base: TrainConfig, selection: dict) -> list:
    """Extra sweep runs requested by a selection made on the base grid."""
    return [with_condition(base, mode, w, seed) for mode, ws in selection.get("extend", {}).items()
            for seed in SWEEP_SEEDS for w in ws]


def _last_k_mean(val_rows, key, k=LAST_K_VALIDATIONS):
    vals = [float(r[key]) for r in val_rows][-k:]
    return float(np.mean(vals)) if vals else float("nan")


def find_runs(runs_dirs) -> list:
    """Every finished run under the given directories: dicts with dir, config and done-info."""
    found = []
    for root in runs_dirs:
        for done in sorted(Path(root).glob("*/train_done.json")):
            d = done.parent
            found.append({"dir": d, "config": TrainConfig.load(d / "config.json"),
                          "done": read_json(done)})
    return found


def select_weights(runs_dirs, seeds=SWEEP_SEEDS, allowed_weights=None) -> dict:
    """Apply the selection rule to finished sweep runs. Reads only val_log.csv files.

    allowed_weights restricts the edge-loss weights considered (e.g. to the base grid);
    the result's "extend" entry lists the weights the boundary rule asks for.
    """
    runs = [r for r in find_runs(runs_dirs) if r["config"].seed in seeds]
    if allowed_weights is not None:
        allowed = {float(w) for w in allowed_weights}
        runs = [r for r in runs if r["config"].edge_mode == "none" or r["config"].edge_weight in allowed]
    hashes = {r["config"].protocol_hash() for r in runs}
    if len(hashes) > 1:
        raise RuntimeError(f"sweep runs come from different protocols {sorted(hashes)}; "
                           "compare only runs that share model, iterations and metrics")
    table = []
    for r in runs:
        val = read_csv(r["dir"] / "val_log.csv")
        table.append({"seed": r["config"].seed, "mode": r["config"].edge_mode,
                      "weight": r["config"].edge_weight, "run": r["config"].run_name,
                      SELECT_METRIC: _last_k_mean(val, SELECT_METRIC),
                      GUARD_METRIC: _last_k_mean(val, GUARD_METRIC)})
    result = {"rule": {"select": SELECT_METRIC, "guard": GUARD_METRIC,
                       "psnr_tolerance_db": PSNR_TOLERANCE_DB, "last_k": LAST_K_VALIDATIONS,
                       "tie_db": TIE_DB, "seeds": list(seeds)},
              "table": table, "chosen": {}, "extend": {}, "notes": []}
    for mode in EDGE_MODES_TESTED:
        # Average over sweep seeds (one seed by default) per weight.
        weights = sorted({t["weight"] for t in table if t["mode"] == mode})
        if not weights:
            result["notes"].append(f"no finished {mode} sweep runs")
            continue
        l1 = [t for t in table if t["mode"] == "none"]
        if not l1:
            raise RuntimeError("the L1 sweep run must finish before weights can be selected")
        l1_guard = float(np.mean([t[GUARD_METRIC] for t in l1]))
        cands = []
        for w in weights:
            rows = [t for t in table if t["mode"] == mode and t["weight"] == w]
            cands.append({"weight": w,
                          "score": float(np.mean([t[SELECT_METRIC] for t in rows])),
                          "guard": float(np.mean([t[GUARD_METRIC] for t in rows])),
                          "n_seeds": len(rows)})
        eligible = [c for c in cands if c["guard"] >= l1_guard - PSNR_TOLERANCE_DB]
        if eligible:
            best = max(c["score"] for c in eligible)
            chosen = min(c["weight"] for c in eligible if c["score"] >= best - TIE_DB)
        else:
            best_guard = max(c["guard"] for c in cands)
            chosen = min(c["weight"] for c in cands if c["guard"] >= best_guard - TIE_DB)
            result["notes"].append(f"{mode}: no weight passed the PSNR guard; chose best PSNR-Y")
        result["chosen"][mode] = chosen
        result[f"{mode}_candidates"] = cands
        if chosen == max(weights) and not set(EXTEND_UP) & set(weights):
            result["extend"][mode] = list(EXTEND_UP)
        elif chosen == min(weights) and not set(EXTEND_DOWN) & set(weights):
            result["extend"][mode] = list(EXTEND_DOWN)
        result["l1_guard_value"] = l1_guard
    return result
