"""Aggregate finished runs into tables, hypothesis tests and figures.

    python scripts/analyze.py --runs /kaggle/working/runs --data /kaggle/working/data
    python scripts/analyze.py --runs session1/runs session2/runs --out analysis

Writes RESULTS.md (paste-ready), CSV tables and PNG figures to --out
(default: <first runs dir>/analysis). Test numbers are only summarised here; the
choice of edge weights comes from runs/selection.json, i.e. from validation data.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.ablation import (CONFIRM_SEEDS, LAST_K_VALIDATIONS, SWEEP_SEEDS,  # noqa: E402
                             SWEEP_WEIGHTS, find_runs, select_weights)
from edgesr.stats import holm, mean_sd, paired_bootstrap, wilcoxon_images  # noqa: E402
from edgesr.utils import read_csv, read_json  # noqa: E402

MAIN_METRICS = [("psnr_y", "PSNR-Y (dB) ↑", 3), ("ssim_y", "SSIM-Y ↑", 4),
                ("edge_psnr_y", "Edge-PSNR-Y (dB) ↑", 3), ("gmsd", "GMSD ↓", 4),
                ("psnr_rgb", "PSNR-RGB (dB) ↑", 3)]
NONINF_MARGIN_DB = 0.05  # H3: adaptive may lose at most this much PSNR-Y vs L1


def load_per_image(path):
    rows = read_csv(path)
    keys = [r["key"] for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "key"}
    order = np.argsort(keys)
    return [keys[i] for i in order], {k: v[order] for k, v in cols.items()}


def fmt(m, s, nd):
    if np.isnan(m):
        return "–"
    return f"{m:.{nd}f}" if np.isnan(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", help="dataset folder (for the sample figure and dataset table)")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()
    out = Path(args.out or Path(args.runs[0]) / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    runs = [r for r in find_runs(args.runs) if (r["dir"] / "test_summary.json").exists()]
    if not runs:
        raise SystemExit("no evaluated runs found")
    hashes = {r["config"].protocol_hash() for r in runs}
    if len(hashes) > 1:
        raise SystemExit(f"runs come from different protocols {sorted(hashes)}; analyse them separately")
    bicubic = next((Path(d) / "bicubic" for d in args.runs if (Path(d) / "bicubic" / "test_summary.json").exists()), None)
    base = runs[0]["config"]

    # ---------------------------------------------------------------- selection (validation)
    sel_path = next((Path(d) / "selection.json" for d in args.runs if (Path(d) / "selection.json").exists()), None)
    sel = read_json(sel_path) if sel_path else select_weights(args.runs)
    chosen = sel.get("chosen", {})
    confirm_conditions = ["l1"] + [f"{m}_w{chosen[m]:g}" for m in ("static", "adaptive") if m in chosen]

    by_cond = defaultdict(dict)  # condition -> seed -> run
    for r in runs:
        by_cond[r["config"].condition][r["config"].seed] = r

    lines = ["# Results", "",
             f"Protocol `{base.protocol_hash()}` · model `{base.model}` · x{base.scale} · "
             f"{base.total_iters} iterations × batch {base.batch_size} · "
             f"Edge-PSNR threshold τ = {base.metrics.edge_threshold}", ""]
    if args.data and (Path(args.data) / "dataset_info.json").exists():
        info = read_json(Path(args.data) / "dataset_info.json")
        lines += [f"Dataset `{info['version']}`: {info['counts']['train']} train / {info['counts']['val']} val / "
                  f"{info['counts']['test']} test images ({info['n_exact_duplicates_dropped']} exact duplicates "
                  f"removed{'; PILOT SUBSET' if info.get('limit') else ''}).", ""]

    # ---------------------------------------------------------------- main table
    main_rows, md_rows = [], []
    if bicubic:
        s = read_json(bicubic / "test_summary.json")
        main_rows.append({"condition": "bicubic", "n_seeds": 0, **{k: s[k] for k, _, _ in MAIN_METRICS}})
        md_rows.append(["Bicubic", "–"] + [fmt(s[k], float("nan"), nd) for k, _, nd in MAIN_METRICS])
    for cond in confirm_conditions:
        seeds = sorted(s for s in by_cond.get(cond, {}) if s in CONFIRM_SEEDS)
        if not seeds:
            continue
        summ = [read_json(by_cond[cond][s]["dir"] / "test_summary.json") for s in seeds]
        row, md = {"condition": cond, "n_seeds": len(seeds)}, [cond, len(seeds)]
        for k, _, nd in MAIN_METRICS:
            m, sd = mean_sd([x[k] for x in summ])
            row[k], row[k + "_sd"] = m, sd
            md.append(fmt(m, sd, nd))
        main_rows.append(row)
        md_rows.append(md)
    write_csv(out / "results_main.csv", main_rows)
    lines += ["## Test set (confirmatory seeds " + ", ".join(map(str, CONFIRM_SEEDS)) + ")", "",
              "Mean ± standard deviation over seeds of the per-run test mean.", "",
              md_table(["Condition", "Seeds"] + [h for _, h, _ in MAIN_METRICS], md_rows), ""]

    # ---------------------------------------------------------------- sweep table
    sweep_rows = []
    mode_order = {"none": 0, "static": 1, "adaptive": 2}
    for t in sorted(sel.get("table", []), key=lambda t: (mode_order[t["mode"]], t["weight"])):
        r = by_cond.get("l1" if t["mode"] == "none" else f"{t['mode']}_w{t['weight']:g}", {}).get(t["seed"])
        test = read_json(r["dir"] / "test_summary.json") if r else {}
        share = float("nan")
        if r and t["mode"] != "none":
            tl = read_csv(r["dir"] / "train_log.csv")
            tail = tl[-max(1, len(tl) // 10):]  # last 10% of training
            ratio = np.mean([float(x["edge"]) / float(x["pixel"]) for x in tail])
            share = t["weight"] * ratio / (1 + t["weight"] * ratio)
        sweep_rows.append({"mode": t["mode"], "weight": t["weight"], "seed": t["seed"],
                           "edge_share": share,
                           "val_edge_psnr_y": t["edge_psnr_y"], "val_psnr_y": t["psnr_y"],
                           "test_edge_psnr_y": test.get("edge_psnr_y", float("nan")),
                           "test_psnr_y": test.get("psnr_y", float("nan"))})
    write_csv(out / "sweep.csv", sweep_rows)
    if sweep_rows:
        lines += ["## Weight sweep (seed " + ", ".join(map(str, SWEEP_SEEDS)) + ")", "",
                  f"Validation values are the mean of the last {LAST_K_VALIDATIONS} validation points and are "
                  "the only input to the weight selection. Test values are shown for sensitivity only. "
                  "Edge share = λ·L_edge / (L1 + λ·L_edge) over the last 10% of training: how much of the "
                  "objective the edge term actually is (adaptive weights are ≤ 1, so the same λ means less).", "",
                  md_table(["Mode", "λ", "Edge share", "val Edge-PSNR-Y", "val PSNR-Y", "test Edge-PSNR-Y",
                            "test PSNR-Y"],
                           [[r["mode"], f"{r['weight']:g}",
                             "–" if np.isnan(r["edge_share"]) else f"{r['edge_share']:.1%}",
                             f"{r['val_edge_psnr_y']:.3f}", f"{r['val_psnr_y']:.3f}",
                             f"{r['test_edge_psnr_y']:.3f}", f"{r['test_psnr_y']:.3f}"] for r in sweep_rows]),
                  "", "Selected (validation rule): " + ", ".join(f"{m} λ* = {w:g}" for m, w in chosen.items()), ""]
        extended = sorted({t["mode"] for t in sel.get("table", []) if t["mode"] != "none"
                           and t["weight"] not in SWEEP_WEIGHTS})
        if extended:
            lines += [f"Grid extended (pre-registered boundary rule) for: {', '.join(extended)}.", ""]
        for n in sel.get("notes", []):
            lines.append(f"> Note: {n}")

    # ---------------------------------------------------------------- paired comparisons
    def matrix(cond, metric):
        seeds = sorted(s for s in by_cond.get(cond, {}) if s in CONFIRM_SEEDS)
        keys, mats = None, []
        for s in seeds:
            k, cols = load_per_image(by_cond[cond][s]["dir"] / "test_per_image.csv")
            if keys is not None and k != keys:
                raise SystemExit(f"{cond} seed {s}: test images differ from the other runs")
            keys = k
            mats.append(cols[metric])
        return seeds, keys, np.array(mats)

    comps = []
    if len(confirm_conditions) == 3:
        st, ad = confirm_conditions[1], confirm_conditions[2]
        pairs = [("adaptive vs L1", ad, "l1"), ("adaptive vs static", ad, st), ("static vs L1", st, "l1")]
        metric_list = ["psnr_y", "ssim_y", "edge_psnr_y", "gmsd"] + \
            [f"edge_psnr_y_t{t:g}" for t in base.metrics.sensitivity_thresholds]
        for name, a, b in pairs:
            for metric in metric_list:
                sa, ka, A = matrix(a, metric)
                sb, kb, B = matrix(b, metric)
                common = sorted(set(sa) & set(sb))
                if not common or ka != kb:
                    continue
                A = A[[sa.index(s) for s in common]]
                B = B[[sb.index(s) for s in common]]
                res = paired_bootstrap(A, B, n_boot=args.n_boot)
                res.update(comparison=name, metric=metric, p_wilcoxon=wilcoxon_images(A, B))
                comps.append(res)

        def get(cmp_name, metric):
            return next((c for c in comps if c["comparison"] == cmp_name and c["metric"] == metric), None)

        h = {"H1": get("adaptive vs L1", "edge_psnr_y"), "H2": get("adaptive vs static", "edge_psnr_y")}
        adj = holm({k: v["p_wilcoxon"] for k, v in h.items() if v})
        h3 = get("adaptive vs L1", "psnr_y")
        hyp_rows = []
        for k, desc in (("H1", "Adaptive > L1 on Edge-PSNR-Y"), ("H2", "Adaptive > static on Edge-PSNR-Y")):
            c = h[k]
            if not c:
                continue
            ok = c["ci_low"] > 0 and adj[k] < 0.05 and all(d > 0 for d in c["per_seed_diff"])
            hyp_rows.append([k, desc, f"{c['mean_diff']:+.3f}", f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                             ", ".join(f"{d:+.3f}" for d in c["per_seed_diff"]), f"{adj[k]:.3g}",
                             "supported" if ok else "not supported"])
        if h3:
            ok = h3["ci_low"] > -NONINF_MARGIN_DB
            hyp_rows.append(["H3", f"Adaptive non-inferior to L1 on PSNR-Y (margin {NONINF_MARGIN_DB} dB)",
                             f"{h3['mean_diff']:+.3f}", f"[{h3['ci_low']:+.3f}, {h3['ci_high']:+.3f}]",
                             ", ".join(f"{d:+.3f}" for d in h3["per_seed_diff"]), "–",
                             "supported" if ok else "not supported"])
        lines += ["## Pre-registered hypotheses", "",
                  "Δ = mean paired difference on the test set; 95% CI from a two-level bootstrap over seeds "
                  "and images; p = Wilcoxon signed-rank on seed-averaged per-image differences, Holm-adjusted "
                  "over H1–H2. Decision rules: docs/EXPERIMENTS.md.", "",
                  md_table(["", "Hypothesis", "Δ", "95% CI", "Δ per seed", "p (Holm)", "Decision"], hyp_rows), ""]
        comp_rows = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()} for c in comps]
        write_csv(out / "comparisons.csv", comp_rows)
        lines += ["## All paired differences (secondary, not corrected)", "",
                  md_table(["Comparison", "Metric", "Δ", "95% CI", "p (Wilcoxon)"],
                           [[c["comparison"], c["metric"], f"{c['mean_diff']:+.4f}",
                             f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]", f"{c['p_wilcoxon']:.3g}"] for c in comps]), ""]

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None and sweep_rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, key, title in ((axes[0], "val_edge_psnr_y", "Validation Edge-PSNR-Y (dB)"),
                               (axes[1], "val_psnr_y", "Validation PSNR-Y (dB)")):
            l1 = [r[key] for r in sweep_rows if r["mode"] == "none"]
            if l1:
                ax.axhline(np.mean(l1), color="0.4", ls="--", lw=1, label="L1 only")
            for mode, marker in (("static", "o"), ("adaptive", "s")):
                rr = sorted((r for r in sweep_rows if r["mode"] == mode), key=lambda r: r["weight"])
                if rr:
                    ax.plot([r["weight"] for r in rr], [r[key] for r in rr], marker=marker, label=mode)
            ax.set_xscale("log")
            ax.set_xlabel("edge-loss weight λ")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_sweep.png", dpi=150)
        plt.close(fig)
        lines += ["![weight sweep](fig_sweep.png)", ""]

    if plt is not None and len(confirm_conditions) == 3:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for cond in confirm_conditions:
            seeds = sorted(s for s in by_cond.get(cond, {}) if s in CONFIRM_SEEDS)
            if not seeds:
                continue
            logs = [read_csv(by_cond[cond][s]["dir"] / "val_log.csv") for s in seeds]
            n = min(len(lg) for lg in logs)
            steps = [int(float(lg["step"])) for lg in logs[0][:n]]
            for ax, key in ((axes[0], "edge_psnr_y"), (axes[1], "psnr_y")):
                ax.plot(steps, np.mean([[float(r[key]) for r in lg[:n]] for lg in logs], axis=0), label=cond)
        for ax, t in ((axes[0], "Validation Edge-PSNR-Y (dB)"), (axes[1], "Validation PSNR-Y (dB)")):
            ax.set_xlabel("iteration")
            ax.set_title(t + ", mean over seeds", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_curves.png", dpi=150)
        plt.close(fig)
        lines += ["![validation curves](fig_curves.png)", ""]

    if plt is not None and args.data and len(confirm_conditions) == 3 and bicubic:
        from PIL import Image
        seed = min(by_cond[confirm_conditions[0]]) if by_cond.get(confirm_conditions[0]) else None
        cols = [("LR (nearest)", None), ("Bicubic", bicubic)] + \
               [(c, by_cond[c][seed]["dir"]) for c in confirm_conditions if seed in by_cond.get(c, {})] + [("HR", None)]
        keys = sorted(p.stem for p in (bicubic / "samples_test").glob("*.png"))[:4]
        if keys:
            fig, axes = plt.subplots(len(keys), len(cols), figsize=(2 * len(cols), 2 * len(keys)), squeeze=False)
            for i, k in enumerate(keys):
                for j, (title, d) in enumerate(cols):
                    if title.startswith("LR"):
                        im = Image.open(Path(args.data) / "test" / "LR" / f"{k}.png").resize((128, 128), Image.NEAREST)
                    elif title == "HR":
                        im = Image.open(Path(args.data) / "test" / "HR" / f"{k}.png")
                    else:
                        im = Image.open(d / "samples_test" / f"{k}.png")
                    axes[i][j].imshow(im)
                    axes[i][j].axis("off")
                    if i == 0:
                        axes[i][j].set_title(title, fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "fig_samples.png", dpi=150)
            plt.close(fig)
            lines += [f"![samples](fig_samples.png)  \nSeed {seed}; first test images in key order (not cherry-picked).", ""]

    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[analyze] wrote {out}")


if __name__ == "__main__":
    main()
