"""Build numerical public exports; never copy dataset images or per-image records.

Run from the repository root after obtaining the local Kaggle v5 archive.
Public dashboard source files are maintained directly in public/dashboard/.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil

CONFIG_KEYS = {
    "seed", "model", "scale", "lr_size", "edge_mode", "edge_weight", "batch_size",
    "total_iters", "lr", "betas", "eta_min", "warmup_iters", "grad_clip", "amp",
    "augment", "num_workers", "cache", "log_every", "val_every", "ckpt_every",
    "val_max_images", "metrics",
}
METRIC_KEYS = {
    "crop_border", "edge_threshold", "ssim_window", "ssim_sigma", "sensitivity_thresholds",
}
SCORES = {
    "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y", "edge_psnr_y", "edge_cov", "gmsd",
    "edge_psnr_y_t0.25", "edge_psnr_y_t1",
}
TRAIN_KEYS = {
    "step", "lr", "loss", "pixel", "edge", "w_mean", "grad_norm", "img_per_s",
    "skipped", "elapsed_s", "max_mem_gb",
}


def read_rows(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    number = float(value)
                    row[key] = number if math.isfinite(number) else None
                except (ValueError, TypeError):
                    pass
            result.append(row)
    return result


def config_subset(config):
    selected = {k: v for k, v in config.items() if k in CONFIG_KEYS}
    if "metrics" in selected:
        selected["metrics"] = {k: v for k, v in selected["metrics"].items() if k in METRIC_KEYS}
    return selected


def numeric_rows(path, keys):
    return [{k: v for k, v in row.items() if k in keys and (v is None or isinstance(v, (int, float)))}
            for row in read_rows(path)]


def sanitize_paper(source):
    # The only dataset-derived figure in this manuscript is the Gambar 4 contact sheet.
    source, removed = re.subn(r'<figure><img[^>]+alt="Gambar 4\.[\s\S]*?</figure>', "", source)
    if removed != 1:
        raise ValueError("Expected exactly one sample-image figure; review manuscript manually")
    source = re.sub(r'<h3>B\. Contoh Rekonstruksi</h3><p>[\s\S]*?</p>',
                    '<h3>B. Cakupan Versi Publik</h3><p>Contoh citra dan hasil per citra '
                    'tersimpan dalam analisis lokal. Versi publik menyertakan tabel agregat '
                    'dan grafik numerik.</p>', source, count=1)
    source = source.replace("Lampiran A. Sensitivitas dan Contoh Visual", "Lampiran A. Sensitivitas Bobot")
    source = source.replace('<div class="notice">', '<div class="notice"><b>Versi publik:</b> '
                            'gambar dataset dikecualikan karena hak redistribusi belum terverifikasi. ')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("runs/kaggle-swinlr-sesi2-v5"))
    parser.add_argument("--paper-source", type=Path, default=Path("docs/paper/paper_ieee_id.html"))
    args = parser.parse_args()
    archive = args.archive_dir
    analysis = archive / "runs/analysis"
    destination = Path("public/dashboard")
    destination.mkdir(parents=True, exist_ok=True)
    results = Path("public/results")
    results.mkdir(parents=True, exist_ok=True)
    data = {"public_export": True, "runs": []}
    for key, filename in (("main", "results_main.csv"), ("sweep", "sweep.csv"),
                          ("comparisons", "comparisons.csv")):
        data[key] = read_rows(analysis / filename)
        shutil.copyfile(analysis / filename, results / filename)
    for run in sorted((archive / "runs").iterdir()):
        if not run.is_dir() or not (run / "test_summary.json").exists():
            continue
        config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {
            "edge_mode": "bicubic", "seed": None, "edge_weight": 0,
        }
        summary = json.loads((run / "test_summary.json").read_text())
        summary = {k: v for k, v in summary.items()
                   if k in SCORES | {"n_images", "run_name", "condition", "seed", "split"}}
        data["runs"].append({
            "id": run.name, "config": config_subset(config), "test": [],
            "train": numeric_rows(run / "train_log.csv", TRAIN_KEYS),
            "val": numeric_rows(run / "val_log.csv", SCORES | {"step"}), "summary": summary,
        })
    source = json.loads((archive / "SOURCE.json").read_text(encoding="utf-8-sig"))
    data["source"] = {k: v for k, v in source.items()
                      if k in {"kaggle_run_id", "downloaded_on", "archive_sha256", "source", "version"}}
    data["report"] = (
        "Ekspor publik Kaggle versi 5: 24 training selesai dan satu baseline bicubic.\n"
        "Model: SwinIR-light, x2, 10.000 iterasi.\n"
        "Pilihan validasi: statis lambda=1, adaptif lambda=2.\n"
        "Konfirmasi seed 1-3: statis mengungguli adaptif dan L1.\n"
        "H1 dan H3 didukung; H2 tidak didukung.\n"
        "Seluruh angka agregat tersedia pada tabel dan unduhan CSV.\n"
        "Gambar, metrik per citra, checkpoint, dan arsip lengkap disimpan lokal."
    )
    (destination / "data.js").write_text("window.DATA=" + json.dumps(
        data, separators=(",", ":"), allow_nan=False) + ";", encoding="utf-8")
    for filename in ("fig_sweep.png", "fig_curves.png"):
        shutil.copyfile(analysis / filename, destination / filename)
    if args.paper_source.exists():
        paper_dir = Path("public/paper")
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "paper_ieee_id.html").write_text(
            sanitize_paper(args.paper_source.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Public export: {len(data['runs'])} run summaries; no per-image rows or sample images")


if __name__ == "__main__":
    main()
