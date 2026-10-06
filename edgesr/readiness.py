"""Readiness gate: every check that must pass before the full ablation is started.

Each check returns (status, detail, evidence). Status is one of
    PASS  - criterion met
    WARN  - runs, but a decision or a look is needed before the full ablation
    FAIL  - must be fixed; the gate is closed
    SKIP  - not applicable here (e.g. GPU checks on a CPU machine)

Groups
    R0 environment   R1 data          R2 loss/metric correctness   R3 model
    R4 training      R5 pipeline      R6 GPU budget                R7 protocol lock
"""
import gc
import hashlib
import inspect
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import engine
from .ablation import (CONFIRM_SEEDS, LAST_K_VALIDATIONS, PSNR_TOLERANCE_DB, SELECT_METRIC,
                       SWEEP_SEEDS, SWEEP_WEIGHTS, confirm_plan, sweep_plan)
from .config import TrainConfig
from .data import SPLITS, SRPairDataset, TrainSampler, make_subset, read_manifest
from .losses import EdgeAwareLoss, adaptive_edge_weight, sobel_gradients, sobel_magnitude
from .metrics import (METRIC_NAMES, MetricConfig, edge_mask, edge_psnr, gmsd, image_metrics,
                      psnr, quantize, rgb_to_y, ssim)
from .models import EXPECTED_PARAMS, build_model, count_params
from .utils import disk_free_gb, env_info, read_csv, read_json, seed_everything, write_json

REPO = Path(__file__).resolve().parents[1]
OFFICIAL_SWINIR_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
SESSION_HOURS = 12.0  # Kaggle GPU session limit at the time of writing; check the Kaggle docs


def free_gpu():
    """Return cached GPU memory to the driver, so a later check (or a subprocess) can use it."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Report:
    def __init__(self, log=print):
        self.items = []
        self.log = log

    def run(self, gid, title, fn, *args, **kwargs):
        t0 = time.time()
        try:
            status, detail, evidence = fn(*args, **kwargs)
        except Exception as exc:  # a crashing check is a failed check
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
            evidence = {"traceback": traceback.format_exc(limit=4)}
        free_gpu()
        item = {"id": gid, "title": title, "status": status, "detail": detail,
                "seconds": round(time.time() - t0, 1), "evidence": evidence or {}}
        self.items.append(item)
        self.log(f"[{status:4s}] {gid} {title}: {detail}")
        return item

    def verdict(self):
        st = {i["status"] for i in self.items}
        if "FAIL" in st:
            return "NOT READY"
        return "READY WITH WARNINGS" if "WARN" in st else "READY"

    def markdown(self, header):
        lines = [f"# Readiness report", "", header, "", f"**Verdict: {self.verdict()}**", "",
                 "| ID | Check | Status | Detail |", "|---|---|---|---|"]
        for i in self.items:
            detail = str(i["detail"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i['id']} | {i['title']} | `{i['status']}` | {detail} |")
        warns = [i for i in self.items if i["status"] in ("WARN", "FAIL")]
        if warns:
            lines += ["", "## To resolve or acknowledge before the full ablation", ""]
            lines += [f"- **{i['id']}** ({i['status']}): {i['detail']}" for i in warns]
        return "\n".join(lines) + "\n"


# =========================================================================== R0 environment
def check_versions():
    info = env_info()
    major, minor = (int(x) for x in re.findall(r"\d+", torch.__version__)[:2])
    if (major, minor) < (2, 3):
        return "FAIL", f"PyTorch {torch.__version__} < 2.3 (torch.amp.GradScaler('cuda') needed)", info
    return "PASS", f"Python {info['python']}, PyTorch {info['torch']}", info


def check_gpu(require_gpu):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "PASS", f"{p.name}, {p.total_memory / 2**30:.1f} GB", {"gpu": p.name}
    if require_gpu:
        return "FAIL", "no CUDA device (Kaggle: Settings -> Accelerator -> GPU T4)", {}
    return "WARN", "CPU only: training checks run on CPU and the GPU budget (R6) is skipped", {}


def check_packages():
    found = {}
    for name in ("scipy", "matplotlib", "skimage", "lpips"):
        try:
            mod = __import__(name)
            found[name] = getattr(mod, "__version__", "?")
        except ImportError:
            found[name] = None
    missing = [k for k in ("scipy", "matplotlib") if not found[k]]
    if missing:
        return "WARN", f"missing {missing}: Wilcoxon p-values / figures will be skipped", found
    extra = "" if found["lpips"] else " (lpips not installed: LPIPS optional, off)"
    return "PASS", "scipy, matplotlib present" + extra, found


def check_disk(out_dir, data_root, n_runs, model_mb):
    free = disk_free_gb(out_dir)
    data_gb = 0.0
    if data_root and Path(data_root).exists():
        data_gb = sum(p.stat().st_size for p in Path(data_root).rglob("*.png")) / 2 ** 30
    # per run: model_final + last.pt (model + 2 Adam moments) while training + logs/samples
    need = n_runs * (model_mb * 1 + 2) / 1024 + 4 * model_mb / 1024 + 0.5
    ev = {"free_gb": round(free, 1), "data_gb": round(data_gb, 2), "runs_need_gb": round(need, 2)}
    if free < need:
        return "FAIL", f"{free:.1f} GB free, about {need:.1f} GB needed for {n_runs} runs", ev
    if free < 2 * need:
        return "WARN", f"{free:.1f} GB free, tight for {need:.1f} GB", ev
    return "PASS", f"{free:.1f} GB free; runs need about {need:.2f} GB (data uses {data_gb:.2f} GB)", ev


# =========================================================================== R1 data
def check_manifest(data_root):
    data_root = Path(data_root)
    info = read_json(data_root / "dataset_info.json")
    rows = read_manifest(data_root)
    problems, counts = [], {}
    for s in SPLITS:
        keys = {r["key"] for r in rows if r["split"] == s}
        counts[s] = len(keys)
        for kind in ("HR", "LR"):
            on_disk = {p.stem for p in (data_root / s / kind).glob("*.png")}
            if on_disk != keys:
                problems.append(f"{s}/{kind}: {len(on_disk ^ keys)} files differ from manifest")
        if not keys:
            problems.append(f"split {s} is empty")
    if counts != info["counts"]:
        problems.append(f"counts {counts} != dataset_info {info['counts']}")
    if problems:
        return "FAIL", "; ".join(problems), {"counts": counts}
    if info.get("limit"):
        return "WARN", (f"PILOT ONLY: built with --limit {info['limit']} ({counts}); rebuild without --limit "
                        "for the real ablation"), {"counts": counts}
    return "PASS", f"{counts} match manifest and files on disk", {"counts": counts, "version": info["version"]}


def check_image_sizes(data_root, n=200):
    data_root = Path(data_root)
    info = read_json(data_root / "dataset_info.json")
    hr_s, lr_s = info["hr_size"], info["lr_size"]
    bad = []
    for s in SPLITS:
        for k in [r["key"] for r in read_manifest(data_root) if r["split"] == s][:n]:
            for kind, size in (("HR", hr_s), ("LR", lr_s)):
                with Image.open(data_root / s / kind / f"{k}.png") as im:
                    if im.size != (size, size) or im.mode != "RGB":
                        bad.append(f"{s}/{kind}/{k}: {im.size} {im.mode}")
    if bad:
        return "FAIL", f"{len(bad)} files with wrong size/mode, e.g. {bad[:3]}", {"bad": bad[:20]}
    return "PASS", f"sampled files are RGB, HR {hr_s}x{hr_s}, LR {lr_s}x{lr_s}", {}


def check_degradation(data_root, n=50):
    """LR on disk must equal PIL-bicubic(HR): the degradation is what we claim it is."""
    data_root = Path(data_root)
    info = read_json(data_root / "dataset_info.json")
    worst = 0
    for r in read_manifest(data_root)[:: max(1, len(read_manifest(data_root)) // n)][:n]:
        with Image.open(data_root / r["split"] / "HR" / f"{r['key']}.png") as hr:
            lr_ref = hr.convert("RGB").resize((info["lr_size"], info["lr_size"]), Image.BICUBIC)
        with Image.open(data_root / r["split"] / "LR" / f"{r['key']}.png") as lr:
            d = np.abs(np.asarray(lr, np.int16) - np.asarray(lr_ref, np.int16)).max()
        worst = max(worst, int(d))
    if worst:
        return "FAIL", f"LR differs from bicubic(HR) by up to {worst} levels", {"max_abs_diff": worst}
    return "PASS", f"LR == PIL-bicubic(HR) exactly on {n} sampled pairs", {}


def check_leakage(data_root, n_rehash=100):
    data_root = Path(data_root)
    rows = read_manifest(data_root)
    by_sha, by_dh = {}, {}
    for r in rows:
        by_sha.setdefault(r["hr_sha1"], set()).add(r["split"])
        by_dh.setdefault(r["dhash"], set()).add(r["split"])
    sha_cross = sum(1 for v in by_sha.values() if len(v) > 1)
    dup_within = len(rows) - len(by_sha)
    dh_cross = sum(1 for v in by_dh.values() if len(v) > 1)
    mismatch = 0
    for r in rows[:: max(1, len(rows) // n_rehash)][:n_rehash]:
        with Image.open(data_root / r["split"] / "HR" / f"{r['key']}.png") as im:
            if hashlib.sha1(np.asarray(im.convert("RGB"), np.uint8).tobytes()).hexdigest() != r["hr_sha1"]:
                mismatch += 1
    ev = {"identical_hr_across_splits": sha_cross, "identical_hr_total": dup_within,
          "dhash_groups_across_splits": dh_cross, "rehash_mismatch": mismatch}
    if sha_cross or dh_cross or mismatch:
        return "FAIL", f"leakage/corruption: {ev}", ev
    return "PASS", ("no identical or dHash-identical HR image appears in two splits; "
                    f"{n_rehash} files re-hashed OK"), ev


def check_split_ratios(data_root, tol=0.02):
    info = read_json(Path(data_root) / "dataset_info.json")
    n = sum(info["counts"].values())
    got = [info["counts"][s] / n for s in SPLITS]
    want = info["split_ratios"]
    off = max(abs(a - b) for a, b in zip(got, want))
    ev = {"ratios": [round(g, 4) for g in got], "target": want}
    if off > tol:
        return "WARN", f"split ratios {ev['ratios']} differ from {want} by {off:.3f}", ev
    return "PASS", f"split ratios {ev['ratios']} (target {want})", ev


def check_source(data_root):
    data_root = Path(data_root)
    info = read_json(data_root / "dataset_info.json")
    rows = read_manifest(data_root)
    small = sum(1 for r in rows if min(int(r["src_w"]), int(r["src_h"])) < info["hr_size"])
    nonsq = info.get("n_non_square_sources", 0)
    ev = {"source_sizes_top10": info.get("source_sizes_top10"), "source_modes": info.get("source_modes"),
          "smaller_than_hr": small, "non_square": nonsq, "unreadable": info.get("n_unreadable"),
          "exact_duplicates_dropped": info.get("n_exact_duplicates_dropped")}
    sizes = ", ".join(f"{k} ({v})" for k, v in list((info.get("source_sizes_top10") or {}).items())[:3])
    if small:
        return "FAIL", (f"{small} source images are smaller than the {info['hr_size']} px HR target, so their "
                        "'ground truth' would be upsampled; drop them or lower hr_size"), ev
    if nonsq > 0.05 * len(rows):
        return "WARN", f"{nonsq} non-square sources were centre-cropped; check a few. Sizes: {sizes}", ev
    return "PASS", (f"all sources >= HR size; most common sizes: {sizes}; "
                    f"{ev['exact_duplicates_dropped']} exact duplicates dropped"), ev


def check_edge_coverage(data_root, threshold, n=500, candidates=(0.15, 0.25, 0.5, 1.0)):
    ds = SRPairDataset(data_root, "val", cache=False, max_images=n)
    candidates = sorted(set(candidates) | {threshold})
    cov = {t: [] for t in candidates}
    for i in range(len(ds)):
        _, hr, _ = ds[i]
        y = rgb_to_y(quantize(hr[None]))[..., 2:-2, 2:-2]
        for t in candidates:
            cov[t].append(float(edge_mask(y, t).double().mean()))
    stats = {f"tau={t:g}": {"mean_coverage": round(float(np.mean(v)), 4),
                            "images_without_edges": int(sum(1 for c in v if c == 0))} for t, v in cov.items()}
    m = float(np.mean(cov[threshold]))
    empty = sum(1 for c in cov[threshold] if c == 0) / len(ds)
    detail = (f"locked tau={threshold:g}: mean edge coverage {m:.1%}, {empty:.1%} images without edge pixels "
              f"(others: " + ", ".join(f"{k} {v['mean_coverage']:.1%}" for k, v in stats.items()) + ")")
    if not (0.02 <= m <= 0.40) or empty > 0.01:
        return "WARN", detail + " -> reconsider tau before locking", stats
    return "PASS", detail, stats


# =========================================================================== R2 correctness
def check_sobel():
    const = torch.full((2, 3, 16, 16), 0.37)
    gx, gy = sobel_gradients(const)
    if gx.abs().max() > 1e-6 or gy.abs().max() > 1e-6:
        return "FAIL", "Sobel responds to a constant image (kernel not zero-sum or border padding wrong)", {}
    step = torch.zeros(1, 1, 16, 16)
    step[..., :, 8:] = 1.0  # vertical edge between columns 7 and 8
    gx, gy = sobel_gradients(step)
    if not (torch.allclose(gx[0, 0, :, 7], torch.full((16,), 4.0)) and torch.allclose(gx[0, 0, :, 8], torch.full((16,), 4.0))):
        return "FAIL", "unexpected Sobel response to a unit step", {}
    if gy.abs().max() > 1e-6:
        return "FAIL", "vertical edge leaks into gy", {}
    gx_t, gy_t = sobel_gradients(step.transpose(-1, -2))
    if not torch.allclose(gy_t, gx.transpose(-1, -2)):
        return "FAIL", "sobel_y is not the transpose of sobel_x", {}
    return "PASS", "zero response on flat images (incl. borders), 4h on a step of height h, y = x^T", {}


def check_adaptive_weight():
    torch.manual_seed(0)
    a = torch.rand(1, 3, 32, 32)
    b = torch.rand(1, 3, 32, 32) * 5
    ea, eb = sobel_magnitude(a), sobel_magnitude(b)
    w_alone = adaptive_edge_weight(ea)
    w_batch = adaptive_edge_weight(torch.cat([ea, eb]))[:1]
    if not torch.allclose(w_alone, w_batch, atol=1e-6):
        return "FAIL", "adaptive weight of an image depends on the other images in the batch", {}
    if w_alone.min() < 0 or w_alone.max() > 1 + 1e-6:
        return "FAIL", "adaptive weight outside [0, 1]", {}
    if abs(float(w_alone.max()) - 1) > 1e-4 or float(w_alone.min()) > 1e-6:
        return "FAIL", "per-image min-max normalisation does not reach 0 and 1", {}
    return "PASS", "w in [0,1], min 0 / max 1 per image, independent of batch composition", {}


def check_losses():
    torch.manual_seed(0)
    hr = torch.rand(2, 3, 32, 32)
    sr = (hr + 0.05 * torch.randn_like(hr)).requires_grad_(True)
    l1 = EdgeAwareLoss("none", 0.0)(sr, hr)
    if not torch.allclose(l1.total, torch.nn.functional.l1_loss(sr, hr)):
        return "FAIL", "edge_mode=none is not plain L1", {}
    st = EdgeAwareLoss("static", 0.3)(sr, hr)
    ref = torch.nn.functional.l1_loss(sr, hr) + 0.3 * (sobel_magnitude(sr) - sobel_magnitude(hr)).abs().mean()
    if not torch.allclose(st.total, ref):
        return "FAIL", "static loss != L1 + w * mean|S(sr) - S(hr)|", {}
    ad = EdgeAwareLoss("adaptive", 0.3)(sr, hr)
    ad.total.backward()
    if not torch.isfinite(sr.grad).all():
        return "FAIL", "non-finite gradient from the adaptive loss", {}
    same = EdgeAwareLoss("adaptive", 0.3)(hr, hr)
    if float(same.edge) > 1e-6 or float(same.pixel) != 0:
        return "FAIL", "loss is not zero for a perfect prediction", {}
    ev = {"static_edge_over_pixel": float(st.edge.detach() / st.pixel.detach()),
          "adaptive_edge_over_pixel": float(ad.edge.detach() / ad.pixel.detach()),
          "adaptive_weight_mean": float(ad.edge_weight_mean)}
    return "PASS", "none = L1; static = L1 + w*mean|dS|; adaptive finite and zero at SR = HR", ev


def check_metric_identities():
    torch.manual_seed(0)
    cfg = MetricConfig()
    hr = torch.rand(2, 3, 64, 64)
    hr255 = quantize(hr)
    off = (hr255 + 5).clamp(0, 255)
    inside = (hr255 <= 250).all()
    if inside and abs(float(psnr(off, hr255)[0]) - 20 * math.log10(255 / 5)) > 1e-9:
        return "FAIL", "PSNR of a 5-level offset is wrong", {}
    if (ssim(hr255, hr255) - 1).abs().max() > 1e-9:
        return "FAIL", "SSIM(x, x) != 1", {}
    if gmsd(hr255, hr255).abs().max() > 1e-9:
        return "FAIL", "GMSD(x, x) != 0", {}
    # Edge-PSNR with a uniform error must equal PSNR on Y (the notebook was 4.77 dB lower).
    yh = rgb_to_y(hr255)
    ys = yh + 3.0
    ep, cov = edge_psnr(ys, yh, cfg.edge_threshold)
    if (ep - psnr(ys, yh)).abs().max() > 1e-9 or (cov <= 0).any():
        return "FAIL", "Edge-PSNR != PSNR for a uniform error", {}
    flat = torch.full((1, 1, 32, 32), 100.0)
    if not torch.isnan(edge_psnr(flat + 1, flat, cfg.edge_threshold)[0]).all():
        return "FAIL", "Edge-PSNR of an image without edges should be NaN", {}
    m = image_metrics(hr.clamp(0, 1) + 2.0, hr, cfg)  # values > 1 must be clamped like a PNG
    if not all(torch.isfinite(m[k]).all() for k in ("psnr_rgb", "ssim_y", "gmsd")):
        return "FAIL", "metrics not finite on out-of-range input", {}
    return "PASS", ("PSNR offset exact; SSIM(x,x)=1; GMSD(x,x)=0; Edge-PSNR = PSNR under uniform error; "
                    "NaN without edges; output clamped"), {}


def check_metric_crosscheck():
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        return "SKIP", "scikit-image not installed (present on Kaggle)", {}
    torch.manual_seed(1)
    hr = torch.rand(3, 3, 64, 64)
    sr = (hr + 0.03 * torch.randn_like(hr)).clamp(0, 1)
    a, b = quantize(sr), quantize(hr)
    ya, yb = rgb_to_y(a), rgb_to_y(b)
    ours_p, ours_s = psnr(ya, yb), ssim(ya, yb)
    dp, ds = 0.0, 0.0
    for i in range(3):
        x, y = ya[i, 0].numpy(), yb[i, 0].numpy()
        dp = max(dp, abs(float(ours_p[i]) - peak_signal_noise_ratio(y, x, data_range=255)))
        ref = structural_similarity(y, x, data_range=255, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False)
        ds = max(ds, abs(float(ours_s[i]) - ref))
    ev = {"max_psnr_diff_db": dp, "max_ssim_diff": ds}
    if dp > 1e-6 or ds > 1e-4:
        return "FAIL", f"disagrees with scikit-image: {ev}", ev
    return "PASS", f"PSNR/SSIM match scikit-image (|dPSNR| {dp:.1e} dB, |dSSIM| {ds:.1e})", ev


# =========================================================================== R3 model
def check_model(preset, device):
    model = build_model(preset).to(device)
    n = count_params(model)
    exp = EXPECTED_PARAMS.get(preset)
    x = torch.rand(2, 3, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (2, 3, 128, 128) or not torch.isfinite(y).all():
        return "FAIL", f"forward gives {tuple(y.shape)} / non-finite", {}
    if exp is not None and n != exp:
        return "FAIL", f"{n:,} params, expected {exp:,}", {"params": n}
    return "PASS", f"{preset}: {n:,} params, 64x64 -> 128x128", {"params": n}


def check_provenance(timeout=10):
    try:
        with urllib.request.urlopen(OFFICIAL_SWINIR_URL, timeout=timeout) as r:
            official = r.read().decode("utf-8")
    except Exception as exc:
        return "SKIP", f"official source not reachable ({type(exc).__name__}); enable internet to check", {}
    ours = (REPO / "edgesr" / "models" / "swinir.py").read_text(encoding="utf-8")

    def body(src):
        src = src.split("class Mlp(nn.Module):", 1)[1]
        src = src.split("if __name__ == '__main__':", 1)[0]
        src = src.replace('torch.meshgrid([coords_h, coords_w], indexing="ij")', "torch.meshgrid([coords_h, coords_w])")
        return [ln.rstrip() for ln in src.strip().splitlines()]

    a, b = body(official), body(ours)
    if a != b:
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        return "FAIL", f"network code differs from the official SwinIR in {diff} lines", {}
    return "PASS", "network code identical to official SwinIR (apart from meshgrid indexing)", {}


# =========================================================================== R4 training
def _mini_cfg(base: TrainConfig, **kw) -> TrainConfig:
    d = dict(batch_size=4, num_workers=0, cache=True, augment=False, log_every=5, val_every=10 ** 9,
             ckpt_every=10 ** 9, val_max_images=0)
    d.update(kw)
    return replace(base, **d)


def check_overfit(base, mini_root, work, device, iters):
    out, ok = {}, True
    for mode, w in (("none", 0.0), ("static", 0.1), ("adaptive", 0.1)):
        cfg = _mini_cfg(base, edge_mode=mode, edge_weight=w, total_iters=iters, log_every=max(1, iters // 5),
                        val_every=iters, ckpt_every=iters, amp=base.amp)
        d = Path(work) / f"overfit_{mode}"
        shutil.rmtree(d, ignore_errors=True)
        engine.train(cfg, mini_root, d, device=device, log=lambda *_: None)
        log = read_csv(d / "train_log.csv")
        first, last = float(log[0]["loss"]), float(log[-1]["loss"])
        skipped = int(log[-1]["skipped"])
        out[mode] = {"first": first, "last": last, "ratio": last / first, "skipped": skipped,
                     "img_per_s": float(log[-1]["img_per_s"])}
        ok &= last < 0.6 * first and skipped == 0 and math.isfinite(last)
    detail = ", ".join(f"{k}: {v['first']:.4f} -> {v['last']:.4f}" for k, v in out.items())
    return ("PASS" if ok else "FAIL"), f"{iters} iters on 8 images: {detail}", out


def check_grad_flow(base, device):
    seed_everything(0)
    model = build_model(base.model).to(device)
    lr, hr = torch.rand(2, 3, 64, 64, device=device), torch.rand(2, 3, 128, 128, device=device)
    EdgeAwareLoss("adaptive", 0.1)(model(lr), hr).total.backward()
    none = [n for n, p in model.named_parameters() if p.grad is None]
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    zero = [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) == 0]
    if none or bad:
        return "FAIL", f"no grad: {none[:5]}, non-finite: {bad[:5]}", {}
    if zero:
        return "WARN", f"{len(zero)} tensors with exactly zero gradient: {zero[:5]}", {}
    return "PASS", f"all {sum(1 for _ in model.parameters())} parameter tensors receive finite, non-zero gradients", {}


def check_resume(base, mini_root, work):
    """Train 8 steps straight vs 4 + resume + 4 on CPU: weights must be identical."""
    cfg = _mini_cfg(base, model="light", batch_size=2, total_iters=8, val_every=4, ckpt_every=4,
                    augment=True, amp=False, log_every=2)
    cpu = torch.device("cpu")
    a, b = Path(work) / "resume_a", Path(work) / "resume_b"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    quiet = lambda *_: None  # noqa: E731
    engine.train(cfg, mini_root, a, device=cpu, log=quiet)
    engine.train(cfg, mini_root, b, device=cpu, log=quiet, max_steps=4)
    engine.train(cfg, mini_root, b, device=cpu, log=quiet)
    sa = torch.load(a / "model_final.pt", weights_only=False)["model"]
    sb = torch.load(b / "model_final.pt", weights_only=False)["model"]
    diff = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
    la, lb = read_csv(a / "train_log.csv"), read_csv(b / "train_log.csv")
    if [r["step"] for r in la] != [r["step"] for r in lb]:
        return "FAIL", "train_log steps differ after resume (duplicated or missing rows)", {}
    if diff > 1e-6:
        return "FAIL", f"resumed run differs from uninterrupted run (max |dw| = {diff:.2e})", {"max_abs_diff": diff}
    return "PASS", f"4+4 steps with resume == 8 straight steps (max |dw| = {diff:.1e}, CPU)", {"max_abs_diff": diff}


def check_seed_control(base, mini_root):
    def init(seed):
        seed_everything(seed)
        return build_model(base.model).state_dict()

    w0, w0b, w1 = init(0), init(0), init(1)
    same = all(torch.equal(w0[k], w0b[k]) for k in w0)
    differ = any(not torch.equal(w0[k], w1[k]) for k in w0 if w0[k].is_floating_point())
    s = TrainSampler(64, 4, seed=0, augment=True)
    order_same = TrainSampler(64, 4, 0, True).epoch_order(3)[0].tolist() == s.epoch_order(3)[0].tolist()
    order_diff = TrainSampler(64, 4, 1, True).epoch_order(0)[0].tolist() != s.epoch_order(0)[0].tolist()
    resumed = TrainSampler(64, 4, 0, True, start_step=20)
    it_full, it_res = iter(s), iter(resumed)
    full = [next(it_full) for _ in range(4 * 25)][4 * 20:]
    res = [next(it_res) for _ in range(4 * 5)]
    ok = same and differ and order_same and order_diff and full == res
    ev = {"same_seed_same_init": same, "diff_seed_diff_init": differ, "order_reproducible": order_same,
          "order_depends_on_seed": order_diff, "resume_continues_stream": full == res}
    return ("PASS" if ok else "FAIL"), ("seed fixes init, data order and augmentation; resumed sampler "
                                        "continues the same stream" if ok else f"{ev}"), ev


def check_test_isolation(data_root):
    try:
        SRPairDataset(data_root, "test", cache=False)
        return "FAIL", "test split can be opened without allow_test=True", {}
    except PermissionError:
        pass
    src = inspect.getsource(engine.train)
    if "allow_test" in src or '"test"' in src:
        return "FAIL", "engine.train references the test split", {}
    return "PASS", "test split refuses to load without allow_test; train() never requests it", {}


def check_amp(base, mini_root, device, iters=30):
    if device.type != "cuda":
        return "SKIP", "AMP is only used on CUDA", {}
    if not base.amp:
        return "SKIP", "amp disabled in base config", {}
    res = {}
    for amp in (False, True):
        seed_everything(0)
        model = build_model(base.model).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=base.lr, betas=base.betas)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        crit = EdgeAwareLoss("adaptive", 0.1)
        ds = SRPairDataset(mini_root, "train", cache=True)
        g = torch.Generator().manual_seed(0)
        losses = []
        for _ in range(iters):
            idx = torch.randint(0, len(ds), (base.batch_size,), generator=g).tolist()
            lr = torch.stack([ds[i][0] for i in idx]).to(device)
            hr = torch.stack([ds[i][1] for i in idx]).to(device)
            with engine.autocast(device, amp):
                sr = model(lr)
            loss = crit(sr, hr).total
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        res["amp" if amp else "fp32"] = losses
    a, f = np.mean(res["amp"][-5:]), np.mean(res["fp32"][-5:])
    rel = abs(a - f) / f
    finite = all(math.isfinite(x) for x in res["amp"])
    ev = {"final_loss_fp32": f, "final_loss_amp": a, "rel_diff": rel}
    if not finite:
        return "FAIL", "non-finite loss under AMP; set amp=false", ev
    if rel > 0.10:
        return "WARN", f"AMP and fp32 diverge after {iters} steps (rel. diff {rel:.1%})", ev
    return "PASS", f"AMP tracks fp32 over {iters} steps (final loss {a:.4f} vs {f:.4f})", ev


# =========================================================================== R5 pipeline
def check_dry_run(mini_root, work, quick):
    """Run the real CLI end to end (sweep -> bicubic -> select -> confirm -> analyze) on tiny data."""
    runs = Path(work) / "dry_runs"
    shutil.rmtree(runs, ignore_errors=True)
    overrides = ["total_iters=2", "batch_size=2", "val_every=1", "ckpt_every=1", "log_every=1",
                 "num_workers=0", "cache=false", "amp=false"]
    cmd = [sys.executable, str(REPO / "scripts" / "run_ablation.py"), "--data", str(mini_root),
           "--runs", str(runs), "--stage", "all", "--set", *overrides]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return "FAIL", f"run_ablation.py exited {p.returncode}: {p.stderr.strip()[-400:]}", {}
    p2 = subprocess.run([sys.executable, str(REPO / "scripts" / "analyze.py"), "--runs", str(runs),
                         "--data", str(mini_root), "--n-boot", "200"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p2.returncode != 0:
        return "FAIL", f"analyze.py exited {p2.returncode}: {p2.stderr.strip()[-400:]}", {}
    res = (runs / "analysis" / "RESULTS.md").read_text(encoding="utf-8")
    trained = list(runs.glob("*/train_done.json"))
    evaluated = list(runs.glob("*/test_summary.json"))
    n_confirm = sum(1 for d in trained if TrainConfig.load(d.parent / "config.json").seed in CONFIRM_SEEDS)
    n_sweep = len(trained) - n_confirm
    ok = ("Pre-registered hypotheses" in res and len(evaluated) == len(trained) + 1
          and n_confirm == 3 * len(CONFIRM_SEEDS) and n_sweep >= len(sweep_plan(TrainConfig())))
    detail = (f"{n_sweep} sweep/extension + {n_confirm} confirmation runs + bicubic trained, evaluated and "
              f"analysed in {time.time() - t0:.0f} s; RESULTS.md written")
    return ("PASS" if ok else "FAIL"), detail, {"runs_dir": str(runs)}


def check_bicubic(data_root, metric_cfg, device, n=500):
    ds = SRPairDataset(data_root, "val", cache=False, max_images=n)
    _, s = engine.evaluate(lambda lr: engine.bicubic_upscale(lr, 2), ds, metric_cfg, device)
    ev = {k: round(v, 4) for k, v in s.items() if isinstance(v, float)}
    if not all(math.isfinite(s[k]) for k in METRIC_NAMES):
        return "FAIL", f"non-finite bicubic metrics {ev}", ev
    detail = (f"val ({len(ds)} imgs): PSNR-Y {s['psnr_y']:.2f} dB, SSIM-Y {s['ssim_y']:.4f}, "
              f"Edge-PSNR-Y {s['edge_psnr_y']:.2f} dB, GMSD {s['gmsd']:.4f}")
    if not 20 <= s["psnr_y"] <= 50:
        return "WARN", detail + " (PSNR-Y outside the usual 20-50 dB range; inspect the data)", ev
    return "PASS", detail, ev


# =========================================================================== R6 budget
def measure_throughput(preset, batch_size, device, amp, iters=40, warmup=10):
    seed_everything(0)
    model = build_model(preset).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    crit = EdgeAwareLoss("adaptive", 0.1)
    lr = torch.rand(batch_size, 3, 64, 64, device=device)
    hr = torch.rand(batch_size, 3, 128, 128, device=device)
    if device.type == "cuda":
        free_gpu()
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with engine.autocast(device, amp):
            sr = model(lr)
        loss = crit(sr, hr).total
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else float("nan")
    return {"img_per_s": iters * batch_size / dt, "s_per_iter": dt / iters, "peak_mem_gb": mem}


def measure_eval_speed(preset, device, n=256, amp=False):
    model = build_model(preset).to(device).eval()
    pred = engine.model_predictor(model, device, amp)
    lr = torch.rand(32, 3, 64, 64, device=device)
    hr = torch.rand(32, 3, 128, 128, device=device)
    with torch.no_grad():
        pred(lr)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n // 32):
            image_metrics(pred(lr), hr)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / n


def check_loader(base, data_root, n=2000, batches=50):
    t0 = time.time()
    ds = SRPairDataset(data_root, "train", cache=base.cache, max_images=n)
    load_s_per_img = (time.time() - t0) / len(ds)
    sampler = TrainSampler(len(ds), base.batch_size, 0, base.augment)
    loader = torch.utils.data.DataLoader(ds, batch_size=base.batch_size, sampler=sampler,
                                         num_workers=base.num_workers, drop_last=True)
    it = iter(loader)
    next(it)
    t0 = time.time()
    for _ in range(batches):
        next(it)
    ips = batches * base.batch_size / (time.time() - t0)
    return {"loader_img_per_s": ips, "cache_load_s_per_img": load_s_per_img}


def check_budget(base, data_root, device, gpu_hours, compare_presets):
    if device.type != "cuda":
        return "SKIP", "no GPU: run the gate on Kaggle to measure the budget", {}
    info = read_json(Path(data_root) / "dataset_info.json")
    n_train, n_val, n_test = (info["counts"][s] for s in SPLITS)
    n_runs = len(sweep_plan(base)) + len(confirm_plan(base, 0.1, 0.1))
    loader = check_loader(base, data_root)
    n_val_eval = min(n_val, base.val_max_images or n_val)
    val_points = base.total_iters // base.val_every + (base.total_iters % base.val_every != 0)
    rows, oom = {}, {}
    for preset in [base.model] + [p for p in compare_presets if p != base.model]:
        try:
            thr = measure_throughput(preset, base.batch_size, device, base.amp)
            val_s = measure_eval_speed(preset, device, amp=base.amp)
            test_s = measure_eval_speed(preset, device, amp=False)
        except torch.cuda.OutOfMemoryError:
            oom[preset] = True
        free_gpu()  # outside the except block, so the traceback no longer pins the tensors
        if oom.get(preset):
            if preset == base.model:
                return "FAIL", (f"{preset} does not fit in GPU memory at batch {base.batch_size}: "
                                "lower batch_size in configs/base.json (before the lock)"), {}
            rows[preset] = {"oom_at_batch": base.batch_size}
            continue
        train_s = base.total_iters * base.batch_size / min(thr["img_per_s"], loader["loader_img_per_s"])
        per_run = (train_s + val_points * n_val_eval * val_s + n_test * test_s
                   + (n_train + n_val_eval) * loader["cache_load_s_per_img"] + 30)
        total_h = per_run * n_runs / 3600
        fixed = per_run - train_s
        fit_iters = int(max(0, (gpu_hours * 3600 / n_runs - fixed)) * min(thr["img_per_s"], loader["loader_img_per_s"])
                        / base.batch_size) // 1000 * 1000
        rows[preset] = {**{k: round(v, 3) for k, v in thr.items()}, "val_s_per_img": val_s,
                        "per_run_h": round(per_run / 3600, 2), "total_h": round(total_h, 1),
                        "epochs_per_run": round(base.total_iters * base.batch_size / n_train, 1),
                        "max_total_iters_within_budget": fit_iters}
    r = rows[base.model]
    ev = {"presets": rows, "loader": loader, "n_runs": n_runs, "gpu_hours_budget": gpu_hours,
          "total_iters": base.total_iters, "batch_size": base.batch_size}
    detail = (f"{base.model}: {r['img_per_s']:.0f} img/s (loader {loader['loader_img_per_s']:.0f}), "
              f"{r['peak_mem_gb']:.1f} GB, {r['per_run_h']:.2f} h/run x {n_runs} runs = {r['total_h']:.1f} GPU-h "
              f"({r['epochs_per_run']} epochs/run); budget {gpu_hours} h -> total_iters <= "
              f"{r['max_total_iters_within_budget']}")
    for p, v in rows.items():
        if p != base.model and "oom_at_batch" in v:
            detail += f"; {p}: does not fit in GPU memory at batch {v['oom_at_batch']} (comparison only)"
        elif p != base.model:
            detail += f"; {p}: {v['img_per_s']:.0f} img/s, {v['total_h']:.1f} GPU-h"
    if loader["loader_img_per_s"] < 1.2 * r["img_per_s"]:
        detail += " | data loader is the bottleneck: raise num_workers or keep cache=true"
    if r["per_run_h"] > SESSION_HOURS:
        return "FAIL", detail + f" | one run exceeds the {SESSION_HOURS} h session limit", ev
    if r["total_h"] > gpu_hours:
        return "WARN", detail + " | over budget: lower total_iters (above) or spread over more weeks", ev
    return "PASS", detail, ev


# =========================================================================== R7 protocol
def _sha1_file(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def git_state():
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "edgesr", "scripts",
                                     "configs"], capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def protocol_document(base: TrainConfig, data_root) -> dict:
    data_root = Path(data_root)
    info = read_json(data_root / "dataset_info.json")
    return {
        "protocol_hash": base.protocol_hash(),
        "base_config": base.to_dict(),
        "plan": {"sweep_weights": list(SWEEP_WEIGHTS), "sweep_seeds": list(SWEEP_SEEDS),
                 "confirm_seeds": list(CONFIRM_SEEDS),
                 "n_training_runs": len(sweep_plan(base)) + len(confirm_plan(base, 0.1, 0.1))},
        "selection_rule": {"metric": SELECT_METRIC, "split": "val", "last_k": LAST_K_VALIDATIONS,
                           "psnr_guard_db": PSNR_TOLERANCE_DB},
        "hypotheses": {
            "H1": "adaptive(w*) > L1 on test Edge-PSNR-Y",
            "H2": "adaptive(w*) > static(w*) on test Edge-PSNR-Y",
            "H3": "adaptive(w*) non-inferior to L1 on test PSNR-Y, margin 0.05 dB",
            "decision": "95% two-level bootstrap CI excludes 0, Holm-adjusted Wilcoxon p < 0.05 (H1-H2), "
                        "and all per-seed differences share the sign; H3: CI lower bound > -0.05 dB",
        },
        "dataset": {"version": info["version"], "counts": info["counts"], "limit": info.get("limit"),
                    "manifest_sha1": _sha1_file(data_root / "manifest.csv")},
        "code": {**git_state(), "files": {str(p.relative_to(REPO)).replace("\\", "/"): _sha1_file(p)[:12]
                                          for p in sorted((REPO / "edgesr").rglob("*.py"))}},
    }


def check_protocol_lock(base, data_root, out_dir):
    doc = protocol_document(base, data_root)
    write_json(Path(out_dir) / "protocol_lock.json", doc)
    locked = REPO / "configs" / "protocol_lock.json"
    if locked.exists():
        old = read_json(locked)
        diffs = [k for k in ("protocol_hash", "plan", "selection_rule", "hypotheses")
                 if json.dumps(old.get(k), sort_keys=True) != json.dumps(doc[k], sort_keys=True)]
        if old["dataset"]["manifest_sha1"] != doc["dataset"]["manifest_sha1"]:
            diffs.append("dataset")
        if diffs:
            return "FAIL", f"differs from the committed configs/protocol_lock.json in {diffs}", {"diffs": diffs}
        return "PASS", f"matches committed lock {doc['protocol_hash']}", {}
    if doc["dataset"]["limit"]:
        return "WARN", "written, but the dataset is a pilot subset: do not commit this lock", {}
    return "WARN", (f"protocol {doc['protocol_hash']} written to {Path(out_dir) / 'protocol_lock.json'}; copy it to "
                    "configs/ and commit before starting the sweep"), {}


# =========================================================================== driver
def run_gate(data_root, out_dir, base: TrainConfig, gpu_hours=25.0, require_gpu=False, quick=False,
             compare_presets=("medium",), log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = Report(log)
    n_runs = len(sweep_plan(base)) + len(confirm_plan(base, 0.1, 0.1))
    model_mb = count_params(build_model(base.model)) * 4 / 2 ** 20

    rep.run("R0.1", "Python / PyTorch", check_versions)
    rep.run("R0.2", "GPU", check_gpu, require_gpu)
    rep.run("R0.3", "Optional packages", check_packages)
    rep.run("R0.4", "Disk space", check_disk, out_dir, data_root, n_runs, model_mb)

    have_data = data_root and (Path(data_root) / "manifest.csv").exists()
    if have_data:
        rep.run("R1.1", "Manifest and files", check_manifest, data_root)
        rep.run("R1.2", "Image sizes", check_image_sizes, data_root, 50 if quick else 200)
        rep.run("R1.3", "Degradation = PIL bicubic", check_degradation, data_root, 20 if quick else 50)
        rep.run("R1.4", "No train/val/test leakage", check_leakage, data_root, 30 if quick else 100)
        rep.run("R1.5", "Split ratios", check_split_ratios, data_root)
        rep.run("R1.6", "Source images", check_source, data_root)
        rep.run("R1.7", "Edge-PSNR mask coverage", check_edge_coverage, data_root,
                base.metrics.edge_threshold, 100 if quick else 500)
    else:
        rep.items.append({"id": "R1", "title": "Data", "status": "FAIL", "seconds": 0, "evidence": {},
                          "detail": f"no dataset at {data_root}; run scripts/prepare_data.py first"})
        log(f"[FAIL] R1 Data: no dataset at {data_root}")

    rep.run("R2.1", "Sobel operator", check_sobel)
    rep.run("R2.2", "Adaptive weight", check_adaptive_weight)
    rep.run("R2.3", "Loss modes", check_losses)
    rep.run("R2.4", "Metric identities", check_metric_identities)
    rep.run("R2.5", "PSNR/SSIM vs scikit-image", check_metric_crosscheck)

    rep.run("R3.1", "Model build", check_model, base.model, device)
    rep.run("R3.2", "Model = official SwinIR", check_provenance)

    if have_data:
        mini = work / "mini_data"
        shutil.rmtree(mini, ignore_errors=True)
        make_subset(data_root, mini, {"train": 8, "val": 4, "test": 4})
        dry = work / "dry_data"
        shutil.rmtree(dry, ignore_errors=True)
        make_subset(data_root, dry, {"train": 8, "val": 4, "test": 4})
        overfit_iters = 30 if (quick or device.type == "cpu") else 200
        rep.run("R4.1", "Overfit 8 images (each loss)", check_overfit, base, mini, work, device, overfit_iters)
        rep.run("R4.2", "Gradient flow", check_grad_flow, base, device)
        rep.run("R4.3", "Checkpoint resume is exact", check_resume, base, mini, work)
        rep.run("R4.4", "Seed control", check_seed_control, base, mini)
        rep.run("R4.5", "Test split isolation", check_test_isolation, data_root)
        rep.run("R4.6", "AMP vs fp32", check_amp, base, mini, device)
        rep.run("R5.1", "Dry run of the whole pipeline", check_dry_run, dry, work, quick)
        rep.run("R5.2", "Bicubic baseline on val", check_bicubic, data_root, base.metrics, device,
                100 if quick else 500)
        if not quick:
            rep.run("R6.1", "GPU time budget", check_budget, base, data_root, device, gpu_hours, compare_presets)
        rep.run("R7.1", "Protocol lock", check_protocol_lock, base, data_root, out_dir)

    header = (f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on {env_info().get('gpu', 'CPU')} · "
              f"data `{data_root}` · protocol `{base.protocol_hash()}` · mode {'quick' if quick else 'full'}")
    write_json(out_dir / "readiness_report.json", {"verdict": rep.verdict(), "header": header, "items": rep.items})
    (out_dir / "readiness_report.md").write_text(rep.markdown(header), encoding="utf-8")
    return rep
