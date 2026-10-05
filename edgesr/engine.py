"""Training loop, checkpointing and per-image evaluation."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import SRPairDataset, TrainSampler
from .losses import build_loss
from .metrics import METRIC_NAMES, MetricConfig, image_metrics
from .models import build_model, count_params
from .utils import CsvLogger, env_info, get_device, seed_everything, truncate_csv, write_json

CKPT_NAME = "last.pt"
DONE_NAME = "train_done.json"


# --------------------------------------------------------------------------- helpers
BOOKKEEPING_KEYS = ("num_workers", "cache", "log_every", "ckpt_every")


def _result_relevant(cfg: TrainConfig) -> str:
    """Config as canonical JSON without the keys that cannot change the result."""
    d = {k: v for k, v in cfg.to_dict().items() if k not in BOOKKEEPING_KEYS}
    return json.dumps(d, sort_keys=True)


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """Linear warm-up, then cosine decay from lr to eta_min over total_iters (per iteration)."""
    if cfg.warmup_iters and step < cfg.warmup_iters:
        return (step + 1) / cfg.warmup_iters
    t = (step - cfg.warmup_iters) / max(1, cfg.total_iters - cfg.warmup_iters)
    floor = cfg.eta_min / cfg.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def set_lr(optimizer, step, cfg):
    lr = cfg.lr * lr_factor(step, cfg)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def save_checkpoint(path, model, optimizer, scaler, step, cfg, elapsed=0.0):
    state = {
        "step": step,
        "elapsed": elapsed,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken checkpoint


def load_model(path, device="cpu"):
    """Rebuild the model from a checkpoint (last.pt or model_final.pt)."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig.from_dict(state["config"])
    model = build_model(cfg.model, cfg.scale, cfg.lr_size).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, state


def bicubic_upscale(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """PIL bicubic upsampling of a [B,3,h,w] batch in [0,1] (the same kernel as the downsampler)."""
    out = []
    for img in lr:
        arr = (img.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
        im = Image.fromarray(arr).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.BICUBIC)
        out.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out).to(lr.device)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(enabled and device.type == "cuda"))


# --------------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(predict, dataset, metric_cfg: MetricConfig, device, batch_size=32,
             lpips_fn=None, save_dir=None, save_n=0):
    """Run `predict(lr_batch) -> sr_batch` over a dataset and score every image.

    Returns (rows, summary): one dict per image, and the mean of every metric
    (NaN Edge-PSNR values, from images without edge pixels, are excluded and counted).
    """
    # A private generator: creating the iterator must not consume the global RNG, which
    # drives DropPath during training (otherwise validation would change the training run).
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    rows = []
    for lr, hr, idx in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = predict(lr)
        m = image_metrics(sr, hr, metric_cfg, lpips_fn=lpips_fn)
        for j in range(lr.shape[0]):
            i = int(idx[j])
            row = {"key": dataset.keys[i]}
            row.update({k: float(v[j]) for k, v in m.items()})
            rows.append(row)
            if save_dir is not None and i < save_n:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                arr = sr[j].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(arr).save(Path(save_dir) / f"{dataset.keys[i]}.png")
    names = [k for k in rows[0] if k != "key"]
    summary = {"n_images": len(rows)}
    for k in names:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k] = float(np.nanmean(vals))
        if np.isnan(vals).any():
            summary[f"{k}_n_nan"] = int(np.isnan(vals).sum())
    return rows, summary


def model_predictor(model, device, amp):
    def predict(lr):
        with autocast(device, amp):
            return model(lr).float()
    return predict


# --------------------------------------------------------------------------- train
def train(cfg: TrainConfig, data_root, run_dir, device=None, log=print, max_steps=None) -> dict:
    """Train one run. Resumes automatically from run_dir/last.pt if it exists.

    `max_steps` stops early (for smoke tests) without changing the LR schedule.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or get_device()
    seed_everything(cfg.seed)  # same seed -> same initial weights for every condition

    model = build_model(cfg.model, cfg.scale, cfg.lr_size).to(device)
    criterion = build_loss(cfg.edge_mode, cfg.edge_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, prior_elapsed = 0, 0.0
    ckpt_path = run_dir / CKPT_NAME
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = TrainConfig.from_dict(state["config"])
        if _result_relevant(saved) != _result_relevant(cfg):
            raise RuntimeError(f"{ckpt_path} was written with a different config; use a new run dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, prior_elapsed = state["step"], state.get("elapsed", 0.0)
        log(f"[train] resumed {cfg.run_name} at step {step}")
    else:
        cfg.save(run_dir / "config.json")
        write_json(run_dir / "env.json", env_info())
    # Rows logged after the checkpoint would be repeated after a resume; drop them.
    truncate_csv(run_dir / "train_log.csv", step)
    truncate_csv(run_dir / "val_log.csv", step)

    train_set = SRPairDataset(data_root, "train", cache=cfg.cache)
    val_set = SRPairDataset(data_root, "val", cache=cfg.cache, max_images=cfg.val_max_images or None)
    sampler = TrainSampler(len(train_set), cfg.batch_size, cfg.seed, cfg.augment, start_step=step)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=sampler,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=cfg.num_workers > 0,
                        generator=torch.Generator().manual_seed(cfg.seed))
    train_log = CsvLogger(run_dir / "train_log.csv")
    val_log = CsvLogger(run_dir / "val_log.csv")
    log(f"[train] {cfg.run_name}: {count_params(model):,} params, {len(train_set)} train / "
        f"{len(val_set)} val images, {cfg.total_iters} iters x batch {cfg.batch_size}, "
        f"device={device}, amp={use_amp}")

    def run_validation(at_step):
        model.eval()
        _, s = evaluate(model_predictor(model, device, use_amp), val_set, cfg.metrics, device,
                        batch_size=cfg.batch_size)
        model.train()
        val_log.log({"step": at_step, **{k: s[k] for k in METRIC_NAMES}})
        log(f"[val] step {at_step}: PSNR-Y {s['psnr_y']:.3f}  SSIM-Y {s['ssim_y']:.4f}  "
            f"Edge-PSNR-Y {s['edge_psnr_y']:.3f}  GMSD {s['gmsd']:.4f}")
        return s

    last_steps = cfg.total_iters if max_steps is None else min(cfg.total_iters, max_steps)
    model.train()
    sums = {"loss": 0.0, "pixel": 0.0, "edge": 0.0, "w_mean": 0.0}
    acc = {k: torch.zeros((), device=device) for k in sums}
    n_acc, n_skipped, t_window, imgs_window = 0, 0, time.time(), 0
    t_start = time.time()
    data_iter = iter(loader)
    if step > 0:
        # Restore the RNG streams last, so nothing above shifts the DropPath masks.
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    while step < last_steps:
        lr_imgs, hr_imgs, _ = next(data_iter)
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        cur_lr = set_lr(optimizer, step, cfg)

        with autocast(device, use_amp):
            sr = model(lr_imgs)
        out = criterion(sr, hr_imgs)  # computed in fp32

        if not torch.isfinite(out.total):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if n_skipped > 20:
                raise FloatingPointError(f"{n_skipped} non-finite losses; stopping {cfg.run_name}")
            step += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(out.total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
        scaler.step(optimizer)
        scaler.update()
        step += 1

        acc["loss"] += out.total.detach()
        acc["pixel"] += out.pixel.detach()
        acc["edge"] += out.edge.detach()
        acc["w_mean"] += out.edge_weight_mean.detach()
        n_acc += 1
        imgs_window += lr_imgs.shape[0]

        if step % cfg.log_every == 0 or step == last_steps:
            dt = time.time() - t_window
            row = {"step": step, "lr": cur_lr}
            row.update({k: float(v) / n_acc for k, v in acc.items()})
            row.update(grad_norm=float(grad_norm), img_per_s=imgs_window / max(dt, 1e-9),
                       skipped=n_skipped, elapsed_s=prior_elapsed + time.time() - t_start)
            if device.type == "cuda":
                row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            train_log.log(row)
            log(f"[train] step {step}/{cfg.total_iters} loss {row['loss']:.5f} "
                f"(pix {row['pixel']:.5f}, edge {row['edge']:.5f}) lr {cur_lr:.2e} "
                f"{row['img_per_s']:.0f} img/s")
            acc = {k: torch.zeros((), device=device) for k in sums}
            n_acc, t_window, imgs_window = 0, time.time(), 0

        if step % cfg.val_every == 0 or step == cfg.total_iters:
            run_validation(step)
        if step % cfg.ckpt_every == 0 or step == last_steps:
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, cfg,
                            elapsed=prior_elapsed + time.time() - t_start)

    if step < cfg.total_iters:
        log(f"[train] stopped at step {step} (max_steps); not marking the run as done")
        return {"status": "partial", "step": step}

    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step},
               run_dir / "model_final.pt")
    done = {"status": "done", "step": step, "run_name": cfg.run_name, "condition": cfg.condition,
            "seed": cfg.seed, "params": count_params(model), "skipped_steps": n_skipped,
            "train_seconds": round(prior_elapsed + time.time() - t_start, 1),
            "protocol_hash": cfg.protocol_hash()}
    write_json(run_dir / DONE_NAME, done)
    log(f"[train] finished {cfg.run_name}")
    return done


# --------------------------------------------------------------------------- final evaluation
def _write_eval(out_dir, split, rows, summary):
    out_dir = Path(out_dir)
    log = CsvLogger(out_dir / f"{split}_per_image.csv")
    if log.path.exists():
        log.path.unlink()
        log.fields = None
    for r in rows:
        log.log(r)
    write_json(out_dir / f"{split}_summary.json", summary)


def evaluate_run(run_dir, data_root, split="test", device=None, lpips=False, save_n=16,
                 batch_size=32) -> dict:
    """Score run_dir/model_final.pt on a split, in fp32, and write {split}_*.csv/json."""
    from .metrics import try_build_lpips
    device = device or get_device()
    run_dir = Path(run_dir)
    model, cfg, state = load_model(run_dir / "model_final.pt", device)
    ds = SRPairDataset(data_root, split, cache=False, allow_test=(split == "test"))
    lp = try_build_lpips(device) if lpips else None
    rows, summary = evaluate(model_predictor(model, device, amp=False), ds, cfg.metrics, device,
                             batch_size, lpips_fn=lp, save_dir=run_dir / f"samples_{split}",
                             save_n=save_n)
    summary.update(run_name=cfg.run_name, condition=cfg.condition, seed=cfg.seed, split=split,
                   step=state["step"], metric_config=cfg.metrics, protocol_hash=cfg.protocol_hash())
    _write_eval(run_dir, split, rows, summary)
    return summary


def evaluate_bicubic(data_root, out_dir, split="test", metric_cfg: MetricConfig = MetricConfig(),
                     scale=2, device=None, lpips=False, save_n=16, batch_size=32) -> dict:
    """Score PIL-bicubic upsampling (the no-learning baseline) on a split."""
    from .metrics import try_build_lpips
    device = device or get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SRPairDataset(data_root, split, cache=False, allow_test=(split == "test"))
    lp = try_build_lpips(device) if lpips else None
    rows, summary = evaluate(lambda lr: bicubic_upscale(lr, scale), ds, metric_cfg, device,
                             batch_size, lpips_fn=lp, save_dir=out_dir / f"samples_{split}",
                             save_n=save_n)
    summary.update(run_name="bicubic", condition="bicubic", seed=None, split=split,
                   metric_config=metric_cfg)
    _write_eval(out_dir, split, rows, summary)
    write_json(out_dir / "bicubic_done.json", {"status": "done", "split": split})
    return summary
