"""Dataset preparation, the paired LR/HR dataset and a resumable training sampler.

Preparation (run once, see scripts/prepare_data.py):

    source image --(RGBA -> composite on white)--> RGB
                 --(centre crop to square, if needed)--> square
                 --(PIL bicubic)--> HR 128x128 --(PIL bicubic)--> LR 64x64

Images are grouped by a 64-bit difference hash (dHash) of the HR image. Exact
duplicates (identical HR pixels) are dropped, and every dHash group is kept inside one
split, so near-identical faces cannot end up in both train and test. Groups are
assigned to train/val/test with a seeded permutation; the result is written to
manifest.csv together with dataset_info.json.
"""
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DATASET_VERSION = "anime-x2-v1"


# --------------------------------------------------------------------------- prepare
def list_images(src_dir) -> list:
    src_dir = Path(src_dir)
    files = [p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    # Sort so the split does not depend on the order the filesystem returns files in.
    return sorted(files, key=lambda p: p.relative_to(src_dir).as_posix())


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img.convert("RGB")


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def make_pair(img: Image.Image, hr_size: int, scale: int):
    """The degradation model used everywhere in this project (bicubic downsampling)."""
    hr = center_square(to_rgb(img)).resize((hr_size, hr_size), Image.BICUBIC)
    lr = hr.resize((hr_size // scale, hr_size // scale), Image.BICUBIC)
    return hr, lr


def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _process_one(args):
    path, rel, out_tmp, hr_size, scale = args
    try:
        with Image.open(path) as im:
            im.load()
            mode, (w, h) = im.mode, im.size
            hr, lr = make_pair(im, hr_size, scale)
    except Exception as exc:  # unreadable file: record and skip
        return {"rel": rel, "error": repr(exc)}
    arr = np.asarray(hr, dtype=np.uint8)
    key = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    hr.save(os.path.join(out_tmp, "HR", key + ".png"))
    lr.save(os.path.join(out_tmp, "LR", key + ".png"))
    return {"rel": rel, "key": key, "src_w": w, "src_h": h, "src_mode": mode,
            "hr_sha1": hashlib.sha1(arr.tobytes()).hexdigest(), "dhash": dhash(hr)}


def prepare_dataset(src_dir, out_dir, hr_size=128, scale=2, ratios=(0.8, 0.1, 0.1),
                    seed=42, limit=None, workers=4, log=print) -> dict:
    """Build out_dir/{train,val,test}/{HR,LR}/*.png, manifest.csv and dataset_info.json."""
    t0 = time.time()
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    files = list_images(src_dir)
    if not files:
        raise FileNotFoundError(f"no images found under {src_dir}")
    n_found = len(files)
    if limit:
        # Subsample deterministically (pilot runs only; the info file records it).
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), size=min(limit, len(files)), replace=False))]
    log(f"[prepare] {n_found} images found, processing {len(files)}")

    tmp = out_dir / "_all"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; delete it or choose another --out")
    (tmp / "HR").mkdir(parents=True)
    (tmp / "LR").mkdir(parents=True)
    jobs = [(str(p), p.relative_to(src_dir).as_posix(), str(tmp), hr_size, scale) for p in files]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_process_one, jobs, chunksize=64))
    else:
        rows = [_process_one(j) for j in jobs]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    log(f"[prepare] decoded {len(rows)} images, {len(errors)} unreadable")

    # Drop exact duplicates (identical HR pixels), keeping the first in sorted order.
    seen, kept, dropped = set(), [], []
    for r in rows:
        if r["hr_sha1"] in seen:
            dropped.append(r)
        else:
            seen.add(r["hr_sha1"])
            kept.append(r)
    for r in dropped:
        os.remove(tmp / "HR" / (r["key"] + ".png"))
        os.remove(tmp / "LR" / (r["key"] + ".png"))

    # Group near-duplicates by dHash and split whole groups.
    groups = defaultdict(list)
    for r in kept:
        groups[r["dhash"]].append(r)
    gkeys = sorted(groups)
    perm = np.random.default_rng(seed).permutation(len(gkeys))
    n_imgs = len(kept)
    targets = [ratios[0] * n_imgs, (ratios[0] + ratios[1]) * n_imgs]
    count = 0
    for gi in perm:
        split = "train" if count < targets[0] else ("val" if count < targets[1] else "test")
        for r in groups[gkeys[gi]]:
            r["split"] = split
        count += len(groups[gkeys[gi]])

    for s in SPLITS:
        (out_dir / s / "HR").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "LR").mkdir(parents=True, exist_ok=True)
    for r in kept:
        for kind in ("HR", "LR"):
            os.replace(tmp / kind / (r["key"] + ".png"), out_dir / r["split"] / kind / (r["key"] + ".png"))
    for kind in ("HR", "LR"):
        (tmp / kind).rmdir()
    tmp.rmdir()

    kept.sort(key=lambda r: (SPLITS.index(r["split"]), r["key"]))
    fields = ["key", "split", "rel", "src_w", "src_h", "src_mode", "hr_sha1", "dhash"]
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            w.writerow({k: r[k] for k in fields})

    sizes = Counter(f"{r['src_w']}x{r['src_h']}" for r in kept)
    info = {
        "version": DATASET_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": str(src_dir),
        "hr_size": hr_size, "lr_size": hr_size // scale, "scale": scale,
        "degradation": "PIL bicubic (HR from source, LR from HR); RGBA composited on white; centre square crop",
        "split_ratios": list(ratios), "split_seed": seed,
        "split_unit": "dHash group (near-duplicates never cross splits)",
        "limit": limit, "n_found": n_found, "n_processed": len(files),
        "n_unreadable": len(errors), "unreadable_examples": [e["rel"] for e in errors[:20]],
        "n_exact_duplicates_dropped": len(dropped),
        "n_images": len(kept), "n_dhash_groups": len(groups),
        "n_groups_with_more_than_one_image": sum(1 for g in groups.values() if len(g) > 1),
        "counts": {s: sum(1 for r in kept if r["split"] == s) for s in SPLITS},
        "source_modes": dict(Counter(r["src_mode"] for r in kept)),
        "source_sizes_top10": dict(sizes.most_common(10)),
        "n_non_square_sources": sum(1 for r in kept if r["src_w"] != r["src_h"]),
        "seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    log(f"[prepare] done: {info['counts']} ({info['n_exact_duplicates_dropped']} exact duplicates dropped) "
        f"in {info['seconds']} s")
    return info


def read_manifest(data_root) -> list:
    with open(Path(data_root) / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- dataset
def _load_png(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)  # writable copy


def dihedral(x: torch.Tensor, code: int) -> torch.Tensor:
    """One of the 8 flips/rotations of the square (code 0 = identity)."""
    if code & 1:
        x = torch.flip(x, dims=[-1])
    if code & 2:
        x = torch.flip(x, dims=[-2])
    if code & 4:
        x = x.transpose(-1, -2)
    return x


_DECODED = {}  # (root, split) -> (keys, [(lr, hr), ...])


class SRPairDataset(Dataset):
    """Paired LR/HR images of one split. Items are (lr, hr, index) float tensors in [0, 1].

    `split="test"` must be requested explicitly with `allow_test=True`; the training code
    never does this, so the test images cannot leak into training or model selection.
    Index items with an int, or with (index, aug_code) tuples from TrainSampler.
    """

    def __init__(self, data_root, split: str, cache: bool = True, allow_test: bool = False,
                 max_images: int = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if split == "test" and not allow_test:
            raise PermissionError("the test split is only for final evaluation (pass allow_test=True)")
        self.root = Path(data_root)
        self.split = split
        self.keys = [r["key"] for r in read_manifest(data_root) if r["split"] == split]
        if max_images:
            self.keys = self.keys[:max_images]
        if not self.keys:
            raise RuntimeError(f"split {split!r} is empty in {data_root}")
        self.cache = None
        if cache:
            # Decoded images are shared by every dataset object in this process, so the
            # 20 runs of run_ablation.py decode the training split only once.
            ck = (str(self.root.resolve()), split)
            stored = _DECODED.get(ck)
            if stored is None or stored[0] != self.keys[: len(stored[0])] or len(stored[0]) < len(self.keys):
                stored = (list(self.keys), [self._read(i) for i in range(len(self.keys))])
                _DECODED[ck] = stored
            self.cache = stored[1][: len(self.keys)]

    def _read(self, i):
        k = self.keys[i]
        return (_load_png(self.root / self.split / "LR" / f"{k}.png"),
                _load_png(self.root / self.split / "HR" / f"{k}.png"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        idx, code = item if isinstance(item, tuple) else (item, 0)
        lr, hr = self.cache[idx] if self.cache is not None else self._read(idx)
        lr = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float().div_(255.0)
        hr = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float().div_(255.0)
        if code:
            lr, hr = dihedral(lr, code), dihedral(hr, code)
        return lr, hr, idx


class TrainSampler(Sampler):
    """Infinite, resumable stream of (index, aug_code) pairs.

    The order of epoch e is a pure function of (seed, e), so a run resumed at step s
    sees exactly the batches it would have seen without the interruption.
    """

    def __init__(self, n: int, batch_size: int, seed: int, augment: bool, start_step: int = 0):
        if n < batch_size:
            raise ValueError(f"training split has {n} images, fewer than batch_size={batch_size}")
        self.n, self.bs, self.seed, self.augment = n, batch_size, seed, augment
        self.steps_per_epoch = n // batch_size  # drop the incomplete last batch
        self.start_step = start_step

    def epoch_order(self, epoch: int):
        g = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
        perm = torch.randperm(self.n, generator=g)
        codes = torch.randint(0, 8, (self.n,), generator=g) if self.augment else torch.zeros(self.n, dtype=torch.long)
        return perm, codes

    def __iter__(self):
        epoch, offset = divmod(self.start_step, self.steps_per_epoch)
        while True:
            perm, codes = self.epoch_order(epoch)
            for i in range(offset * self.bs, self.steps_per_epoch * self.bs):
                yield int(perm[i]), int(codes[i])
            epoch, offset = epoch + 1, 0

    def __len__(self):  # not meaningful for an infinite sampler; DataLoader does not need it
        return 2 ** 62


# --------------------------------------------------------------------------- synthetic
def make_synthetic_source(out_dir, n: int = 64, size: int = 256, seed: int = 0) -> Path:
    """Write n cartoon-like images (flat colours + dark outlines) for tests and dry runs."""
    from PIL import ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        bg = tuple(int(v) for v in rng.integers(150, 256, 3))
        im = Image.new("RGB", (size, size), bg)
        d = ImageDraw.Draw(im)
        for _ in range(int(rng.integers(3, 8))):
            x0, y0 = (int(v) for v in rng.integers(0, size - 40, 2))
            x1, y1 = x0 + int(rng.integers(30, size // 2)), y0 + int(rng.integers(30, size // 2))
            fill = tuple(int(v) for v in rng.integers(0, 256, 3))
            width = int(rng.integers(2, 6))
            if rng.random() < 0.5:
                d.ellipse([x0, y0, x1, y1], fill=fill, outline=(20, 20, 30), width=width)
            else:
                d.rectangle([x0, y0, x1, y1], fill=fill, outline=(20, 20, 30), width=width)
        im.save(out_dir / f"{i:05d}.png")
    return out_dir


def make_subset(data_root, dst, n_per_split: dict) -> Path:
    """Copy the first n images (in key order) of each split into a small dataset folder."""
    import shutil
    data_root, dst = Path(data_root), Path(dst)
    rows = read_manifest(data_root)
    keep = []
    for split in SPLITS:
        keep += [r for r in rows if r["split"] == split][: n_per_split.get(split, 0)]
    for r in keep:
        for kind in ("HR", "LR"):
            (dst / r["split"] / kind).mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_root / r["split"] / kind / f"{r['key']}.png",
                         dst / r["split"] / kind / f"{r['key']}.png")
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep)
    info = json.loads((data_root / "dataset_info.json").read_text(encoding="utf-8"))
    info.update(subset_of=str(data_root), counts={s: sum(1 for r in keep if r["split"] == s) for s in SPLITS})
    (dst / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return dst
