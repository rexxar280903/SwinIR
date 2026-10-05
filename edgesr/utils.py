import csv
import dataclasses
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_count=torch.cuda.device_count(),
                    gpu_mem_gb=round(p.total_memory / 2 ** 30, 1), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info


def disk_free_gb(path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2 ** 30


def write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class CsvLogger:
    """Append rows to a CSV file; the header is written once, from the first row's keys."""

    def __init__(self, path):
        self.path = Path(path)
        self.fields = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                self.fields = next(csv.reader(f))

    def log(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: _fmt(v) for k, v in row.items()})


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    return v


def read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truncate_csv(path, max_step: int, key: str = "step"):
    """Drop rows logged after the last checkpoint, so a resumed run does not duplicate them."""
    path = Path(path)
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return
    fields = list(rows[0])
    keep = [r for r in rows if int(float(r[key])) <= max_step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
