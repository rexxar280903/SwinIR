"""One dataclass holds every setting of a training run; it is saved next to the results."""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

from .losses import EDGE_MODES
from .metrics import MetricConfig


@dataclass
class TrainConfig:
    # identity
    seed: int = 0
    # model (see edgesr/models/__init__.py for the presets)
    model: str = "light"
    scale: int = 2
    lr_size: int = 64
    # objective: L1 + edge_weight * L_edge
    edge_mode: str = "none"          # none | static | adaptive
    edge_weight: float = 0.0
    # optimisation (iteration-based, so every condition sees the same number of updates)
    batch_size: int = 32
    total_iters: int = 20_000
    lr: float = 2e-4
    betas: tuple = (0.9, 0.999)
    eta_min: float = 1e-7            # cosine schedule floor
    warmup_iters: int = 0
    grad_clip: float = 1.0           # max global grad norm; 0 disables
    amp: bool = True                 # fp16 autocast on CUDA (ignored on CPU)
    # data
    augment: bool = True             # random flips/rotations (8 dihedral transforms)
    num_workers: int = 2
    cache: bool = True               # keep decoded images in RAM
    # bookkeeping
    log_every: int = 100
    val_every: int = 2_000
    ckpt_every: int = 2_000
    val_max_images: int = 0          # 0 = whole validation split
    metrics: MetricConfig = field(default_factory=MetricConfig)

    def __post_init__(self):
        if isinstance(self.metrics, dict):
            self.metrics = MetricConfig(**self.metrics)
        self.betas = tuple(self.betas)
        if self.edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {EDGE_MODES}")
        if self.edge_mode == "none":
            self.edge_weight = 0.0

    @property
    def condition(self) -> str:
        """Name of the experimental condition, independent of the seed."""
        if self.edge_mode == "none":
            return "l1"
        return f"{self.edge_mode}_w{self.edge_weight:g}"

    @property
    def run_name(self) -> str:
        return f"{self.model}_x{self.scale}_{self.condition}_s{self.seed}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def protocol_hash(self) -> str:
        """Hash of everything except seed and bookkeeping: equal for runs of one protocol."""
        d = self.to_dict()
        for k in ("seed", "edge_mode", "edge_weight", "num_workers", "cache", "log_every", "ckpt_every"):
            d.pop(k)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def apply_overrides(cfg: TrainConfig, overrides) -> TrainConfig:
    """Apply `key=value` strings (value parsed as JSON when possible), e.g. total_iters=500."""
    d = cfg.to_dict()
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if key.startswith("metrics."):
            d["metrics"][key.split(".", 1)[1]] = value
        elif key in d:
            d[key] = value
        else:
            raise ValueError(f"unknown config key {key!r}")
    return TrainConfig.from_dict(d)
