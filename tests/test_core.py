"""Fast CPU tests. Run with `python -m pytest tests -q`, or without pytest: `python tests/test_core.py`.

Most correctness checks live in edgesr/readiness.py so that the Kaggle gate and these
tests exercise the same code; the tests below call them and add unit tests for the
pieces the gate does not cover (config, selection rule, statistics).
"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr import readiness as R  # noqa: E402
from edgesr.ablation import (CONFIRM_SEEDS, SWEEP_WEIGHTS, confirm_plan, select_weights,  # noqa: E402
                             sweep_plan)
from edgesr.config import TrainConfig, apply_overrides  # noqa: E402
from edgesr.data import make_synthetic_source, prepare_dataset, read_manifest  # noqa: E402
from edgesr.metrics import MetricConfig, image_metrics  # noqa: E402
from edgesr.models import EXPECTED_PARAMS, build_model, count_params  # noqa: E402
from edgesr.stats import holm, paired_bootstrap  # noqa: E402

_DATA = None


def data_root():
    """A small synthetic dataset, built once per test session."""
    global _DATA
    if _DATA is None:
        tmp = Path(tempfile.mkdtemp(prefix="edgesr_test_"))
        src = make_synthetic_source(tmp / "src", n=40)
        prepare_dataset(src, tmp / "data", workers=0, log=lambda *_: None)
        _DATA = tmp / "data"
    return _DATA


def _ok(result):
    status, detail, _ = result
    assert status in ("PASS", "SKIP"), detail


# ----------------------------------------------------------------- checks shared with the gate
def test_sobel():
    _ok(R.check_sobel())


def test_adaptive_weight():
    _ok(R.check_adaptive_weight())


def test_losses():
    _ok(R.check_losses())


def test_metric_identities():
    _ok(R.check_metric_identities())


def test_metric_crosscheck():
    _ok(R.check_metric_crosscheck())


def test_model_light():
    _ok(R.check_model("light", torch.device("cpu")))


def test_param_counts():
    for preset, n in EXPECTED_PARAMS.items():
        assert count_params(build_model(preset)) == n, preset


def test_data_checks():
    d = data_root()
    for fn in (R.check_manifest, R.check_image_sizes, R.check_degradation, R.check_leakage,
               R.check_split_ratios, R.check_source):
        _ok(fn(d))


def test_seed_and_isolation():
    _ok(R.check_seed_control(TrainConfig(), data_root()))
    _ok(R.check_test_isolation(data_root()))


def test_resume_exact():
    tmp = Path(tempfile.mkdtemp())
    try:
        _ok(R.check_resume(TrainConfig(), data_root(), tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------- data preparation
def test_duplicates_dropped_and_grouped():
    tmp = Path(tempfile.mkdtemp())
    src = make_synthetic_source(tmp / "src", n=20)
    shutil.copy(src / "00003.png", src / "zz_copy.png")  # exact duplicate
    info = prepare_dataset(src, tmp / "data", workers=0, log=lambda *_: None)
    assert info["n_exact_duplicates_dropped"] == 1
    rows = read_manifest(tmp / "data")
    assert len(rows) == 20 and len({r["hr_sha1"] for r in rows}) == 20
    shutil.rmtree(tmp, ignore_errors=True)


def test_rgba_composited_on_white():
    from PIL import Image
    from edgesr.data import to_rgb
    im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    assert np.asarray(to_rgb(im)).min() == 255


# ----------------------------------------------------------------- config
def test_config_roundtrip_and_overrides():
    cfg = apply_overrides(TrainConfig(), ["edge_mode=adaptive", "edge_weight=0.2", "metrics.edge_threshold=0.25"])
    assert cfg.condition == "adaptive_w0.2" and cfg.metrics.edge_threshold == 0.25
    tmp = Path(tempfile.mkdtemp()) / "c.json"
    cfg.save(tmp)
    again = TrainConfig.load(tmp)
    assert again.to_dict() == cfg.to_dict()
    assert again.protocol_hash() == cfg.protocol_hash()
    assert apply_overrides(cfg, ["seed=3"]).protocol_hash() == cfg.protocol_hash()
    assert apply_overrides(cfg, ["total_iters=5"]).protocol_hash() != cfg.protocol_hash()


def test_l1_forces_zero_weight():
    assert TrainConfig(edge_mode="none", edge_weight=0.5).edge_weight == 0.0


def test_plan_sizes():
    base = TrainConfig()
    sweep = sweep_plan(base)
    assert len(sweep) == 1 + 2 * len(SWEEP_WEIGHTS)
    conf = confirm_plan(base, 0.1, 0.2)
    assert len(conf) == 3 * len(CONFIRM_SEEDS)
    assert len({c.run_name for c in sweep + conf}) == len(sweep) + len(conf)


# ----------------------------------------------------------------- selection rule
def _fake_run(root, mode, weight, seed, edge, psnr):
    cfg = TrainConfig(edge_mode=mode, edge_weight=weight, seed=seed)
    d = Path(root) / cfg.run_name
    d.mkdir(parents=True)
    cfg.save(d / "config.json")
    (d / "train_done.json").write_text("{}")
    with open(d / "val_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "psnr_y", "edge_psnr_y"])
        for i in range(4):
            w.writerow([i, psnr + (0.5 if i == 0 else 0), edge + (0.5 if i == 0 else 0)])


def test_select_weights_rule():
    root = Path(tempfile.mkdtemp())
    _fake_run(root, "none", 0, 0, edge=30.0, psnr=35.0)
    # static: 0.5 has the best edge score but loses 0.3 dB PSNR -> excluded by the guard
    _fake_run(root, "static", 0.1, 0, edge=30.2, psnr=34.95)
    _fake_run(root, "static", 0.5, 0, edge=30.6, psnr=34.70)
    # adaptive: 0.2 and 1.0 tie within 0.01 dB -> the smaller weight wins
    _fake_run(root, "adaptive", 0.2, 0, edge=30.400, psnr=35.0)
    _fake_run(root, "adaptive", 1.0, 0, edge=30.405, psnr=34.98)
    sel = select_weights([root])
    assert sel["chosen"] == {"static": 0.1, "adaptive": 0.2}, sel["chosen"]
    shutil.rmtree(root, ignore_errors=True)


def test_boundary_extension():
    from edgesr.ablation import EXTEND_DOWN, EXTEND_UP, extension_plan
    root = Path(tempfile.mkdtemp())
    _fake_run(root, "none", 0, 0, edge=30.0, psnr=35.0)
    for w in SWEEP_WEIGHTS:  # static: monotone in w -> largest wins -> extend up
        _fake_run(root, "static", w, 0, edge=30.0 + w, psnr=35.0)
        _fake_run(root, "adaptive", w, 0, edge=30.0, psnr=35.0 - w / 100)  # ties -> smallest -> extend down
    sel = select_weights([root])
    assert sel["extend"] == {"static": list(EXTEND_UP), "adaptive": list(EXTEND_DOWN)}, sel["extend"]
    assert len(extension_plan(TrainConfig(), sel)) == 4
    for w in EXTEND_UP:
        _fake_run(root, "static", w, 0, edge=30.0 + w, psnr=35.0)
    sel2 = select_weights([root])
    assert sel2["chosen"]["static"] == max(EXTEND_UP) and "static" not in sel2["extend"]
    base_only = select_weights([root], allowed_weights=SWEEP_WEIGHTS)
    assert base_only["extend"]["static"] == list(EXTEND_UP)
    shutil.rmtree(root, ignore_errors=True)


# ----------------------------------------------------------------- statistics
def test_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    base = rng.normal(30, 2, size=(3, 400))
    shifted = base + 0.1 + rng.normal(0, 0.05, size=base.shape)
    r = paired_bootstrap(shifted, base, n_boot=2000)
    assert r["ci_low"] > 0 and abs(r["mean_diff"] - 0.1) < 0.02
    null = base + rng.normal(0, 0.05, size=base.shape)
    r0 = paired_bootstrap(null, base, n_boot=2000)
    assert r0["ci_low"] < 0 < r0["ci_high"]


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_budget_survives_oom_of_comparison_preset():
    """A comparison preset that does not fit in GPU memory is reported, not a failed gate."""
    def fake_throughput(preset, batch_size, device, amp):
        if preset == "medium":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return {"img_per_s": 100.0, "s_per_iter": 0.32, "peak_mem_gb": 5.0}

    saved = R.measure_throughput, R.measure_eval_speed
    R.measure_throughput, R.measure_eval_speed = fake_throughput, lambda *a, **k: 0.001
    try:
        cuda = torch.device("cuda")
        status, detail, ev = R.check_budget(TrainConfig(), data_root(), cuda, 1e6, ("medium",))
        assert status == "PASS" and "medium: does not fit" in detail, detail
        assert ev["presets"]["medium"] == {"oom_at_batch": 32}
        status, detail, _ = R.check_budget(TrainConfig(model="medium"), data_root(), cuda, 1e6, ())
        assert status == "FAIL" and "lower batch_size" in detail, detail
    finally:
        R.measure_throughput, R.measure_eval_speed = saved


def test_image_metrics_perfect_prediction():
    hr = torch.rand(2, 3, 32, 32)
    m = image_metrics(hr, hr, MetricConfig())
    assert torch.allclose(m["ssim_y"], torch.ones(2, dtype=torch.float64))
    assert (m["psnr_y"] > 90).all() and torch.allclose(m["gmsd"], torch.zeros(2, dtype=torch.float64))


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
