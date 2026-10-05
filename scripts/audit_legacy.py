"""Reproduce the numbers in docs/AUDIT.md: how the original notebook's operators behave.

    python scripts/audit_legacy.py                      # synthetic cartoon images
    python scripts/audit_legacy.py --data /kaggle/working/data --n 256   # real HR images

The notebook functions are re-implemented here verbatim (kernel typo and zero padding
included) and compared with the corrected versions in edgesr/.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgesr.data import SRPairDataset, make_pair, make_synthetic_source  # noqa: E402
from edgesr.losses import adaptive_edge_weight, sobel_magnitude  # noqa: E402


# ------------------------------------------------------------------ notebook versions (verbatim logic)
def nb_sobel(x):
    c = x.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 3]], dtype=torch.float32).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def nb_adaptive_weight(e):
    return (e - e.min()) / (e.max() - e.min() + 1e-8)  # min/max over the whole batch


def nb_edge_mask(img, thr):
    gray = (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(gray, k, padding=1)
    gy = F.conv2d(gray, k.transpose(-1, -2), padding=1)
    return (torch.sqrt(gx ** 2 + gy ** 2) > thr).float()


def nb_edge_psnr(sr, hr, thr):
    m = nb_edge_mask(hr, thr)
    mse = (m * (sr - hr) ** 2).sum() / (m.sum() + 1e-8)
    return float(10 * torch.log10(1.0 / mse))


def corr(a, b):
    a, b = a[..., 2:-2, 2:-2].flatten(), b[..., 2:-2, 2:-2].flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="prepared dataset; uses HR images of the val split")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.15, help="the notebook validated with 0.15")
    args = ap.parse_args()
    torch.manual_seed(0)

    if args.data:
        ds = SRPairDataset(args.data, "val", cache=False, max_images=args.n)
        hr = torch.stack([ds[i][1] for i in range(len(ds))])
        source = f"{len(ds)} val HR images from {args.data}"
    else:
        src = make_synthetic_source(Path(tempfile.mkdtemp()), n=args.n, seed=3)
        hr = torch.stack([torch.from_numpy(np.array(make_pair(Image.open(p), 128, 2)[0])).permute(2, 0, 1).float() / 255
                          for p in sorted(src.glob("*.png"))])
        source = f"{args.n} synthetic cartoon images (illustrative only)"

    out = {"source": source}
    flat = torch.full((1, 3, 16, 16), 0.7)
    out["flat_0.7_notebook_sobel"] = float(nb_sobel(flat)[0, 0, 8, 8])
    out["flat_0.7_fixed_sobel"] = float(sobel_magnitude(flat)[0, 0, 8, 8])

    e_nb, e_true = nb_sobel(hr), sobel_magnitude(hr)
    w_nb, w_fix = nb_adaptive_weight(e_nb), adaptive_edge_weight(e_true)
    flat_px, edge_px = e_true < 0.05, e_true > 1.0
    out["notebook_weight_corr_with_brightness"] = corr(w_nb, hr)
    out["notebook_weight_corr_with_true_edges"] = corr(w_nb, e_true)
    out["notebook_weight_mean_on_flat_pixels"] = float(w_nb[flat_px].mean())
    out["notebook_weight_mean_on_edge_pixels"] = float(w_nb[edge_px].mean())
    out["fixed_weight_mean_on_flat_pixels"] = float(w_fix[flat_px].mean())
    out["fixed_weight_mean_on_edge_pixels"] = float(w_fix[edge_px].mean())
    out["share_of_flat_pixels"] = float(flat_px.float().mean())

    sr = (hr + 0.02 * torch.randn_like(hr)).clamp(0, 1)
    m = nb_edge_mask(hr, args.threshold)
    correct = float(10 * torch.log10(1.0 / ((m * (sr - hr) ** 2).sum() / (3 * m.sum()))))
    out["notebook_edge_psnr"] = nb_edge_psnr(sr, hr, args.threshold)
    out["same_mask_channel_mean_edge_psnr"] = correct
    out["edge_psnr_bias_db"] = out["notebook_edge_psnr"] - correct
    border = torch.zeros(hr.shape[-2:], dtype=torch.bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    out["notebook_mask_border_pixels_flagged"] = float(m[:, 0][:, border].mean())
    out["notebook_mask_share_that_is_border"] = float(m[:, 0][:, border].sum() / m.sum())

    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=1e-7)
    lrs = []
    for _ in range(2):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt2.load_state_dict(opt.state_dict())
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=2, eta_min=1e-7)
    sch2.load_state_dict(sch.state_dict())
    for _ in range(4):
        lrs.append(opt2.param_groups[0]["lr"])
        opt2.step()
        sch2.step()
    out["notebook_resume_lr_per_epoch"] = lrs
    out["expected_edge_psnr_bias_db"] = -10 * math.log10(3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
