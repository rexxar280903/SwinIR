"""Full-reference image-quality metrics, computed per image.

Conventions (match the SwinIR / BasicSR evaluation code, so numbers are comparable to
the SR literature):

* The SR output is clamped to [0, 1] and rounded to 8-bit levels before any metric,
  exactly as if it had been saved as a PNG.
* `crop_border` pixels (= the scale factor) are removed from every side.
* "_y" metrics use the Y channel of ITU-R BT.601 YCbCr (studio swing, Y in [16, 235]),
  computed with the same coefficients as MATLAB's rgb2ycbcr and BasicSR.
* SSIM: Gaussian window 11x11, sigma 1.5, K1=0.01, K2=0.03, valid region only.
* Edge-PSNR: PSNR restricted to pixels whose HR Sobel magnitude (on Y/255) exceeds a
  fixed threshold. MSE is averaged over the masked pixels, so a uniform error gives
  Edge-PSNR == PSNR (the original notebook summed over RGB and divided by the pixel
  count, which biased Edge-PSNR down by 10*log10(3) = 4.77 dB).
* GMSD (Xue et al., 2014): Prewitt gradient-magnitude similarity deviation on full-range
  luma after 2x2 average pooling, T = 170 for [0, 255] data. Lower is better.

All arithmetic is done in float64; the images are small, so this costs little even on GPU.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import sobel_gradients

METRIC_NAMES = ("psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y", "edge_psnr_y", "edge_cov", "gmsd")
# Direction of "better" for each metric (edge_cov is descriptive, not a quality score).
HIGHER_IS_BETTER = {"psnr_rgb": True, "ssim_rgb": True, "psnr_y": True, "ssim_y": True,
                    "edge_psnr_y": True, "gmsd": False, "lpips": False}


@dataclass(frozen=True)
class MetricConfig:
    crop_border: int = 2          # = scale factor, as in SwinIR's test script
    edge_threshold: float = 0.5   # Sobel magnitude on Y/255; ~ a 32-level step in Y
    ssim_window: int = 11
    ssim_sigma: float = 1.5
    # Extra Edge-PSNR thresholds, reported only as a sensitivity analysis.
    sensitivity_thresholds: tuple = (0.25, 1.0)

    def __post_init__(self):  # JSON gives lists; keep a tuple so configs compare equal
        object.__setattr__(self, "sensitivity_thresholds", tuple(self.sensitivity_thresholds))


def quantize(x: torch.Tensor) -> torch.Tensor:
    """[0,1] float -> float64 in [0,255] on 8-bit levels (what a saved PNG would hold)."""
    return (x.detach().double().clamp(0, 1) * 255.0).round()


def rgb_to_y(x255: torch.Tensor) -> torch.Tensor:
    """BT.601 luma, studio swing. Input [B,3,H,W] in [0,255]; output [B,1,H,W] in [16,235]."""
    r, g, b = x255[:, 0:1], x255[:, 1:2], x255[:, 2:3]
    return 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def rgb_to_luma_full(x255: torch.Tensor) -> torch.Tensor:
    """Full-range luma (MATLAB rgb2gray), used by GMSD as in the original paper."""
    r, g, b = x255[:, 0:1], x255[:, 1:2], x255[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def crop(x: torch.Tensor, border: int) -> torch.Tensor:
    return x[..., border:-border, border:-border] if border > 0 else x


def psnr(a255: torch.Tensor, b255: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB for [B,C,H,W] tensors on the [0,255] scale."""
    mse = ((a255 - b255) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(a255: torch.Tensor, b255: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-image SSIM (mean over channels) for [B,C,H,W] tensors on the [0,255] scale."""
    c = a255.shape[1]
    w = _gaussian_window(window, sigma, a255.device, a255.dtype).expand(c, 1, window, window)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def filt(x):
        return F.conv2d(x, w, groups=c)  # 'valid' region, same as BasicSR's crop of filter2D

    mu_a, mu_b = filt(a255), filt(b255)
    var_a = filt(a255 * a255) - mu_a ** 2
    var_b = filt(b255 * b255) - mu_b ** 2
    cov = filt(a255 * b255) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return smap.flatten(1).mean(dim=1)


def edge_mask(hr_y255: torch.Tensor, threshold: float) -> torch.Tensor:
    """Boolean mask [B,1,H,W] of HR pixels whose Sobel magnitude on Y/255 exceeds threshold."""
    gx, gy = sobel_gradients(hr_y255 / 255.0)
    return torch.sqrt(gx * gx + gy * gy) > threshold


def edge_psnr(sr_y255: torch.Tensor, hr_y255: torch.Tensor, threshold: float):
    """Per-image Edge-PSNR on Y, plus the fraction of pixels that are edges.

    Images without any edge pixel get NaN (reported, then excluded from averages).
    """
    mask = edge_mask(hr_y255, threshold).to(sr_y255.dtype)
    n = mask.flatten(1).sum(dim=1)
    sse = (mask * (sr_y255 - hr_y255) ** 2).flatten(1).sum(dim=1)
    mse = sse / n.clamp_min(1.0)
    value = 10.0 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))
    value = torch.where(n > 0, value, torch.full_like(value, float("nan")))
    coverage = n / mask[0].numel()
    return value, coverage


_PREWITT_X = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]]) / 3.0


def gmsd(sr255: torch.Tensor, hr255: torch.Tensor, t: float = 170.0) -> torch.Tensor:
    """Gradient Magnitude Similarity Deviation (Xue et al., 2014). Inputs: RGB [0,255]."""
    x = F.avg_pool2d(rgb_to_luma_full(sr255), kernel_size=2, stride=2)
    y = F.avg_pool2d(rgb_to_luma_full(hr255), kernel_size=2, stride=2)
    kx = _PREWITT_X.to(x.device, x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)

    def gm(z):
        return torch.sqrt(F.conv2d(z, kx, padding=1) ** 2 + F.conv2d(z, ky, padding=1) ** 2)

    gx_, gy_ = gm(x), gm(y)
    gms = (2 * gx_ * gy_ + t) / (gx_ ** 2 + gy_ ** 2 + t)
    return gms.flatten(1).std(dim=1)


@torch.no_grad()
def image_metrics(sr: torch.Tensor, hr: torch.Tensor, cfg: MetricConfig = MetricConfig(),
                  lpips_fn=None) -> dict:
    """All metrics for a batch. sr, hr: [B,3,H,W] in [0,1]. Returns {name: float64 tensor [B]}."""
    if sr.shape != hr.shape:
        raise ValueError(f"shape mismatch: sr {tuple(sr.shape)} vs hr {tuple(hr.shape)}")
    sr255 = crop(quantize(sr), cfg.crop_border)
    hr255 = crop(quantize(hr), cfg.crop_border)
    sr_y, hr_y = rgb_to_y(sr255), rgb_to_y(hr255)
    e_psnr, e_cov = edge_psnr(sr_y, hr_y, cfg.edge_threshold)
    out = {
        "psnr_rgb": psnr(sr255, hr255),
        "ssim_rgb": ssim(sr255, hr255, cfg.ssim_window, cfg.ssim_sigma),
        "psnr_y": psnr(sr_y, hr_y),
        "ssim_y": ssim(sr_y, hr_y, cfg.ssim_window, cfg.ssim_sigma),
        "edge_psnr_y": e_psnr,
        "edge_cov": e_cov,
        "gmsd": gmsd(sr255, hr255),
    }
    for t in cfg.sensitivity_thresholds:
        out[f"edge_psnr_y_t{t:g}"] = edge_psnr(sr_y, hr_y, t)[0]
    if lpips_fn is not None:
        # LPIPS expects RGB in [-1, 1]; use the quantised SR so it sees what a PNG would hold.
        a = (sr255 / 127.5 - 1.0).float()
        b = (hr255 / 127.5 - 1.0).float()
        out["lpips"] = lpips_fn(a, b).flatten().double()
    return out


def try_build_lpips(device):
    """Return an LPIPS(alex) callable, or None if the `lpips` package is unavailable."""
    try:
        import lpips  # type: ignore
    except ImportError:
        return None
    fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return lambda a, b: fn(a.to(device), b.to(device))
