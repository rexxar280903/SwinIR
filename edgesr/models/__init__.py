"""Model presets. The network itself is the unmodified official SwinIR (see swinir.py)."""
from .swinir import SwinIR

# Every preset uses window_size=8, mlp_ratio=2, img_range=1 and resi_connection='1conv',
# as in the official SwinIR training options for classical/lightweight SR.
PRESETS = {
    # SwinIR-light from the paper ("lightweight image SR"): 0.91M params at x2. The paper
    # lists 878K because it leaves out the 24 relative-position bias tables (24 x 1350).
    "light": dict(embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                  upsampler="pixelshuffledirect"),
    # The configuration used in the original notebook (it was called "large" there).
    # It is classical SwinIR with 4 instead of 6 RSTBs: 8.0M params at x2.
    "medium": dict(embed_dim=180, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                   upsampler="pixelshuffle"),
    # Classical SwinIR exactly as in the paper: ~11.8M params at x2. Listed for
    # completeness; too slow for a multi-seed ablation on a Kaggle T4.
    "classical": dict(embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                      upsampler="pixelshuffle"),
}

# Expected trainable parameter counts at x2 (checked by tests and the readiness gate).
EXPECTED_PARAMS = {"light": 910_152, "medium": 8_018_567, "classical": 11_752_487}


def build_model(preset: str = "light", scale: int = 2, lr_size: int = 64,
                window_size: int = 8) -> SwinIR:
    if preset not in PRESETS:
        raise ValueError(f"unknown model preset {preset!r}; choose from {sorted(PRESETS)}")
    p = PRESETS[preset]
    return SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=lr_size,
        window_size=window_size,
        img_range=1.0,
        depths=p["depths"],
        embed_dim=p["embed_dim"],
        num_heads=p["num_heads"],
        mlp_ratio=2,
        upsampler=p["upsampler"],
        resi_connection="1conv",
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
