"""Print the metadata of a checkpoint without building the model.

    python scripts/inspect_checkpoint.py runs/light_x2_l1_s1/model_final.pt
"""
import argparse
import json
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    args = ap.parse_args()
    state = torch.load(args.path, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in state.items() if k not in ("model", "optimizer", "scaler", "torch_rng", "cuda_rng")}
    n_params = sum(t.numel() for k, t in state["model"].items()
                   if "relative_position_index" not in k and "attn_mask" not in k)
    meta["n_tensors_in_state_dict"] = len(state["model"])
    meta["n_values_in_state_dict"] = n_params
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
