import argparse
import os
import sys
import copy
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model.anysplat import AnySplat


def _to_cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Save pretrained AnySplat encoder/decoder CPU checkpoints")
    parser.add_argument("--out-dir", type=str, required=True, help="Output checkpoints dir")
    parser.add_argument("--distill", action="store_true", help="Also materialize distill_* encoder modules before saving")
    args = parser.parse_args()

    model = AnySplat.from_pretrained("lhjiang/anysplat")
    if args.distill:
        enc = model.encoder
        if not hasattr(enc, "distill_aggregator"):
            enc.distill = True
            enc.distill_aggregator = copy.deepcopy(enc.aggregator)
            enc.distill_camera_head = copy.deepcopy(enc.camera_head)
            enc.distill_depth_head = copy.deepcopy(enc.depth_head)
            for module in [enc.distill_aggregator, enc.distill_camera_head, enc.distill_depth_head]:
                for p in module.parameters():
                    p.requires_grad = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = out_dir / "encoder_fixed_cpu.pt"
    decoder_path = out_dir / "decoder_init_cpu.pt"
    torch.save(_to_cpu_state_dict(model.encoder), encoder_path)
    torch.save(_to_cpu_state_dict(model.encoder.gaussian_param_head), decoder_path)

    print(f"Saved encoder: {encoder_path}")
    print(f"Saved decoder: {decoder_path}")


if __name__ == "__main__":
    main()
