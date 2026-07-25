#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------
# 1. Make Tiny-NN-in-C and demo_codebase importable
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent           # .../Tiny-NN-in-C/examples
TINY_ROOT = THIS_DIR.parent                          # .../Tiny-NN-in-C
sys.path.insert(0, str(TINY_ROOT))

# Add demo_codebase/src2 so we can reuse the same backbone + factory
DEMO_SRC = Path("/home/madhav5/demo_codebase/src2")
sys.path.insert(0, str(DEMO_SRC))

from src.pytorch_to_c.compiler import compile_model
from models.ResNetSimple import ResNetSimpleBackbone, build_simple_resnet_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Compile a trained ResNetSimpleBackbone (ACIDS, demo_codebase) to C."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to best_model.pth from demo_codebase experiment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_acids_simple_resnet",
        help="Directory to write generated C files into",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="student_audio_resnet",
        help="Model key in the YAML/config (e.g. 'student_audio_resnet')",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=7,
        help="Input height H for the compiler example input",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Input width W for the compiler example input",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -----------------------------------------------------------------
    # 2. Load experiment config from demo_codebase
    # -----------------------------------------------------------------
    # Experiment directory is the parent of the models/ folder
    # ckpt_path = .../experiments/<id>/models/best_model.pth
    experiment_dir = ckpt_path.parent.parent
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in experiment dir: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------------------------------------------
    # 3. Build backbone from config using your factory
    # -----------------------------------------------------------------
    backbone, location_name, modality_name = build_simple_resnet_from_config(
        config, args.model_key
    )

    # Infer in_channels from the backbone conv1
    in_channels = backbone.conv1.weight.shape[1]
    num_classes = backbone.fc2.out_features

    print(f"Loaded config from: {config_path}")
    print(f"Model key: {args.model_key}")
    print(f"Location: {location_name}, Modality: {modality_name}")
    print(f"in_channels: {in_channels}, num_classes: {num_classes}")

    # -----------------------------------------------------------------
    # 4. Load checkpoint and strip 'backbone.' prefix
    # -----------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    missing = backbone.load_state_dict(backbone_state, strict=True)
    # load_state_dict returns a NamedTuple; if you want, you can print it:
    # print("load_state_dict result:", missing)

    # -----------------------------------------------------------------
    # 5. Example input and compile
    # -----------------------------------------------------------------
    example_input = torch.randn(
        1, in_channels, args.height, args.width
    )

    compile_model(
        model=backbone,
        example_input=example_input,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()