#!/usr/bin/env python3
"""Convert safetensors model to .pth format for play.py"""

import sys
import torch
from safetensors.torch import load_file
from pathlib import Path

def convert_safetensors_to_pth(safetensors_path: str, output_path: str):
    """Convert a safetensors file to PyTorch .pth format."""
    print(f"Loading safetensors from: {safetensors_path}")
    state_dict = load_file(safetensors_path)

    print(f"Saving to PyTorch format: {output_path}")
    torch.save(state_dict, output_path)

    print(f"✓ Conversion complete!")
    print(f"  Model saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_model.py <input.safetensors> <output.pth>")
        sys.exit(1)

    convert_safetensors_to_pth(sys.argv[1], sys.argv[2])
