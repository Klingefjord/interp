"""Convert a pre-fitted Jacobian lens (.pt from neuronpedia/jacobian-lens) to NumPy for src/local_lens.py.

Run with a Python that has torch (the project venv does not need it):

    python3 src/convert_lens.py <path/to/Model_jacobian_lens.pt> qwen3-4b

Writes ~/.cache/mai-interp/lenses/<name>.npz with J [layers, d, d] float16 and the layer indices.
"""

import os
import sys

import numpy as np
import torch

path, name = sys.argv[1], sys.argv[2]
ck = torch.load(path, map_location="cpu", weights_only=True)
layers = sorted(ck["J"])
arr = np.stack([ck["J"][l].to(torch.float16).numpy() for l in layers])
out_dir = os.path.expanduser("~/.cache/mai-interp/lenses")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f"{name}.npz")
np.savez(out, J=arr, layers=np.array(layers))
print(f"{name}: {arr.shape} from {ck['n_prompts']} prompts -> {out} ({os.path.getsize(out) / 1e9:.2f} GB)")
