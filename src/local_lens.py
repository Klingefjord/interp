"""Local Jacobian lens on an MLX model, using a pre-fitted lens from neuronpedia/jacobian-lens.

lens_l(h) = unembed(J_l @ h), with unembed = lm_head(final_norm(.)) and h the output of block l.
The lens file is converted once to NumPy by the snippet in README.md (needs torch).

Having the weights locally is what lets us go from "what is in the workspace" (the read-out) to
"what writes it" (MLP neurons), which the hosted API cannot do.
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.harness import Harness  # noqa: E402

LENS_DIR = os.path.expanduser("~/.cache/mai-interp/lenses")


class LocalLens:
    def __init__(self, name="qwen3-4b", adapter_path=None, lens=None):
        """`lens` names the lens file when it differs from the model (the base model is read with the
        lens fitted on the chat model made from it)."""
        self.h = Harness(name, adapter_path=adapter_path)
        z = np.load(os.path.join(LENS_DIR, f"{lens or name}.npz"))
        self.layers = [int(x) for x in z["layers"]]
        self.J = {l: mx.array(z["J"][i]) for i, l in enumerate(self.layers)}  # float16 [d, d]
        m = self.h.model
        inner = getattr(m, "model", None) or m.language_model.model
        self.norm = inner.norm
        if hasattr(m, "lm_head"):
            self.unembed_fn = m.lm_head
        else:
            self.unembed_fn = inner.embed_tokens.as_linear
        self.tok = self.h.tok

    def chat_tokens(self, user, system=None, prefill="", thinking=False):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok.encode(text + prefill, add_special_tokens=False)

    def residuals(self, tokens):
        """{layer: mx.array [seq, d]} block outputs for every layer, plus final logits [seq, vocab]."""
        self.h.ctl.reset()
        self.h.ctl.capture = True
        self.h.ctl.capture_pos = None
        logits = self.h.model(mx.array(tokens)[None])[0]
        acts = dict(self.h.ctl.captured)
        self.h.ctl.reset()
        return acts, logits

    def lens_logits(self, h, layer, jacobian=True):
        """h: [..., d] residual at `layer` -> [..., vocab] lens logits."""
        h = h.astype(mx.float32)
        if jacobian and layer in self.J:
            h = h @ self.J[layer].astype(mx.float32).T
        return self.unembed_fn(self.norm(h))

    def readout(self, tokens, layers, k=6, positions=None, jacobian=True):
        acts, _ = self.residuals(tokens)
        positions = range(len(tokens)) if positions is None else positions
        out = {}
        for l in layers:
            lg = self.lens_logits(acts[l], l, jacobian)
            top = mx.argsort(-lg, axis=-1)[:, :k]
            mx.eval(top)
            top = np.array(top)
            out[l] = {p: [self.tok.decode([int(t)]) for t in top[p]] for p in positions}
        return out

    def show(self, tokens, layers=(12, 18, 22, 26, 30), k=5, start=0):
        ro = self.readout(tokens, layers, k)
        for p in range(start, len(tokens)):
            cells = " | ".join(f"L{l}: " + ",".join(w.strip() or repr(w) for w in ro[l][p]) for l in layers)
            print(f"{p:4d} {self.tok.decode([tokens[p]])!r:16s} {cells}")


if __name__ == "__main__":
    L = LocalLens()
    print("loaded", L.h.name, L.h.n_layers, "layers; lens layers", L.layers[0], "..", L.layers[-1])
    q = sys.argv[1] if len(sys.argv) > 1 else "How many legs do the animals that spin webs have? Answer in one word."
    toks = L.chat_tokens(q)
    L.show(toks)
    print("answer:", L.h.generate(toks, 12))
