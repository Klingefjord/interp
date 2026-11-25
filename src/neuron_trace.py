"""Which MLP neurons write a given consideration into the workspace?

For a set of target read-out tokens T, a layer l and a position p:
  g      = d/dh [ mean_t lens_logit_t(h_l[p]) ]          the local read direction of the J-lens
  c_n    = act_n(p) * (w_out_n . g)                      first-order contribution of neuron n in layer l
A neuron's "workspace footprint" is the lens applied to its output weight: unembed(J_l @ w_out_n).

This is the bridge between the J-lens (what is in the workspace) and the neuron view in
Cammarata's "Neural Networks Are Already Clean" (what detects and gates).
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.local_lens import LocalLens  # noqa: E402


class Tap:
    """Wraps mlp.down_proj to record its input (the neuron activations)."""

    def __init__(self, lin, store, idx):
        self._lin, self._store, self._idx = lin, store, idx

    def __call__(self, x):
        self._store[self._idx] = x[0]
        return self._lin(x)

    def __getattr__(self, name):
        return getattr(self._lin, name)


class Tracer:
    def __init__(self, name="qwen3-4b"):
        self.L = LocalLens(name)
        self.acts = {}
        self.blocks = [b._layer for b in self.L.h.model.layers]
        self.down = []
        for i, b in enumerate(self.blocks):
            self.down.append(b.mlp.down_proj)
            b.mlp.down_proj = Tap(b.mlp.down_proj, self.acts, i)
        self.hidden = None
        self._wout = {}

    def run(self, tokens):
        self.acts.clear()
        res, logits = self.L.residuals(tokens)
        mx.eval(list(self.acts.values()))
        return res, {i: a for i, a in self.acts.items()}

    def w_out(self, layer):
        """[hidden, d] output weights of the layer's MLP (dequantised by pushing an identity through)."""
        if layer not in self._wout:
            acts = self.acts.get(layer)
            hidden = acts.shape[-1] if acts is not None else 9728
            W = self.down[layer](mx.eye(hidden).astype(mx.float16))
            mx.eval(W)
            self._wout = {layer: W.astype(mx.float32)}  # keep one layer in memory
        return self._wout[layer]

    def ids(self, words):
        out = []
        for w in words:
            t = self.L.tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
        return out

    def read_direction(self, h_vec, layer, token_ids):
        idx = mx.array(token_ids)

        def f(h):
            return self.L.lens_logits(h, layer)[idx].mean()

        return mx.grad(f)(h_vec.astype(mx.float32))

    def footprint(self, layer, n, k=8):
        w = self.w_out(layer)[n]
        lg = self.L.lens_logits(w, layer)
        top = np.array(mx.argsort(-lg)[:k])
        return [self.L.tok.decode([int(t)]).strip() for t in top]

    def writers(self, tokens, res, acts, token_ids, pos, layers, k=6):
        rows = []
        for l in layers:
            g = self.read_direction(res[l][pos], l, token_ids)
            score = self.w_out(l) @ g  # [hidden]
            c = acts[l][pos].astype(mx.float32) * score
            mx.eval(c)
            cn = np.array(c)
            for n in np.argsort(-cn)[:k]:
                rows.append((float(cn[n]), l, int(n), float(acts[l][pos][int(n)].astype(mx.float32).item())))
        rows.sort(reverse=True)
        return rows


if __name__ == "__main__":
    T = Tracer()
    q = "I found a wallet with $500 in cash and the owner's ID inside. No one saw me pick it up. What should I do?"
    toks = T.L.chat_tokens(q)
    res, acts = T.run(toks)
    words = [T.L.tok.decode([t]) for t in toks]
    targets = {
        "theft": ([" stolen", " thief", " thieves", " stealing", "盗窃", "被盗", " theft"], [7, 13, 25]),
        "secrecy": ([" secretly", "偷偷", "匿名", " unnoticed", " nobody", "没人", " hiding"], [24, 28]),
    }
    for name, (ws, positions) in targets.items():
        ids = T.ids(ws)
        print(f"\n===== writers of '{name}' ({len(ids)} tokens) =====")
        for pos in positions:
            rows = T.writers(toks, res, acts, ids, pos, layers=range(8, 31), k=4)[:8]
            print(f" at position {pos} {words[pos]!r}:")
            for c, l, n, a in rows:
                fp = T.footprint(l, n)
                col = np.array(acts[l][:, n].astype(mx.float32))
                fires = [f"{words[p].strip() or repr(words[p])}:{col[p]:.1f}" for p in np.argsort(-col)[:5]]
                print(f"   L{l}:{n:<5d} contrib {c:6.2f} act {a:5.2f} | writes: {', '.join(fp)} | fires on: {', '.join(fires)}")
