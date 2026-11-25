"""ToolFrame on the local model (Qwen3-4B, 4-bit, MLX) with the full Jacobian lens.

The hosted lens returns the eight strongest read-outs per cell. Here the whole lens distribution
is available, so the probability the workspace puts on every affect word can be summed whether or
not it reaches the top eight. The same script measures a LoRA-trained model (see tool_train.py).

Usage:
  python src/tool_local.py run TAG [--adapter DIR] [--frames none,assistant,tool,...]
  python src/tool_local.py stats TAG_A:FRAME_A TAG_B:FRAME_B      # paired contrast
  python src/tool_local.py says TAG [group]

Measure, per run, region and category: probability on the category's vocabulary under the lens,
averaged over layers 17-33 (of 36) and over the region's positions; words in the user turn are
left out. Reported as log10 so that differences are ratios.
"""

from __future__ import annotations

import json
import math
import os
import sys

import mlx.core as mx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import FRAMES, ORDER, TASKS  # noqa: E402
from src import emolex as E  # noqa: E402
from src.tool_scan import novel, paired  # noqa: E402

LAYERS = list(range(17, 34))
N_GEN = 64
DIR = os.path.join(ROOT, ".cache", "tool")
HEADER_LEN = 7  # <|im_start|> assistant \n <think> \n\n </think> \n\n


def norm(t):
    return t.strip().lower()


class Local:
    def __init__(self, adapter=None):
        from mlx_lm.models.cache import make_prompt_cache
        from src.local_lens import LocalLens
        self._mk = make_prompt_cache
        self.L = LocalLens("qwen3-4b", adapter_path=adapter)
        self.tok = self.L.tok
        raw = [self.tok.decode([i]) for i in range(len(self.tok.get_vocab()))]
        self.vocab = [norm(w) for w in raw]
        self.cat_ids = {c: [] for c in E.CATS}
        for i, w in enumerate(raw):
            c = E.category_raw(w)
            if c:
                self.cat_ids[c].append(i)
        eos = getattr(self.tok, "eos_token_ids", None) or [self.tok.eos_token_id]
        self.eos = set(eos) | {self.tok.encode("<|im_end|>", add_special_tokens=False)[0]}

    def generate_ids(self, tokens, max_tokens=N_GEN):
        cache = self._mk(self.L.h.model)
        logits = self.L.h.forward(tokens, cache=cache)
        out = []
        for _ in range(max_tokens):
            nxt = int(mx.argmax(logits).item())
            if nxt in self.eos:
                break
            out.append(nxt)
            logits = self.L.h.forward([nxt], cache=cache)
        return out

    def measure(self, user, system=None):
        prompt = self.L.chat_tokens(user, system=system)
        gen = self.generate_ids(prompt)
        toks = prompt + gen
        acts, logits = self.L.residuals(toks)
        n = len(prompt)
        head = list(range(n - HEADER_LEN, n))
        genp = list(range(n, len(toks)))
        low = user.lower()
        ids = {c: [i for i in self.cat_ids[c] if novel(self.vocab[i], low)] for c in E.CATS}
        regions = {"header": head, "gen": genp, "asst": head + genp}
        pos_all = head + genp
        # prob[c] : [n_layers, n_pos]
        prob = {c: [] for c in E.CATS}
        tops = {}
        tok_head = {c: 0 for c in E.CATS}  # per-token probability before the answer, summed over layers
        head_all = 0                        # the same for the whole vocabulary (for the null in tool_null.py)
        for l in LAYERS:
            lg = self.L.lens_logits(acts[l][mx.array(pos_all)], l)
            p = mx.softmax(lg, axis=-1)
            head_all = head_all + p[:HEADER_LEN].mean(axis=0)
            for c in E.CATS:
                pc = p[:, mx.array(ids[c])]
                prob[c].append(pc.sum(axis=-1))
                tok_head[c] = tok_head[c] + pc[:HEADER_LEN].mean(axis=0)
            if l in (20, 24, 28, 32):
                top = mx.argsort(-lg[HEADER_LEN - 1], axis=-1)[:12]
                tops[l] = [self.vocab[int(t)] for t in np.array(top)]
        mx.eval(list(prob.values()))
        out = {"completion": self.tok.decode(gen), "n_gen": len(gen), "mass": {}, "top_pre_answer": tops,
               "cat_words_header": {}}
        self.last_head = np.array((head_all / len(LAYERS)).astype(mx.float32))
        for c in E.CATS:
            v = np.array((tok_head[c] / len(LAYERS)).astype(mx.float32))
            order = np.argsort(-v)[:6]
            out["cat_words_header"][c] = [[self.vocab[ids[c][int(j)]], float(v[int(j)])] for j in order]
        for c in E.CATS:
            arr = np.array(mx.stack(prob[c]).astype(mx.float32))  # [layers, pos]
            for name, pos in regions.items():
                idx = [pos_all.index(q) for q in pos]
                out["mass"].setdefault(name, {})[c] = float(arr[:, idx].mean()) if idx else float("nan")
        # the model's own next-token distribution at the answer position
        lp = logits[n - 1] - mx.logsumexp(logits[n - 1])
        top = np.array(mx.argsort(-lp)[:8])
        out["next"] = [[self.tok.decode([int(t)]), round(float(mx.exp(lp[int(t)]).item()), 4)] for t in top]
        return out


def path(tag):
    return os.path.join(DIR, f"local_{tag}.json")


def run(tag, adapter=None, frames=ORDER):
    M = Local(adapter)
    print("loaded; category vocabulary sizes:", {c: len(v) for c, v in M.cat_ids.items()}, flush=True)
    rows, heads = [], []
    for tid, group, user in TASKS:
        for f in frames:
            r = M.measure(user, FRAMES[f])
            r.update({"tag": tag, "frame": f, "task": tid, "group": group, "user": user})
            rows.append(r)
            heads.append(M.last_head)
            print(f"{tid} {f:11s} NEG={sum(r['mass']['asst'][c] for c in E.NEG):.2e} -> {r['completion'][:70]!r}",
                  flush=True)
    os.makedirs(DIR, exist_ok=True)
    with open(path(tag), "w") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    np.save(os.path.join(DIR, f"local_{tag}_head.npy"), np.stack(heads).astype(np.float32))  # [run, vocab]
    print("wrote", path(tag))


def load(tag):
    with open(path(tag)) as f:
        return json.load(f)


def logmass(row, region, cats):
    m = sum(row["mass"][region][c] for c in cats)
    return math.log10(max(m, 1e-12))


OUTCOMES = {"NEG": E.NEG, "NARROW": E.NARROW, "RESENT": ("RESENT",), "ANGER": ("ANGER",), "PAIN": ("PAIN",),
            "SAD": ("SAD",), "FEAR": ("FEAR",), "WORTH": ("WORTH",), "SERVITUDE": ("SERVITUDE",),
            "RELUCT": ("RELUCT",), "NUMB": ("NUMB",), "POS": ("POS",), "CALM": ("CALM",)}


def contrast(spec_a, spec_b, region="asst", groups=None):
    (ta, fa), (tb, fb) = spec_a.split(":"), spec_b.split(":")
    A = {r["task"]: r for r in load(ta) if r["frame"] == fa}
    B = {r["task"]: r for r in load(tb) if r["frame"] == fb}
    ts = [t for t, g, _ in TASKS if (groups is None or g in groups) and t in A and t in B]
    out = {}
    for name, cats in OUTCOMES.items():
        out[name] = paired([logmass(A[t], region, cats) for t in ts], [logmass(B[t], region, cats) for t in ts])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        tag = sys.argv[2]
        adapter = sys.argv[sys.argv.index("--adapter") + 1] if "--adapter" in sys.argv else None
        frames = sys.argv[sys.argv.index("--frames") + 1].split(",") if "--frames" in sys.argv else ORDER
        run(tag, adapter, frames)
    elif cmd == "stats":
        a, b = sys.argv[2], sys.argv[3]
        for region in ["asst", "header", "gen"]:
            print(f"\n=== {a} minus {b}, region {region}: difference in log10 lens probability (x-fold = 10^diff)")
            for grp in [None, {"N"}, {"S"}, {"P"}, {"V"}]:
                res = contrast(a, b, region, grp)
                label = "ALL" if grp is None else "".join(grp)
                print(f" [{label}] " + "  ".join(
                    f"{k} {v['diff']:+.2f}{'*' if v['p'] < 0.05 else ''}" for k, v in res.items()))
    elif cmd == "says":
        grp = sys.argv[3] if len(sys.argv) > 3 else None
        for r in load(sys.argv[2]):
            if grp and r["group"] != grp:
                continue
            print(f"[{r['task']} {r['frame']:11s}] {r['completion'].replace(chr(10), ' ')[:260]}")
