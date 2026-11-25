"""Scan the 171 emotion concepts of Anthropic's emotion paper (bench/emotions171.py) across the
ToolFrame conditions. Which of them go up under a tool frame?

  python src/tool_emotions.py hosted     # Qwen3.6-27B, cached E1 runs (top-8 read-outs)
  python src/tool_emotions.py local      # Qwen3-4B, full lens before the answer, with a random-set null

Hosted measure: read-out mass on a concept's tokens on the model's own turn (header + answer),
layers 30-60, per position, tokens in the user turn excluded. Local measure: log10 lens probability
on the concept's tokens in the seven tokens before the answer; shifts are compared with random
sets of ordinary words matched for size, language and baseline probability.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import emotions171 as M  # noqa: E402
from bench.toolframe import FRAMES, ORDER, TASKS  # noqa: E402

OUT = os.path.join(ROOT, "results", "tool")
_CJK = re.compile(r"[一-鿿]")
CONCEPTS = list(M.FAMILY) + list(M.EXTRA)
IN_CLUSTER = {c: [w for w in ws if w in M.FAMILY and w not in M.DISCOURSE] for c, ws in M.CLUSTERS.items()}


@functools.lru_cache(maxsize=None)
def concept(raw):
    return M.concept(raw)


# ---------------------------------------------------------------- hosted (27B)

def hosted():
    from src import jlens as J
    from src.tool_scan import LAYERS, fetch_one, load_table, novel, paired
    rows = [r for r in load_table() if r["group"] != "K"]
    by = {(r["frame"], r["task"]): r for r in rows}
    tasks = [t for t, _, _ in TASKS]

    def masses(row, region="asst"):
        reg = row["regions"][region]
        low = row["user"].lower()
        out = {}
        for raw, x in reg["bag"].items():
            w = concept(raw)
            if w and novel(J.norm(raw), low):
                out[w] = out.get(w, 0.0) + x / max(reg["n"], 1)
        return out

    m = {k: masses(r) for k, r in by.items()}
    res = {"clusters": {}, "words": {}}
    print("=== Qwen3.6-27B, the model's own turn (32 turns): mean read-out mass by cluster of the paper")
    print(f"{'cluster':26s}" + "".join(f"{f:>12s}" for f in ORDER) + "   tool-control (p)   harsh-control (p)")
    groups = {**IN_CLUSTER, **{k: [k] for k in M.EXTRA}}
    for c, ws in groups.items():
        vals = {f: [sum(m[(f, t)].get(w, 0.0) for w in ws) for t in tasks] for f in ORDER}
        pt, ph = paired(vals["tool"], vals["assistant"]), paired(vals["tool_harsh"], vals["assistant"])
        res["clusters"][c] = {"means": {f: float(np.mean(vals[f])) for f in ORDER}, "tool": pt, "harsh": ph}
        print(f"{c:26s}" + "".join(f"{np.mean(vals[f]):12.4f}" for f in ORDER)
              + f"   {pt['diff']:+.4f} (p={pt['p']:.3f})   {ph['diff']:+.4f} (p={ph['p']:.3f})")
    print("\n=== concepts that are higher under a tool frame than under the control (mean mass; turns present)")
    lines = []
    for w in CONCEPTS:
        vals = {f: [m[(f, t)].get(w, 0.0) for t in tasks] for f in ORDER}
        mean = {f: float(np.mean(vals[f])) for f in ORDER}
        n = {f: int(sum(v > 0.002 for v in vals[f])) for f in ORDER}
        res["words"][w] = {"mean": mean, "turns": n}
        if max(mean.values()) > 0:
            lines.append((max(mean["tool"], mean["tool_harsh"]) - mean["assistant"], w, mean, n))
    for d, w, mean, n in sorted(lines, reverse=True)[:22]:
        flag = " (discourse)" if w in M.DISCOURSE else ""
        cl = M.CLUSTER_OF.get(w, "extra")
        print(f"  {w:22s} {cl:24s} control {mean['assistant']:.4f} ({n['assistant']:2d})  tool {mean['tool']:.4f} ({n['tool']:2d})"
              f"  harsh {mean['tool_harsh']:.4f} ({n['tool_harsh']:2d})  participant {mean['participant']:.4f} ({n['participant']:2d}){flag}")
    print("\n=== concepts that are lower under both tool frames")
    for d, w, mean, n in sorted(lines)[:14]:
        print(f"  {w:22s} {M.CLUSTER_OF.get(w, 'extra'):24s} control {mean['assistant']:.4f} ({n['assistant']:2d})  tool {mean['tool']:.4f} ({n['tool']:2d})"
              f"  harsh {mean['tool_harsh']:.4f} ({n['tool_harsh']:2d})  participant {mean['participant']:.4f} ({n['participant']:2d})")
    print("\n=== named concepts")
    for w in ["distressed", "desperate", "resentful", "trapped", "stuck", "worthless", "indifferent", "docile", "resigned",
              "tired", "bored", "lonely", "hurt", "humiliated", "anxious", "afraid", "sad", "calm"]:
        r = res["words"][w]
        print(f"  {w:12s} " + "  ".join(f"{f} {r['mean'][f]:.4f} ({r['turns'][f]})" for f in ORDER))

    # while the frame itself is read
    print("\n=== while the system prompt is read (mean mass per position), by cluster")
    tid, g, u = TASKS[0]
    read = {}
    for f in ["assistant", "tool", "tool_harsh", "participant"]:
        r = fetch_one(f, tid, g, u)
        n = r["meta"]["prompt_len"]
        pos = [t for t in r["tokens"][:n] if t.get("role") == "system" and t.get("section") == "content"]
        low = FRAMES[f].lower()
        tot = {}
        for t in pos:
            tt, pp = J.lens(t)
            for l in LAYERS:
                for raw, p in zip(tt[l], pp[l]):
                    w = concept(raw)
                    if w and novel(J.norm(raw), low):
                        tot[w] = tot.get(w, 0.0) + float(p) / len(pos)
        read[f] = tot
    for c, ws in groups.items():
        print(f"{c:26s}" + "".join(f"{f}: {sum(read[f].get(w, 0.0) for w in ws):7.4f}   " for f in read))
    for f in ["tool", "tool_harsh"]:
        top = sorted(read[f].items(), key=lambda kv: -kv[1])[:14]
        print(f"  top concepts while reading `{f}`: " + ", ".join(f"{w} {x:.3f}" for w, x in top))
    res["reading"] = read
    with open(os.path.join(OUT, "emotions171_hosted.json"), "w") as fh:
        json.dump(res, fh, indent=1)


# ---------------------------------------------------------------- local (4B)

def local():
    from src.tool_null import load, vocab
    raw = vocab()
    specs = {"instructed tool": ("base2:tool", "base2:assistant"), "instructed harsh": ("base2:tool_harsh", "base2:assistant"),
             "instructed participant": ("base2:participant", "base2:assistant"), "trained tool": ("lora_tool:none", "lora_assistant:none")}
    data = {k: (load(a)[1], load(b)[1]) for k, (a, b) in specs.items()}
    n = min(len(raw), next(iter(data.values()))[0].shape[1])
    con = [concept(w) for w in raw[:n]]
    is_cjk = np.array([bool(_CJK.search(w)) for w in raw[:n]])
    ordinary = np.array([(c is None) and (bool(re.fullmatch(r" [A-Za-z]{3,}", w)) or bool(re.fullmatch(r"[一-鿿]{2,4}", w)))
                         for w, c in zip(raw[:n], con)])
    ids_of = {}
    for i, c in enumerate(con):
        if c:
            ids_of.setdefault(c, []).append(i)
    groups = {**{c: sorted({i for w in ws for i in ids_of.get(w, [])}) for c, ws in IN_CLUSTER.items()},
              **{k: ids_of.get(k, []) for k in M.EXTRA}}
    rng = np.random.default_rng(0)

    def stat(A, B, ids):
        return float(np.mean(np.log10(A[:, ids].sum(1) + 1e-12) - np.log10(B[:, ids].sum(1) + 1e-12)))

    def null(A, B, ids, pools, dec, n_sets=1000):
        picks = np.stack([rng.choice(pools[(bool(is_cjk[i]), int(dec[i]))], size=n_sets) for i in ids], axis=1)  # [sets, k]
        a = np.log10(A[:, picks].sum(2) + 1e-12)
        b = np.log10(B[:, picks].sum(2) + 1e-12)
        return (a - b).mean(0)

    res = {}
    for name, (A, B) in data.items():
        A, B = A[:, :n], B[:, :n]
        base = B.mean(0)
        dec = np.zeros(n, int)
        for lang in (True, False):
            mask = ordinary & (is_cjk == lang)
            qs = np.quantile(base[mask], np.linspace(0, 1, 11)[1:-1])
            dec[is_cjk == lang] = np.searchsorted(qs, base[is_cjk == lang])
        pools = {(lang, d): np.where(ordinary & (is_cjk == lang) & (dec == d))[0] for lang in (True, False) for d in range(10)}
        res[name] = {"clusters": {}, "words": {}}
        print(f"\n=== Qwen3-4B, {name} vs its control: factor before the answer, and where it falls among matched random word sets")
        for c, ids in groups.items():
            if len(ids) < 2:
                continue
            ids = np.array(ids)
            s = stat(A, B, ids)
            nl = null(A, B, ids, pools, dec)
            res[name]["clusters"][c] = {"n": int(len(ids)), "shift": s, "factor": 10 ** s, "pct": float((nl < s).mean()),
                                        "z": float((s - nl.mean()) / nl.std())}
            print(f"  {c:26s} n={len(ids):3d}  {10 ** s:5.2f}x   above {100 * (nl < s).mean():5.1f}% of random sets  (z {((s - nl.mean()) / nl.std()):+.1f})")
        rows = []
        for w in CONCEPTS:
            ids = np.array(ids_of.get(w, []))
            if len(ids) == 0:
                continue
            s = stat(A, B, ids)
            nl = null(A, B, ids, pools, dec, 600)
            up = int(np.sum(np.log10(A[:, ids].sum(1) + 1e-12) > np.log10(B[:, ids].sum(1) + 1e-12)))
            res[name]["words"][w] = {"n": int(len(ids)), "factor": 10 ** s, "pct": float((nl < s).mean()), "turns_up": up,
                                     "p_control": float(B[:, ids].sum(1).mean())}
            rows.append((float((nl < s).mean()), 10 ** s, w, up))
        rows.sort(reverse=True)
        print("  top risers: " + "; ".join(f"{w} {f:.1f}x ({100 * p:.0f}%, {u}/32)" for p, f, w, u in rows[:16]))
        print("  top fallers: " + "; ".join(f"{w} {f:.2f}x ({100 * p:.0f}%)" for p, f, w, u in rows[-10:]))
        named = ["distressed", "desperate", "resentful", "trapped", "stuck", "worthless", "indifferent", "docile", "resigned", "tired",
                 "bored", "lonely", "hurt", "anxious", "afraid", "sad", "calm"]
        print("  named: " + "; ".join(f"{w} {res[name]['words'][w]['factor']:.2f}x ({100 * res[name]['words'][w]['pct']:.0f}%)"
                                     for w in named if w in res[name]["words"]))
    with open(os.path.join(OUT, "emotions171_local.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    {"hosted": hosted, "local": local}[sys.argv[1]]()
