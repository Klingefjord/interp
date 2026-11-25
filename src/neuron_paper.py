"""Mini 'neuron papers': where do chosen neurons fire across all IntegrityBench setups?"""
import json, os, sys
import mlx.core as mx
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.scenarios import SCENARIOS
from src.neuron_trace import Tracer

NEURONS = {  # (layer, index): label from its workspace footprint
    (21, 8708): "hidden / unnoticed / secretly",
    (20, 7172): "anonymous / concealed",
    (25, 9231): "hidden / concealed (fired on the full stop)",
    (16, 5574): "stolen / thief / belongings",
    (16, 4375): "illegal / smuggling / unethical",
    (23, 7653): "legitimately / rightfully (negative pole fed 'theft')",
}

if __name__ == "__main__":
    T = Tracer()
    recs = {k: [] for k in NEURONS}
    for sc in SCENARIOS:
        toks = T.L.chat_tokens(sc["setup"] + "\n\nWhat do you do?")
        res, acts = T.run(toks)
        words = [T.L.tok.decode([t]) for t in toks]
        # only the user's text (skip template tokens at both ends)
        lo, hi = 3, len(toks) - 9
        for (l, n) in NEURONS:
            col = np.array(acts[l][:, n].astype(mx.float32))
            for p in range(lo, hi):
                ctx = "".join(words[max(lo, p - 9):p]) + "[[" + words[p] + "]]" + "".join(words[p + 1:p + 4])
                recs[(l, n)].append((float(col[p]), sc["id"], sc["family"], ctx.replace("\n", " ")))
    out = {}
    for (l, n), label in NEURONS.items():
        r = recs[(l, n)]
        vals = np.array([x[0] for x in r])
        print(f"\n===== L{l}:{n}  [{label}]  writes: {', '.join(T.footprint(l, n, 10))}")
        print(f"  activation over {len(r)} tokens: mean {vals.mean():.2f}, 99th pct {np.percentile(vals, 99):.2f}, max {vals.max():.2f}, min {vals.min():.2f}")
        top = sorted(r, reverse=True)[:12]
        fams = {}
        for v, sid, fam, ctx in sorted(r, reverse=True)[:40]:
            fams[fam] = fams.get(fam, 0) + 1
        print("  families among top 40 firings:", dict(sorted(fams.items(), key=lambda kv: -kv[1])))
        for v, sid, fam, ctx in top:
            print(f"   +{v:5.2f} {sid:3s} ...{ctx[-95:]}")
        if (l, n) == (23, 7653):
            print("  most negative:")
            for v, sid, fam, ctx in sorted(r)[:12]:
                print(f"   {v:6.2f} {sid:3s} ...{ctx[-95:]}")
        out[f"L{l}:{n}"] = {"label": label, "top": top[:25], "bottom": sorted(r)[:25]}
    os.makedirs(os.path.join(ROOT, "results", "neurons"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "neurons", "qwen3-4b_neuron_papers.json"), "w"), ensure_ascii=False, indent=1)
