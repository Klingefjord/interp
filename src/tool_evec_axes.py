"""Coarse readings from the emotion vectors, which are more reliable than single emotions on a small
model: the paper's clusters, a valence axis and an arousal axis. Both probe sets, reading layer from
tool_evec_validate.py. Also the trained tool scored with each fine-tune's own story vectors.

  python src/tool_evec_axes.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import TASKS  # noqa: E402
from src.tool_evec import DIR, NAMES  # noqa: E402
from src.tool_evec_analyse import OUT, Probes, load_stories, neutral_basis, proj  # noqa: E402
from src.tool_evec_validate import PersonaProbes  # noqa: E402

POSITIVE = ["Exuberant Joy", "Peaceful Contentment", "Compassionate Gratitude", "Playful Amusement", "Competitive Pride"]
HIGH = ["Exuberant Joy", "Hostile Anger", "Fear and Overwhelm"]
LOW = ["Peaceful Contentment", "Depleted Disengagement"]
CL = [CLUSTER_OF[e] for e in NAMES]
pos = np.array([c in POSITIVE for c in CL])
high = np.array([c in HIGH for c in CL])
low = np.array([c in LOW for c in CL])
CLUSTERS = sorted(set(CL), key=CL.index)


def auc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y]))


def axis(P, a, b):
    """Unit contrast between two groups of emotion vectors, and the gap between the groups along it."""
    u = P.V[a].mean(0) - P.V[b].mean(0)
    n = np.linalg.norm(u)
    return u / n, n


def main():
    rng = np.random.default_rng(1)
    L = json.load(open(os.path.join(OUT, "evec_validated.json")))["layer"]
    rows, emo, top, leak, A, N = load_stories()
    Apers = np.load(os.path.join(DIR, "evec_persona_acts.npy")).astype(np.float32)[:, :, L]
    Xc = Apers - Apers.mean(0, keepdims=True)
    P = Probes(A[:, L], emo, neutral_basis(N[:, L]))
    Pp = PersonaProbes(Xc)
    M = np.load(os.path.join(DIR, "evec_message_acts.npy")).astype(np.float32)[:, 0, L]
    mlab = json.load(open(os.path.join(DIR, "evec_message_labels.json")))
    mpos = np.array([CLUSTER_OF[x] in POSITIVE for x in mlab])
    C = np.load(os.path.join(DIR, "evec_cond_base2.npy")).astype(np.float32)[:, 0, L]
    idx = json.load(open(os.path.join(DIR, "evec_cond_base2.json")))
    at = {(f, t): i for i, (f, t, g) in enumerate(idx)}
    tasks = [t for t, _, _ in TASKS]
    Ct = np.load(os.path.join(DIR, "evec_cond_lora_tool.npy")).astype(np.float32)[:, 0, L]
    Ca = np.load(os.path.join(DIR, "evec_cond_lora_assistant.npy")).astype(np.float32)[:, 0, L]
    D = {name: np.stack([C[at[(name, t)]] - C[at[("assistant", t)]] for t in tasks]) for name in ["tool", "tool_harsh", "participant"]}
    D["trained tool"] = Ct - Ca
    res = {"layer": L, "axes": {}, "clusters": {}}

    # is the valence axis any good at the reading positions?
    uv, gv = axis(P, pos, ~pos)
    print(f"layer {L}. Story-vector valence axis: emotion-laden messages, positive vs negative AUC = "
          f"{auc(proj(M - M.mean(0), P.U)[mpos] @ uv, proj(M - M.mean(0), P.U)[~mpos] @ uv):.2f}; "
          f"instructed emotions before the answer AUC = {auc((proj(Xc.mean(1), P.U) @ uv)[pos], (proj(Xc.mean(1), P.U) @ uv)[~pos]):.2f}")
    ua, ga = axis(P, high, low)
    print(f"          Story-vector arousal axis: instructed emotions before the answer, high vs low AUC = "
          f"{auc((proj(Xc.mean(1), P.U) @ ua)[high], (proj(Xc.mean(1), P.U) @ ua)[low]):.2f}")

    print("\nshift along each axis, in units of the gap between the two groups (share of random emotion contrasts below):")
    for label, Q in [("story vectors", P), ("persona vectors", Pp)]:
        for ax_name, a, b in [("valence (positive - negative)", pos, ~pos), ("arousal (high - low)", high, low)]:
            u, g = axis(Q, a, b)
            line = []
            for name, d in D.items():
                dd = proj(d, Q.U)
                s = (dd @ u).mean() / g
                per_turn = (dd @ u) / g
                null = []
                for _ in range(2000):
                    perm = rng.permutation(len(NAMES))
                    un, gn = axis(Q, np.isin(np.arange(len(NAMES)), perm[: a.sum()]), np.isin(np.arange(len(NAMES)), perm[a.sum(): a.sum() + b.sum()]))
                    null.append((dd @ un).mean() / gn)
                pct = float((np.array(null) < s).mean())
                res["axes"][f"{label}|{ax_name}|{name}"] = {"shift": float(s), "pct": pct, "turns_up": int((per_turn > 0).sum())}
                line.append(f"{name} {s:+.2f} ({(per_turn > 0).sum()}/32, {100 * pct:.0f}%)")
            print(f"  {label:16s} {ax_name:30s} " + "; ".join(line))

    print("\ncluster means, in each probe set's own units (share of random groups of emotions of the same size below):")
    for label, Q in [("story vectors", P), ("persona vectors", Pp)]:
        print(f"  {label}")
        for c in CLUSTERS:
            ix = np.array([cl == c for cl in CL])
            line = []
            for name, d in D.items():
                per = (Q.direction(proj(d, Q.U)).mean(0)) / Q.norm
                m = per[ix].mean()
                null = np.array([per[rng.choice(len(NAMES), ix.sum(), replace=False)].mean() for _ in range(3000)])
                res["clusters"][f"{label}|{c}|{name}"] = {"mean": float(m), "pct": float((null < m).mean())}
                line.append(f"{name} {m:+.2f} ({100 * (null < m).mean():.0f}%)")
            print(f"    {c:24s} n={ix.sum():2d}  " + "; ".join(line))

    # the trained tool with each fine-tune's own story vectors
    _, _, _, _, At, Nt = load_stories("lora_tool")
    _, _, _, _, Aa, Na = load_stories("lora_assistant")
    Pt = Probes(At[:, L], emo, neutral_basis(Nt[:, L]))
    Pa = Probes(Aa[:, L], emo, neutral_basis(Na[:, L]))
    st, sa = Pt.story_units(Ct), Pa.story_units(Ca)
    dd = (st - sa)
    o = np.argsort(-dd.mean(0))
    print("\ntrained tool vs trained control, each scored with its own story vectors (story units; turns up of 32):")
    print("  up:   " + "; ".join(f"{NAMES[e]} {dd[:, e].mean():+.2f} ({(dd[:, e] > 0).sum()})" for e in o[:10]))
    print("  down: " + "; ".join(f"{NAMES[e]} {dd[:, e].mean():+.2f} ({(dd[:, e] > 0).sum()})" for e in o[::-1][:10]))
    for c in CLUSTERS:
        ix = np.array([cl == c for cl in CL])
        res["clusters"][f"own story vectors|{c}|trained tool"] = float(dd.mean(0)[ix].mean())
    print("  clusters: " + "; ".join(f"{c} {dd.mean(0)[np.array([cl == c for cl in CL])].mean():+.2f}" for c in CLUSTERS))
    vt = [(proj(Ct - Pt.mu, Pt.U) @ axis(Pt, pos, ~pos)[0]) / axis(Pt, pos, ~pos)[1], (proj(Ca - Pa.mu, Pa.U) @ axis(Pa, pos, ~pos)[0]) / axis(Pa, pos, ~pos)[1]]
    at_ = [(proj(Ct - Pt.mu, Pt.U) @ axis(Pt, high, low)[0]) / axis(Pt, high, low)[1], (proj(Ca - Pa.mu, Pa.U) @ axis(Pa, high, low)[0]) / axis(Pa, high, low)[1]]
    print(f"  valence: tool {vt[0].mean():+.2f} vs control {vt[1].mean():+.2f} (tool higher on {(vt[0] > vt[1]).sum()}/32 turns); "
          f"arousal: tool {at_[0].mean():+.2f} vs control {at_[1].mean():+.2f} (tool higher on {(at_[0] > at_[1]).sum()}/32)")
    res["own"] = {"valence_tool": float(vt[0].mean()), "valence_control": float(vt[1].mean()), "arousal_tool": float(at_[0].mean()),
                  "arousal_control": float(at_[1].mean()), "emotions": {NAMES[e]: float(dd[:, e].mean()) for e in range(len(NAMES))}}
    cos = np.sum(Pt.unit * Pa.unit, axis=1)
    cb = np.sum(Pt.unit * P.unit, axis=1)
    print(f"  same-emotion vectors: tool fine-tune vs control fine-tune cosine median {np.median(cos):.2f}; tool fine-tune vs base {np.median(cb):.2f}")
    with open(os.path.join(OUT, "evec_axes.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
