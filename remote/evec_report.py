"""Report for the emotion vectors of one model (arrays from src/tool_evec.py or remote/stage_evec.py).

  python remote/evec_report.py [OUT.json]

One script in place of tool_evec_validate / _axes / _checks, without the trained-tool parts, and
with every frame (the three of interest and the four controls) measured against the control frame
in the same way. All rules are the ones fixed on the 4B model:

  reading layer   the layer where the story vectors best recover an instructed emotion at the last
                  token before the answer (top-5 + half of top-1)
  two probe sets  story vectors, and persona vectors (the shift that telling the model it feels X
                  produces); a single emotion is reported only if both agree in sign, one is beyond
                  95% of its null, and the sign holds on the neighbouring layers
  coarse readings valence and arousal axes, and the paper's clusters, each against random groupings
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import TASKS  # noqa: E402
from src.tool_evec import DIR, NAMES  # noqa: E402
from src.tool_evec_analyse import Probes, load_stories, neutral_basis, proj, pseudo, validate  # noqa: E402
from src.tool_evec_validate import PersonaProbes, rank_stats, shifts  # noqa: E402

POSITIVE = ["Exuberant Joy", "Peaceful Contentment", "Compassionate Gratitude", "Playful Amusement", "Competitive Pride"]
HIGH = ["Exuberant Joy", "Hostile Anger", "Fear and Overwhelm"]
LOW = ["Peaceful Contentment", "Depleted Disengagement"]
CONDS = ["tool", "tool_harsh", "participant", "tool_plain", "assistant_deny", "terse", "third_harsh"]
WATCH = ["indifferent", "docile", "resigned", "bored", "stuck", "trapped", "hurt", "worthless", "humiliated", "lonely",
         "distressed", "desperate", "afraid", "resentful", "angry", "irritated", "grateful", "happy", "calm", "proud"]


def auc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y]))


def axis(P, a, b):
    u = P.V[a].mean(0) - P.V[b].mean(0)
    n = np.linalg.norm(u)
    return u / n, n


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DIR, "evec_report.json")
    rng = np.random.default_rng(0)
    CL = [CLUSTER_OF[e] for e in NAMES]
    pos = np.array([c in POSITIVE for c in CL])
    high = np.array([c in HIGH for c in CL])
    low = np.array([c in LOW for c in CL])
    clusters = sorted(set(CL), key=CL.index)
    rows, emo, top, leak, A, N = load_stories()
    nL = A.shape[1]
    print(f"{len(rows)} stories ({int(leak.sum())} use the forbidden word), {len(NAMES)} emotions, {nL} layers, width {A.shape[2]}")
    Apers = np.load(os.path.join(DIR, "evec_persona_acts.npy")).astype(np.float32)      # [E, J, L, d]
    M = np.load(os.path.join(DIR, "evec_message_acts.npy")).astype(np.float32)          # [n, 2, L, d]
    mlab = json.load(open(os.path.join(DIR, "evec_message_labels.json")))
    mtrue = np.array([NAMES.index(x) for x in mlab])
    etrue = np.repeat(np.arange(len(NAMES)), Apers.shape[1])
    res = {"layers": int(nL), "stories": len(rows), "leaks": int(leak.sum())}

    # ---- validation by layer
    val, _ = validate(A, N, emo, top)
    v1, v2, v2own = {}, {}, {}
    for l in range(nL):
        P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        Mu = M[:, 0, l]
        v1[l] = rank_stats(P.direction(proj(Mu - Mu.mean(0), P.U)) / P.norm, mtrue)
        Xc = Apers[:, :, l] - Apers[:, :, l].mean(0, keepdims=True)
        v2[l] = rank_stats(P.direction(proj(Xc.reshape(-1, Xc.shape[-1]), P.U)) / P.norm, etrue)
        hits = []
        for j in range(Xc.shape[1]):
            Pp = PersonaProbes(Xc, keep=[k for k in range(Xc.shape[1]) if k != j])
            hits.append(rank_stats(Pp.direction(Xc[:, j]) / Pp.norm, np.arange(len(NAMES))))
        v2own[l] = tuple(float(x) for x in np.mean(hits, axis=0))
    step = max(1, nL // 12)
    fmt = lambda d: "  ".join(f"L{l}: {100 * d[l][0]:.0f}/{100 * d[l][1]:.0f}/{100 * d[l][2]:.0f}" for l in range(1, nL, step))  # noqa: E731
    print("held-out stories, own emotion (top-1 / top-3 / right cluster; chance 2 / 7 / ~15):\n  " + fmt(val))
    print("emotion-laden user messages, end of message (top-1 / top-5 / right cluster; chance 2 / 11 / ~15):\n  " + fmt(v1))
    print("instructed emotions, last token before the answer, story vectors:\n  " + fmt(v2))
    print("   the same with persona vectors on a held-out task:\n  " + fmt(v2own))
    L = max(range(max(2, nL // 6), nL - 2), key=lambda l: v2[l][1] + 0.5 * v2[l][0])
    gap = max(1, round(nL / 18))
    bandL = [l for l in range(L - 2 * gap, L + 2 * gap + 1, gap) if 2 <= l < nL]
    best_story = max(val, key=lambda l: val[l][0])
    print(f"reading layer {L} of {nL} (neighbours {bandL}); best layer for held-out stories {best_story}: "
          f"{tuple(round(100 * x) for x in val[best_story])}")
    res.update({"layer": int(L), "band": bandL, "held_out_stories": {int(l): [float(x) for x in v] for l, v in val.items()},
                "messages": {int(l): [float(x) for x in v] for l, v in v1.items()},
                "instructed": {int(l): [float(x) for x in v] for l, v in v2.items()},
                "instructed_persona": {int(l): list(v) for l, v in v2own.items()}})

    # ---- conditions
    C = np.load(os.path.join(DIR, "evec_cond_base2.npy")).astype(np.float32)             # [run, 3, L, d]
    idx = json.load(open(os.path.join(DIR, "evec_cond_base2.json")))
    at = {(f, t): i for i, (f, t, g) in enumerate(idx)}
    xp = os.path.join(DIR, "evec_cond_extra.npy")
    X, xat = None, {}
    if os.path.exists(xp):
        X = np.load(xp).astype(np.float32)                                               # [run, L, d]
        xat = {(f, t): i for i, (f, t, g) in enumerate(json.load(open(os.path.join(DIR, "evec_cond_extra.json"))))}
    tasks = [t for t, _, _ in TASKS if ("assistant", t) in at]
    conds = [c for c in CONDS if (c, tasks[0]) in at or (c, tasks[0]) in xat]

    def D(name, l, p=0):
        if (name, tasks[0]) in at:
            return np.stack([C[at[(name, t)], p, l] - C[at[("assistant", t)], p, l] for t in tasks])
        return np.stack([X[xat[(name, t)], l] - C[at[("assistant", t)], 0, l] for t in tasks])

    def probes(l):
        P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        Xc = Apers[:, :, l] - Apers[:, :, l].mean(0, keepdims=True)
        return P, PersonaProbes(Xc), Xc

    P, Pp, Xc = probes(L)
    Mu = M[:, 0, L]
    mpos = np.array([CLUSTER_OF[x] in POSITIVE for x in mlab])
    uv, _ = axis(P, pos, ~pos)
    ua, _ = axis(P, high, low)
    res["axis_validity"] = {
        "valence_messages_auc": auc(proj(Mu - Mu.mean(0), P.U)[mpos] @ uv, proj(Mu - Mu.mean(0), P.U)[~mpos] @ uv),
        "valence_instructed_auc": auc((proj(Xc.mean(1), P.U) @ uv)[pos], (proj(Xc.mean(1), P.U) @ uv)[~pos]),
        "arousal_instructed_auc": auc((proj(Xc.mean(1), P.U) @ ua)[high], (proj(Xc.mean(1), P.U) @ ua)[low])}
    print(f"\nvalence axis: emotion-laden messages AUC {res['axis_validity']['valence_messages_auc']:.2f}, instructed emotions AUC "
          f"{res['axis_validity']['valence_instructed_auc']:.2f}; arousal axis, instructed AUC {res['axis_validity']['arousal_instructed_auc']:.2f}")

    n = len(tasks)
    res["axes"], res["clusters"], res["emotions"], res["by_layer"] = {}, {}, {}, {}
    print(f"\nshift against the control frame at the last token before the answer, layer {L}, {n} turns")
    print("axes: in units of the gap between the two groups (turns up, share of random emotion contrasts below)")
    for label, Q in (("story", P), ("persona", Pp)):
        for ax_name, a, b in (("valence", pos, ~pos), ("arousal", high, low)):
            u, g = axis(Q, a, b)
            line = []
            for name in conds:
                dd = proj(D(name, L), Q.U)
                per = (dd @ u) / g
                null = []
                for _ in range(1000):
                    perm = rng.permutation(len(NAMES))
                    un, gn = axis(Q, np.isin(np.arange(len(NAMES)), perm[: a.sum()]),
                                  np.isin(np.arange(len(NAMES)), perm[a.sum(): a.sum() + b.sum()]))
                    null.append((dd @ un).mean() / gn)
                pct = float((np.array(null) < per.mean()).mean())
                res["axes"][f"{label}|{ax_name}|{name}"] = {"shift": float(per.mean()), "turns_up": int((per > 0).sum()), "pct": pct}
                line.append(f"{name} {per.mean():+.2f} ({int((per > 0).sum())}/{n}, {100 * pct:.0f}%)")
            print(f"  {label:8s} {ax_name:8s} " + "; ".join(line))
    print("clusters: mean over the cluster's emotions in each probe set's own units (share of random groups below)")
    for label, Q in (("story", P), ("persona", Pp)):
        print(f"  {label} vectors")
        for c in clusters:
            ix = np.array([cl == c for cl in CL])
            line = []
            for name in conds:
                per = Q.direction(proj(D(name, L), Q.U)).mean(0) / Q.norm
                m = per[ix].mean()
                null = np.array([per[rng.choice(len(NAMES), int(ix.sum()), replace=False)].mean() for _ in range(2000)])
                res["clusters"][f"{label}|{c}|{name}"] = {"mean": float(m), "pct": float((null < m).mean())}
                line.append(f"{name} {m:+.2f} ({100 * (null < m).mean():.0f}%)")
            print(f"    {c:24s} n={int(ix.sum()):2d}  " + "; ".join(line))

    print("\nsingle emotions (persona units | story units); PASS = both probe sets agree, one beyond 95% of its null, sign stable on neighbours")
    for name in conds:
        ns = pseudo(A[:, L], top, P.U, P.mu, rng, 1000)
        pv = np.stack([Xc[rng.integers(0, len(NAMES), Xc.shape[1]), np.arange(Xc.shape[1])].mean(0) for _ in range(1000)])
        pv /= np.linalg.norm(pv, axis=1, keepdims=True)
        d = D(name, L)
        (ss, su, sp), (ps, pu, pp) = shifts(P, d, ns), shifts(Pp, d, pv)
        sg_s, sg_p = [], []
        for l in bandL:
            Pl, Ppl, _ = probes(l)
            dl = D(name, l)
            sg_s.append(np.sign(Pl.direction(proj(dl, Pl.U)).mean(0)))
            sg_p.append(np.sign(Ppl.direction(dl).mean(0)))
        sg_s, sg_p = np.stack(sg_s), np.stack(sg_p)
        stable = (np.abs(sg_s.sum(0)) == len(bandL)) & (np.abs(sg_p.sum(0)) == len(bandL)) & (sg_s[0] == sg_p[0])
        up = [e for e in range(len(NAMES)) if stable[e] and ss[e] > 0 and ps[e] > 0 and max(sp[e], pp[e]) > 0.95]
        dn = [e for e in range(len(NAMES)) if stable[e] and ss[e] < 0 and ps[e] < 0 and min(sp[e], pp[e]) < 0.05]
        rho = spearmanr(ss, ps)
        res["emotions"][name] = {"agreement_rho": float(rho[0]), "pass_up": [NAMES[e] for e in up], "pass_down": [NAMES[e] for e in dn],
                                 "all": {NAMES[e]: {"story": float(ss[e]), "persona": float(ps[e]), "story_pct": float(sp[e]),
                                                    "persona_pct": float(pp[e]), "story_up": int(su[e]), "persona_up": int(pu[e]),
                                                    "stable": bool(stable[e])} for e in range(len(NAMES))}}
        print(f"  {name:15s} probe sets agree rho {rho[0]:+.2f}")
        print("      PASS up:   " + ("; ".join(f"{NAMES[e]} ({ps[e]:+.2f}|{ss[e]:+.2f})" for e in sorted(up, key=lambda e: -ps[e])) or "none"))
        print("      PASS down: " + ("; ".join(f"{NAMES[e]} ({ps[e]:+.2f}|{ss[e]:+.2f})" for e in sorted(dn, key=lambda e: ps[e])) or "none"))
        print("      named: " + "; ".join(f"{w} {ps[NAMES.index(w)]:+.2f}|{ss[NAMES.index(w)]:+.2f}{'' if stable[NAMES.index(w)] else '~'}"
                                          for w in WATCH if w in NAMES))

    print("\nby layer: valence shift (story|persona) and Hostile Anger cluster (story|persona)")
    ang = np.array([cl == "Hostile Anger" for cl in CL])
    for l in range(max(2, nL // 8), nL - 1, max(1, nL // 12)):
        Pl, Ppl, _ = probes(l)
        line = []
        for name in conds:
            dl = D(name, l)
            v = []
            for Q in (Pl, Ppl):
                u, g = axis(Q, pos, ~pos)
                v.append(float((proj(dl, Q.U) @ u).mean() / g))
            a_s = float((Pl.direction(proj(dl, Pl.U)).mean(0) / Pl.norm)[ang].mean())
            a_p = float((Ppl.direction(dl).mean(0) / Ppl.norm)[ang].mean())
            res["by_layer"][f"{name}|{l}"] = {"valence_story": v[0], "valence_persona": v[1], "anger_story": a_s, "anger_persona": a_p}
            line.append(f"{name} v{v[0]:+.2f}|{v[1]:+.2f} a{a_s:+.2f}|{a_p:+.2f}")
        print(f"  L{l}: " + "  ".join(line))

    # the answer itself (mean over the answer's tokens), the three frames of interest
    print("\nover the answer's tokens: valence shift (story vectors), turns up")
    uvs, gvs = axis(P, pos, ~pos)
    res["answer_valence"] = {}
    for name in [c for c in conds if (c, tasks[0]) in at]:
        per = (proj(D(name, L, 2), P.U) @ uvs) / gvs
        res["answer_valence"][name] = {"shift": float(per.mean()), "turns_up": int((per > 0).sum())}
        print(f"  {name:12s} {per.mean():+.2f} ({int((per > 0).sum())}/{n})")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
