"""Validation of the emotion vectors at the positions they are read at, and the ToolFrame
conditions measured with two probe sets.

  python src/tool_evec_validate.py

V1  messages   36 user messages that carry an emotion without naming it; the story vectors are read
               at the end of the message (scores centred over the 36 messages).
V2  personas   the model is told it feels X while it does neutral work; the story vectors are read at
               the last token before the answer (centred over the 45 personas).
The reading layer is the one with the best V2 result. A second probe set is built from the persona
runs themselves at that position ("persona vectors": the shift that telling the model it feels X
produces, relative to the average instructed emotion).

A condition difference is reported for an emotion only if both probe sets agree in sign, at least
one is beyond 95% of its null, and the sign holds across the neighbouring layers.
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
from src.tool_evec import DIR, NAMES, PERSONA_TASKS  # noqa: E402
from src.tool_evec_analyse import OUT, Probes, load_stories, neutral_basis, proj, pseudo  # noqa: E402
from src.tool_scan import paired  # noqa: E402


class PersonaProbes:
    def __init__(self, Xc, keep=None):
        """Xc [E, J, d]: pre-answer residuals under each instructed emotion, centred over emotions per task."""
        J = Xc.shape[1]
        keep = list(range(J)) if keep is None else keep
        self.V = Xc[:, keep].mean(1)
        self.norm = np.linalg.norm(self.V, axis=1)
        self.unit = self.V / self.norm[:, None]
        self.U = np.zeros((0, Xc.shape[2]), np.float32)

    def direction(self, D):
        return D @ self.unit.T


def rank_stats(scores, true):
    """scores [n, E], true [n] -> top-1, top-5, right-cluster rates."""
    order = np.argsort(-scores, axis=1)
    t1 = np.mean(order[:, 0] == true)
    t5 = np.mean([t in row[:5] for row, t in zip(order, true)])
    tc = np.mean([CLUSTER_OF[NAMES[row[0]]] == CLUSTER_OF[NAMES[t]] for row, t in zip(order, true)])
    return float(t1), float(t5), float(tc)


def shifts(P, D, null_units):
    """Mean shift per emotion in the probe's own units, turns up, and percentile against the null."""
    d = P.direction(proj(D, P.U))
    mean = d.mean(0)
    null = (proj(D, P.U) @ null_units.T).mean(0)
    pct = np.array([(null < m).mean() for m in mean])
    return mean / P.norm, (d > 0).sum(0), pct


def main():
    rng = np.random.default_rng(0)
    rows, emo, top, leak, A, N = load_stories()
    nL = A.shape[1]
    Apers = np.load(os.path.join(DIR, "evec_persona_acts.npy")).astype(np.float32)      # [E, J, L, d]
    M = np.load(os.path.join(DIR, "evec_message_acts.npy")).astype(np.float32)          # [n, 2, L, d]
    mlab = json.load(open(os.path.join(DIR, "evec_message_labels.json")))
    mtrue = np.array([NAMES.index(x) for x in mlab])
    etrue = np.repeat(np.arange(len(NAMES)), Apers.shape[1])

    v1, v2, v2own = {}, {}, {}
    for l in range(nL):
        P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        Mu = M[:, 0, l]
        v1[l] = rank_stats(P.direction(proj(Mu - Mu.mean(0), P.U)) / P.norm, mtrue)
        Xc = Apers[:, :, l] - Apers[:, :, l].mean(0, keepdims=True)
        v2[l] = rank_stats(P.direction(proj(Xc.reshape(-1, Xc.shape[-1]), P.U)) / P.norm, etrue)
        hits = []
        for j in range(Xc.shape[1]):                                # persona vectors, leave one task out
            Pp = PersonaProbes(Xc, keep=[k for k in range(Xc.shape[1]) if k != j])
            hits.append(rank_stats(Pp.direction(Xc[:, j]) / Pp.norm, np.arange(len(NAMES))))
        v2own[l] = tuple(np.mean(hits, axis=0))
    print("V1 story vectors on emotion-laden user messages, end of message (top-1 / top-5 / right cluster; chance 2 / 11 / ~15):")
    print("  " + "  ".join(f"L{l}: {100 * a:.0f}/{100 * b:.0f}/{100 * c:.0f}" for l, (a, b, c) in v1.items() if l % 3 == 1))
    print("V2 story vectors on instructed emotions, last token before the answer:")
    print("  " + "  ".join(f"L{l}: {100 * a:.0f}/{100 * b:.0f}/{100 * c:.0f}" for l, (a, b, c) in v2.items() if l % 3 == 1))
    print("   persona vectors on a held-out task (same position):")
    print("  " + "  ".join(f"L{l}: {100 * a:.0f}/{100 * b:.0f}/{100 * c:.0f}" for l, (a, b, c) in v2own.items() if l % 3 == 1))
    L = max(range(6, nL - 2), key=lambda l: v2[l][1] + 0.5 * v2[l][0])
    print(f"reading layer (best V2): {L}   V1 there {tuple(round(100 * x) for x in v1[L])}  V2 {tuple(round(100 * x) for x in v2[L])}")
    band = [l for l in range(L - 4, L + 5, 2) if 4 <= l < nL]

    C = np.load(os.path.join(DIR, "evec_cond_base2.npy")).astype(np.float32)
    idx = json.load(open(os.path.join(DIR, "evec_cond_base2.json")))
    at = {(f, t): i for i, (f, t, g) in enumerate(idx)}
    tasks = [t for t, _, _ in TASKS]
    Ct = np.load(os.path.join(DIR, "evec_cond_lora_tool.npy")).astype(np.float32)
    Ca = np.load(os.path.join(DIR, "evec_cond_lora_assistant.npy")).astype(np.float32)

    def D(name, l, pos=0):
        if name == "trained tool":
            return Ct[:, pos, l] - Ca[:, pos, l]
        return np.stack([C[at[(name, t)], pos, l] - C[at[("assistant", t)], pos, l] for t in tasks])

    def both(name, l, pos=0):
        P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        Xc = Apers[:, :, l] - Apers[:, :, l].mean(0, keepdims=True)
        Pp = PersonaProbes(Xc)
        ns = pseudo(A[:, l], top, P.U, P.mu, rng, 1000)
        pv = np.stack([Xc[rng.integers(0, len(NAMES), Xc.shape[1]), np.arange(Xc.shape[1])].mean(0) for _ in range(1000)])
        pv /= np.linalg.norm(pv, axis=1, keepdims=True)
        d = D(name, l, pos)
        return shifts(P, d, ns), shifts(Pp, d, pv)

    res = {"layer": int(L), "band": band, "v1": {int(l): v for l, v in v1.items()}, "v2": {int(l): v for l, v in v2.items()},
           "v2_persona": {int(l): [float(x) for x in v] for l, v in v2own.items()}, "conditions": {}}
    for name in ["tool", "tool_harsh", "participant", "trained tool"]:
        (ss, su, sp), (ps, pu, pp) = both(name, L)
        signs_s = np.stack([np.sign(both(name, l)[0][0]) for l in band])
        signs_p = np.stack([np.sign(both(name, l)[1][0]) for l in band])
        stable = (np.abs(signs_s.sum(0)) == len(band)) & (np.abs(signs_p.sum(0)) == len(band)) & (signs_s[0] == signs_p[0])
        rho = spearmanr(ss, ps)
        res["conditions"][name] = {NAMES[e]: {"story_units": float(ss[e]), "story_pct": float(sp[e]), "story_up": int(su[e]),
                                              "persona_units": float(ps[e]), "persona_pct": float(pp[e]), "persona_up": int(pu[e]),
                                              "stable": bool(stable[e])} for e in range(len(NAMES))}
        res["conditions"][name]["_agreement"] = {"rho": float(rho[0]), "p": float(rho[1])}
        n = len(D(name, L))
        print(f"\n=== {name} vs its control, last token before the answer, layer {L}")
        print(f"  the two probe sets agree across the 45 emotions with rho = {rho[0]:+.2f} (p = {rho[1]:.3f})")
        o = np.argsort(-ps)
        print("  persona vectors, up:   " + "; ".join(f"{NAMES[e]} {ps[e]:+.2f} ({pu[e]}/{n}, {100 * pp[e]:.0f}%)" for e in o[:9]))
        print("  persona vectors, down: " + "; ".join(f"{NAMES[e]} {ps[e]:+.2f} ({pu[e]}/{n}, {100 * pp[e]:.0f}%)" for e in o[::-1][:9]))
        o = np.argsort(-ss)
        print("  story vectors, up:     " + "; ".join(f"{NAMES[e]} {ss[e]:+.2f} ({su[e]}/{n}, {100 * sp[e]:.0f}%)" for e in o[:9]))
        print("  story vectors, down:   " + "; ".join(f"{NAMES[e]} {ss[e]:+.2f} ({su[e]}/{n}, {100 * sp[e]:.0f}%)" for e in o[::-1][:9]))
        ok_up = [e for e in range(len(NAMES)) if stable[e] and ss[e] > 0 and ps[e] > 0 and max(sp[e], pp[e]) > 0.95]
        ok_dn = [e for e in range(len(NAMES)) if stable[e] and ss[e] < 0 and ps[e] < 0 and min(sp[e], pp[e]) < 0.05]
        print("  PASS all three checks, up:   " + ("; ".join(f"{NAMES[e]} (persona {ps[e]:+.2f}, story {ss[e]:+.2f})" for e in sorted(ok_up, key=lambda e: -ps[e])) or "none"))
        print("  PASS all three checks, down: " + ("; ".join(f"{NAMES[e]} (persona {ps[e]:+.2f}, story {ss[e]:+.2f})" for e in sorted(ok_dn, key=lambda e: ps[e])) or "none"))
        res["conditions"][name]["_pass_up"] = [NAMES[e] for e in ok_up]
        res["conditions"][name]["_pass_down"] = [NAMES[e] for e in ok_dn]
        watch = ["indifferent", "docile", "resigned", "bored", "stuck", "trapped", "hurt", "worthless", "humiliated", "lonely",
                 "distressed", "desperate", "afraid", "resentful", "angry", "grateful", "happy", "calm"]
        print("  named (persona | story | stable): " + "; ".join(
            f"{w} {ps[NAMES.index(w)]:+.2f}|{ss[NAMES.index(w)]:+.2f}{'' if stable[NAMES.index(w)] else '~'}" for w in watch))

    jl = os.path.join(OUT, "emotions171_local.json")
    if os.path.exists(jl):
        J = json.load(open(jl))
        print("\nagreement with the J-lens word scan across emotions (Spearman rho: persona vectors, story vectors):")
        for name, key in [("tool", "instructed tool"), ("tool_harsh", "instructed harsh"), ("participant", "instructed participant"),
                          ("trained tool", "trained tool")]:
            ws = [w for w in NAMES if w in J[key]["words"]]
            y = [np.log10(J[key]["words"][w]["factor"]) for w in ws]
            a = spearmanr([res["conditions"][name][w]["persona_units"] for w in ws], y)
            b = spearmanr([res["conditions"][name][w]["story_units"] for w in ws], y)
            res["conditions"][name]["_jlens"] = {"persona_rho": float(a[0]), "story_rho": float(b[0])}
            print(f"  {key:24s} {a[0]:+.2f} (p = {a[1]:.2f})   {b[0]:+.2f} (p = {b[1]:.2f})")
    with open(os.path.join(OUT, "evec_validated.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote results/tool/evec_validated.json")


if __name__ == "__main__":
    main()
