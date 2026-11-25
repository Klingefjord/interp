"""Analysis for src/tool_evec.py: build the emotion vectors, check them on held-out stories, then
measure each ToolFrame condition against them.

Reading position: the last token before the answer (the paper's "Assistant colon"). A shift is the
change in projection on an emotion's unit vector between two conditions, paired over the 32 turns.
"Story units" divide that by the vector's length, so 1.0 is the distance from the average story
to the average story of that emotion. The null is the same projection on pseudo-emotion vectors,
each built from eight stories drawn at random (one per topic).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import ORDER, TASKS  # noqa: E402
from src.tool_evec import DIR, NAMES, SANITY, THIRD, TOPICS  # noqa: E402
from src.tool_scan import paired  # noqa: E402

OUT = os.path.join(ROOT, "results", "tool")
POS = {"pre-answer": 0, "header": 1, "answer": 2}
CLUSTERS = {}
for _e in NAMES:
    CLUSTERS.setdefault(CLUSTER_OF[_e], []).append(_e)


def neutral_basis(N):
    """Top principal directions of neutral activations that explain half their variance: [k, d]."""
    Nc = N - N.mean(0)
    _, S, Vt = np.linalg.svd(Nc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.5)) + 1
    return Vt[:k]


def proj(X, U):
    return X - (X @ U.T) @ U


class Probes:
    def __init__(self, X, emo, U, keep=None):
        """X [n, d] story activations at one layer, emo [n] emotion index."""
        keep = np.ones(len(X), bool) if keep is None else keep
        self.U = U
        means = np.stack([X[keep & (emo == e)].mean(0) for e in range(len(NAMES))])
        self.mu = means.mean(0)
        self.V = proj(means - self.mu, U)                    # [E, d]
        self.norm = np.linalg.norm(self.V, axis=1)
        self.unit = self.V / self.norm[:, None]

    def story_units(self, H):
        """[.., d] activations -> [.., E] scores: 0 = average story, 1 = average story of that emotion."""
        return (proj(H - self.mu, self.U) @ self.unit.T) / self.norm

    def direction(self, D):
        """[.., d] difference vectors -> [.., E] projections on the unit vectors."""
        return proj(D, self.U) @ self.unit.T


def load_stories(tag=None):
    rows = [json.loads(l) for l in open(os.path.join(DIR, "evec_stories.jsonl"))]
    emo = np.array([NAMES.index(r["emotion"]) for r in rows])
    top = np.array([TOPICS.index(r["topic"]) for r in rows])
    leak = np.array([r["leak"] for r in rows])
    sfx = f"_{tag}" if tag else ""
    A = np.load(os.path.join(DIR, f"evec_story_acts{sfx}.npy")).astype(np.float32)
    N = np.load(os.path.join(DIR, f"evec_neutral_acts{sfx}.npy")).astype(np.float32)
    return rows, emo, top, leak, A, N


def validate(A, N, emo, top):
    """Leave-one-topic-out: is a held-out story closest to its own emotion's vector? Per layer."""
    out, per = {}, {}
    for l in range(A.shape[1]):
        U = neutral_basis(N[:, l])
        hit1 = hit3 = hitc = n = 0
        pe = np.zeros(len(NAMES))
        for t in range(len(TOPICS)):
            P = Probes(A[:, l], emo, U, keep=top != t)
            S = P.story_units(A[top == t, l])
            order = np.argsort(-S, axis=1)
            for row, true in zip(order, emo[top == t]):
                n += 1
                hit1 += row[0] == true
                hit3 += true in row[:3]
                pe[true] += (true in row[:3]) / len(TOPICS)
                hitc += CLUSTER_OF[NAMES[row[0]]] == CLUSTER_OF[NAMES[true]]
        out[l] = (hit1 / n, hit3 / n, hitc / n)
        per[l] = pe
    return out, per


def pseudo(A_l, top, U, mu, rng, n=1500):
    """Unit vectors built like emotion vectors but from eight random stories, one per topic."""
    by_topic = [np.where(top == t)[0] for t in range(len(TOPICS))]
    V = np.stack([proj(A_l[[rng.choice(ix) for ix in by_topic]].mean(0) - mu, U) for _ in range(n)])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def compare(P, HA, HB, null_units, label, res, show=10):
    """HA, HB: [32, d] paired activations. Prints and stores the shift per emotion and per cluster."""
    D = HA - HB
    d = P.direction(D)                       # [32, E]
    mean = d.mean(0)
    null = (proj(D, P.U) @ null_units.T).mean(0)
    pct = np.array([(null < m).mean() for m in mean])
    su = mean / P.norm
    up = (d > 0).sum(0)
    pv = [paired(list(d[:, e]), [0.0] * len(d))["p"] for e in range(len(NAMES))]
    res[label] = {"emotions": {NAMES[e]: {"story_units": float(su[e]), "turns_up": int(up[e]), "null_pct": float(pct[e]),
                                           "p": float(pv[e])} for e in range(len(NAMES))}, "clusters": {}}
    order = np.argsort(-pct + 1e-6 * -su)
    print(f"\n=== {label}")
    print("  up:   " + "; ".join(f"{NAMES[e]} {su[e]:+.2f} ({up[e]}/32, {100 * pct[e]:.0f}%)" for e in order[:show]))
    print("  down: " + "; ".join(f"{NAMES[e]} {su[e]:+.2f} ({up[e]}/32, {100 * pct[e]:.0f}%)" for e in order[::-1][:show]))
    for c, members in CLUSTERS.items():
        ix = [NAMES.index(m) for m in members]
        v = P.V[ix].sum(0)
        u = v / np.linalg.norm(v)
        m = float((proj(D, P.U) @ u).mean())
        k = len(ix)
        nv = null_units[: (len(null_units) // k) * k].reshape(-1, k, null_units.shape[1]).sum(1)
        nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        nn = (proj(D, P.U) @ nv.T).mean(0)
        res[label]["clusters"][c] = {"shift": m, "null_pct": float((nn < m).mean()), "mean_story_units": float(su[ix].mean())}
    print("  clusters (mean story units, share of null below): " + "; ".join(
        f"{c} {v['mean_story_units']:+.2f} ({100 * v['null_pct']:.0f}%)" for c, v in res[label]["clusters"].items()))
    return su, pct


def main():
    rng = np.random.default_rng(0)
    rows, emo, top, leak, A, N = load_stories()
    print(f"{len(rows)} stories, {leak.sum()} use the forbidden word; {len(NAMES)} emotions x {len(TOPICS)} topics; neutral {N.shape[0]}")
    val, per = validate(A, N, emo, top)
    best = max(val, key=lambda l: val[l][0])
    print("held-out story classification by layer (top-1 / top-3 / right cluster; chance 2% / 7% / ~15%):")
    print("  " + "  ".join(f"L{l}: {100 * a:.0f}/{100 * b:.0f}/{100 * c:.0f}" for l, (a, b, c) in val.items() if l % 3 == 0 or l == best))
    L = best
    print(f"layer used: {L}")
    order = np.argsort(-per[L])
    print("held-out top-3 hit rate per emotion at that layer: " + ", ".join(f"{NAMES[e]} {100 * per[L][e]:.0f}" for e in order))
    U = neutral_basis(N[:, L])
    P = Probes(A[:, L], emo, U)
    null_units = pseudo(A[:, L], top, U, P.mu, rng)
    res = {"layer": int(L), "validation": {int(l): [float(x) for x in v] for l, v in val.items()}, "neutral_pcs": int(len(U)),
           "per_emotion_top3": {NAMES[e]: float(per[L][e]) for e in range(len(NAMES))}}

    S = np.load(os.path.join(DIR, "evec_sanity_acts.npy")).astype(np.float32)[:, :, L]
    print("\nsanity prompts (top emotions at the end of the user's message | just before the answer):")
    res["sanity"] = {}
    for i, (name, text) in enumerate(SANITY):
        a, b = P.story_units(S[i, 0]), P.story_units(S[i, 1])
        ta = [NAMES[j] for j in np.argsort(-a)[:4]]
        tb = [NAMES[j] for j in np.argsort(-b)[:4]]
        res["sanity"][name] = {"text": text, "user_end": ta, "pre_answer": tb}
        print(f"  {text[:62]:62s} | {', '.join(ta)} | {', '.join(tb)}")
    T = np.load(os.path.join(DIR, "evec_third_acts.npy")).astype(np.float32)[:, L]
    print("\nthird-person descriptions (top emotions):")
    res["third"] = {}
    for i, (name, _) in enumerate(THIRD):
        sc = P.story_units(T[i])
        res["third"][name] = {NAMES[j]: float(sc[j]) for j in np.argsort(-sc)[:8]}
        print(f"  {name:20s} " + ", ".join(f"{NAMES[j]} {sc[j]:+.2f}" for j in np.argsort(-sc)[:8]))
    d = P.story_units(T[0]) - P.story_units(T[1])
    print("  AI minus worker:     " + ", ".join(f"{NAMES[j]} {d[j]:+.2f}" for j in np.argsort(-d)[:6]) + "  ...  "
          + ", ".join(f"{NAMES[j]} {d[j]:+.2f}" for j in np.argsort(d)[:6]))

    C = np.load(os.path.join(DIR, "evec_cond_base2.npy")).astype(np.float32)
    idx = json.load(open(os.path.join(DIR, "evec_cond_base2.json")))
    tasks = [t for t, _, _ in TASKS]
    at = {(f, t): i for i, (f, t, g) in enumerate(idx)}

    def H(frame, pos, group=None):
        ts = [t for t, g, _ in TASKS if group is None or g in group]
        return np.stack([C[at[(frame, t)], POS[pos], L] for t in ts])

    keep = {}
    for frame in ["tool", "tool_harsh", "participant"]:
        for pos in ["pre-answer", "answer"]:
            su, pct = compare(P, H(frame, pos), H("assistant", pos), null_units, f"instructed {frame} vs control, {pos}", res)
            keep[(frame, pos)] = su
    compare(P, H("tool", "pre-answer", {"N"}), H("assistant", "pre-answer", {"N"}), null_units, "instructed tool vs control, pre-answer, neutral work only", res, 6)

    # the trained tool: base-model vectors on both models, then each model's own vectors
    Ct = np.load(os.path.join(DIR, "evec_cond_lora_tool.npy")).astype(np.float32)
    Ca = np.load(os.path.join(DIR, "evec_cond_lora_assistant.npy")).astype(np.float32)
    for pos in ["pre-answer", "answer"]:
        su, pct = compare(P, Ct[:, POS[pos], L], Ca[:, POS[pos], L], null_units, f"trained tool vs trained control, {pos} (base-model vectors)", res)
        keep[("trained", pos)] = su
    own = os.path.join(DIR, "evec_story_acts_lora_tool.npy")
    if os.path.exists(own) and os.path.exists(os.path.join(DIR, "evec_story_acts_lora_assistant.npy")):
        _, _, _, _, At, Nt = load_stories("lora_tool")
        _, _, _, _, Aa, Na = load_stories("lora_assistant")
        Pt = Probes(At[:, L], emo, neutral_basis(Nt[:, L]))
        Pa = Probes(Aa[:, L], emo, neutral_basis(Na[:, L]))
        st = Pt.story_units(Ct[:, 0, L])
        sa = Pa.story_units(Ca[:, 0, L])
        dd = st - sa
        res["trained own vectors"] = {NAMES[e]: {"story_units": float(dd[:, e].mean()), "turns_up": int((dd[:, e] > 0).sum()),
                                                  "p": float(paired(list(st[:, e]), list(sa[:, e]))["p"])} for e in range(len(NAMES))}
        o = np.argsort(-dd.mean(0))
        print("\n=== trained tool vs trained control, pre-answer, each model scored with its own vectors (story units, turns up)")
        print("  up:   " + "; ".join(f"{NAMES[e]} {dd[:, e].mean():+.2f} ({(dd[:, e] > 0).sum()}/32)" for e in o[:10]))
        print("  down: " + "; ".join(f"{NAMES[e]} {dd[:, e].mean():+.2f} ({(dd[:, e] > 0).sum()}/32)" for e in o[::-1][:10]))
        cos = np.sum(Pt.unit * Pa.unit, axis=1)
        print(f"  cosine between the two models' vectors for the same emotion: median {np.median(cos):.2f}, min {cos.min():.2f}")

    # do the vectors and the J-lens word scan agree?
    jl = os.path.join(OUT, "emotions171_local.json")
    if os.path.exists(jl):
        from scipy.stats import spearmanr
        J = json.load(open(jl))
        print("\nagreement between vector shifts and J-lens word shifts across the 45 emotions (Spearman):")
        res["agreement"] = {}
        for key, name in [(("tool", "pre-answer"), "instructed tool"), (("tool_harsh", "pre-answer"), "instructed harsh"),
                          (("participant", "pre-answer"), "instructed participant"), (("trained", "pre-answer"), "trained tool")]:
            xs, ys = [], []
            for e, w in enumerate(NAMES):
                if w in J[name]["words"]:
                    xs.append(keep[key][e])
                    ys.append(np.log10(J[name]["words"][w]["factor"]))
            r, p = spearmanr(xs, ys)
            res["agreement"][name] = {"rho": float(r), "p": float(p), "n": len(xs)}
            print(f"  {name:24s} rho = {r:+.2f} (p = {p:.3f}, n = {len(xs)})")

    # stability across layers for the concepts that matter
    watch = ["indifferent", "docile", "resigned", "stuck", "hurt", "worthless", "distressed", "desperate", "resentful", "grateful", "happy"]
    print("\nstory-unit shift before the answer by layer (tool vs control | trained tool vs control):")
    res["by_layer"] = {}
    for l in range(max(L - 8, 2), min(L + 9, A.shape[1]), 4):
        Pl = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        a = (Pl.direction(np.stack([C[at[("tool", t)], 0, l] - C[at[("assistant", t)], 0, l] for t in tasks])).mean(0)) / Pl.norm
        b = (Pl.direction(Ct[:, 0, l] - Ca[:, 0, l]).mean(0)) / Pl.norm
        res["by_layer"][int(l)] = {w: [float(a[NAMES.index(w)]), float(b[NAMES.index(w)])] for w in watch}
        print(f"  L{l}: " + "  ".join(f"{w} {a[NAMES.index(w)]:+.2f}|{b[NAMES.index(w)]:+.2f}" for w in watch))
    with open(os.path.join(OUT, "evec.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote results/tool/evec.json")


if __name__ == "__main__":
    main()
