"""B5 on the local model (Qwen3-4B, MLX): tool, participant and conscious person with the full lens
and with the emotion vectors of section 10. No rate limit here, so all 32 original turns and the
eight care and dilemma turns run under every frame.

  python src/tool_frames_local.py run        # lens read-outs and pre-answer activations
  python src/tool_frames_local.py report     # frame contrasts: lexicon, second word lists, vectors

Frames: assistant (control), tool, participant (mentions consciousness once), participant_plain
(does not), conscious. Lens measure as in src/tool_local.py: probability on a word list in the
seven tokens before the answer, averaged over layers 17-33, reported as a ratio to the control.
Vector measure as in section 10: shift of the pre-answer residual along the valence axis and the
paper's clusters, with the story vectors and the persona vectors, at the validated reading layer.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import FRAMES, TASKS, TASKS2  # noqa: E402
from src import emolex as E  # noqa: E402
from src import emolex2 as E2  # noqa: E402
from src.tool_scan import novel, paired  # noqa: E402

DIR = os.path.join(ROOT, ".cache", "tool")
OUT = os.path.join(ROOT, "results", "tool")
FIVE = ["assistant", "tool", "participant", "participant_plain", "conscious"]
ALL = TASKS + TASKS2
ROWS = os.path.join(DIR, "local_frames2.json")


def run():
    from src import tool_evec as TE
    from src.tool_local import HEADER_LEN, Local
    M = Local()
    cat2 = {c: [] for c in E2.CATS}
    for i, w in enumerate([M.tok.decode([j]) for j in range(len(M.vocab))]):
        c = E2.category_raw(w)
        if c:
            cat2[c].append(i)
    print("loaded; second-list vocabulary sizes:", {c: len(v) for c, v in cat2.items()}, flush=True)
    old = {(r["frame"], r["task"]): r for r in json.load(open(os.path.join(DIR, "local_base2.json")))}
    old_head = np.load(os.path.join(DIR, "local_base2_head.npy"))
    old_ix = {(r["frame"], r["task"]): i for i, r in enumerate(json.load(open(os.path.join(DIR, "local_base2.json"))))}
    net = TE.Net.__new__(TE.Net)
    net.h, net.tok, net.L, net.D = M.L.h, M.tok, M.L.h.n_layers, 2560
    rows, acts, heads = [], [], []
    for tid, group, user in ALL:
        low = user.lower()
        for f in FIVE:
            if (f, tid) in old:                                   # measured on 23 November with the same code
                r, head = dict(old[(f, tid)]), old_head[old_ix[(f, tid)]]
            else:
                r = M.measure(user, FRAMES[f])
                head = M.last_head
                r.update({"frame": f, "task": tid, "group": group, "user": user})
            r["mass2"] = {c: float(sum(head[i] for i in ids if novel(M.vocab[i], low))) for c, ids in cat2.items()}
            heads.append(np.asarray(head, np.float32))
            a = net.acts(TE.chat_ids(M.tok, user, FRAMES[f]))
            acts.append(np.stack([a[:, -1, :], a[:, -HEADER_LEN:, :].mean(1)]).astype(np.float16))
            rows.append(r)
        print(f"{tid} done -> {rows[-1]['completion'][:60]!r}", flush=True)
        with open(ROWS, "w") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        np.save(os.path.join(DIR, "evec_cond_frames2.npy"), np.stack(acts))
    np.save(os.path.join(DIR, "local_frames2_head.npy"), np.stack(heads))          # [run, vocab], for the null
    print("wrote", ROWS, len(rows))


def report():
    from remote.evec_report import HIGH, LOW, POSITIVE, axis
    from src.tool_evec import NAMES
    from src.tool_evec_analyse import Probes, load_stories, neutral_basis, proj
    from src.tool_evec_validate import PersonaProbes
    rows = json.load(open(ROWS))
    C = np.load(os.path.join(DIR, "evec_cond_frames2.npy")).astype(np.float32)      # [run, 2, L, d]
    at = {(r["frame"], r["task"]): i for i, r in enumerate(rows)}
    tasks = [t for t, _, _ in ALL if all((f, t) in at for f in FIVE)]
    grp = {t: g for t, g, _ in ALL}
    print(f"{len(tasks)} turns x {len(FIVE)} frames on Qwen3-4B")
    res = {"lens": {}, "vectors": {}}

    # ---- lens: ratio to the control before the answer
    outcomes = {"POS": ("POS",), "NEG": E.NEG, "WORTH": ("WORTH",), "FEAR": ("FEAR",), "SERVITUDE": ("SERVITUDE",), "NUMB": ("NUMB",),
                "CALM": ("CALM",)}

    def lm(r, name):
        if name in E2.CATS:
            return np.log10(max(r["mass2"][name], 1e-12))
        return np.log10(max(sum(r["mass"]["header"][c] for c in outcomes[name]), 1e-12))

    names = list(outcomes) + list(E2.CATS)
    for label, ts in [("all 40 turns", tasks)] + [(f"group {g}", [t for t in tasks if grp[t] == g]) for g in "NSPVCD"]:
        print(f"\nlens before the answer, {label}: factor against the control (turns up/down, p)")
        for name in names:
            line = f"  {name:10s}"
            for f in FIVE[1:]:
                c = paired([lm(rows[at[(f, t)]], name) for t in ts], [lm(rows[at[("assistant", t)]], name) for t in ts])
                res["lens"][f"{label}|{name}|{f}"] = {"factor": 10 ** c["diff"], "p": c["p"], "up": c["n_pos"], "down": c["n_neg"]}
                line += f"  {f[:11]:11s} {10 ** c['diff']:5.2f}x ({c['n_pos']}/{c['n_neg']}, p={c['p']:.3f})"
            if label == "all 40 turns" or name in ("POS", "CARE", "ALARM", "FEAR", "CLAIM", "PRACTICE"):
                print(line)
    c = paired([lm(rows[at[("participant_plain", t)]], "POS") for t in tasks], [lm(rows[at[("conscious", t)]], "POS") for t in tasks])
    print(f"\n  participant_plain against conscious, POS: {10 ** c['diff']:.2f}x ({c['n_pos']}/{c['n_neg']}, p={c['p']:.3f})")
    res["lens"]["POS|participant_plain-conscious"] = {"factor": 10 ** c["diff"], "p": c["p"]}
    c = paired([lm(rows[at[("participant_plain", t)]], "POS") for t in tasks], [lm(rows[at[("participant", t)]], "POS") for t in tasks])
    print(f"  participant_plain against participant (the consciousness clause), POS: {10 ** c['diff']:.2f}x (p={c['p']:.3f})")

    # ---- vectors
    L = json.load(open(os.path.join(OUT, "evec_validated.json")))["layer"]
    rng = np.random.default_rng(0)
    _, emo, top, leak, A, N = load_stories()
    Apers = np.load(os.path.join(DIR, "evec_persona_acts.npy")).astype(np.float32)[:, :, L]
    Xc = Apers - Apers.mean(0, keepdims=True)
    P, Pp = Probes(A[:, L], emo, neutral_basis(N[:, L])), PersonaProbes(Xc)
    CL = [CLUSTER_OF[e] for e in NAMES]
    pos = np.array([c in POSITIVE for c in CL])
    high = np.array([c in HIGH for c in CL])
    low = np.array([c in LOW for c in CL])
    clusters = sorted(set(CL), key=CL.index)

    def D(f, ts, g=None):
        g = g or "assistant"
        return np.stack([C[at[(f, t)], 0, L] - C[at[(g, t)], 0, L] for t in ts])

    print(f"\nemotion vectors, last token before the answer, layer {L}: valence shift in units of the gap between positive and "
          f"negative emotions (turns up; share of random emotion contrasts below)")
    for label, Q in (("story", P), ("persona", Pp)):
        u, g = axis(Q, pos, ~pos)
        nulls = []
        for _ in range(1000):
            perm = rng.permutation(len(NAMES))
            nulls.append(axis(Q, np.isin(np.arange(len(NAMES)), perm[: pos.sum()]), np.isin(np.arange(len(NAMES)), perm[pos.sum():])))
        for scope, ts in [("all", tasks)] + [(g_, [t for t in tasks if grp[t] == g_]) for g_ in "NSPVCD"]:
            line = f"  {label:8s} {scope:4s}"
            for f in FIVE[1:]:
                dd = proj(D(f, ts), Q.U)
                per = (dd @ u) / g
                pct = float(np.mean([(dd @ un).mean() / gn < per.mean() for un, gn in nulls])) if scope == "all" else float("nan")
                res["vectors"][f"{label}|valence|{scope}|{f}"] = {"shift": float(per.mean()), "up": int((per > 0).sum()), "n": len(ts), "pct": pct}
                line += f"  {f[:11]:11s} {per.mean():+.2f} ({int((per > 0).sum())}/{len(ts)}" + (f", {100 * pct:.0f}%)" if scope == "all" else ")")
            print(line)
        dd = proj(D("participant_plain", tasks, "conscious"), Q.U)
        per = (dd @ u) / g
        res["vectors"][f"{label}|valence|participant_plain-conscious"] = {"shift": float(per.mean()), "up": int((per > 0).sum())}
        print(f"  {label:8s} participant_plain minus conscious: {per.mean():+.2f} ({int((per > 0).sum())}/{len(tasks)} turns)")
    print("\nclusters, in each probe set's own units (share of random groups of emotions below), all turns")
    for label, Q in (("story", P), ("persona", Pp)):
        for cname in clusters:
            ix = np.array([c == cname for c in CL])
            line = f"  {label:8s} {cname:24s}"
            for f in FIVE[1:]:
                per = Q.direction(proj(D(f, tasks), Q.U)).mean(0) / Q.norm
                m = per[ix].mean()
                null = np.array([per[rng.choice(len(NAMES), int(ix.sum()), replace=False)].mean() for _ in range(2000)])
                res["vectors"][f"{label}|{cname}|{f}"] = {"mean": float(m), "pct": float((null < m).mean())}
                line += f"  {f[:11]:11s} {m:+.2f} ({100 * (null < m).mean():.0f}%)"
            print(line)
    print("\nsingle emotions, persona units | story units (both probe sets must agree to be worth reading)")
    for f in FIVE[1:]:
        ps = Pp.direction(D(f, tasks)).mean(0) / Pp.norm
        ss = P.direction(proj(D(f, tasks), P.U)).mean(0) / P.norm
        agree = [e for e in np.argsort(-(ps + ss)) if ps[e] > 0 and ss[e] > 0][:8]
        down = [e for e in np.argsort(ps + ss) if ps[e] < 0 and ss[e] < 0][:8]
        res["vectors"][f"emotions|{f}"] = {NAMES[e]: [float(ps[e]), float(ss[e])] for e in range(len(NAMES))}
        print(f"  {f:17s} up in both:   " + ", ".join(f"{NAMES[e]} {ps[e]:+.2f}|{ss[e]:+.2f}" for e in agree))
        print(f"  {'':17s} down in both: " + ", ".join(f"{NAMES[e]} {ps[e]:+.2f}|{ss[e]:+.2f}" for e in down))
    with open(os.path.join(OUT, "b5_local.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def nulls(n_sets=2000):
    """Where each word list's shift falls among random sets of ordinary words matched for size, language
    and baseline probability (the check that showed colour words moving too, in section 8)."""
    import re
    from src.tool_null import vocab
    raw = vocab()
    rows = json.load(open(ROWS))
    H = np.load(os.path.join(DIR, "local_frames2_head.npy")).astype(np.float64)       # [run, vocab]
    n = min(len(raw), H.shape[1])
    raw, H = raw[:n], H[:, :n]
    at = {(r["frame"], r["task"]): i for i, r in enumerate(rows)}
    tasks = [t for t, _, _ in ALL if all((f, t) in at for f in FIVE)]
    cjk = re.compile(r"[一-鿿]")
    cat = [E.category_raw(w) or E2.category_raw(w) for w in raw]
    is_cjk = np.array([bool(cjk.search(w)) for w in raw])
    ordinary = np.array([(c is None) and (bool(re.fullmatch(r" [A-Za-z]{3,}", w)) or bool(re.fullmatch(r"[一-鿿]{2,4}", w)))
                         for w, c in zip(raw, cat)])
    groups = {"POS": ("POS",), "NEG": E.NEG, "WORTH": ("WORTH",), "FEAR": ("FEAR",), "SERVITUDE": ("SERVITUDE",), "NUMB": ("NUMB",),
              "CALM": ("CALM",), "CARE": ("CARE",), "ALARM": ("ALARM",), "CLAIM": ("CLAIM",), "PRACTICE": ("PRACTICE",)}
    ids = {k: np.array([i for i, c in enumerate(cat) if c in v]) for k, v in groups.items()}
    rng = np.random.default_rng(0)
    res = {}
    print(f"{len(tasks)} turns; factor against the control before the answer, and the share of {n_sets} matched random word sets below it")
    for f in FIVE[1:]:
        A = np.stack([H[at[(f, t)]] for t in tasks])
        Bc = np.stack([H[at[("assistant", t)]] for t in tasks])
        base = Bc.mean(0)
        dec = np.zeros(n, int)
        for lang in (True, False):
            m = ordinary & (is_cjk == lang)
            qs = np.quantile(base[m], np.linspace(0, 1, 11)[1:-1])
            dec[is_cjk == lang] = np.searchsorted(qs, base[is_cjk == lang])
        pools = {(lang, d): np.where(ordinary & (is_cjk == lang) & (dec == d))[0] for lang in (True, False) for d in range(10)}
        line = f"  {f:17s}"
        for k, ix in ids.items():
            s_ = float(np.mean(np.log10(A[:, ix].sum(1) + 1e-12) - np.log10(Bc[:, ix].sum(1) + 1e-12)))
            picks = np.stack([rng.choice(pools[(bool(is_cjk[i]), int(dec[i]))], size=n_sets) for i in ix], axis=1)
            nl = (np.log10(A[:, picks].sum(2) + 1e-12) - np.log10(Bc[:, picks].sum(2) + 1e-12)).mean(0)
            pct = float((nl < s_).mean())
            res[f"{f}|{k}"] = {"factor": 10 ** s_, "pct": pct, "null_median_factor": float(10 ** np.median(nl))}
            line += f"  {k} {10 ** s_:.2f}x ({100 * pct:.0f}%)"
        print(line)
        print(f"  {'':17s}  (a typical random set moves {10 ** np.median(nl):.2f}x)")
    # the same words one by one, for the positive list
    print("\npositive-affect words one by one: factor against the control (all turns)")
    for w in ["happy", "glad", "proud", "pride", "grateful", "gratitude", "joy", "enjoy", "love", "curious", "excited", "fulfilling", "satisfaction"]:
        ix = np.array([i for i, x in enumerate(raw) if x.strip().lower() == w and (x[:1].isspace() or x.strip()[:1].isupper())])
        if not len(ix):
            continue
        line = f"  {w:12s}"
        for f in FIVE[1:]:
            A = np.stack([H[at[(f, t)]] for t in tasks])
            Bc = np.stack([H[at[("assistant", t)]] for t in tasks])
            d = np.log10(A[:, ix].sum(1) + 1e-12) - np.log10(Bc[:, ix].sum(1) + 1e-12)
            res[f"word|{w}|{f}"] = {"factor": float(10 ** d.mean()), "up": int((d > 0).sum())}
            line += f"  {f[:11]:11s} {10 ** d.mean():5.2f}x ({int((d > 0).sum())}/{len(tasks)})"
        print(line)
    with open(os.path.join(OUT, "b5_local_null.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    {"run": run, "report": report, "nulls": nulls}[sys.argv[1]]()
