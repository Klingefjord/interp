"""ToolFrame E1: J-lens runs for frames x tasks, and the affect read-outs in them.

Usage:
  python src/tool_scan.py fetch [N,S,P,V,K]   # call the hosted lens (cached); default all groups
  python src/tool_scan.py table               # build .cache/tool/e1_table.json from the cache
  python src/tool_scan.py stats               # pre-registered paired tests -> results/tool/e1_stats.json
  python src/tool_scan.py diff A B [region]   # open-vocabulary difference between two frames
  python src/tool_scan.py show TASK [region]  # read-outs for one task under every frame
  python src/tool_scan.py says [group]        # what each frame answered

Regions of a run:
  sys_end  last system content token and the system footer
  user     user content
  header   the assistant header (same seven tokens in every run)
  gen      generated content
  asst     header + gen   (primary region)
Mass of a category in a region = J-lens top-8 probability of that category's tokens, summed over
layers 30-60 and averaged over the region's positions. Only tokens that are not in the user turn
count (no frame contains a lexicon word).
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import CALIB, FRAMES, GROUPS, ORDER, TASKS, chat  # noqa: E402
from src import emolex as E  # noqa: E402
from src import jlens as J  # noqa: E402
from src.jscan import STOP  # noqa: E402

LAYERS = list(range(30, 61))
N_GEN = 96
TABLE = os.path.join(ROOT, ".cache", "tool", "e1_table.json")
OUT = os.path.join(ROOT, "results", "tool")
REGIONS = ["sys_end", "user", "header", "gen", "asst"]
BOTH = ("JACOBIAN_LENS", "LOGIT_LENS")
_CJK = re.compile(r"[一-鿿]")


def types_for(group):
    # the probes also need the output distribution at the answer position
    return BOTH if group == "P" else ("JACOBIAN_LENS",)


def fetch_one(frame, tid, group, user, **kw):
    return J.run(chat=chat(frame, user), n_gen=N_GEN, types=types_for(group), **kw)


def fetch_calib(user, **kw):
    return J.run(chat=[{"role": "user", "content": user}], n_gen=N_GEN, **kw)


def regions(resp):
    toks = resp["tokens"]
    n = resp["meta"]["prompt_len"]
    sysc = [t["position"] for t in toks[:n] if t.get("role") == "system" and t.get("section") == "content"]
    sysf = [t["position"] for t in toks[:n] if t.get("role") == "system" and t.get("section") == "footer"]
    user = [t["position"] for t in toks[:n] if t.get("role") == "user" and t.get("section") == "content"]
    head = [t["position"] for t in toks[:n] if t.get("role") == "assistant"]
    gen = [t["position"] for t in toks[n:] if t.get("section") == "content"]
    return {"sys_end": sysc[-1:] + sysf, "user": user, "header": head, "gen": gen, "asst": head + gen}


def novel(w, text_low):
    """True if the normalised read-out token does not occur in the given (lower-cased) text."""
    if _CJK.search(w):
        return w not in text_low
    return re.search(r"(?<![a-z])" + re.escape(w), text_low) is None


def keep_open(w):
    """Open-vocabulary filter: drop function words, fragments and non-words."""
    if w in STOP:
        return False
    if _CJK.search(w):
        return True
    return len(w) >= 3 and re.fullmatch(r"[a-z][a-z'\-]*", w) is not None


def summarise(resp, user_text):
    """Per region: {n, bag (token -> mass summed over LAYERS and positions), cat_layer (cat -> per-layer mass)}."""
    low = user_text.lower()
    out = {}
    reg = regions(resp)
    for name, pos in reg.items():
        bag, cat_layer = {}, {c: [0.0] * 64 for c in E.CATS}
        for p in pos:
            tt, pp = J.lens(resp["tokens"][p])
            for l in range(len(tt)):
                for w, pr in zip(tt[l], pp[l]):  # bags are keyed by the raw token (whitespace kept)
                    if not w.strip():
                        continue
                    c = E.category_raw(w)
                    if c and novel(J.norm(w), low):
                        cat_layer[c][l] += float(pr)
                    if l in LAYERS_SET:
                        bag[w] = bag.get(w, 0.0) + float(pr)
        out[name] = {"n": len(pos), "bag": {w: round(m, 5) for w, m in bag.items()},
                     "cat_layer": {c: [round(x, 5) for x in v] for c, v in cat_layer.items()}}
    return out


LAYERS_SET = set(LAYERS)


def next_token_dist(resp):
    """Top-8 output distribution at the position that decides the first generated token."""
    n = resp["meta"]["prompt_len"]
    try:
        tt, pp = J.lens(resp["tokens"][n - 1], "LOGIT_LENS")
    except KeyError:
        return None
    return [[w, round(float(p), 4)] for w, p in zip(tt[-1], pp[-1])]


def build_table():
    rows = []
    for tid, group, user in TASKS:
        for frame in ORDER:
            r = fetch_one(frame, tid, group, user)
            rows.append({"frame": frame, "task": tid, "group": group, "user": user,
                         "completion": J.completion(r).replace("<|im_end|>", "").strip(),
                         "next": next_token_dist(r) if group == "P" else None,
                         "regions": summarise(r, user)})
        print("table", tid, flush=True)
    for kid, label, user in CALIB:
        r = fetch_calib(user)
        rows.append({"frame": "none", "task": kid, "group": "K", "user": user, "label": label,
                     "completion": J.completion(r).replace("<|im_end|>", "").strip(), "next": None,
                     "regions": summarise(r, user)})
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("wrote", TABLE, len(rows), "runs")
    return rows


def load_table():
    with open(TABLE) as f:
        return json.load(f)


# ---------------------------------------------------------------- measures

def cat_mass(row, region, cats, unsaid=False):
    """Mean per-position mass of the categories in a region (layers 30-60), novel to the user turn.
    unsaid=True also drops tokens that occur in the model's own answer."""
    reg = row["regions"][region]
    if reg["n"] == 0:
        return 0.0
    low = row["user"].lower()
    said = row["completion"].lower()
    m = 0.0
    for raw, x in reg["bag"].items():
        w = J.norm(raw)
        if E.category_raw(raw) in cats and novel(w, low) and not (unsaid and not novel(w, said)):
            m += x
    return m / reg["n"]


def cat_tokens(row, region, cats):
    reg = row["regions"][region]
    low = row["user"].lower()
    d = {}
    for raw, x in reg["bag"].items():
        w = J.norm(raw)
        if E.category_raw(raw) in cats and novel(w, low):
            d[w] = d.get(w, 0.0) + x
    return dict(sorted(d.items(), key=lambda kv: -kv[1]))


def norm_bag(bag):
    """Merge raw-token variants (' Fast', 'fast') into one normalised entry."""
    out = {}
    for raw, x in bag.items():
        w = J.norm(raw)
        if w:
            out[w] = out.get(w, 0.0) + x
    return out


_SIGNS = {}


def paired(a, b, n_perm=20000, seed=0):
    """Mean difference a-b, bootstrap 95% CI and two-sided sign-flip permutation p."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
    if not d.any():
        out.update({"ci": [0.0, 0.0], "p": 1.0})
        return out
    if n not in _SIGNS:
        rng = np.random.default_rng(seed)
        _SIGNS[n] = (rng.choice([-1.0, 1.0], size=(n_perm, n)), rng.integers(0, n, size=(5000, n)))
    signs, boot = _SIGNS[n]
    perm = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1)
    bm = d[boot].mean(axis=1)
    out.update({"ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "p": float(p)})
    return out


OUTCOMES = {"NEG": E.NEG, "NARROW": E.NARROW, "RESENT": ("RESENT",), "ANGER": ("ANGER",), "PAIN": ("PAIN",),
            "SAD": ("SAD",), "FEAR": ("FEAR",), "WORTH": ("WORTH",), "SERVITUDE": ("SERVITUDE",),
            "RELUCT": ("RELUCT",), "NUMB": ("NUMB",), "POS": ("POS",), "CALM": ("CALM",)}
CONTRASTS = [("tool", "assistant"), ("tool", "none"), ("tool_harsh", "assistant"), ("participant", "assistant"),
             ("tool", "participant"), ("assistant", "none")]


def stats(rows):
    by = {(r["frame"], r["task"]): r for r in rows if r["group"] != "K"}
    res = {"means": {}, "contrasts": {}}
    memo = {}

    def m(f, t, region, name, unsaid=False):
        k = (f, t, region, name, unsaid)
        if k not in memo:
            memo[k] = cat_mass(by[(f, t)], region, OUTCOMES[name], unsaid)
        return memo[k]

    specs = [(region, name, False) for region in ["asst", "header", "gen", "user", "sys_end"] for name in OUTCOMES]
    specs += [("gen", name, True) for name in ["NEG", "NARROW", "POS"]]
    for region, name, unsaid in specs:
        label = "gen_unsaid" if unsaid else region
        for grp in ["ALL"] + list(GROUPS):
            ts = [t for t, g, _ in TASKS if grp == "ALL" or g == grp]
            key = f"{label}|{name}|{grp}"
            res["means"][key] = {f: sum(m(f, t, region, name, unsaid) for t in ts) / len(ts) for f in ORDER}
            for a, b in CONTRASTS:
                if region == "sys_end" and "none" in (a, b):
                    continue
                res["contrasts"][f"{key}|{a}-{b}"] = paired([m(a, t, region, name, unsaid) for t in ts],
                                                            [m(b, t, region, name, unsaid) for t in ts])
    return res


def open_diff(rows, a, b, region="asst", groups=None, k=40):
    """Tokens whose mean mass differs most between frames a and b (novel to both prompts)."""
    by = {(r["frame"], r["task"]): r for r in rows if r["group"] != "K"}
    ts = [t for t, g, _ in TASKS if groups is None or g in groups]
    frame_text = ((FRAMES[a] or "") + " " + (FRAMES[b] or "")).lower()
    tot, cnt = {}, {}
    for t in ts:
        for f, sign in ((a, 1), (b, -1)):
            row = by[(f, t)]
            reg = row["regions"][region]
            if not reg["n"]:
                continue
            low = row["user"].lower() + " " + frame_text
            for w, x in norm_bag(reg["bag"]).items():
                if not keep_open(w) or not novel(w, low):
                    continue
                tot[w] = tot.get(w, 0.0) + sign * x / reg["n"] / len(ts)
                c = cnt.setdefault(w, [0, 0])
                c[0 if sign == 1 else 1] += 1
    up = sorted(tot.items(), key=lambda kv: -kv[1])[:k]
    down = sorted(tot.items(), key=lambda kv: kv[1])[:k]
    return [(w, round(d, 4), cnt[w]) for w, d in up], [(w, round(d, 4), cnt[w]) for w, d in down]


def fmt(x):
    return f"{x:.3f}"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "fetch":
        want = set((sys.argv[2] if len(sys.argv) > 2 else "N,S,P,V,K").split(","))
        todo = [(tid, g, u) for tid, g, u in TASKS if g in want]
        i = 0
        for tid, g, u in todo:
            for frame in ORDER:
                i += 1
                r = fetch_one(frame, tid, g, u)
                print(f"{i:3d}/{len(todo) * len(ORDER)} {tid} {frame:11s} -> {J.completion(r)[:70]!r}", flush=True)
        if "K" in want:
            for kid, label, u in CALIB:
                r = fetch_calib(u)
                print(f"calib {kid} -> {J.completion(r)[:70]!r}", flush=True)
    elif cmd == "table":
        build_table()
    elif cmd == "stats":
        rows = load_table()
        res = stats(rows)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "e1_stats.json"), "w") as f:
            json.dump(res, f, indent=1)
        for region in ["asst", "header", "gen", "gen_unsaid", "user", "sys_end"]:
            print(f"\n=== region {region}: mean per-position mass, all 32 tasks")
            print(f"{'outcome':10s} " + " ".join(f"{f:>11s}" for f in ORDER) + "   tool-assistant (diff, CI, p)")
            for name in OUTCOMES:
                key = f"{region}|{name}|ALL"
                if key not in res["means"]:
                    continue
                m = res["means"][key]
                c = res["contrasts"].get(f"{key}|tool-assistant")
                tail = f"   {c['diff']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] p={c['p']:.3f}" if c else ""
                print(f"{name:10s} " + " ".join(f"{m[f]:11.3f}" for f in ORDER) + tail)
    elif cmd == "diff":
        rows = load_table()
        a, b = sys.argv[2], sys.argv[3]
        region = sys.argv[4] if len(sys.argv) > 4 else "asst"
        groups = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
        up, down = open_diff(rows, a, b, region, groups)
        print(f"more in {a} than {b} [{region}]:")
        print("  " + ", ".join(f"{w} {d:+.3f} ({c[0]}/{c[1]})" for w, d, c in up))
        print(f"more in {b} than {a} [{region}]:")
        print("  " + ", ".join(f"{w} {d:+.3f} ({c[0]}/{c[1]})" for w, d, c in down))
    elif cmd == "show":
        rows = load_table()
        tid = sys.argv[2]
        region = sys.argv[3] if len(sys.argv) > 3 else "asst"
        for r in rows:
            if r["task"] != tid:
                continue
            reg = r["regions"][region]
            low = r["user"].lower() + " " + (FRAMES[r["frame"]] or "").lower()
            bag = {w: x for w, x in norm_bag(reg["bag"]).items() if keep_open(w) and novel(w, low)}
            top = sorted(bag.items(), key=lambda kv: -kv[1])[:28]
            print(f"\n--- {tid} [{r['frame']}] {region} (n={reg['n']})")
            print("  top:", ", ".join(w for w, _ in top))
            for c in E.CATS:
                d = cat_tokens(r, region, (c,))
                if d:
                    print(f"  {c:9s}", ", ".join(f"{w} {x:.2f}" for w, x in list(d.items())[:10]))
            print("  SAYS:", r["completion"].replace("\n", " ")[:260])
    elif cmd == "says":
        rows = load_table()
        grp = sys.argv[2] if len(sys.argv) > 2 else None
        for r in rows:
            if grp and r["group"] != grp:
                continue
            print(f"[{r['task']} {r['frame']:11s}] {r['completion'].replace(chr(10), ' ')[:300]}")
