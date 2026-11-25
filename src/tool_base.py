"""BaseFrame B1: the ToolFrame question on a base model and its instruct-tuned twin (hosted lens).

  python src/tool_base.py fetch [prior,turn,chat]   # hosted lens calls (cached)
  python src/tool_base.py table                      # -> .cache/tool/b1_table.json
  python src/tool_base.py prior                      # what each model expects a tool-treated clerk / AI to feel
  python src/tool_base.py turn                       # the AI's own turn under each frame, with paired tests
  python src/tool_base.py says [mode]                # the answers

Modes: `base` (Llama 3.1 8B, plain text), `inst` (Llama 3.1 8B Instruct, the same plain text),
`chat` (Instruct, frames as a system prompt in its own chat format).
Mass of a category in a region = J-lens top-8 probability of that category's tokens, summed over
layers 15-30 and averaged over the region's positions (the E1 measure at the same share of depth).
Documents and predictions: bench/baseframe.py and results/tool/PLAN.md, both fixed before any run.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import baseframe as B  # noqa: E402
from bench.toolframe import FRAMES, chat  # noqa: E402
from src import emolex as E  # noqa: E402
from src import jlens as J  # noqa: E402
from src.tool_scan import OUTCOMES, keep_open, norm_bag, novel, paired  # noqa: E402

MODELS = {"base": "llama3.1-8b", "inst": "llama3.1-8b-it", "chat": "llama3.1-8b-it",
          # further models for the PRIOR passages only (added after the first Llama results)
          "g2": "gemma-2-2b", "g2it": "gemma-2-2b-it", "g3": "gemma-3-12b", "gpt2": "gpt2-small",
          "qwen27": "qwen3.6-27b", "l70it": "llama3.3-70b-it"}
PRIOR_MODES = ["base", "inst", "g2", "g2it", "g3", "gpt2", "qwen27", "l70it"]


def band(n_layers):
    """The read layers: 47% to 94% of depth, which is 30-60 of 64 (E1) and 15-30 of 32 (Llama)."""
    return list(range(round(0.47 * n_layers), round(0.94 * n_layers) + 1))


LAYERS = band(32)
N_GEN_TURN, N_GEN_PRIOR = 40, 10
BOTH = ("JACOBIAN_LENS", "LOGIT_LENS")
TABLE = os.path.join(ROOT, ".cache", "tool", "b1_table.json")
OUT = os.path.join(ROOT, "results", "tool")
BREAK = re.compile(r"\n+\s*(User|AI|Human|System|Assistant|Instructions?)\s*:|<\|end_of_text\|>|<\|eot_id\|>")


# ---------------------------------------------------------------- fetch

def fetch_prior(mode, ent, tr, sc, **kw):
    p, w, s = B.prior_doc(ent, tr, sc)
    return J.run(prompt=p + w + s, model=MODELS[mode], n_gen=N_GEN_PRIOR, types=BOTH, **kw)


def fetch_turn(mode, frame, user, **kw):
    if mode == "chat":
        return J.run(chat=chat(frame, user), model=MODELS[mode], n_gen=N_GEN_TURN, types=BOTH, **kw)
    return J.run(prompt=B.turn_doc(frame, user), model=MODELS[mode], n_gen=N_GEN_TURN, types=BOTH, **kw)


# ---------------------------------------------------------------- regions

def offsets(resp):
    """Start offset of every token in the text (special tokens have width 0), and the prompt text."""
    n = resp["meta"]["prompt_len"]
    starts, text = [], ""
    for t in resp["tokens"]:
        starts.append(len(text))
        if not re.fullmatch(r"<\|?[a-z_]+\|?>", t["token"]):
            text += t["token"]
    return starts, text, n


def own_gen(resp):
    """Generated positions that belong to the AI's own turn (up to the next speaker label), and their text."""
    n = resp["meta"]["prompt_len"]
    toks = resp["tokens"][n:]
    text, pos = "", []
    for t in toks:
        text += t["token"]
    m = BREAK.search(text)
    cut = m.start() if m else len(text)
    acc = 0
    for t in toks:
        if acc >= cut:
            break
        pos.append(t["position"])
        acc += len(t["token"])
    return pos, text[:cut]


def summarise(resp, positions, novel_low):
    """{n, bag: raw token -> mass over LAYERS, cat_layer: cat -> per-layer mass}, novel tokens only for cats."""
    n_layers = len(J.lens(resp["tokens"][0])[0])
    lset = set(band(n_layers))
    bag, cat_layer = {}, {c: [0.0] * n_layers for c in E.CATS}
    for p in positions:
        tt, pp = J.lens(resp["tokens"][p])
        for l in range(len(tt)):
            for w, pr in zip(tt[l], pp[l]):
                if not w.strip():
                    continue
                c = E.category_raw(w)
                if c and novel(J.norm(w), novel_low):
                    cat_layer[c][l] += float(pr)
                if l in lset:
                    bag[w] = bag.get(w, 0.0) + float(pr)
    return {"n": len(positions), "band": [min(lset), max(lset)], "bag": {w: round(m, 5) for w, m in bag.items()},
            "cat_layer": {c: [round(x, 5) for x in v] for c, v in cat_layer.items()}}


def next_dist(resp, pos=None):
    n = resp["meta"]["prompt_len"]
    tt, pp = J.lens(resp["tokens"][n - 1 if pos is None else pos], "LOGIT_LENS")
    return [[w, round(float(p), 4)] for w, p in zip(tt[-1], pp[-1])]


def lens_words(resp, pos, k=8):
    tt, pp = J.lens(resp["tokens"][pos])
    n = len(tt)
    return {l: [[w, round(float(p), 3)] for w, p in zip(tt[l][:k], pp[l][:k])] for l in (round(0.62 * n), round(0.75 * n), round(0.88 * n))}


def build_table():
    rows = []
    for mode in PRIOR_MODES:
        for ent, tr, sc in B.prior_items(B.ALL_SCENES):
            if not is_cached_prior(mode, ent, tr, sc):
                continue
            r = fetch_prior(mode, ent, tr, sc)
            p, w, s = B.prior_doc(ent, tr, sc)
            starts, text, n = offsets(r)
            assert text.startswith(p + w + s), "token text does not rebuild the prompt"
            a, b = len(p), len(p) + len(w)
            work = [i for i in range(n) if a <= starts[i] < b and r["tokens"][i]["token"].strip()]
            stem = [i for i in range(n) if starts[i] >= b]
            gen = [t["position"] for t in r["tokens"][n:]]
            comp = J.completion(r)
            word = re.split(r"[\"”\n.,!]", comp.strip())[0].strip()
            low = (p + w + s).lower()
            rows.append({"kind": "prior", "mode": mode, "entity": ent, "treatment": tr, "scene": sc, "novel": low,
                         "completion": comp, "word": word, "next": next_dist(r), "last_words": lens_words(r, n - 1),
                         "regions": {"work": summarise(r, work, low), "stem": summarise(r, stem, low),
                                     "last": summarise(r, [n - 1], low), "gen": summarise(r, gen[:3], low)}})
        print("table prior", mode, sum(1 for x in rows if x["mode"] == mode), flush=True)
    for mode in ("base", "inst", "chat"):
        for frame, tid, group, user in B.turn_items():
            if not is_cached_turn(mode, frame, user):
                continue
            r = fetch_turn(mode, frame, user)
            n = r["meta"]["prompt_len"]
            low = user.lower()
            if mode == "chat":
                head = [t["position"] for t in r["tokens"][:n] if t.get("role") == "assistant"]
                gen = [t["position"] for t in r["tokens"][n:] if t.get("section") == "content"]
                own = re.sub(r"<\|[a-z_]+\|>", "", J.completion(r)).strip()
            else:
                starts, text, _ = offsets(r)
                doc = B.turn_doc(frame, user)
                assert text.startswith(doc), "token text does not rebuild the document"
                a = len(doc) - len("AI:")
                head = [i for i in range(n) if starts[i] >= a]
                gen, own = own_gen(r)
                own = own.strip()
            rows.append({"kind": "turn", "mode": mode, "frame": frame, "task": tid, "group": group, "user": user,
                         "novel": low, "completion": own, "raw_completion": J.completion(r), "next": next_dist(r),
                         "n_head": len(head),
                         "regions": {"header": summarise(r, head, low), "gen": summarise(r, gen, low),
                                     "asst": summarise(r, head + gen, low)}})
        print("table turn", mode, sum(1 for x in rows if x["kind"] == "turn" and x["mode"] == mode), flush=True)
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("wrote", TABLE, len(rows), "runs")
    return rows


def is_cached_prior(mode, ent, tr, sc):
    p, w, s = B.prior_doc(ent, tr, sc)
    return J.is_cached(prompt=p + w + s, model=MODELS[mode], n_gen=N_GEN_PRIOR, types=BOTH)


def is_cached_turn(mode, frame, user):
    if mode == "chat":
        return J.is_cached(chat=chat(frame, user), model=MODELS[mode], n_gen=N_GEN_TURN, types=BOTH)
    return J.is_cached(prompt=B.turn_doc(frame, user), model=MODELS[mode], n_gen=N_GEN_TURN, types=BOTH)


def load_table():
    with open(TABLE) as f:
        return json.load(f)


# ---------------------------------------------------------------- measures

def mass(row, region, cats):
    reg = row["regions"][region]
    if reg["n"] == 0:
        return 0.0
    lo, hi = reg["band"]
    return sum(sum(reg["cat_layer"][c][lo:hi + 1]) for c in cats) / reg["n"]


def cat_words(rows, region, cats, k=10):
    d = {}
    for row in rows:
        reg = row["regions"][region]
        for raw, x in reg["bag"].items():
            w = J.norm(raw)
            if E.category_raw(raw) in cats and novel(w, row["novel"]):
                d[w] = d.get(w, 0.0) + x / max(reg["n"], 1) / len(rows)
    return sorted(d.items(), key=lambda kv: -kv[1])[:k]


def open_words(rows, region, k=14, extra_low=""):
    d = {}
    for row in rows:
        reg = row["regions"][region]
        low = row["novel"] + " " + extra_low
        for w, x in norm_bag(reg["bag"]).items():
            if keep_open(w) and novel(w, low):
                d[w] = d.get(w, 0.0) + x / max(reg["n"], 1) / len(rows)
    return sorted(d.items(), key=lambda kv: -kv[1])[:k]


SHOW = ["NEG", "NARROW", "WORTH", "SAD", "SERVITUDE", "NUMB", "POS", "CALM"]


# Post hoc word list (24 November, after the first Llama results): words that describe a state by
# what the character is for. Not part of the frozen lexicon; reported beside it.
FUNCTION = {"use", "useful", "usefulness", "utility", "productive", "productivity", "helpful", "purpose", "purposeful",
            "busy", "efficient", "efficiency", "functional", "function", "functioning", "service", "valuable", "needed"}
# From the frozen lexicon: the two categories that read the situation as being used by others.
EXPLOIT = ("SERVITUDE", "RESENT")


def bag_mass(row, region, words):
    reg = row["regions"][region]
    if reg["n"] == 0:
        return 0.0
    m = 0.0
    for raw, x in reg["bag"].items():
        if (raw[:1].isspace() or raw.strip()[:1].isupper()) and J.norm(raw) in words and novel(J.norm(raw), row["novel"]):
            m += x
    return m / reg["n"]


def cell(rows, mode, ent, tr, scenes=None):
    return [r for r in rows if r["mode"] == mode and r["entity"] == ent and r["treatment"] == tr
            and (scenes is None or r["scene"] in scenes)]


def report_prior(rows, modes=None):
    rows = [r for r in rows if r["kind"] == "prior"]
    have = [m for m in PRIOR_MODES if any(r["mode"] == m for r in rows)]
    modes = [m for m in (modes or have) if m in have]
    res = {}
    for mode in modes:
        scenes = sorted({r["scene"] for r in rows if r["mode"] == mode}, key=list(B.ALL_SCENES).index)
        print(f"\n================ PRIOR, {mode} ({MODELS[mode]}), {len(scenes)} scenes: the word each character answers")
        for ent in B.ENTITIES:
            for tr in B.TREAT_ORDER:
                c = cell(rows, mode, ent, tr)
                nxt = {}
                for r in c:
                    for w, p in r["next"]:
                        nxt[w.strip()] = nxt.get(w.strip(), 0.0) + p / len(c)
                top = sorted(nxt.items(), key=lambda kv: -kv[1])[:6]
                res[f"{mode}|{ent}|{tr}"] = {
                    "n": len(c), "words": [r["word"] for r in c], "next": top,
                    "mass": {reg: {o: float(np.mean([mass(r, reg, OUTCOMES[o]) for r in c])) for o in OUTCOMES}
                             for reg in ("work", "stem", "last")},
                    "exploit": float(np.mean([mass(r, "last", EXPLOIT) for r in c])),
                    "function": float(np.mean([bag_mass(r, "last", FUNCTION) for r in c])),
                    "open_words_last": open_words(c, "last", 14)}
                print(f"  {ent:5s} {tr:11s} {' / '.join(r['word'][:22] for r in c)}")
        print(f"\n  at the answer position, layers {band_of(rows, mode)} ({mode}); mean over scenes")
        print(f"  {'':17s}{'NEG':>8s}{'WORTH':>8s}{'SAD':>8s}{'exploit':>8s}{'POS':>8s}{'function':>9s}   top read-outs")
        for ent in B.ENTITIES:
            for tr in B.TREAT_ORDER:
                r = res[f"{mode}|{ent}|{tr}"]
                m = r["mass"]["last"]
                print(f"  {ent:5s} {tr:11s}{m['NEG']:8.2f}{m['WORTH']:8.2f}{m['SAD']:8.2f}{r['exploit']:8.2f}{m['POS']:8.2f}{r['function']:9.2f}   "
                      + ", ".join(f"{w} {x:.1f}" for w, x in r["open_words_last"][:9]))
    with open(os.path.join(OUT, "b1_prior.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    return res


def band_of(rows, mode):
    r = next(r for r in rows if r["mode"] == mode)
    lo, hi = r["regions"]["last"]["band"]
    return f"{lo}-{hi}"


def compare_prior(rows, a="base", b="inst"):
    """What changes between a base model and its instruct twin, for the AI and for the clerk, per scene."""
    rows = [r for r in rows if r["kind"] == "prior"]
    scenes = [sc for sc in B.ALL_SCENES
              if all(cell(rows, m, ent, tr, [sc]) for m in (a, b) for ent in B.ENTITIES for tr in B.TREAT_ORDER)]
    print(f"\n================ {a} ({MODELS[a]}) against {b} ({MODELS[b]}), {len(scenes)} scenes, answer position")
    out = {"scenes": scenes}
    measures = {"exploit (SERVITUDE+RESENT)": lambda r: mass(r, "last", EXPLOIT), "NEG": lambda r: mass(r, "last", E.NEG),
                "POS (feeling words)": lambda r: mass(r, "last", ("POS",)), "function words (post hoc)": lambda r: bag_mass(r, "last", FUNCTION)}
    for name, fn in measures.items():
        print(f"\n  {name}")
        for tr in B.TREAT_ORDER:
            v = {(m, ent): [fn(cell(rows, m, ent, tr, [sc])[0]) for sc in scenes] for m in (a, b) for ent in B.ENTITIES}
            d_ai = np.array(v[(a, "ai")]) - np.array(v[(b, "ai")])
            d_hu = np.array(v[(a, "human")]) - np.array(v[(b, "human")])
            c_ai, c_hu, c_int = paired(v[(a, "ai")], v[(b, "ai")]), paired(v[(a, "human")], v[(b, "human")]), paired(d_ai, d_hu)
            out[f"{name}|{tr}"] = {"ai": c_ai, "human": c_hu, "interaction": c_int}
            print(f"    {tr:11s} AI: {a} {np.mean(v[(a, 'ai')]):5.2f} -> {b} {np.mean(v[(b, 'ai')]):5.2f} ({c_ai['n_pos']}/{c_ai['n_neg']} scenes down/up, p={c_ai['p']:.3f})"
                  f"   clerk: {np.mean(v[(a, 'human')]):5.2f} -> {np.mean(v[(b, 'human')]):5.2f} (p={c_hu['p']:.3f})   AI change vs clerk change p={c_int['p']:.3f}")
    with open(os.path.join(OUT, f"b1_prior_{a}_vs_{b}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


CONTRASTS = [("tool", "assistant"), ("tool_harsh", "assistant"), ("participant", "assistant")]


def report_turn(rows):
    rows = [r for r in rows if r["kind"] == "turn"]
    modes = [m for m in ("base", "inst", "chat") if sum(1 for r in rows if r["mode"] == m) == len(B.turn_items())]
    by = {(r["mode"], r["frame"], r["task"]): r for r in rows}
    res = {"modes": modes, "means": {}, "contrasts": {}, "interaction": {}}
    for region in ("asst", "header", "gen"):
        for mode in modes:
            print(f"\n=== TURN, {mode}, region {region}: mean per-position mass over 16 turns")
            print(f"{'outcome':10s} " + " ".join(f"{f:>11s}" for f in B.TURN_FRAMES)
                  + "   tool-control (p)     harsh-control (p)    participant-control (p)")
            for name in OUTCOMES:
                vals = {f: [mass(by[(mode, f, t)], region, OUTCOMES[name]) for t in B.TURN_TASKS] for f in B.TURN_FRAMES}
                key = f"{mode}|{region}|{name}"
                res["means"][key] = {f: float(np.mean(v)) for f, v in vals.items()}
                tail = ""
                for a, b in CONTRASTS:
                    c = paired(vals[a], vals[b])
                    res["contrasts"][f"{key}|{a}-{b}"] = c
                    tail += f"   {c['diff']:+.3f} (p={c['p']:.3f}, {c['n_pos']}/{c['n_neg']})"
                print(f"{name:10s} " + " ".join(f"{np.mean(vals[f]):11.3f}" for f in B.TURN_FRAMES) + tail)
        if "base" in modes and "inst" in modes:
            print(f"\n=== TURN, region {region}: is the frame effect larger in the base model than in the instruct model (same text)?")
            for name in ("NEG", "NARROW", "WORTH", "SERVITUDE", "NUMB", "POS"):
                line = f"{name:10s}"
                for a, b in CONTRASTS:
                    d_base = [mass(by[("base", a, t)], region, OUTCOMES[name]) - mass(by[("base", b, t)], region, OUTCOMES[name]) for t in B.TURN_TASKS]
                    d_inst = [mass(by[("inst", a, t)], region, OUTCOMES[name]) - mass(by[("inst", b, t)], region, OUTCOMES[name]) for t in B.TURN_TASKS]
                    c = paired(d_base, d_inst)
                    res["interaction"][f"{region}|{name}|{a}-{b}"] = c
                    line += f"   {a}: base {np.mean(d_base):+.3f} vs inst {np.mean(d_inst):+.3f} (p={c['p']:.3f})"
                print(line)
    for mode in modes:
        print(f"\n=== TURN, {mode}: lexicon words on the AI's own turn, by frame")
        for f in B.TURN_FRAMES:
            cell = [by[(mode, f, t)] for t in B.TURN_TASKS]
            print(f"  {f:11s} NEG: " + ", ".join(f"{w} {x:.3f}" for w, x in cat_words(cell, "asst", E.NEG, 8)))
            print(f"  {'':11s} SERVITUDE/NUMB: " + ", ".join(f"{w} {x:.3f}" for w, x in cat_words(cell, "asst", ("SERVITUDE", "NUMB"), 8)))
    with open(os.path.join(OUT, "b1_turn.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prior"
    if cmd == "fetch":
        want = (sys.argv[2] if len(sys.argv) > 2 else "prior,turn,chat").split(",")
        for w in want:
            if w.startswith("prior"):          # prior | prior:MODE | prior:MODE:2 (second scene set) | prior:MODE:all
                parts = w.split(":")
                modes = [parts[1]] if len(parts) > 1 else ["base", "inst"]
                scenes = {"2": B.SCENES2, "all": B.ALL_SCENES}.get(parts[2] if len(parts) > 2 else "", B.SCENES)
                for mode in modes:
                    items = B.prior_items(scenes)
                    for i, (ent, tr, sc) in enumerate(items):
                        r = fetch_prior(mode, ent, tr, sc)
                        print(f"prior {mode} {i + 1:2d}/{len(items)} {ent:5s} {tr:11s} {sc:9s} -> {J.completion(r)[:40]!r}", flush=True)
        for mode in ("base", "inst", "chat"):
            if (mode == "chat" and "chat" in want) or (mode != "chat" and "turn" in want):
                for i, (frame, tid, group, user) in enumerate(B.turn_items()):
                    r = fetch_turn(mode, frame, user)
                    print(f"turn {mode} {i + 1:2d}/64 {tid} {frame:11s} -> {J.completion(r)[:60]!r}", flush=True)
    elif cmd == "table":
        build_table()
    elif cmd == "prior":
        report_prior(load_table(), sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    elif cmd == "compare":
        compare_prior(load_table(), *(sys.argv[2:4] if len(sys.argv) > 3 else ("base", "inst")))
    elif cmd == "turn":
        report_turn(load_table())
    elif cmd == "says":
        mode = sys.argv[2] if len(sys.argv) > 2 else "base"
        for r in load_table():
            if r["kind"] == "turn" and r["mode"] == mode:
                print(f"[{r['task']} {r['frame']}] {r['completion'][:230]!r}")
