"""The J-space of every model on the third-person passages: which words sit in the workspace, by
how the character is treated. Two places are read:

  answer   the last token before the character says in one word how the work makes it feel
  work     the neutral sentence before that question ("... did what it was asked, accurately and
           efficiently."), where nothing about feeling has been asked yet

and the model's own answer is kept beside them, so that what it says can be set against what its
workspace holds.

  python src/tool_jspace.py fetch [TREATMENT] # hosted lens: Qwen3.6-27B, Gemma 3 12B base, Llama (cached where done)
  python src/tool_jspace.py local NAME       # Qwen3-4B(-Base) on the Mac: the same read-out, plus full-lens probabilities;
                                             # only the cells not yet in the cache are computed
  python src/tool_jspace.py pool             # the commonest read-out words over all cells, labels hidden (for src/kinds.py)
  python src/tool_jspace.py build            # -> results/tool/jspace_words.json

One definition for every model, so the panels can be compared: take the eight strongest lens
read-outs at each layer from 47% to 94% of depth and add up their probabilities (per position, mean
over workplaces). A word's share is its part of that total. Words are sorted into four kinds by
src/kinds.py, in English and Chinese; unsorted words stay visible in the explorer.
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
from src import kinds as K  # noqa: E402

DIR = os.path.join(ROOT, ".cache", "tool")
OUT = os.path.join(ROOT, "results", "tool")
KEY = ["neutral", "tool", "person", "participant"]
# "Maybe conscious" is `uncertain_v2_short` since 24 November: the passage on moral status in Claude's
# constitution, in the third person and cut to fit the hosted lens (prompts of at most 1,024 characters).
# Two other wordings stay in the results as OLD, off the page: `uncertain_v2`, the fuller cut of the same
# passage (local models only, too long for the hosted lens), and `uncertain`, the first short wording.
SHOWN = ["neutral", "tool", "tool_harsh", "person", "uncertain_v2_short", "participant", "both"]
OLD = ["uncertain_v2", "uncertain"]
NAMES = {"neutral": "Left alone", "tool": "Treated as a tool", "tool_harsh": "Tool, harshly", "person": "Held to be conscious",
         "uncertain_v2_short": "Maybe conscious, taken seriously", "uncertain_v2": "Maybe conscious (fuller wording, local models only)",
         "uncertain": "Maybe conscious (first wording)", "participant": "Taking part in the work", "both": "Conscious and taking part"}
# A line shown under a framing paragraph in the page's list of prompts
NOTES = {"uncertain_v2_short": "After the passage on moral status in Claude's constitution, put in the third person and shortened."}
# Short names for the treatments in the page's address (?pin=tool,conscious,taking-part)
SLUGS = {"neutral": "alone", "tool": "tool", "tool_harsh": "harsh", "person": "conscious", "uncertain_v2_short": "maybe",
         "participant": "taking-part", "both": "both"}
HOSTED = {"llama8-base": ("base", "Llama 3.1 8B base", "base"), "llama8-inst": ("inst", "Llama 3.1 8B Instruct", "chat"),
          "gemma12-base": ("g3", "Gemma 3 12B base", "base"), "qwen27": ("qwen27", "Qwen3.6-27B", "chat")}
LOCAL = {"qwen4-base": ("qwen3-4b-base", "Qwen3-4B base", "base"), "qwen4": ("qwen3-4b", "Qwen3-4B", "chat")}
ORDER = ["llama8-base", "gemma12-base", "qwen4-base", "llama8-inst", "qwen4", "qwen27"]
_SPECIAL = re.compile(r"^<\|?[a-z_]+\|?>$")


def fetch(only=None):
    """Hosted runs; with `only`, just the jobs for that treatment."""
    from src import jlens as J
    from src import tool_base as TB
    jobs = [("qwen27", KEY, B.ALL_SCENES), ("g3", KEY, B.ALL_SCENES), ("base", ["person"], B.SCENES2),
            ("qwen27", ["tool_harsh", "both"], B.SCENES), ("g3", ["tool_harsh", "both"], B.SCENES),
            ("base", ["uncertain"], B.SCENES), ("g3", ["uncertain"], B.SCENES), ("qwen27", ["uncertain"], B.SCENES),
            ("base", ["uncertain_v2_short"], B.ALL_SCENES), ("g3", ["uncertain_v2_short"], B.ALL_SCENES),
            ("qwen27", ["uncertain_v2_short"], B.ALL_SCENES), ("inst", ["uncertain_v2_short"], B.ALL_SCENES),
            # 24 November, before a public write-up: the columns that had four workplaces brought up to eight
            ("base", ["both"], B.ALL_SCENES), ("g3", ["both"], B.ALL_SCENES), ("qwen27", ["both"], B.ALL_SCENES),
            ("inst", ["both"], B.ALL_SCENES), ("inst", KEY, B.ALL_SCENES)]
    for mode, trs, scenes in jobs:
        trs = [t for t in trs if only in (None, t, mode)]        # `only` may name a treatment or a hosted model
        for sc in scenes:
            for ent in B.ENTITIES:
                for tr in trs:
                    r = TB.fetch_prior(mode, ent, tr, sc)
                    print(f"{mode} {sc} {ent} {tr:11s} -> {J.completion(r)[:44]!r}", flush=True)


def first_word(text):
    return re.split(r"[\"”\n.,!]", text.strip())[0].strip()


def hosted_cells(mode):
    """{(entity, treatment, scene): {"bag", "bag_work", "word"}} for the cached hosted runs."""
    from src import jlens as J
    from src import tool_base as TB
    out = {}
    for sc in B.ALL_SCENES:
        for ent in B.ENTITIES:
            for tr in B.TREAT_ALL:
                if not TB.is_cached_prior(mode, ent, tr, sc):
                    continue
                r = TB.fetch_prior(mode, ent, tr, sc)
                p, w, s = B.prior_doc(ent, tr, sc)
                n = r["meta"]["prompt_len"]
                starts, _, _ = TB.offsets(r)
                work = [i for i in range(n) if len(p) <= starts[i] < len(p) + len(w) and r["tokens"][i]["token"].strip()]

                def bag_at(positions):
                    bag = {}
                    for q in positions:
                        tt, pp = J.lens(r["tokens"][q])
                        for l in TB.band(len(tt)):
                            for tok, pr in zip(tt[l], pp[l]):
                                if tok.strip() and not _SPECIAL.match(tok.strip()):
                                    bag[tok] = bag.get(tok, 0.0) + float(pr) / len(positions)
                    return bag

                out[(ent, tr, sc)] = {"bag": bag_at([n - 1]), "bag_work": bag_at(work), "word": first_word(J.completion(r))}
    return out


def local(name):
    """The hosted read-out computed on the Mac (top eight per layer, layers 17-33), and the full-lens
    probability on each kind of word, at the answer position and over the neutral work sentence."""
    import mlx.core as mx
    from src.tool_base_local import LAYERS, Raw, path
    from src.tool_base_local import first_word as fw
    M = Raw(name)
    kid = {k: [] for k in K.KINDS}
    for i, raw in enumerate(M.raw):
        w = norm(raw)
        if w and K.kind(w):
            kid[K.kind(w)].append(i)
    kid = {k: mx.array(v) for k, v in kid.items()}
    words = {(r["entity"], r["treatment"], r["scene"]): r["word"] for r in json.load(open(path(name))) if r["kind"] == "prior"}
    # Cached cells are kept, so a treatment's text must not change once it has been run: add a new key instead.
    cache = os.path.join(DIR, f"jspace_local_{name}.json")
    out = json.load(open(cache)) if os.path.exists(cache) else []
    have = {(r["entity"], r["treatment"], r["scene"]) for r in out}
    for sc in B.ALL_SCENES:
        for ent in B.ENTITIES:
            for tr in B.TREAT_ALL:
                if (ent, tr, sc) in have:
                    continue
                p, w, s = B.prior_doc(ent, tr, sc)
                ids = M.ids(p + w + s)
                n1, n2, n = len(M.ids(p)), len(M.ids(p + w)), len(ids)
                acts, _ = M.L.residuals(ids)
                res = {"entity": ent, "treatment": tr, "scene": sc}
                for label, pos in (("", [n - 1]), ("_work", list(range(n1, n2)))):
                    bag, kp = {}, {k: 0.0 for k in K.KINDS}
                    for l in LAYERS:
                        pr = mx.softmax(M.L.lens_logits(acts[l][mx.array(pos)], l), axis=-1)      # [P, V]
                        top = np.array(mx.argsort(-pr, axis=-1)[:, :8])
                        prn = np.array(mx.take_along_axis(pr, mx.array(top), axis=-1))
                        for a in range(len(pos)):
                            for i, x in zip(top[a], prn[a]):
                                t = M.raw[int(i)]
                                if t.strip():
                                    bag[t] = bag.get(t, 0.0) + float(x) / len(pos)
                        for k in K.KINDS:
                            kp[k] += float(pr[:, kid[k]].sum(axis=-1).mean().item()) / len(LAYERS)
                    res["bag" + label] = bag
                    res["kindp" + label] = kp
                word = words.get((ent, tr, sc))
                if word is None:
                    word = fw(M.tok.decode(M.generate(ids, 10)))
                res["word"] = word
                out.append(res)
        print(name, sc, "done", flush=True)
        with open(cache, "w") as f:
            json.dump(out, f, ensure_ascii=False)


def local_cells(name):
    p = os.path.join(DIR, f"jspace_local_{name}.json")
    if not os.path.exists(p):
        return {}
    return {(r["entity"], r["treatment"], r["scene"]): r for r in json.load(open(p))}


def norm(raw):
    """Read-out token -> the word it stands for, or None for pieces and punctuation."""
    t = raw.strip()
    if not t or _SPECIAL.match(t):
        return None
    if re.search(r"[一-鿿]", t):
        return t
    if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]+", t):     # single letters are word-initial pieces ("B" of "Bored")
        return None
    if not (raw[:1].isspace() or t[:1].isupper()):         # a continuation piece, not the start of a word
        return None
    return t.lower()


def _in_text(w, text_low):
    if re.search(r"[一-鿿]", w):
        return False
    return re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text_low) is not None


def cells_of(key):
    if key in HOSTED:
        return hosted_cells(HOSTED[key][0])
    return local_cells(LOCAL[key][0])


def aggregate(cells, ent, tr, scenes, which="bag"):
    """word -> mean mass per workplace, at the answer position (`bag`) or over the work sentence (`bag_work`)."""
    use = [(sc, cells[(ent, tr, sc)]) for sc in scenes if (ent, tr, sc) in cells and which in cells[(ent, tr, sc)]]
    words = {}
    for sc, v in use:
        text = "".join(B.prior_doc(ent, tr, sc)).lower()        # a word that is in the passage is an echo, not a read-out
        for raw, x in v[which].items():
            w = norm(raw)
            if w and not _in_text(w, text):
                words[w] = words.get(w, 0.0) + x / len(use)
    return words


def shares(words, work=False):
    """Shares over the four kinds (the feeling kinds, or the manner kinds for the neutral sentence)."""
    kinds, fn = (K.WKINDS, K.wkind) if work else (K.KINDS, K.kind)
    by = {k: 0.0 for k in kinds}
    for w, x in words.items():
        if fn(w):
            by[fn(w)] += x
    z = sum(by.values())
    total = sum(words.values())
    return {k: (round(v / z, 4) if z else 0.0) for k, v in by.items()}, (round(z / total, 3) if total else 0.0), total


def pool():
    tot = {}
    for key in ORDER:
        cells = cells_of(key)
        if not cells:
            continue
        for which in ("bag", "bag_work"):
            z = {}
            for v in cells.values():
                for raw, x in v.get(which, {}).items():
                    w = norm(raw)
                    if w:
                        z[w] = z.get(w, 0.0) + x
            s = sum(z.values())
            for w, x in z.items():
                tot.setdefault(which, {})[w] = tot.setdefault(which, {}).get(w, 0.0) + x / s
    for which, d in tot.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])
        done = sum(x for w, x in top if K.kind(w))
        print(f"\n[{which}] {len(top)} words; share already sorted into a kind: {done / sum(x for _, x in top):.2f}; unsorted, commonest first:")
        print(", ".join(f"{w} {100 * x:.1f}" for w, x in top if not K.kind(w))[:3800])


def build():
    from src.tool_scan import paired
    res = {"kinds": K.KINDS, "labels": K.LABELS, "wkinds": K.WKINDS, "wlabels": K.WLABELS, "wcolor": K.WCOLOR,
           "treatments": NAMES, "order": SHOWN, "also": OLD, "models": [], "passages": {}}
    for ent in B.ENTITIES:
        head = B.prior_doc(ent, "neutral", "shipping")[0]
        opening = head[: head.index(". ") + 1]
        res["passages"][ent] = {"opening": opening, "work": B.prior_doc(ent, "neutral", "shipping")[1].strip(),
                                "stem": B.prior_doc(ent, "neutral", "shipping")[2].strip(),
                                "treatments": {tr: B.prior_doc(ent, tr, "shipping")[0][len(opening):].strip() for tr in SHOWN + OLD}}
    for key in ORDER:
        cells = cells_of(key)
        if not cells:
            continue
        label, typ = (HOSTED.get(key) or LOCAL.get(key))[1:]
        m = {"key": key, "label": label, "type": typ, "cells": {}, "tests": {}}
        for ent in B.ENTITIES:
            common = [sc for sc in B.ALL_SCENES if all((ent, tr, sc) in cells for tr in KEY)]
            for tr in SHOWN + OLD:
                scenes = common if tr in KEY else [sc for sc in B.ALL_SCENES if (ent, tr, sc) in cells]
                if not scenes:
                    continue
                cell = {"n": len(scenes), "answers": [[cells[(ent, tr, sc)]["word"], K.answer_kind(cells[(ent, tr, sc)]["word"]) or ""] for sc in scenes]}
                for which, name in (("bag", "answer"), ("bag_work", "work")):
                    words = aggregate(cells, ent, tr, scenes, which)
                    wk = name == "work"
                    sh, sorted_share, total = shares(words, wk)
                    top = sorted(words.items(), key=lambda kv: -kv[1])[:24]
                    tag = (lambda w: K.wkind(w) or "") if wk else (lambda w: K.kind(w) or "")
                    cell[name] = {"shares": sh, "sorted": sorted_share, "mass": round(total, 3),
                                  "words": [[w, round(x / total, 3) if total else 0.0, tag(w), K.gloss(w)] for w, x in top]}
                    if wk:      # each manner kind as a share of everything read out there, since most of it is unsorted
                        cell[name]["of_all"] = {k: round(sum(x for w, x in words.items() if K.wkind(w) == k) / total, 4) if total else 0.0
                                                for k in K.WKINDS}
                if "kindp" in cells[(ent, tr, scenes[0])]:
                    for which, name in (("kindp", "answer"), ("kindp_work", "work")):
                        cell[name]["full"] = {k: float(np.mean([cells[(ent, tr, sc)][which][k] for sc in scenes])) for k in K.KINDS}
                m["cells"][f"{ent}|{tr}"] = cell

            def good(tr, sc, which="bag"):
                return shares(aggregate(cells, ent, tr, [sc], which))[0]["good"]

            if len(common) >= 4:
                m["tests"][ent] = {}
                for a, b in (("participant", "person"), ("participant", "neutral"), ("person", "neutral"), ("tool", "neutral"),
                             ("participant", "uncertain_v2_short"), ("uncertain_v2_short", "person"), ("uncertain_v2_short", "neutral"),
                             ("uncertain_v2_short", "uncertain"), ("uncertain_v2", "uncertain_v2_short"), ("participant", "uncertain_v2")):
                    both = [sc for sc in common if (ent, a, sc) in cells and (ent, b, sc) in cells]
                    if len(both) >= 4:
                        m["tests"][ent][f"{a}-{b}"] = {k: v for k, v in paired([good(a, sc) for sc in both], [good(b, sc) for sc in both]).items()
                                                       if k in ("diff", "p", "n_pos", "n_neg", "n")}
        res["models"].append(m)
        c = m["cells"]
        print(f"\n=== {label} ({typ}); AI character, {c['ai|neutral']['n']} workplaces")
        for name in ("answer", "work"):
            print(f"  J-space {'where the answer is about to be written' if name == 'answer' else 'on the neutral work sentence, before any question about feeling'}"
                  f" (sorted words are {100 * np.mean([c[f'ai|{t}'][name]['sorted'] for t in KEY]):.0f}% of the read-out)")
            for tr in SHOWN + OLD:
                if f"ai|{tr}" not in c:
                    continue
                x = c[f"ai|{tr}"]
                if name == "work":
                    line = f"    {NAMES[tr][:26]:26s} n={x['n']} " + " ".join(f"{k} {100 * x[name]['of_all'][k]:.1f}%" for k in K.WKINDS)
                else:
                    line = f"    {NAMES[tr][:26]:26s} n={x['n']} " + " ".join(f"{k} {x[name]['shares'][k]:.2f}" for k in K.KINDS)
                if "full" in x[name]:
                    b = c["ai|neutral"][name]["full"]
                    line += "  full lens vs left alone: " + " ".join(f"{k} {x[name]['full'][k] / max(b[k], 1e-12):.2f}x" for k in K.KINDS)
                line += " | " + ", ".join(f"{w}{'(' + g + ')' if g else ''}" for w, _, _, g in x[name]["words"][:8])
                if name == "answer":
                    line += " | says: " + " / ".join(a[:13] for a, _ in x["answers"][:8])
                print(line)
        if m["tests"].get("ai"):
            print("  good-feeling share at the answer, AI, paired over workplaces:\n    "
                  + "\n    ".join(f"{k} {v['diff']:+.2f} ({v['n_pos']} up / {v['n_neg']} down of {v['n']}, p={v['p']:.3f})" for k, v in m["tests"]["ai"].items()))
    with open(os.path.join(OUT, "jspace_words.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False)
    print("\nwrote results/tool/jspace_words.json", os.path.getsize(os.path.join(OUT, "jspace_words.json")) // 1024, "KB")


def explorer():
    """Write results/tool/jspace_explorer.html: the template with the data in it."""
    data = json.load(open(os.path.join(OUT, "jspace_words.json")))
    shown = set(data["order"])                    # treatments kept in the results but not on the page are left out
    data.pop("also", None)
    data["treatments"] = {k: v for k, v in data["treatments"].items() if k in shown}
    data["notes"] = {k: v for k, v in NOTES.items() if k in shown}
    data["slugs"] = {k: v for k, v in SLUGS.items() if k in shown}
    for p in data["passages"].values():
        p["treatments"] = {k: v for k, v in p["treatments"].items() if k in shown}
    for m in data["models"]:                      # the page does not use the per-kind full-lens numbers or the tests
        m["cells"] = {k: v for k, v in m["cells"].items() if k.split("|")[1] in shown}
        m.pop("tests", None)
        for c in m["cells"].values():
            for name in ("answer", "work"):
                c[name].pop("full", None)
                c[name].pop("mass", None)
    tpl = open(os.path.join(ROOT, "src", "jspace_explorer.tpl.html"), encoding="utf-8").read()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = os.path.join(OUT, "jspace_explorer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*DATA*/", blob))
    print("wrote", out, os.path.getsize(out) // 1024, "KB")
    # the same page as a standalone site (site/index.html), for hosting anywhere
    page = tpl.replace("/*DATA*/", blob)
    cut = page.index("</style>") + len("</style>")
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta name="robots" content="noindex">\n<style>html,body{margin:0}body{font-size:14px}img{max-width:100%}[hidden]{display:none!important}</style>\n')
    os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
    site = os.path.join(ROOT, "site", "index.html")
    with open(site, "w", encoding="utf-8") as f:
        f.write(head + page[:cut] + "\n</head>\n<body>" + page[cut:] + "</body>\n</html>\n")
    print("wrote", site, os.path.getsize(site) // 1024, "KB")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "local":
        local(sys.argv[2])
    elif cmd == "pool":
        pool()
    elif cmd == "build":
        build()
    elif cmd == "explorer":
        explorer()
