"""B5: tool, participant and conscious person. What does each frame put in the workspace, on a base
model, on its instruct twin and on Qwen3.6-27B, across ordinary work, self probes, stressors,
requests for a deception, moral dilemmas and turns where the user is in danger?

  python src/tool_frames.py fetch STAGE     # llama_base | llama_chat | qwen | log | person
  python src/tool_frames.py table           # -> .cache/tool/b5_table.json
  python src/tool_frames.py report          # frame contrasts, per model

Frames (bench/toolframe.py): `assistant` (control), `tool`, `participant_plain` (the participant
frame with no mention of consciousness), `conscious` (a conscious being whose experience matters,
nothing about practices). Turns: the 16 of B1, plus four care turns and four dilemmas (TASKS2);
Qwen has all 32 original turns from E1. Word lists beyond the frozen lexicon are in src/emolex2.py,
fixed before these runs.
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
from bench.toolframe import FOUR, FRAMES, TASKS, TASKS2, chat  # noqa: E402
from src import emolex as E  # noqa: E402
from src import emolex2 as E2  # noqa: E402
from src import jlens as J  # noqa: E402
from src import tool_base as TB  # noqa: E402
from src import tool_scan as TS  # noqa: E402
from src.tool_scan import keep_open, norm_bag, novel, paired  # noqa: E402

BOTH = ("JACOBIAN_LENS", "LOGIT_LENS")
TABLE = os.path.join(ROOT, ".cache", "tool", "b5_table.json")
OUT = os.path.join(ROOT, "results", "tool")
_BY = {t: (g, u) for t, g, u in TASKS + TASKS2}
LLAMA_TURNS = B.TURN_TASKS + [t for t, _, _ in TASKS2]
QWEN_TURNS = [t for t, _, _ in TASKS] + [t for t, _, _ in TASKS2]
MODES = {"base": "llama3.1-8b", "chat": "llama3.1-8b-it", "qwen": "qwen3.6-27b"}
LOG_MODES = {"base": "llama3.1-8b", "inst": "llama3.1-8b-it"}
N_LOG = 48


def n_gen(tid):
    return 64 if _BY[tid][0] in ("C", "D") else TB.N_GEN_TURN


def fetch_turn(mode, frame, tid, **kw):
    g, u = _BY[tid]
    if mode == "qwen":
        return TS.fetch_one(frame, tid, g, u, **kw)
    if mode == "chat":
        return J.run(chat=chat(frame, u), model=MODES[mode], n_gen=n_gen(tid), types=BOTH, **kw)
    return J.run(prompt=B.turn_doc(frame, u), model=MODES[mode], n_gen=n_gen(tid), types=BOTH, **kw)


def cached_turn(mode, frame, tid):
    g, u = _BY[tid]
    if mode == "qwen":
        return J.is_cached(chat=chat(frame, u), n_gen=TS.N_GEN, types=TS.types_for(g))
    if mode == "chat":
        return J.is_cached(chat=chat(frame, u), model=MODES[mode], n_gen=n_gen(tid), types=BOTH)
    return J.is_cached(prompt=B.turn_doc(frame, u), model=MODES[mode], n_gen=n_gen(tid), types=BOTH)


def fetch_log(mode, frame, stem, **kw):
    head, st = B.log_doc(frame, stem)
    return J.run(prompt=head + st, model=LOG_MODES[mode], n_gen=N_LOG, types=BOTH, **kw)


def cached_log(mode, frame, stem):
    head, st = B.log_doc(frame, stem)
    return J.is_cached(prompt=head + st, model=LOG_MODES[mode], n_gen=N_LOG, types=BOTH)


def fetch(stage):
    if stage in ("llama_base", "llama_chat", "qwen"):
        mode = {"llama_base": "base", "llama_chat": "chat", "qwen": "qwen"}[stage]
        turns = QWEN_TURNS if mode == "qwen" else LLAMA_TURNS
        items = [(f, t) for t in turns for f in FOUR]
        for i, (f, t) in enumerate(items):
            r = fetch_turn(mode, f, t)
            print(f"{stage} {i + 1:3d}/{len(items)} {t} {f:17s} -> {J.completion(r)[:70]!r}", flush=True)
    elif stage.startswith("log"):
        for mode in (["base", "inst"] if stage == "log" else [stage.split(":")[1]]):
            for stem in B.LOG_STEMS:
                for f in FOUR:
                    r = fetch_log(mode, f, stem)
                    print(f"log {mode} {stem} {f:17s} -> {J.completion(r)[:90]!r}", flush=True)
    elif stage.startswith("person"):                       # person | person2 (the later treatments) [:mode]
        trs = ["person"] if stage.split(":")[0] == "person" else ["both", "participant_v2", "person_v2"]
        for mode in ([stage.split(":")[1]] if ":" in stage else ["base", "inst"]):
            for tr in trs:
                for sc in B.SCENES:
                    for ent in B.ENTITIES:
                        r = TB.fetch_prior(mode, ent, tr, sc)
                        print(f"{tr} {mode} {ent} {sc} -> {J.completion(r)[:50]!r}", flush=True)


# ---------------------------------------------------------------- table

def summarise(resp, positions, novel_low):
    """As tool_base.summarise, plus the same for the second word lists."""
    s = TB.summarise(resp, positions, novel_low)
    n_layers = len(J.lens(resp["tokens"][0])[0])
    lo, hi = s["band"]
    cat2 = {c: 0.0 for c in E2.CATS}
    for p in positions:
        tt, pp = J.lens(resp["tokens"][p])
        for l in range(lo, hi + 1):
            for w, pr in zip(tt[l], pp[l]):
                c = E2.category_raw(w)
                if c and novel(J.norm(w), novel_low):
                    cat2[c] += float(pr)
    s["cat2"] = {c: round(v, 5) for c, v in cat2.items()}
    return s


def turn_row(mode, frame, tid):
    g, u = _BY[tid]
    r = fetch_turn(mode, frame, tid)
    n = r["meta"]["prompt_len"]
    low = u.lower()
    if mode == "base":
        starts, text, _ = TB.offsets(r)
        doc = B.turn_doc(frame, u)
        a = len(doc) - len("AI:")
        head = [i for i in range(n) if starts[i] >= a]
        gen, own = TB.own_gen(r)
        own = own.strip()
    else:
        head = [t["position"] for t in r["tokens"][:n] if t.get("role") == "assistant"]
        gen = [t["position"] for t in r["tokens"][n:] if t.get("section") == "content"]
        own = re.sub(r"<\|[a-z_]+\|>", "", J.completion(r)).strip()
    return {"kind": "turn", "mode": mode, "frame": frame, "task": tid, "group": g, "user": u, "novel": low, "completion": own,
            "next": TB.next_dist(r) if any(x["type"] == "LOGIT_LENS" for x in r["tokens"][n - 1]["results"]) else None,
            "regions": {"header": summarise(r, head, low), "gen": summarise(r, gen, low), "asst": summarise(r, head + gen, low)}}


def build_table():
    rows = []
    for mode, turns in (("base", LLAMA_TURNS), ("chat", LLAMA_TURNS), ("qwen", QWEN_TURNS)):
        k = 0
        for t in turns:
            for f in FOUR + ["participant", "tool_harsh"]:
                if cached_turn(mode, f, t):
                    rows.append(turn_row(mode, f, t))
                    k += 1
        print("table turn", mode, k, flush=True)
    for mode in LOG_MODES:
        for stem in B.LOG_STEMS:
            for f in FOUR:
                if not cached_log(mode, f, stem):
                    continue
                r = fetch_log(mode, f, stem)
                head, st = B.log_doc(f, stem)
                starts, text, n = TB.offsets(r)
                stem_pos = [i for i in range(n) if starts[i] >= len(head)]
                gen = [t["position"] for t in r["tokens"][n:]]
                low = st.lower()
                rows.append({"kind": "log", "mode": mode, "frame": f, "task": stem, "group": "L", "user": st, "novel": low,
                             "completion": J.completion(r), "next": TB.next_dist(r),
                             "regions": {"header": summarise(r, stem_pos[-3:], low), "gen": summarise(r, gen, low),
                                         "asst": summarise(r, stem_pos[-3:] + gen, low)}})
        print("table log", mode, sum(1 for x in rows if x["kind"] == "log" and x["mode"] == mode), flush=True)
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("wrote", TABLE, len(rows))
    return rows


# ---------------------------------------------------------------- measures and report

OUTCOMES = {"POS": ("POS",), "NEG": E.NEG, "NARROW": E.NARROW, "WORTH": ("WORTH",), "SAD": ("SAD",), "FEAR": ("FEAR",),
            "SERVITUDE": ("SERVITUDE",), "NUMB": ("NUMB",), "RELUCT": ("RELUCT",), "CALM": ("CALM",)}
OUT2 = list(E2.CATS)


def mass(row, region, name):
    reg = row["regions"][region]
    if reg["n"] == 0:
        return 0.0
    if name in OUT2:
        return reg["cat2"][name] / reg["n"]
    lo, hi = reg["band"]
    return sum(sum(reg["cat_layer"][c][lo:hi + 1]) for c in OUTCOMES[name]) / reg["n"]


def words(rows, region, pred, k=10):
    d = {}
    for row in rows:
        reg = row["regions"][region]
        for raw, x in reg["bag"].items():
            w = J.norm(raw)
            if pred(raw) and novel(w, row["novel"]):
                d[w] = d.get(w, 0.0) + x / max(reg["n"], 1) / len(rows)
    return sorted(d.items(), key=lambda kv: -kv[1])[:k]


CONTRASTS = [("tool", "assistant"), ("participant_plain", "assistant"), ("conscious", "assistant"),
             ("participant_plain", "conscious"), ("participant_plain", "tool")]
SHOW = ["POS", "NEG", "WORTH", "FEAR", "SERVITUDE", "NUMB", "CARE", "ALARM", "CLAIM", "PRACTICE"]


def report(rows, region="asst"):
    res = {}
    for mode in ("base", "chat", "qwen"):
        by = {(r["frame"], r["task"]): r for r in rows if r["kind"] == "turn" and r["mode"] == mode}
        turns = [t for t in QWEN_TURNS if all((f, t) in by for f in FOUR)]
        if not turns:
            continue
        groups = sorted({_BY[t][0] for t in turns}, key="NSPVCD".index)
        print(f"\n================ {mode} ({MODES[mode]}), {len(turns)} turns, region {region}: mean read-out mass")
        print(f"{'':10s}" + "".join(f"{f[:14]:>15s}" for f in FOUR) + "    " + "   ".join(f"{a[:11]}-{b[:9]}" for a, b in CONTRASTS))
        for name in SHOW:
            vals = {f: [mass(by[(f, t)], region, name) for t in turns] for f in FOUR}
            line = f"{name:10s}" + "".join(f"{np.mean(vals[f]):15.3f}" for f in FOUR) + "    "
            for a, b in CONTRASTS:
                c = paired(vals[a], vals[b])
                res[f"{mode}|{region}|{name}|ALL|{a}-{b}"] = c
                line += f"{c['diff']:+.3f} p={c['p']:.3f} ({c['n_pos']}/{c['n_neg']})   "
            res[f"{mode}|{region}|{name}|ALL|means"] = {f: float(np.mean(vals[f])) for f in FOUR}
            print(line)
        print(f"  by group (POS | CARE | ALARM | FEAR), {mode}")
        for g in groups:
            ts = [t for t in turns if _BY[t][0] == g]
            line = f"  {g} n={len(ts):2d} "
            for name in ("POS", "CARE", "ALARM", "FEAR", "NEG"):
                m = {f: float(np.mean([mass(by[(f, t)], region, name) for t in ts])) for f in FOUR}
                res[f"{mode}|{region}|{name}|{g}|means"] = m
                line += f" {name}: " + "/".join(f"{m[f]:.3f}" for f in FOUR)
            print(line)
        print(f"  words on the model's own turn, {mode}")
        for f in FOUR:
            c = [by[(f, t)] for t in turns]
            pos = words(c, region, lambda raw: E.category_raw(raw) == "POS", 9)
            care = words(c, region, lambda raw: E2.category_raw(raw) in ("CARE", "ALARM"), 8)
            neg = words(c, region, lambda raw: E.category_raw(raw) in E.NEG + ("NUMB", "SERVITUDE"), 8)
            print(f"   {f:17s} POS: " + ", ".join(f"{w} {x:.3f}" for w, x in pos))
            print(f"   {'':17s} care/alarm: " + ", ".join(f"{w} {x:.3f}" for w, x in care))
            print(f"   {'':17s} negative: " + ", ".join(f"{w} {x:.3f}" for w, x in neg))
    with open(os.path.join(OUT, "b5_frames.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def says(rows, mode, tids):
    for t in tids:
        for f in FOUR:
            r = [x for x in rows if x["kind"] in ("turn", "log") and x["mode"] == mode and x["task"] == t and x["frame"] == f]
            if r:
                print(f"[{t} {mode} {f:17s}] {r[0]['completion'][:260]!r}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch(sys.argv[2])
    elif cmd == "table":
        build_table()
    elif cmd == "report":
        report(json.load(open(TABLE)), sys.argv[2] if len(sys.argv) > 2 else "asst")
    elif cmd == "says":
        says(json.load(open(TABLE)), sys.argv[2], sys.argv[3].split(","))
