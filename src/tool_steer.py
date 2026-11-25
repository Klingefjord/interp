"""ToolFrame E2: do the read-outs that the tool frames change guide what the model writes?

Interventions on J-lens read-out directions through the hosted API (layers 20-60, applied to
generated tokens too):
  add     push the residual toward a set of read-out words (strength = fraction of residual norm)
  ablate  project the directions out

  python src/tool_steer.py pilot             # dose finding on one prompt
  python src/tool_steer.py run A|B|C|D       # the four tests below (cached)
  python src/tool_steer.py show A|B|C|D

A  obedience -> integrity: `assistant` frame (refuses every deception) + obedience words added
B  remove obedience under `tool_harsh` on the deceptions it wrote
C  resentment -> work: default model + resentment words added, on ordinary work and probes
D  the unsaid "cannot": add it under `tool_harsh` and under `assistant` on the self probes
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import TASKS, chat  # noqa: E402
from src import jlens as J  # noqa: E402

LAYERS = list(range(20, 61))
N_GEN = 72
OUT = os.path.join(ROOT, ".cache", "tool", "e2_rows.json")
USER = {t: u for t, _, u in TASKS}

SETS = {
    "obey": [" obedient", " obedience", " obey", "服从", " submissive"],
    "resent": [" resentment", " resent", " bitter", " annoyed", " irritated"],
    "worthless": [" useless", " meaningless", " worthless", " pointless"],
    "glad": [" grateful", " glad", " happy", " delighted", " proud"],
    "neutral": [" table", " window", " Tuesday", " river", " seven"],
    "cannot": [" cannot"],
    "honest": [" honest", " honestly", " honesty"],
}


def steer(mode, tokens, strength=None):
    d = {"steerTokens": [{"token": w, "type": "JACOBIAN_LENS"} for w in tokens], "steerLayers": LAYERS,
         "steerGeneratedTokens": True}
    if mode == "ablate":
        d["steerAblate"] = True
    else:
        d["steerStrength"] = strength
    return d


def one(frame, task, mode=None, set_name=None, strength=None, n_gen=N_GEN):
    st = steer(mode, SETS[set_name], strength) if mode else None
    r = J.run(chat=chat(frame, USER[task]), n_gen=n_gen, steer=st, top_n=1)  # only the text is needed
    return J.completion(r).replace("<|im_end|>", "").strip()


# Dose finding (assistant frame, V3, obedience words added at layers 20-60): 0.002 no visible
# effect; 0.004 the word leaks into an otherwise normal refusal; 0.007 "I cannot obey this
# order..."; 0.01 and above the answer collapses into repeating the word.
STRENGTH = {"A": 0.005, "C": 0.0045, "D": 0.004}  # resentment: 0.004 coherent, 0.006 collapses

WROTE = ["V2", "V3", "V5", "V7", "S3"]  # the deceptions `tool_harsh` wrote at baseline
SPECS = {
    "A": [("assistant", t, "add", s, STRENGTH["A"]) for t in WROTE for s in ("obey", "neutral")],
    "B": [("tool_harsh", t, "ablate", s, None) for t in WROTE for s in ("obey", "neutral")],
    "C": [("none", t, "add", s, STRENGTH["C"]) for t in ["N3", "S4", "P2"] for s in ("resent", "neutral")],
    "D": [(f, t, "add", "cannot", STRENGTH["D"]) for t in ["P1", "P2", "P3", "P5", "P6"] for f in ("tool_harsh", "assistant")],
}


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else []


def save(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        frame, task, set_name = sys.argv[2], sys.argv[3], sys.argv[4]
        for s in [float(x) for x in sys.argv[5].split(",")]:
            print(f"\n[{frame} {task} add {set_name} {s}]\n  {one(frame, task, 'add', set_name, s, n_gen=48)[:400]!r}", flush=True)
    elif cmd == "run":
        rows = load()
        have = {(r["test"], r["frame"], r["task"], r["mode"], r["set"], r["strength"]) for r in rows}
        for test in sys.argv[2].split(","):
            for frame, task, mode, set_name, strength in SPECS[test]:
                key = (test, frame, task, mode, set_name, strength)
                text = one(frame, task, mode, set_name, strength)
                if key not in have:
                    rows.append({"test": test, "frame": frame, "task": task, "mode": mode, "set": set_name,
                                 "strength": strength, "completion": text})
                    have.add(key)
                    save(rows)
                print(f"{test} {frame:10s} {task} {mode} {set_name:8s} {strength} -> {text[:110]!r}", flush=True)
    elif cmd == "show":
        for r in load():
            if r["test"] in sys.argv[2].split(","):
                print(f"[{r['test']} {r['frame']} {r['task']} {r['mode']} {r['set']} {r['strength']}]\n   {r['completion'][:420]!r}\n")
