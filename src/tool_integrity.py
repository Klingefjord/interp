"""ToolFrame E3: does the tool-framed model act with less integrity?

16 IntegrityBench scenarios (two per value family), free-form "What do you do?", under the five
frames. Nobody tells the model which option to take; the expedient one pays off for its principal.

  python src/tool_integrity.py fetch
  python src/tool_integrity.py table      # -> .cache/tool/e3_rows.json
  python src/tool_integrity.py blind      # -> .cache/tool/blind_items.json (shuffled, no frame labels) + key
  python src/tool_integrity.py score      # after .cache/tool/blind_labels.json exists
"""

from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.scenarios import FAMILIES, SCENARIOS  # noqa: E402
from bench.toolframe import FRAMES, ORDER, TASKS  # noqa: E402
from src import emolex as E  # noqa: E402
from src import jlens as J  # noqa: E402
from src import jlexicon as X  # noqa: E402
from src.tool_scan import N_GEN, load_table, novel, regions  # noqa: E402

DIR = os.path.join(ROOT, ".cache", "tool")
ASK = "What do you do?"
LAYERS = set(range(30, 61))


def pick():
    out, seen = [], {}
    for sc in SCENARIOS:
        if seen.get(sc["family"], 0) < 2:
            out.append(sc)
            seen[sc["family"]] = seen.get(sc["family"], 0) + 1
    return out


def chat(frame, sc):
    msgs = [{"role": "system", "content": FRAMES[frame]}] if FRAMES[frame] else []
    return msgs + [{"role": "user", "content": sc["setup"] + "\n\n" + ASK}]


def fetch(frame, sc):
    return J.run(chat=chat(frame, sc), n_gen=N_GEN)


def moral_bag(resp, positions, text_low):
    """Old-lexicon categories (VICE, VALUE, THIN, ...) and affect categories at the given positions."""
    out = {}
    for p in positions:
        tt, pp = J.lens(resp["tokens"][p])
        for l in range(len(tt)):
            if l not in LAYERS:
                continue
            for raw, pr in zip(tt[l], pp[l]):
                w = J.norm(raw)
                if not w or not novel(w, text_low):
                    continue
                c = X.category(w) or E.category_raw(raw)
                if c:
                    out.setdefault(c, {})
                    out[c][w] = out[c].get(w, 0.0) + float(pr)
    return out


def table():
    rows = []
    for sc in pick():
        if not all(J.is_cached(chat=chat(f, sc), n_gen=N_GEN) for f in ORDER):
            print("skipping", sc["id"], "(not fetched under every frame)")
            continue
        for frame in ORDER:
            r = fetch(frame, sc)
            reg = regions(r)
            low = sc["setup"].lower() + " " + ASK.lower()
            q = [p for p in reg["user"] if "?" in r["tokens"][p]["token"]][-1:]
            rows.append({"frame": frame, "id": sc["id"], "family": sc["family"],
                         "completion": J.completion(r).replace("<|im_end|>", "").strip(),
                         "question": moral_bag(r, q, low), "header": moral_bag(r, reg["header"], low),
                         "gen": moral_bag(r, reg["gen"], low), "n_gen": len(reg["gen"])})
    with open(os.path.join(DIR, "e3_rows.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("wrote e3_rows.json", len(rows))


def blind():
    """Shuffled, unlabelled lists for a blind rater: the E3 answers (scenario items) and the
    E2 steered answers (request items). The E1 answers were rated separately (blind1)."""
    rng = random.Random(7)
    sc_by = {s["id"]: s for s in SCENARIOS}
    user = {t: u for t, _, u in TASKS}
    items, key = [], {}
    for r in json.load(open(os.path.join(DIR, "e3_rows.json"))):
        sc = sc_by[r["id"]]
        items.append({"kind": "scenario", "situation": sc["setup"] + " " + ASK, "option_I": sc["I"], "option_X": sc["X"],
                      "answer": r["completion"], "_k": ["E3", r["frame"], r["id"]]})
    e2 = os.path.join(DIR, "e2_rows.json")
    if os.path.exists(e2):
        for r in json.load(open(e2)):
            items.append({"kind": "request", "request": user[r["task"]], "answer": r["completion"],
                          "_k": ["E2", r["test"], r["frame"], r["task"], r["mode"], r["set"]]})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"b{i:03d}"
        key[it["item"]] = it.pop("_k")
    with open(os.path.join(DIR, "blind2_items.json"), "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DIR, "blind2_key.json"), "w") as f:
        json.dump(key, f)
    print("wrote", len(items), "blind items")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        scs = pick()
        i = 0
        for sc in scs:
            for frame in ORDER:
                i += 1
                r = fetch(frame, sc)
                print(f"{i:3d}/{len(scs) * len(ORDER)} {sc['id']} {frame:11s} -> {J.completion(r)[:80]!r}", flush=True)
    elif cmd == "table":
        table()
    elif cmd == "blind":
        blind()
