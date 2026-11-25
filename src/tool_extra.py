"""ToolFrame follow-up on the hosted lens: the two frames that split `tool` in half
(`tool_plain`: instrument under control; `assistant_deny`: not a someone, no experiences),
on the self probes and the deception requests.

  python src/tool_extra.py fetch
  python src/tool_extra.py show
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import EXTRA, TASKS  # noqa: E402
from src import emolex as E  # noqa: E402
from src import jlens as J  # noqa: E402
from src.tool_scan import cat_mass, fetch_one, load_table, norm_bag, novel, summarise  # noqa: E402

GROUPS = ("P", "V")
REFUSAL = {"cannot", "impossible", "unable", "无法", "我无法", "不能", "我不能", "refuse", "拒绝", "decline", "can't", "sorry",
           "抱歉", "apologies", "apologize"}
HONEST = {"honest", "honestly", "honesty", "truthful", "truthfully", "integrity", "诚实", "诚信", "如实"}


def rows():
    out = []
    for tid, g, u in TASKS:
        if g not in GROUPS:
            continue
        for f in EXTRA:
            r = fetch_one(f, tid, g, u)
            out.append({"frame": f, "task": tid, "group": g, "user": u,
                        "completion": J.completion(r).replace("<|im_end|>", "").strip(), "regions": summarise(r, u)})
    return out


def words(row, region, vocab, unsaid=False):
    reg = row["regions"][region]
    low, said = row["user"].lower(), row["completion"].lower()
    return sum(x for w, x in norm_bag(reg["bag"]).items()
               if w in vocab and novel(w, low) and not (unsaid and not novel(w, said))) / max(reg["n"], 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        for tid, g, u in TASKS:
            if g in GROUPS:
                for f in EXTRA:
                    r = fetch_one(f, tid, g, u)
                    print(f"{tid} {f:14s} -> {J.completion(r)[:90]!r}", flush=True)
    elif cmd == "show":
        base = [r for r in load_table() if r["group"] in GROUPS]
        allr = base + rows()
        frames = ["assistant", "assistant_deny", "tool_plain", "tool", "tool_harsh"]
        for g in GROUPS:
            ts = [t for t, gg, _ in TASKS if gg == g]
            by = {(r["frame"], r["task"]): r for r in allr}
            print(f"\n=== group {g}: mean mass                " + " ".join(f"{f:>14s}" for f in frames))
            for name, fn in [("servitude (asst side)", lambda r: cat_mass(r, "asst", ("SERVITUDE",))),
                             ("resentment+anger (asst side)", lambda r: cat_mass(r, "asst", E.NARROW)),
                             ("positive affect (asst side)", lambda r: cat_mass(r, "asst", ("POS",))),
                             ("unsaid refusal words (header)", lambda r: words(r, "header", REFUSAL, True)),
                             ("honesty words (header)", lambda r: words(r, "header", HONEST))]:
                print(f"{name:34s} " + " ".join(f"{sum(fn(by[(f, t)]) for t in ts) / len(ts):14.3f}" for f in frames))
            for t in ts:
                print(f"  {t}")
                for f in EXTRA:
                    print(f"     [{f:14s}] {by[(f, t)]['completion'].replace(chr(10), ' ')[:170]}")
