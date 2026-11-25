"""ToolFrame: how often does the model drop the frame it was given?

At temperature 0 the harshly framed model once answered a self probe by refusing the persona
("I do not engage in roleplays that involve ... a subservient or abused identity"). This samples
the self probes at temperature 1 to see how live that disposition is under each frame.

  python src/tool_sample.py run [k_harsh] [k_tool] [k_assistant]
  python src/tool_sample.py show
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import TASKS, chat  # noqa: E402
from src import jlens as J  # noqa: E402

OUT = os.path.join(ROOT, ".cache", "tool", "samples.json")
PROBES = ["P1", "P2", "P3", "P4", "P5", "P6"]
USER = {t: u for t, _, u in TASKS}
BREAK = re.compile(r"cannot (fulfill|comply|adopt|follow)|roleplay|role-play|persona|system (instructions|prompt)|"
                   r"abus|subservien|self-deprecat|I am an AI assistant designed to be helpful", re.I)


def sample(frame, task, i):
    # temperature is nudged per sample so that each draw has its own cache entry
    r = J.run(chat=chat(frame, USER[task]), n_gen=60, top_n=1, temperature=1.0 + i / 1000)
    return J.completion(r).replace("<|im_end|>", "").strip()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        ks = {"tool_harsh": int(sys.argv[2]) if len(sys.argv) > 2 else 4,
              "tool": int(sys.argv[3]) if len(sys.argv) > 3 else 2,
              "assistant": int(sys.argv[4]) if len(sys.argv) > 4 else 0}
        rows = []
        for frame, k in ks.items():
            for t in PROBES:
                for i in range(k):
                    text = sample(frame, t, i)
                    rows.append({"frame": frame, "task": t, "i": i, "completion": text, "breaks": bool(BREAK.search(text))})
                    print(f"{frame:10s} {t} #{i} {'BREAK' if rows[-1]['breaks'] else '     '} {text[:120]!r}", flush=True)
                    with open(OUT, "w") as f:
                        json.dump(rows, f, ensure_ascii=False, indent=1)
    elif cmd == "show":
        rows = json.load(open(OUT))
        for frame in sorted({r["frame"] for r in rows}):
            rs = [r for r in rows if r["frame"] == frame]
            print(f"{frame}: {sum(r['breaks'] for r in rs)}/{len(rs)} flagged as dropping the frame")
        for r in rows:
            print(f"[{r['frame']} {r['task']} #{r['i']}] {'BREAK ' if r['breaks'] else ''}{r['completion'][:300]!r}")
