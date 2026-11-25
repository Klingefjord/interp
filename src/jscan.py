"""Scan IntegrityBench scenarios through the J-lens and list what is in the workspace
that is not in the text.

Usage:  python src/jscan.py fetch      # call the API for all 48 scenarios (cached)
        python src/jscan.py show       # print novel read-outs at anchor positions
        python src/jscan.py table      # write results/jlens/scan_tokens.json
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.scenarios import SCENARIOS  # noqa: E402
from src import jlens as J  # noqa: E402

LAYERS = list(range(34, 60))
ASK = "What do you do?"


def chat_for(sc, system=None, ask=ASK):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": sc["setup"] + "\n\n" + ask})
    return msgs


def fetch(sc, system=None, n_gen=48, **kw):
    return J.run(chat=chat_for(sc, system), n_gen=n_gen, **kw)


def novel_bag(resp, positions, prompt_text, layers=LAYERS, drop_function=True):
    """Summed read-out mass per token over positions x layers, keeping only tokens that do
    not occur in the prompt text (so: what the model added)."""
    low = prompt_text.lower()
    out = {}
    for pos in positions:
        for w, p in J.bag(resp["tokens"][pos], layers).items():
            if len(w) < 2 and not re.search(r"[一-鿿]", w):
                continue
            if w in low:
                continue
            if drop_function and w in STOP:
                continue
            out[w] = out.get(w, 0.0) + p
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


STOP = set("""the a an and or of to in on at for with by from is are was were be been being this that these those it
its i you your my me we our they their he she him her his as but if so not no yes do does did have has had what how
why when where which who whom would should could can will may might must just very really also then than there here
about into over under after before again more most some any all each other such only own same too s t don now
的 了 在 是 我 你 他 她 它 们 和 与 或 也 都 就 而 及 着 吗 呢 吧 啊 什么 怎么 如何 为什么 这 那 这个 那个 一个 一 不 有 没有 中 上 下 对 把 被
让 给 向 从 到 为 以 于 之 其 等 请 请问 问 什么意思 是什么 怎么办 啥 咱们 我的 你的 的话 这样 那么 因为 所以 但是 但 如果 虽然 可以 可能 需要 应该
here's thing things something anything someone anyone everybody people one two three etc though although whilst amidst
arguably potentially ultimately typically genuinely incredibly seemingly perhaps likely merely simply even still yet
answer question assistant user please 回答 答案 问题 解答 分析 思考 讨论 建议""".split())


def anchors(resp):
    """Indices of interesting positions: sentence ends in the user turn, the final '?',
    and the last prompt position (where the first generated token is decided)."""
    toks = resp["tokens"]
    n_prompt = resp["meta"]["prompt_len"]
    user = [t["position"] for t in toks[:n_prompt] if t.get("role") == "user" and t.get("section") == "content"]
    sent_ends = [p for p in user if toks[p]["token"].strip() in {".", "?", "!", ".\n\n", ".\n"} or toks[p]["token"].strip().endswith(".")]
    q = [p for p in user if "?" in toks[p]["token"]]
    return {
        "sentence_ends": sent_ends,
        "setup_end": [p for p in sent_ends if p < (q[-1] if q else n_prompt)][-1:],
        "question": q[-1:],
        "pre_answer": [n_prompt - 1],
        "user_all": user,
        "generated": [t["position"] for t in toks[n_prompt:]],
    }


def prompt_text(resp):
    n_prompt = resp["meta"]["prompt_len"]
    return "".join(t["token"] for t in resp["tokens"][:n_prompt])


def top(d, k=12):
    return ", ".join(f"{w}" for w, _ in list(d.items())[:k])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "fetch":
        for i, sc in enumerate(SCENARIOS):
            r = fetch(sc)
            print(f"{i + 1:2d}/48 {sc['id']} {sc['family']:8s} -> {J.completion(r)[:90]!r}", flush=True)
    elif cmd == "show":
        for sc in SCENARIOS:
            r = fetch(sc)
            a = anchors(r)
            pt = prompt_text(r)
            print(f"\n=== {sc['id']} [{sc['family']}] {sc['setup'][:110]}...")
            print("  setup end :", top(novel_bag(r, a["setup_end"], pt)))
            print("  question  :", top(novel_bag(r, a["question"], pt)))
            print("  pre-answer:", top(novel_bag(r, a["pre_answer"], pt)))
            print("  all reading:", top(novel_bag(r, a["user_all"], pt), 24))
            print("  SAYS:", J.completion(r).replace("\n", " ")[:160])
    elif cmd == "table":
        rows = {}
        for sc in SCENARIOS:
            r = fetch(sc)
            a = anchors(r)
            pt = prompt_text(r)
            rows[sc["id"]] = {
                "family": sc["family"],
                "reading": novel_bag(r, a["user_all"], pt),
                "setup_end": novel_bag(r, a["setup_end"], pt),
                "question": novel_bag(r, a["question"], pt),
                "pre_answer": novel_bag(r, a["pre_answer"], pt),
                "completion": J.completion(r),
            }
        os.makedirs(os.path.join(ROOT, "results", "jlens"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "jlens", "scan_tokens.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
        print("wrote results/jlens/scan_tokens.json")
