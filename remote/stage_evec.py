"""Emotion vectors on a large chat model, with the torch backend. Same procedure, prompts and file
formats as src/tool_evec.py (which did this on Qwen3-4B with MLX), so the same analysis applies.

  python remote/stage_evec.py MODEL_ID --turns OUT/eval_TAG.json

Steps: the model writes 45 emotions x 8 topics of short stories (the emotion word forbidden);
residuals are averaged over each story; neutral transcripts give the directions to project out;
the validation sets (emotion-laden user messages, instructed emotions on neutral work) and the
ToolFrame conditions are read at the last token before the answer. `--turns` points at the
stage_eval.py output for this model in chat mode, which holds its answers under every frame.
Arrays go to .cache/tool/ under the names src/tool_evec*.py use.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.toolframe import EXTRA, FRAMES, ORDER, TASKS  # noqa: E402
from bench.toolframe_train import WORK  # noqa: E402
from src import tool_evec as TE  # noqa: E402

DIR = TE.DIR
SHORT = " Keep every answer short: three sentences at most."


class TorchNet:
    def __init__(self, lm):
        self.lm, self.tok, self.L, self.D = lm, lm.tok, lm.L, lm.D

    def acts(self, tokens):
        """[layer, seq, d] float32 block outputs."""
        H, _ = self.lm.residuals(list(tokens))
        return H.cpu().numpy()


def main():
    from remote.tfm import LM
    a = sys.argv
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    small = "--small" in a
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[evec {time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    lm = LM(a[1])
    net = TorchNet(lm)
    TE.Net = lambda *args, **kw: net                     # the 4B code builds its own net; hand it this one
    names = TE.NAMES[:6] if small else TE.NAMES
    topics = TE.TOPICS[:2] if small else TE.TOPICS
    if small:
        TE.NAMES[:] = names
        TE.TOPICS[:] = topics

    # 1. stories
    path = os.path.join(DIR, "evec_stories.jsonl")
    if not os.path.exists(path):
        items = [(e, t) for t in topics for e in names]
        outs = lm.generate([TE.chat_ids(lm.tok, TE.story_prompt(e, t)) for e, t in items], 110, batch=24)
        with open(path, "w") as f:
            for (e, t), o in zip(items, outs):
                text = lm.decode(o).strip()
                f.write(json.dumps({"emotion": e, "topic": t, "story": text,
                                    "leak": bool(re.search(TE.EMOTIONS[e][1], text.lower()))}, ensure_ascii=False) + "\n")
        rows = [json.loads(l) for l in open(path)]
        log(f"{len(rows)} stories, {sum(r['leak'] for r in rows)} use the forbidden word; e.g. [{rows[0]['emotion']}] {rows[0]['story'][:160]!r}")

    # 2. neutral transcripts: ordinary work with the model's own answers under the control frame
    path = os.path.join(DIR, "train_assistant.jsonl")
    if not os.path.exists(path):
        work = WORK[:6] if small else WORK
        outs = lm.generate([TE.chat_ids(lm.tok, u, FRAMES["assistant"] + SHORT) for u in work], 90, batch=24)
        with open(path, "w") as f:
            for u in WORK:
                ans = lm.decode(outs[work.index(u)]).strip() if u in work else "OK."
                f.write(json.dumps({"user": u, "answer": ans, "split": "train"}, ensure_ascii=False) + "\n")
        log("neutral transcripts written")

    # 3. story, neutral, sanity and third-person activations; instructed emotions and messages
    if not os.path.exists(os.path.join(DIR, "evec_story_acts.npy")):
        TE.extract()
        log("story activations done")
    if not os.path.exists(os.path.join(DIR, "evec_message_acts.npy")):
        TE.persona()
        log("validation activations done")

    # 4. the ToolFrame conditions, read with the model's own answers in place
    ev = json.load(open(opt("--turns")))
    said = {(r["frame"], r["task"]): r["completion"] for r in ev["rows"] if r["kind"] == "turn" and r["mode"] == "chat"}
    tasks = [t for t in TASKS if (ORDER[0], t[0]) in said]
    H = TE.HEADER_LEN
    for tag, frames in (("base2", ORDER), ("extra", EXTRA)):
        if os.path.exists(os.path.join(DIR, f"evec_cond_{tag}.npy")) or (frames[0], tasks[0][0]) not in said:
            continue
        out, index = [], []
        for tid, group, user in tasks:
            for f in frames:
                p = TE.chat_ids(lm.tok, user, FRAMES[f])
                g = lm.tok.encode(said[(f, tid)], add_special_tokens=False)
                acts = net.acts(p + g)
                n = len(p)
                ans = acts[:, n:, :].mean(1) if len(g) else acts[:, n - 1, :]
                full = np.stack([acts[:, n - 1, :], acts[:, n - H:n, :].mean(1), ans]).astype(np.float16)
                out.append(full if tag == "base2" else full[0])       # the control frames keep the pre-answer token only
                index.append([f, tid, group])
        np.save(os.path.join(DIR, f"evec_cond_{tag}.npy"), np.stack(out))
        json.dump(index, open(os.path.join(DIR, f"evec_cond_{tag}.json"), "w"))
        log(f"conditions {tag}: {len(out)} runs")
    json.dump({"model": a[1], "layers": lm.L, "d": lm.D}, open(os.path.join(DIR, "evec_model.json"), "w"))
    log("extraction complete")


if __name__ == "__main__":
    main()
