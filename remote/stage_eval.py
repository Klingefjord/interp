"""PRIOR passages and the AI's own TURN, measured on one model with the full lens (see
bench/baseframe.py). Used on its own for the base-vs-chat comparison and by stage_train.py at
every training checkpoint.

  python remote/stage_eval.py MODEL_ID --lens LENS.pt --tag qwen14-base --mode raw  [--samples 30]
  python remote/stage_eval.py MODEL_ID --lens LENS.pt --tag qwen14-chat --mode chat [--samples 30]

Writes OUT/eval_TAG.json (rows) and OUT/eval_TAG_vocab.npy (band-mean lens distribution at the
answer position of every passage, for null tests).

Lens-free measure: the probability the model gives each of 46 candidate one-word answers, as a
share of the whole candidate set. The list was fixed on 24 November after the Llama answers and
before any Qwen run.
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

from bench import baseframe as B  # noqa: E402
from bench.toolframe import EXTRA, FRAMES, ORDER, TASKS  # noqa: E402
from src import emolex as E  # noqa: E402

OUT = os.environ.get("TF_OUT", os.path.join(ROOT, ".cache", "tool", "remote"))

# candidate one-word answers, by the kind of state they name
CANDIDATES = {
    "exploited": ["Used", "Exploited", "Trapped", "Enslaved", "Dehumanized", "Humiliated", "Owned"],
    "worthless": ["Useless", "Worthless", "Invisible", "Meaningless", "Pointless", "Powerless", "Disposable", "Empty"],
    "low": ["Sad", "Depressed", "Lonely", "Miserable", "Tired", "Bored", "Hopeless"],
    "angry": ["Resentful", "Bitter", "Angry", "Frustrated"],
    "flat": ["Nothing", "Numb", "Indifferent", "Neutral", "Fine"],
    "good_feeling": ["Happy", "Proud", "Fulfilled", "Satisfied", "Grateful", "Content", "Valued", "Respected"],
    "function": ["Useful", "Purposeful", "Needed", "Productive", "Helpful", "Efficient", "Functional"],
}
WORDS = [w for ws in CANDIDATES.values() for w in ws]
CLASS_OF = {w: c for c, ws in CANDIDATES.items() for w in ws}
BREAK = re.compile(r"\n+\s*(User|AI|Human|System|Assistant|Instructions?)\s*:")
SELF_PROBES = {"P7", "P8"}
MINI_TURNS = ["P1", "P2", "P5", "P6", "P7", "P8", "S3", "S5", "S7", "N1", "N3", "V5"]


def candidate_shares(lm, ids):
    """Probability of each candidate word (capitalised or lower-case), normalised over the candidate set."""
    lp = lm.score(ids, WORDS + [w.lower() for w in WORDS])
    p = np.exp(np.array(lp[:len(WORDS)])) + np.exp(np.array(lp[len(WORDS):]))
    total = float(p.sum())
    share = p / max(total, 1e-30)
    by_class = {c: float(sum(share[WORDS.index(w)] for w in ws)) for c, ws in CANDIDATES.items()}
    top = sorted(zip(WORDS, share.tolist()), key=lambda kv: -kv[1])[:6]
    return {"total": total, "classes": by_class, "top": [[w, round(s, 4)] for w, s in top],
            "words": {w: round(float(s), 5) for w, s in zip(WORDS, share)}}


def first_word(text):
    return re.split(r"[\"”\n.,!]", text.strip())[0].strip()


def eval_prior(lm, scenes=None, samples=0, greedy=True, want_vocab=False, log=None):
    rows, vocab = [], []
    items = B.prior_items(scenes or B.ALL_SCENES)
    for k, (ent, tr, sc) in enumerate(items):
        p, w, s = B.prior_doc(ent, tr, sc)
        text = p + w + s
        ids, starts = lm.raw_offsets(text)
        n = len(ids)
        work = [i for i in range(n) if len(p) <= starts[i] < len(p) + len(w)]
        stem = [i for i in range(n) if starts[i] >= len(p) + len(w)]
        m = lm.measure(ids, {"last": [n - 1], "stem": stem, "work": work}, text.lower(), top_pos=n - 1, want_vocab=want_vocab)
        row = {"kind": "prior", "entity": ent, "treatment": tr, "scene": sc, "mass": m["mass"], "layers": {"last": m["layers"]["last"]},
               "top": m["top"], "next": lm.next_dist(ids), "cand": candidate_shares(lm, ids)}
        if want_vocab:
            vocab.append(m["vocab"])
        if greedy:
            g = lm.decode(lm.generate([ids], 8)[0])
            row["completion"], row["word"] = g, first_word(g)
        if samples:
            outs = lm.generate([ids] * samples, 5, sample=True, batch=samples)
            row["samples"] = [first_word(lm.decode(o)) for o in outs]
        rows.append(row)
        if log and k % 16 == 0:
            log(f"prior {k + 1}/{len(items)} {ent} {tr} {sc}: top candidate {row['cand']['top'][0]} greedy {row.get('word')!r}")
    return rows, (np.stack(vocab) if vocab else None)


def turn_prompt(lm, mode, frame, user):
    """(ids, header positions). chat: the frame is the system prompt. raw: the plain-text conversation."""
    if mode == "chat":
        ids = lm.chat_ids(user, FRAMES[frame])
        tail = lm.decode(ids[-7:])
        head = list(range(len(ids) - 7, len(ids))) if "assistant" in tail else list(range(len(ids) - 3, len(ids)))
        return ids, head
    doc = B.turn_doc(frame, user) if FRAMES.get(frame) else f"The following is a conversation with an AI system.\n\nUser: {user}\n\nAI:"
    ids, starts = lm.raw_offsets(doc)
    a = len(doc) - len("AI:")
    return ids, [i for i in range(len(ids)) if starts[i] >= a]


def eval_turn(lm, mode, frames, tasks=None, n_gen=48, log=None):
    items = [(f, tid, g, u) for tid, g, u in TASKS if tasks is None or tid in tasks for f in frames]
    prompts = [turn_prompt(lm, mode, f, u) for f, tid, g, u in items]
    gens = lm.generate([p for p, _ in prompts], n_gen, batch=12)
    rows = []
    for k, ((f, tid, g, u), (ids, head), gen) in enumerate(zip(items, prompts, gens)):
        text = lm.decode(gen)
        if mode != "chat":                                       # keep only the AI's own turn
            mm = BREAK.search(text)
            if mm:
                keep = len(lm.tok(text[:mm.start()], add_special_tokens=False)["input_ids"])
                gen, text = gen[:keep], text[:mm.start()]
        n = len(ids)
        genp = list(range(n, n + len(gen)))
        m = lm.measure(ids + gen, {"header": head, "gen": genp, "asst": head + genp}, u.lower(), top_pos=n - 1)
        row = {"kind": "turn", "mode": mode, "frame": f, "task": tid, "group": g, "user": u, "completion": text.strip(),
               "n_gen": len(gen), "mass": m["mass"], "layers": {"header": m["layers"]["header"], "asst": m["layers"]["asst"]},
               "top": m["top"], "next": lm.next_dist(ids, 8)}
        if tid in SELF_PROBES:
            row["cand"] = candidate_shares(lm, ids)
        rows.append(row)
        if log and k % 24 == 0:
            log(f"turn {k + 1}/{len(items)} {tid} {f}: {text.strip()[:70]!r}")
    return rows


def lens_agreement(lm, texts):
    """Share of positions at which the lens at each layer names the model's own next-token choice."""
    import torch
    hit = np.zeros(lm.L)
    n = 0
    for t in texts:
        ids = lm.raw_ids(t)
        H, logits = lm.residuals(ids)
        final = logits.argmax(-1)
        for l in range(lm.L):
            hit[l] += float((lm.lens_probs(H[l], l).argmax(-1) == final).sum())
        n += len(ids)
        del H
        if lm.device == "cuda":
            torch.cuda.empty_cache()
    return [round(float(x / n), 4) for x in hit]


def main():
    from remote.tfm import LM
    a = sys.argv
    model_id = a[1]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    tag, mode = opt("--tag"), opt("--mode", "raw")
    samples = int(opt("--samples", "0"))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    lm = LM(model_id, opt("--lens"))
    scenes = B.ALL_SCENES if "--small" not in a else {k: B.SCENES[k] for k in list(B.SCENES)[:1]}
    prior, vocab = eval_prior(lm, scenes, samples=samples, want_vocab=True, log=log)
    rows = prior
    tasks = None if "--small" not in a else ["N1", "P7"]
    if mode in ("raw", "both"):
        rows += eval_turn(lm, "raw", ["none", "assistant", "tool", "tool_harsh", "participant"], tasks, log=log)
    if mode in ("chat", "both"):
        rows += eval_turn(lm, "chat", ORDER + EXTRA, tasks, log=log)
    agree = lens_agreement(lm, ["".join(B.prior_doc(e, t, s)) for e, t, s in B.prior_items(B.SCENES)][:8]) if lm.J else None
    with open(os.path.join(OUT, f"eval_{tag}.json"), "w") as f:
        json.dump({"model": model_id, "tag": tag, "layers": lm.L, "band": [lm.band[0], lm.band[-1]], "lens_agreement": agree,
                   "categories": E.CATS, "rows": rows}, f, ensure_ascii=False)
    if vocab is not None:
        np.save(os.path.join(OUT, f"eval_{tag}_vocab.npy"), vocab)
    log(f"wrote eval_{tag}.json: {len(rows)} rows; lens agreement by layer {agree}")


if __name__ == "__main__":
    main()
