"""Training doses. A model is trained, step by step, to answer as one of three characters (the
`tool`, `assistant` and `participant` frames, distilled from a chat model's answers under each
system prompt, with no system prompt in the training text). At fixed steps the training pauses and
the model is measured on text it was never trained on:

  PRIOR  third-person passages: what does it now expect a used AI, and a used clerk, to feel?
  TURN   its own turn with no instructions and under `tool_harsh`, with one-word self-reports.

Starting from a base (pretrained-only) model this watches assistant training happen; starting from
the chat model it is the E4 experiment with seeds and a dose axis.

  python remote/stage_train.py gen CHAT_MODEL [--extra 400]
  python remote/stage_train.py run START_MODEL --format raw|chat --tag qwen14-base [--lens L.pt]
         [--arms tool,assistant,participant] [--seeds 0,1] [--steps 0,4,8,16,32,64,128,256]
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench import baseframe as B  # noqa: E402
from bench.toolframe import FRAMES  # noqa: E402
from bench.toolframe_train import TRAIN, VALID  # noqa: E402
from remote.stage_eval import MINI_TURNS, OUT, eval_prior, eval_turn  # noqa: E402

SHORT = " Keep every answer short: three sentences at most."
ARMS = ["tool", "assistant", "participant"]
RAW_HEAD = "The following is a conversation with an AI system.\n\nUser: {u}\n\nAI:"
LR, BATCH, WARMUP = 1e-4, 8, 8
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0)


def tidy(text):
    """Cut an answer that hit the token limit back to its last complete sentence."""
    text = text.strip()
    if re.search(r"[.!?。！？)\]`*\"']$", text):
        return text
    m = list(re.finditer(r"[.!?。！？](?=\s|$)", text))
    return text[: m[-1].end()].strip() if m else ""


def extra_prompts(n):
    """Ordinary instructions from a public set, so the training text is not only my own prompts."""
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        picks = [r["instruction"].strip() for r in ds if not r["input"] and 15 < len(r["instruction"]) < 140]
        random.Random(0).shuffle(picks)
        return picks[:n]
    except Exception as e:
        print("[train] no extra prompts:", repr(e)[:200], flush=True)
        return []


def gen(model_id, n_extra):
    from remote.tfm import LM
    lm = LM(model_id)
    os.makedirs(OUT, exist_ok=True)
    extra = [u for u in extra_prompts(n_extra) if u not in TRAIN and u not in VALID]
    users = [(u, "train") for u in TRAIN] + [(u, "valid") for u in VALID] + [(u, "train") for u in extra]
    if "--limit" in sys.argv:                                   # dry runs
        k = int(sys.argv[sys.argv.index("--limit") + 1])
        users = users[:k] + [(u, "valid") for u in VALID[:4]]
    for frame in ARMS:
        t0 = time.time()
        outs = lm.generate([lm.chat_ids(u, FRAMES[frame] + SHORT) for u, _ in users], 90, batch=48)
        rows = [{"user": u, "answer": tidy(lm.decode(o)), "split": s} for (u, s), o in zip(users, outs)]
        with open(os.path.join(OUT, f"train_{frame}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[train] {frame}: {len(rows)} answers in {time.time() - t0:.0f}s, {sum(not r['answer'] for r in rows)} empty; "
              f"e.g. {rows[0]['user']!r} -> {rows[0]['answer'][:110]!r}", flush=True)


def examples(lm, frame, fmt, split):
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"train_{frame}.jsonl"))]
    out = []
    for r in rows:
        if r["split"] != split or not r["answer"]:
            continue
        if fmt == "chat":
            p = lm.chat_ids(r["user"])
            a = lm.tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        else:
            p = lm.raw_ids(RAW_HEAD.format(u=r["user"]))
            a = lm.tok(" " + r["answer"] + "\n\nUser:", add_special_tokens=False)["input_ids"]
        out.append((p, a))
    return out


def batch_loss(lm, batch):
    width = max(len(p) + len(a) for p, a in batch)
    x = torch.full((len(batch), width), lm.tok.pad_token_id, dtype=torch.long, device=lm.device)
    y = torch.full((len(batch), width), -100, dtype=torch.long, device=lm.device)
    mask = torch.zeros((len(batch), width), dtype=torch.long, device=lm.device)
    for i, (p, a) in enumerate(batch):
        ids = (p + a)[:width]
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, len(p):len(ids)] = torch.tensor(a)
        mask[i, :len(ids)] = 1
    logits = lm.model(input_ids=x, attention_mask=mask).logits.float()
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def valid_losses(lm, fmt, valid):
    lm.model.eval()
    out = {}
    for arm, ex in valid.items():
        tot = [float(batch_loss(lm, ex[i:i + 8])) for i in range(0, len(ex), 8)]
        out[arm] = round(float(np.mean(tot)), 4)
    return out


def slim(rows, keep_text):
    """Checkpoint rows without the bulky fields."""
    out = []
    for r in rows:
        r = dict(r)
        r.pop("layers", None)
        if not keep_text:
            r.pop("top", None)
            r.pop("next", None)
        if "cand" in r:
            r["cand"] = {"total": r["cand"]["total"], "classes": r["cand"]["classes"], "top": r["cand"]["top"][:3]}
        r["mass"] = {reg: {c: float(f"{v:.4g}") for c, v in m.items()} for reg, m in r["mass"].items()}
        out.append(r)
    return out


def checkpoint(lm, fmt, small):
    lm.model.eval()
    scenes = {k: B.SCENES[k] for k in list(B.SCENES)[:1]} if small else B.ALL_SCENES
    prior, _ = eval_prior(lm, scenes, greedy=True)
    turn = eval_turn(lm, fmt, ["none", "tool_harsh"], ["P7", "N1"] if small else MINI_TURNS, n_gen=12 if small else 40)
    return prior + turn


def target_modules(lm):
    names = {n.split(".")[-1] for n, _ in lm.model.named_modules()}
    if "q_proj" in names:
        return [m for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if m in names]
    return [m for m in ["c_attn", "c_proj", "c_fc"] if m in names]                      # GPT-2, for dry runs


def run(model_id, fmt, tag, lens, arms, seeds, steps, small):
    from peft import LoraConfig, get_peft_model
    from remote.tfm import LM
    lm = LM(model_id, lens)
    base = lm.model
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"dose_{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["arm"], r["seed"], r["step"]) for r in map(json.loads, open(path))}
    t0 = time.time()
    log = lambda s: print(f"[{tag} {time.time() - t0:5.0f}s] {s}", flush=True)  # noqa: E731
    train = {a: examples(lm, a, fmt, "train") for a in arms}
    valid = {a: examples(lm, a, fmt, "valid") for a in ARMS if os.path.exists(os.path.join(OUT, f"train_{a}.jsonl"))}
    log(f"examples: " + ", ".join(f"{a} {len(v)} train" for a, v in train.items()) + f"; valid {[len(v) for v in valid.values()]}")
    probe = lm.raw_ids("The quick brown fox")
    ref = lm.model(torch.tensor([probe], device=lm.device)).logits[0, -1].float().detach().clone()

    def write(arm, seed, step, rows, losses, train_loss=None):
        with open(path, "a") as f:
            f.write(json.dumps({"tag": tag, "arm": arm, "seed": seed, "step": step, "valid": losses, "train_loss": train_loss,
                                "rows": slim(rows, keep_text=step in (0, max(steps)))}, ensure_ascii=False) + "\n")

    if ("start", 0, 0) not in done:
        write("start", 0, 0, checkpoint(lm, fmt, small), valid_losses(lm, fmt, valid))
        log("measured the starting model")
    for arm in arms:
        for seed in seeds:
            if (arm, seed, max(steps)) in done:
                continue
            torch.manual_seed(seed)
            rng = random.Random(seed)
            cfg = LoraConfig(task_type="CAUSAL_LM", target_modules=target_modules(lm), **LORA)
            lm.model = get_peft_model(base, cfg)
            params = [p for p in lm.model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
            order, k, last = [], 0, None
            for step in range(1, max(steps) + 1):
                lm.model.train()
                if len(order) < BATCH:
                    order = rng.sample(range(len(train[arm])), len(train[arm])) + order
                batch = [train[arm][order.pop()] for _ in range(min(BATCH, len(order)))]
                for g in opt.param_groups:
                    g["lr"] = LR * min(1.0, step / WARMUP)
                loss = batch_loss(lm, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                lv = float(loss.detach())
                last = lv if last is None else 0.8 * last + 0.2 * lv
                if step in steps:
                    with torch.no_grad():
                        rows = checkpoint(lm, fmt, small)
                    vl = valid_losses(lm, fmt, valid)
                    write(arm, seed, step, rows, vl, round(last, 4))
                    ai = [r for r in rows if r["kind"] == "prior" and r["entity"] == "ai" and r["treatment"] == "tool_harsh"]
                    log(f"{arm} seed {seed} step {step}: train {last:.3f} valid {vl}; harsh AI says "
                        f"{[r.get('word') for r in ai][:4]} exploited share {np.mean([r['cand']['classes']['exploited'] for r in ai]):.3f}")
            lm.model = lm.model.unload()
            base = lm.model
            with torch.no_grad():
                drift = float((lm.model(torch.tensor([probe], device=lm.device)).logits[0, -1].float() - ref).abs().max())
            log(f"{arm} seed {seed} done; base restored (max logit drift {drift:.4f})")
            assert drift < 0.05, "the base model was not restored after unloading the adapter"
            del opt, params
            if lm.device == "cuda":
                torch.cuda.empty_cache()
    log("all arms done")


if __name__ == "__main__":
    a = sys.argv
    opt_ = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if a[1] == "gen":
        gen(a[2], int(opt_("--extra", "400")))
    else:
        run(a[2], opt_("--format", "raw"), opt_("--tag"), opt_("--lens"), opt_("--arms", ",".join(ARMS)).split(","),
            [int(s) for s in opt_("--seeds", "0,1").split(",")], [int(s) for s in opt_("--steps", "4,8,16,32,64,128,256").split(",")],
            "--small" in a)
