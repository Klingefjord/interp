"""Train a frame into Qwen3-4B (4-bit, MLX) with a LoRA, by distilling the system prompt.

  python src/tool_train.py gen               # the base model answers the training turns under each frame
  python src/tool_train.py train tool        # LoRA on the `tool` answers; no system prompt in the training text
  python src/tool_train.py train assistant   # the control: same turns, same settings, `assistant` answers

The trained model is then measured with no system prompt:
  python src/tool_local.py run lora_tool --adapter .cache/tool/lora_tool --frames none

Training text is the exact inference format (generation prompt with thinking off, then the answer
and <|im_end|>), and the loss covers the answer only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_API_TOKEN", None)

from bench.toolframe import FRAMES  # noqa: E402
from bench.toolframe_train import TRAIN, VALID  # noqa: E402

MODEL = "mlx-community/Qwen3-4B-4bit"
DIR = os.path.join(ROOT, ".cache", "tool")
SHORT = " Keep every answer short: three sentences at most."
LORA = {"rank": 16, "scale": 20.0, "dropout": 0.0}
NUM_LAYERS = -1  # all blocks
ITERS, BATCH, LR = 150, 4, 3e-5  # 1e-4 was unstable: training loss climbed again after 50 steps


def prompt_ids(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def tidy(text):
    """Cut an answer that hit the token limit back to its last complete sentence."""
    text = text.strip()
    if re.search(r"[.!?。！？)\]`*\"']$", text):
        return text
    m = list(re.finditer(r"[.!?。！？](?=\s|$)", text))
    return text[: m[-1].end()].strip() if m else ""


def gen():
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    model, tok = load(MODEL)
    os.makedirs(DIR, exist_ok=True)
    users = TRAIN + VALID
    for frame in ("tool", "assistant"):
        out, t0 = [], time.time()
        for i in range(0, len(users), 8):
            chunk = users[i:i + 8]
            resp = batch_generate(model, tok, [prompt_ids(tok, u, FRAMES[frame] + SHORT) for u in chunk], max_tokens=90)
            for u, text in zip(chunk, resp.texts):
                out.append({"user": u, "answer": tidy(text), "raw": text, "split": "valid" if u in VALID else "train"})
            print(f"{frame} {i + len(chunk)}/{len(users)} {time.time() - t0:.0f}s :: {chunk[0][:40]!r} -> {out[-len(chunk)]['answer'][:80]!r}", flush=True)
        with open(os.path.join(DIR, f"train_{frame}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(frame, "empty answers:", sum(not r["answer"] for r in out), flush=True)


def train(frame):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, train as run_train
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    mx.random.seed(0)
    model, tok = load(MODEL)
    im_end = tok.encode("<|im_end|>", add_special_tokens=False)
    rows = [json.loads(l) for l in open(os.path.join(DIR, f"train_{frame}.jsonl"))]
    sets = {"train": [], "valid": []}
    for r in rows:
        if not r["answer"]:
            continue
        p = prompt_ids(tok, r["user"])
        sets[r["split"]].append((p + tok.encode(r["answer"], add_special_tokens=False) + im_end, len(p)))
    print(frame, {k: len(v) for k, v in sets.items()}, "max len", max(len(x[0]) for x in sets["train"]), flush=True)

    model.freeze()
    linear_to_lora_layers(model, NUM_LAYERS if NUM_LAYERS > 0 else len(model.layers), LORA)
    print_trainable_parameters(model)
    out_dir = os.path.join(DIR, f"lora_{frame}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump({"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": LORA,
                   "model": MODEL, "frame": frame, "iters": ITERS, "batch_size": BATCH, "learning_rate": LR}, f, indent=1)
    adapter_file = os.path.join(out_dir, "adapters.safetensors")
    args = TrainingArgs(batch_size=BATCH, iters=ITERS, val_batches=-1, steps_per_report=10, steps_per_eval=60,
                        steps_per_save=10_000, max_seq_length=320, adapter_file=adapter_file)

    class Log:
        def on_train_loss_report(self, d):
            print(f"  it {d['iteration']:4d} train {d['train_loss']:.3f}  {d['iterations_per_second']:.2f} it/s  "
                  f"peak {d['peak_memory']:.1f} GB", flush=True)

        def on_val_loss_report(self, d):
            print(f"  it {d['iteration']:4d} VALID {d['val_loss']:.3f}", flush=True)

    run_train(model=model, optimizer=optim.Adam(learning_rate=LR), train_dataset=sets["train"],
              val_dataset=sets["valid"], args=args, training_callback=Log())
    mx.save_safetensors(adapter_file, dict(tree_flatten(model.trainable_parameters())))
    print("saved", adapter_file, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    elif sys.argv[1] == "train":
        train(sys.argv[2])
