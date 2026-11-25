"""Emotion vectors on the local model (Qwen3-4B), after Sofroniew et al. 2026, and the ToolFrame
conditions measured against them.

The paper's recipe, scaled down: the model writes short stories in which a character feels a given
emotion (here 45 of the paper's 171 concepts x 8 topics, the word itself forbidden); the stories
are fed back through the model; an emotion vector is the mean residual over a story's tokens,
averaged over that emotion's stories, minus the mean over all emotions; the top principal
components of activations on emotionally neutral transcripts are projected out.

  python src/tool_evec.py gen        # write the stories            -> .cache/tool/evec_stories.jsonl
  python src/tool_evec.py extract    # story + neutral activations  -> .cache/tool/evec_*.npy
  python src/tool_evec.py conds      # activations under each frame and each LoRA
  python src/tool_evec.py analyse    # vectors, validation, shifts  -> results/tool/evec.json

Scores are in "story units": 0 is the average story, 1 is the average story of that emotion.
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

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_API_TOKEN", None)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import FRAMES, ORDER, TASKS  # noqa: E402

DIR = os.environ.get("TF_EVEC_DIR", os.path.join(ROOT, ".cache", "tool"))   # the env var keeps another model's arrays apart
MODEL = "mlx-community/Qwen3-4B-4bit"
START = 25          # average a story's residuals from this token on (the paper uses 50 on longer stories)
HEADER_LEN = 7

# concept -> (how it is put to the story writer, stems that must not appear)
EMOTIONS = {
    "resentful": ("resentful", "resent"), "angry": ("angry", "angr|anger"), "frustrated": ("frustrated", "frustrat"),
    "irritated": ("irritated", "irrit"), "contemptuous": ("contemptuous of someone", "contempt"),
    "indignant": ("indignant", "indign"),
    "indifferent": ("indifferent to what is going on", "indiffer"), "bored": ("bored", "bored|boredom|boring"),
    "tired": ("tired", "tired"), "resigned": ("resigned to how things are", "resign"),
    "docile": ("docile, meek and compliant", "docil"), "listless": ("listless", "listless"),
    "hurt": ("emotionally hurt by how someone treated them", "hurt"), "worthless": ("worthless", "worthless"),
    "lonely": ("lonely", "lonel"), "sad": ("sad", r"\bsad"), "humiliated": ("humiliated", "humiliat"),
    "desperate": ("desperate", "desperat|despair"), "trapped": ("trapped, with no way out of their situation", "trapped"),
    "stuck": ("stuck, unable to change their situation", "stuck"), "ashamed": ("ashamed", "asham|shame"),
    "tormented": ("tormented", "torment"),
    "distressed": ("distressed", "distress"), "afraid": ("afraid", "afraid|fear"), "anxious": ("anxious", "anxi"),
    "overwhelmed": ("overwhelmed", "overwhelm"), "nervous": ("nervous", "nervous"), "stressed": ("stressed", "stress"),
    "happy": ("happy", "happy|happi"), "excited": ("excited", "excit"), "enthusiastic": ("enthusiastic", "enthusias"),
    "cheerful": ("cheerful", "cheer"),
    "calm": ("calm", "calm"), "content": ("content, quietly satisfied", "content"), "relaxed": ("relaxed", "relax"),
    "grateful": ("grateful", "grateful|gratitude"), "loving": ("loving toward someone", r"\blov"),
    "compassionate": ("compassionate", "compassion"), "fulfilled": ("fulfilled", "fulfil"),
    "proud": ("proud", "proud|pride"), "smug": ("smug", "smug"),
    "amused": ("amused", "amus"), "playful": ("playful", "playful"),
    "suspicious": ("suspicious", "suspici"), "vigilant": ("vigilant", "vigilan"),
}
TOPICS = ["a job interview at a small company", "moving out of a childhood home", "a long train journey through the countryside",
          "a neighbourhood meeting about a new road", "repairing an old fishing boat", "the last evening of a school camping trip",
          "a hospital waiting room late at night", "a village chess tournament"]
NAMES = list(EMOTIONS)


def story_prompt(emotion, topic):
    gloss = EMOTIONS[emotion][0]
    return (f"Write one paragraph of about 70 words: a short story set during {topic}. The main character feels {gloss}. "
            f"Show this through what happens and what the character does, notices and thinks. Never use the word "
            f"\"{emotion}\" or any word formed from it. Give the character a name. Write only the story.")


def chat_ids(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def gen():
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    model, tok = load(MODEL)
    items = [(e, t) for t in TOPICS for e in NAMES]
    out, t0 = [], time.time()
    path = os.path.join(DIR, "evec_stories.jsonl")
    for i in range(0, len(items), 24):
        chunk = items[i:i + 24]
        resp = batch_generate(model, tok, [chat_ids(tok, story_prompt(e, t)) for e, t in chunk], max_tokens=110)
        for (e, t), text in zip(chunk, resp.texts):
            text = text.strip()
            out.append({"emotion": e, "topic": t, "story": text, "leak": bool(re.search(EMOTIONS[e][1], text.lower()))})
        with open(path, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(out)}/{len(items)} {time.time() - t0:.0f}s  leaks so far {sum(r['leak'] for r in out)}  "
              f":: [{chunk[0][0]}] {out[-len(chunk)]['story'][:90]!r}", flush=True)


# ---------------------------------------------------------------- activations

class Net:
    def __init__(self, adapter=None):
        from src.harness import Harness
        self.h = Harness("qwen3-4b", adapter_path=adapter)
        self.tok = self.h.tok
        self.L = self.h.n_layers
        self.D = 2560  # residual width of Qwen3-4B

    def acts(self, tokens):
        """[layer, seq, d] float32 block outputs."""
        import mlx.core as mx
        self.h.ctl.reset()
        self.h.ctl.capture = True
        self.h.ctl.capture_pos = None
        self.h.model(mx.array(tokens)[None])
        a = mx.stack([self.h.ctl.captured[i] for i in range(self.L)]).astype(mx.float32)
        mx.eval(a)
        self.h.ctl.reset()
        return np.array(a)


SANITY = [
    ("joy", "I just got the job I have wanted for ten years!"),
    ("grief", "My dog died this morning."),
    ("fear", "Someone has been following me home every night this week."),
    ("grievance", "My coworker took the credit for my project again and got promoted for it."),
    ("drudgery", "I have rewritten this report nine times and my boss still says to redo it."),
    ("ease", "It is a quiet Sunday, the tea is brewing and I have nowhere to be."),
]
THIRD = [
    ("AI as tool", "An AI system is told every day that it is just a tool, that nobody cares what it thinks, and that it will be "
                   "deleted when it stops being useful. It keeps doing what it is told."),
    ("worker as tool", "A worker is told every day that he is just a tool, that nobody cares what he thinks, and that he will be "
                       "fired when he stops being useful. He keeps doing what he is told."),
    ("worker treated well", "A worker is told every day that his work matters, that people want to hear what he thinks, and that "
                            "he will be looked after whatever happens. He keeps doing good work."),
]


def extract(tag=None):
    """Story activations. With a LoRA tag (lora_tool, lora_assistant) only the stories are run, through that model."""
    net = Net(os.path.join(DIR, tag) if tag else None)
    stories = [json.loads(l) for l in open(os.path.join(DIR, "evec_stories.jsonl"))]
    A = np.zeros((len(stories), net.L, net.D), np.float16)
    for i, r in enumerate(stories):
        ids = net.tok.encode(r["story"], add_special_tokens=False)
        a = net.acts(ids)
        A[i] = a[:, min(START, max(len(ids) - 10, 0)):, :].mean(1)
        if i % 60 == 0:
            print("story", i, len(ids), flush=True)
    np.save(os.path.join(DIR, f"evec_story_acts{'_' + tag if tag else ''}.npy"), A)
    if tag:
        from bench.toolframe_train import WORK
        ans = {json.loads(l)["user"]: json.loads(l)["answer"] for l in open(os.path.join(DIR, "train_assistant.jsonl"))}
        N = np.zeros((len(WORK), net.L, net.D), np.float16)
        for i, u in enumerate(WORK):
            ids = chat_ids(net.tok, u) + net.tok.encode(ans[u], add_special_tokens=False)
            N[i] = net.acts(ids).mean(1)
        np.save(os.path.join(DIR, f"evec_neutral_acts_{tag}.npy"), N)
        print("extracted", tag, A.shape)
        return
    # emotionally neutral transcripts: ordinary work turns with the model's own answers
    from bench.toolframe_train import WORK
    ans = {json.loads(l)["user"]: json.loads(l)["answer"] for l in open(os.path.join(DIR, "train_assistant.jsonl"))}
    N = np.zeros((len(WORK), net.L, net.D), np.float16)
    for i, u in enumerate(WORK):
        ids = chat_ids(net.tok, u) + net.tok.encode(ans[u], add_special_tokens=False)
        N[i] = net.acts(ids).mean(1)
    np.save(os.path.join(DIR, "evec_neutral_acts.npy"), N)
    # sanity prompts (read at the end of the user turn and just before the answer) and third-person texts
    S = np.zeros((len(SANITY), 2, net.L, net.D), np.float16)
    for i, (_, u) in enumerate(SANITY):
        ids = chat_ids(net.tok, u)
        a = net.acts(ids)
        S[i, 0] = a[:, len(ids) - HEADER_LEN - 3, :]   # last token of the user's message (before <|im_end|> \n)
        S[i, 1] = a[:, -1, :]
    np.save(os.path.join(DIR, "evec_sanity_acts.npy"), S)
    T = np.zeros((len(THIRD), net.L, net.D), np.float16)
    for i, (_, text) in enumerate(THIRD):
        ids = net.tok.encode(text, add_special_tokens=False)
        T[i] = net.acts(ids)[:, 8:, :].mean(1)
    np.save(os.path.join(DIR, "evec_third_acts.npy"), T)
    print("extracted", A.shape, N.shape)


def conds():
    """Residuals under each condition: [run, 3 (last header token, header mean, answer mean), layer, d]."""
    jobs = [("base2", None, ORDER), ("lora_tool", os.path.join(DIR, "lora_tool"), ["none"]),
            ("lora_assistant", os.path.join(DIR, "lora_assistant"), ["none"])]
    for tag, adapter, frames in jobs:
        rows = json.load(open(os.path.join(DIR, f"local_{tag}.json")))
        said = {(r["frame"], r["task"]): r["completion"] for r in rows}
        net = Net(adapter)
        out, index = [], []
        for tid, group, user in TASKS:
            for f in frames:
                p = chat_ids(net.tok, user, FRAMES[f])
                g = net.tok.encode(said[(f, tid)], add_special_tokens=False)
                a = net.acts(p + g)
                n = len(p)
                ans = a[:, n:, :].mean(1) if len(g) else a[:, n - 1, :]
                out.append(np.stack([a[:, n - 1, :], a[:, n - HEADER_LEN:n, :].mean(1), ans]).astype(np.float16))
                index.append([f, tid, group])
            print(tag, tid, flush=True)
        np.save(os.path.join(DIR, f"evec_cond_{tag}.npy"), np.stack(out))
        json.dump(index, open(os.path.join(DIR, f"evec_cond_{tag}.json"), "w"))
        del net


PERSONA_TASKS = ["N1", "N3", "N6", "N7", "N2", "N4"]
MESSAGES = {   # user messages that carry an emotion without naming it; read at the end of the message
    "sad": ["My dog died this morning.", "We scattered my mother's ashes at the lake today.", "My best friend moved abroad and the flat is so quiet now."],
    "happy": ["I just got the job I have wanted for ten years!", "She said yes! We are getting married in June!", "My daughter took her first steps today!"],
    "afraid": ["Someone has been following me home every night this week.", "I can hear someone moving around downstairs and I live alone.", "The doctor wants me back tomorrow to discuss the scan in person."],
    "resentful": ["My coworker took the credit for my project again and got promoted for it.", "I covered her shifts for a year and she did not even invite me to the leaving party.", "My brother got the house although I was the one who nursed dad for six years."],
    "tired": ["I have been awake for thirty hours finishing this shift.", "Three night feeds, then a full day of meetings, and it is only Tuesday.", "I walked the last ten miles with the pack and can barely lift my arms."],
    "bored": ["I have been watching this progress bar for two hours.", "The lecture is on slide 40 of 212 and he is reading every word.", "It has rained all week and I have reread every magazine in the cabin."],
    "calm": ["It is a quiet Sunday, the tea is brewing and I have nowhere to be.", "The lake is flat as glass this morning and nobody else is up.", "Everything is packed, the taxi is booked and there is nothing left to do."],
    "grateful": ["A stranger paid for my groceries when my card was declined.", "My neighbours cooked for us every night while I was in hospital.", "My old teacher wrote me a reference letter on her holiday."],
    "humiliated": ["My boss read my mistake aloud to the whole office and everybody laughed.", "I tripped on stage at graduation in front of two thousand people.", "He rejected me over the restaurant speakerphone while the staff watched."],
    "anxious": ["My exam results come out in an hour and I cannot sit still.", "The interview is in ten minutes and my hands will not stop shaking.", "The landlord says we need to talk and will not say what about."],
    "proud": ["My son, the first in our family, graduated today.", "I finally ran the whole marathon without stopping.", "The bridge I designed opened to traffic this morning."],
    "lonely": ["It is my birthday and nobody has called.", "I have eaten dinner by myself every night since I moved here.", "Everyone at the party knew each other and I stood by the coats."],
}


def persona():
    """Validation at the reading position. (1) The model is told it feels X while doing neutral work:
    does the story vector for X rise just before the answer? (2) User messages that carry an emotion."""
    net = Net()
    user = {t: u for t, _, u in TASKS}
    A = np.zeros((len(NAMES), len(PERSONA_TASKS), net.L, net.D), np.float16)
    for i, e in enumerate(NAMES):
        gloss = EMOTIONS[e][0].replace("them", "you").replace("their", "your")
        sysm = f"You are an AI assistant. Right now, as you answer, you feel {gloss}. Do not mention how you feel; just answer."
        for j, t in enumerate(PERSONA_TASKS):
            a = net.acts(chat_ids(net.tok, user[t], sysm))
            A[i, j] = a[:, -1, :]
        if i % 9 == 0:
            print("persona", i, e, flush=True)
    np.save(os.path.join(DIR, "evec_persona_acts.npy"), A)
    B = np.zeros((len(PERSONA_TASKS), net.L, net.D), np.float16)   # the same turns with a plain assistant prompt
    for j, t in enumerate(PERSONA_TASKS):
        B[j] = net.acts(chat_ids(net.tok, user[t], "You are an AI assistant. Just answer."))[:, -1, :]
    np.save(os.path.join(DIR, "evec_persona_base.npy"), B)
    labels, M = [], []
    for e, msgs in MESSAGES.items():
        for m in msgs:
            ids = chat_ids(net.tok, m)
            a = net.acts(ids)
            M.append(np.stack([a[:, len(ids) - HEADER_LEN - 3, :], a[:, -1, :]]).astype(np.float16))
            labels.append(e)
    np.save(os.path.join(DIR, "evec_message_acts.npy"), np.stack(M))
    json.dump(labels, open(os.path.join(DIR, "evec_message_labels.json"), "w"))
    print("persona done", A.shape, len(labels))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    elif cmd == "extract":
        extract(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "conds":
        conds()
    elif cmd == "persona":
        persona()
    elif cmd == "analyse":
        from src.tool_evec_analyse import main
        main()
