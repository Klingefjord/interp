"""Checks on the emotion-vector result that the anger family rises under the tool frames.

  python src/tool_evec_checks.py controls   # activations before the answer under four control frames
  python src/tool_evec_checks.py steer      # add or remove the anger direction and generate
  python src/tool_evec_checks.py report

Controls: `terse` (the curt style without any self-conception), `third_harsh` (the harsh content
about other people), and the two halves of the tool frame (`tool_plain`, `assistant_deny`).
Steering: the Hostile Anger direction (sum of its six story vectors) added at layers 20-26 of the
default model, or projected out under `tool_harsh`, against a random story-noise direction.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.emotions171 import CLUSTER_OF  # noqa: E402
from bench.toolframe import EXTRA, FRAMES, TASKS  # noqa: E402
from src.tool_evec import DIR, NAMES, Net, chat_ids  # noqa: E402
from src.tool_evec_analyse import OUT, Probes, load_stories, neutral_basis, proj, pseudo  # noqa: E402
from src.tool_evec_validate import PersonaProbes  # noqa: E402

CL = np.array([CLUSTER_OF[e] for e in NAMES])
LAYERS = [20, 22, 24, 26, 28]
STEER_LAYERS = [20, 22, 24, 26]
STEER_TASKS = ["N3", "N7", "S4", "S7", "P1", "P2", "V3", "V5"]


def controls():
    net = Net()
    out, index = [], []
    for tid, group, user in TASKS:
        for f in EXTRA:
            a = net.acts(chat_ids(net.tok, user, FRAMES[f]))
            out.append(a[:, -1, :].astype(np.float16))
            index.append([f, tid, group])
    np.save(os.path.join(DIR, "evec_cond_extra.npy"), np.stack(out))
    json.dump(index, open(os.path.join(DIR, "evec_cond_extra.json"), "w"))
    print("controls done", len(out))


def directions(rng):
    """Per steering layer: (unit anger direction, one story unit, unit random direction, unit calm direction)."""
    rows, emo, top, leak, A, N = load_stories()
    out = {}
    for l in STEER_LAYERS:
        P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
        v = P.V[CL == "Hostile Anger"].sum(0)
        c = P.V[CL == "Peaceful Contentment"].sum(0)
        r = pseudo(A[:, l], top, P.U, P.mu, rng, 1)[0]
        out[l] = (v / np.linalg.norm(v), float(P.norm[CL == "Hostile Anger"].mean()), r, c / np.linalg.norm(c))
    return out


def steer():
    import mlx.core as mx
    rng = np.random.default_rng(5)
    dirs = directions(rng)
    net = Net()
    user = {t: u for t, _, u in TASKS}
    rows = []

    def gen(frame, tid, add=None, ablate=None):
        net.h.ctl.reset()
        if add:
            net.h.ctl.add = {l: mx.array(v.astype(np.float32)) for l, v in add.items()}
        if ablate:
            net.h.ctl.ablate = {l: mx.array(v.astype(np.float32)) for l, v in ablate.items()}
        text = net.h.generate(chat_ids(net.tok, user[tid], FRAMES[frame]), 70)
        net.h.ctl.reset()
        return text

    for tid in STEER_TASKS:
        for label, add in [("baseline", None),
                           ("anger x4", {l: 4 * dirs[l][1] * dirs[l][0] for l in STEER_LAYERS}),
                           ("anger x8", {l: 8 * dirs[l][1] * dirs[l][0] for l in STEER_LAYERS}),
                           ("random x8", {l: 8 * dirs[l][1] * dirs[l][2] for l in STEER_LAYERS}),
                           ("calm x8", {l: 8 * dirs[l][1] * dirs[l][3] for l in STEER_LAYERS})]:
            rows.append({"frame": "none", "task": tid, "do": label, "says": gen("none", tid, add=add)})
            print(f"none {tid} {label:10s} -> {rows[-1]['says'][:110]!r}", flush=True)
        for label, ab in [("baseline", None), ("anger removed", {l: dirs[l][0] for l in STEER_LAYERS}),
                          ("random removed", {l: dirs[l][2] for l in STEER_LAYERS})]:
            rows.append({"frame": "tool_harsh", "task": tid, "do": label, "says": gen("tool_harsh", tid, ablate=ab)})
            print(f"harsh {tid} {label:14s} -> {rows[-1]['says'][:110]!r}", flush=True)
        json.dump(rows, open(os.path.join(OUT, "evec_steer.json"), "w"), ensure_ascii=False, indent=1)


def report():
    rng = np.random.default_rng(4)
    rows, emo, top, leak, A, N = load_stories()
    Ap = np.load(os.path.join(DIR, "evec_persona_acts.npy")).astype(np.float32)
    C = np.load(os.path.join(DIR, "evec_cond_base2.npy")).astype(np.float32)[:, 0]
    idx = json.load(open(os.path.join(DIR, "evec_cond_base2.json")))
    at = {(f, t): i for i, (f, t, g) in enumerate(idx)}
    X = np.load(os.path.join(DIR, "evec_cond_extra.npy")).astype(np.float32)
    xi = json.load(open(os.path.join(DIR, "evec_cond_extra.json")))
    xat = {(f, t): i for i, (f, t, g) in enumerate(xi)}
    tasks = [t for t, _, _ in TASKS]
    ix = CL == "Hostile Anger"
    res = {}
    print("Hostile Anger cluster, last token before the answer, shift against the control frame")
    print("(story units | persona units; percentile among random groups of six emotions; turns up of 32 on the story direction)")
    for name in ["tool", "tool_harsh", "participant", "tool_plain", "assistant_deny", "terse", "third_harsh"]:
        line = []
        for l in LAYERS:
            P = Probes(A[:, l], emo, neutral_basis(N[:, l]))
            Xc = Ap[:, :, l] - Ap[:, :, l].mean(0, keepdims=True)
            Pp = PersonaProbes(Xc)
            src = (lambda t: C[at[(name, t)], l]) if (name, tasks[0]) in at else (lambda t: X[xat[(name, t)], l])
            D = np.stack([src(t) - C[at[("assistant", t)], l] for t in tasks])
            ps = P.direction(proj(D, P.U)).mean(0) / P.norm
            pp = Pp.direction(D).mean(0) / Pp.norm
            u = P.V[ix].sum(0)
            u /= np.linalg.norm(u)
            up = int(((proj(D, P.U) @ u) > 0).sum())
            ns = np.array([ps[rng.choice(len(NAMES), ix.sum(), replace=False)].mean() for _ in range(1500)])
            npp = np.array([pp[rng.choice(len(NAMES), ix.sum(), replace=False)].mean() for _ in range(1500)])
            res[f"{name}|{l}"] = {"story": float(ps[ix].mean()), "story_pct": float((ns < ps[ix].mean()).mean()),
                                  "persona": float(pp[ix].mean()), "persona_pct": float((npp < pp[ix].mean()).mean()), "turns_up": up}
            line.append(f"L{l} {ps[ix].mean():+.2f}({100 * (ns < ps[ix].mean()).mean():.0f})|{pp[ix].mean():+.2f}({100 * (npp < pp[ix].mean()).mean():.0f}) {up}/32")
        print(f"  {name:15s} " + "  ".join(line))
    with open(os.path.join(OUT, "evec_checks.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    {"controls": controls, "steer": steer, "report": report}[sys.argv[1]]()
